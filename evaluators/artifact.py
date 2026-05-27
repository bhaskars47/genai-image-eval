# evaluators/artifact.py
#
# Detects common AI image artifacts using BLIP VQA.
#
# Why BLIP instead of CLIP?
#   CLIP scores the whole image as a single embedding — it cannot localise.
#   Asking CLIP "are the hands deformed?" on a headshot with no visible hands
#   returns a near-zero score that crosses the old 0.0 threshold, causing false
#   positives. BLIP actually answers questions about the image content,
#   so it can first confirm whether hands are visible before assessing them.
#
# Question design — "ask about breakage, not perfection":
#   Questions are phrased to detect OBVIOUS artifacts (flag_on="yes"),
#   not to confirm perfection (flag_on="no"). Asking "is it perfect?" causes
#   BLIP to answer "no" even for acceptable AI images, producing false positives.
#   Asking "is it obviously broken?" only flags genuine problems.
#
# Categories checked:
#   hands_fingers : deformed/extra/missing fingers — only if hands visible
#   face_structure: severe AI face deformations (extra eyes, melted features)
#   eyes          : obvious AI eye artifacts (extra, merged, severely unnatural)
#
# Limitations:
#   - BLIP-VQA-base may miss subtle artifacts — it's a first-pass filter.
#   - Treat "flagged" as a cue for human review, not a hard fail.
#   - CPU inference: ~2–5s per question.
#
# Model is shared from run_eval.py / batch_runner.py to avoid loading it twice.

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VQA questions per category
# ---------------------------------------------------------------------------

# Each category has:
#   presence  (optional) — asked first; if "no", category is skipped entirely
#   quality   — the main artifact check
#   flag_on   — "yes" means flag when BLIP answers yes (artifact present)
#               "no"  means flag when BLIP answers no  (quality not confirmed)
#
# All quality questions are now phrased to ask about OBVIOUS BREAKAGE,
# so flag_on="yes" for all of them. This avoids false positives from asking
# "is it perfect?" which BLIP answers "no" for even acceptable AI images.
ARTIFACT_CHECKS = [
    {
        "name":     "hands_fingers",
        "presence": "Are any hands or fingers clearly visible in this image?",
        "quality":  "Does this person have more than five fingers on one hand, or do their fingers appear fused, melted, or sprouting from the wrong places?",
        "flag_on":  "yes",   # flag if YES — obvious hand/finger deformity detected
    },
    {
        "name":     "face_structure",
        "quality":  "Does this person's face have any severe AI artifacts such as extra eyes, melted or merged facial features, or completely unnatural deformations?",
        "flag_on":  "yes",   # flag if YES — obvious face breakage detected
    },
    {
        "name":     "eyes",
        "quality":  "Does this person have more than two eyes, or do their eyes appear fused together, floating off the face, or growing from an unexpected location?",
        "flag_on":  "yes",   # flag if YES — structural eye placement error detected
    },
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ArtifactResult:
    overall_status: str         # pass | flagged | error
    flagged_categories: list    # category names that failed
    category_scores: dict       # {category: "pass" | "flagged" | "skipped"}
    answers: dict               # {category: raw Moondream answer strings}
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ArtifactEvaluator:
    """
    BLIP-VQA-based artifact detector.

    Questions are phrased to detect obvious breakage (flag_on="yes") rather
    than confirm perfection (flag_on="no"), which reduces false positives on
    AI-generated images that look acceptable but not pixel-perfect.

    Accepts the pre-loaded BLIP model and processor from the shared loader
    to avoid loading the model multiple times.
    """

    def __init__(self, model, tokenizer) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        logger.info("ArtifactEvaluator ready (BLIP VQA — flag_on=yes strategy)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: Path):
        """Load and return a PIL Image (BLIP processes image + question together)."""
        return Image.open(image_path).convert("RGB")

    def _ask(self, pil_image, question: str) -> str:
        """Run a single VQA query via BLIP and return the answer string."""
        import torch
        inputs = self._tokenizer(pil_image, question, return_tensors="pt")
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=64)
        answer = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.strip()

    @staticmethod
    def _is_yes(answer: str) -> bool:
        a = answer.lower().strip()
        return (
            a.startswith("yes")
            or a.startswith("i can see")
            or "yes" in a.split()[:6]
        )

    @staticmethod
    def _is_no(answer: str) -> bool:
        a = answer.lower().strip()
        return (
            a.startswith("no")
            or a.startswith("there are no")
            or a.startswith("this is a portrait")
            or "not visible" in a
            or "not present" in a
            or "cannot see" in a
            or "no " in a[:30]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, generated_image_path: str | Path) -> ArtifactResult:
        """
        Assess the generated image for common AI artifacts via Moondream VQA.

        Parameters
        ----------
        generated_image_path : path to the generated image

        Returns
        -------
        ArtifactResult
        """
        path = Path(generated_image_path)

        try:
            enc_image       = self._encode_image(path)
            category_scores = {}
            answers         = {}
            flagged         = []

            for check in ARTIFACT_CHECKS:
                name = check["name"]

                # Step 1 — presence check (optional, only for hands)
                if "presence" in check:
                    presence_ans = self._ask(enc_image, check["presence"])
                    answers[f"{name}_presence"] = presence_ans
                    logger.debug("[%s] presence → %s", name, presence_ans)

                    if self._is_no(presence_ans):
                        # Body part not in frame — skip, don't flag
                        category_scores[name] = "skipped"
                        logger.debug("[%s] skipped — not visible in frame", name)
                        continue

                # Step 2 — quality check
                quality_ans = self._ask(enc_image, check["quality"])
                answers[f"{name}_quality"] = quality_ans
                logger.debug("[%s] quality → %s", name, quality_ans)

                # flag_on="yes"  → flag when BLIP says YES (artifact present)
                # flag_on="no"   → flag when BLIP says NO  (quality not confirmed)
                flag_on = check.get("flag_on", "no")
                is_flagged = (
                    self._is_yes(quality_ans) if flag_on == "yes"
                    else not self._is_yes(quality_ans)
                )

                if is_flagged:
                    category_scores[name] = "flagged"
                    flagged.append(name)
                    logger.warning(
                        "Artifact flagged [%s] for %s — answer: %s",
                        name, path.name, quality_ans,
                    )
                else:
                    category_scores[name] = "pass"

            overall = "flagged" if flagged else "pass"

            return ArtifactResult(
                overall_status=overall,
                flagged_categories=flagged,
                category_scores=category_scores,
                answers=answers,
                error=None,
            )

        except Exception as exc:
            logger.exception("ArtifactEvaluator error for %s: %s", path.name, exc)
            return ArtifactResult(
                overall_status="error",
                flagged_categories=[],
                category_scores={},
                answers={},
                error=str(exc),
            )
