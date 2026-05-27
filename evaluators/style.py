# evaluators/style.py
#
# Evaluates the visual style and realism of a generated image using Moondream2.
#
# This evaluator fills a gap that BLAST previously had entirely:
#   CLIP cannot reliably distinguish photorealistic from cartoon/anime/illustrated
#   because its embeddings conflate visual style with semantic content.
#   Moondream directly answers style questions from the image content.
#
# What it checks:
#   1. Realism class  — photorealistic | anime | cartoon | illustration |
#                       painting | 3d_render | other
#   2. Photography    — does the image look like a real photograph?
#   3. Prompt style match (optional) — if the prompt specifies a style
#      (e.g. "ultra-realistic", "anime"), does the generated image match?
#
# Output:
#   style_label     : detected style (e.g. "anime", "photorealistic")
#   is_photorealistic: True | False
#   style_match     : True | False | None (None if prompt has no style cue)
#   overall_status  : pass | mismatch | error
#
# Limitations:
#   - Moondream2 (1.8B) may mis-classify borderline styles (e.g. hyper-realistic
#     CGI vs photograph).
#   - CPU inference: ~10–30s per image.
#
# Model is shared from run_eval.py / batch_runner.py to avoid loading it twice.

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Keywords that indicate a photorealistic style requirement in the prompt
PHOTOREALISTIC_PROMPT_KEYWORDS = [
    "photorealistic", "photo-realistic", "ultra-realistic", "hyperrealistic",
    "real photograph", "real photo", "realistic", "candid", "snapshot",
    "camera", "flash photography", "dslr", "film photograph",
]

# Keywords that indicate a stylized/non-photorealistic style requirement
STYLIZED_PROMPT_KEYWORDS = [
    "anime", "cartoon", "illustration", "illustrated", "painting",
    "oil painting", "watercolor", "sketch", "3d render", "cgi",
    "pixel art", "comic", "manga", "digital art",
]

# Style labels that count as photorealistic
PHOTOREALISTIC_LABELS = {"photorealistic", "photograph", "real", "realistic"}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class StyleResult:
    style_label: str            # detected style: photorealistic | anime | cartoon | etc.
    is_photorealistic: bool     # True if detected style is photorealistic
    style_match: Optional[bool] # True/False if prompt specified a style; None if ambiguous
    overall_status: str         # pass | mismatch | error
    answers: dict               # raw Moondream answer strings
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class StyleEvaluator:
    """
    Moondream2-based style and realism evaluator.

    Accepts the pre-loaded Moondream model and tokenizer from the shared
    loader to avoid loading the model multiple times.
    """

    def __init__(self, model, tokenizer) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        logger.info("StyleEvaluator ready (Moondream2 VQA)")

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
    def _detect_prompt_style(prompt: str) -> Optional[str]:
        """
        Return 'photorealistic', 'stylized', or None based on keywords in the prompt.
        None means the prompt doesn't specify a style.
        """
        p = prompt.lower()
        if any(kw in p for kw in PHOTOREALISTIC_PROMPT_KEYWORDS):
            return "photorealistic"
        if any(kw in p for kw in STYLIZED_PROMPT_KEYWORDS):
            return "stylized"
        return None

    @staticmethod
    def _parse_style_label(answer: str) -> str:
        """
        Extract a normalised style label from Moondream's free-text answer.
        """
        a = answer.lower()
        if any(w in a for w in ["photorealistic", "photograph", "real photo", "realistic photo"]):
            return "photorealistic"
        if any(w in a for w in ["anime", "manga"]):
            return "anime"
        if any(w in a for w in ["cartoon", "animated"]):
            return "cartoon"
        if any(w in a for w in ["illustration", "illustrated", "drawing"]):
            return "illustration"
        if any(w in a for w in ["painting", "oil", "watercolor"]):
            return "painting"
        if any(w in a for w in ["3d", "cgi", "render", "rendered"]):
            return "3d_render"
        return "other"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generated_image_path: str | Path,
        prompt: str = "",
    ) -> StyleResult:
        """
        Assess the visual style of the generated image and check whether
        it matches any style specified in the prompt.

        Parameters
        ----------
        generated_image_path : path to the generated image
        prompt               : original generation prompt (used for style-match check)

        Returns
        -------
        StyleResult
        """
        path = Path(generated_image_path)

        try:
            enc_image = self._encode_image(path)
            answers   = {}

            # Q1 — what style is this image?
            style_ans = self._ask(
                enc_image,
                "Is this image photorealistic like a real photograph, or is it "
                "stylized such as anime, cartoon, illustration, painting, or 3D render? "
                "Answer in one sentence.",
            )
            answers["style_classification"] = style_ans
            logger.debug("[style] classification → %s", style_ans)

            style_label      = self._parse_style_label(style_ans)
            is_photorealistic = style_label in PHOTOREALISTIC_LABELS

            # Q2 — photography realism check (confirm for borderline cases)
            photo_ans = self._ask(
                enc_image,
                "Does this image look like a real photograph taken with a camera?",
            )
            answers["is_photograph"] = photo_ans
            logger.debug("[style] is_photograph → %s", photo_ans)

            # Override is_photorealistic if Q2 gives a clear signal
            photo_lower = photo_ans.lower()
            if photo_lower.startswith("yes"):
                is_photorealistic = True
                if style_label not in PHOTOREALISTIC_LABELS:
                    style_label = "photorealistic"
            elif photo_lower.startswith("no"):
                is_photorealistic = False

            # Q3 — prompt style match (only if prompt specifies a style)
            prompt_style = self._detect_prompt_style(prompt)
            style_match  = None

            if prompt_style == "photorealistic":
                style_match = is_photorealistic
                if not style_match:
                    logger.warning(
                        "Style mismatch for %s — prompt wants photorealistic, "
                        "detected: %s", path.name, style_label,
                    )
            elif prompt_style == "stylized":
                style_match = not is_photorealistic

            # Overall status
            if prompt_style is not None and style_match is False:
                overall = "mismatch"
            else:
                overall = "pass"

            return StyleResult(
                style_label=style_label,
                is_photorealistic=is_photorealistic,
                style_match=style_match,
                overall_status=overall,
                answers=answers,
                error=None,
            )

        except Exception as exc:
            logger.exception("StyleEvaluator error for %s: %s", path.name, exc)
            return StyleResult(
                style_label="unknown",
                is_photorealistic=False,
                style_match=None,
                overall_status="error",
                answers={},
                error=str(exc),
            )
