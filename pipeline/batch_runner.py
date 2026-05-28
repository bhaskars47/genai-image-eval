# pipeline/batch_runner.py
#
# Reads a CSV manifest, runs both evaluators per row, writes one JSON
# output file per image into the output directory.
#
# CSV expected columns (order doesn't matter, names must match exactly):
#   prompt              — the text prompt used for generation
#   identity_image      — path to the identity reference image
#   generated_image     — path to the generated image to evaluate
#
# Optional columns:
#   id                  — a stable identifier for this row (used as output filename)
#                         if absent, row index (zero-padded) is used instead
#   LLM used            — name of the model used for generation (e.g. "Gemini", "Flux")
#                         passed through to the output JSON as-is
#
# One bad row never kills the batch — errors are captured per-row in the JSON.
# Models are loaded once before the loop, not per-image.

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"prompt", "identity_image", "generated_image"}

# Optional columns that are passed through to the output JSON if present.
# Keys here are lowercase-stripped for case-insensitive matching.
OPTIONAL_PASSTHROUGH_COLUMNS = {"llm used": "llm_used"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_csv_headers(headers: list[str]) -> None:
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {headers}"
        )


def _row_id(row: dict, index: int) -> str:
    """Stable identifier for this row — used as the output filename."""
    return str(row.get("id", f"row_{index:04d}")).strip().replace(" ", "_")


def _resolve_path(raw: str, csv_dir: Path) -> Path:
    """
    Resolve an image path relative to the CSV's directory.
    Absolute paths pass through unchanged.
    """
    p = Path(raw.strip())
    if p.is_absolute():
        return p
    return (csv_dir / p).resolve()


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_batch(
    csv_path: str | Path,
    output_dir: str | Path,
    face_evaluator=None,
    prompt_evaluator=None,
    quality_evaluator=None,
    artifact_evaluator=None,
    safety_evaluator=None,
    style_evaluator=None,
    row_filter: Optional[set[str]] = None,
) -> list[dict]:
    """
    Process every row in the CSV and write one JSON result file per row.

    Parameters
    ----------
    csv_path        : path to the input CSV manifest
    output_dir      : directory where JSON result files will be written
    face_evaluator  : FaceSimilarityEvaluator instance (loaded externally for reuse)
    prompt_evaluator: PromptAdherenceEvaluator instance (loaded externally for reuse)
    row_filter      : if provided, only rows whose id is in this set are evaluated

    Returns
    -------
    List of result dicts (same content as the JSON files) — useful for
    printing a summary after the batch.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = csv_path.parent

    # Lazy-load evaluators if not injected (useful for standalone testing)
    if face_evaluator is None:
        from evaluators.face_similarity import FaceSimilarityEvaluator
        logger.info("Loading FaceSimilarityEvaluator…")
        face_evaluator = FaceSimilarityEvaluator()

    if prompt_evaluator is None:
        from evaluators.prompt_adherence import PromptAdherenceEvaluator
        logger.info("Loading PromptAdherenceEvaluator…")
        prompt_evaluator = PromptAdherenceEvaluator()

    if quality_evaluator is None:
        from evaluators.quality import QualityEvaluator
        logger.info("Loading QualityEvaluator…")
        quality_evaluator = QualityEvaluator()

    if artifact_evaluator is None or safety_evaluator is None or style_evaluator is None:
        from pipeline.vqa_loader import load_vqa
        logger.info("Loading BLIP-VQA for artifact / safety / style evaluators…")
        vqa_model, vqa_processor = load_vqa()

    if artifact_evaluator is None:
        from evaluators.artifact import ArtifactEvaluator
        artifact_evaluator = ArtifactEvaluator(model=vqa_model, tokenizer=vqa_processor)

    if safety_evaluator is None:
        from evaluators.safety import SafetyEvaluator
        safety_evaluator = SafetyEvaluator(model=vqa_model, tokenizer=vqa_processor)

    if style_evaluator is None:
        from evaluators.style import StyleEvaluator
        style_evaluator = StyleEvaluator(model=vqa_model, tokenizer=vqa_processor)

    results = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _validate_csv_headers(reader.fieldnames or [])

        for index, row in enumerate(reader):
            row_id = _row_id(row, index)

            if row_filter and row_id not in row_filter:
                logger.debug("Skipping row %s (not in --rows filter)", row_id)
                continue

            logger.info("Evaluating row %s (%d)…", row_id, index)

            result = _evaluate_row(
                row_id=row_id,
                row=row,
                csv_dir=csv_dir,
                face_evaluator=face_evaluator,
                prompt_evaluator=prompt_evaluator,
                quality_evaluator=quality_evaluator,
                artifact_evaluator=artifact_evaluator,
                safety_evaluator=safety_evaluator,
                style_evaluator=style_evaluator,
            )
            results.append(result)

            out_file = output_dir / f"{row_id}.json"
            with out_file.open("w", encoding="utf-8") as jf:
                json.dump(result, jf, indent=2)

            logger.info(
                "  face=%s (%.3f)  prompt=%s (%.3f)  → %s",
                result["face_similarity"]["status"],
                result["face_similarity"].get("score") or 0.0,
                result["prompt_adherence"]["status"],
                result["prompt_adherence"].get("score") or 0.0,
                out_file.name,
            )

    logger.info("Batch complete. %d rows processed → %s", len(results), output_dir)
    return results


def _evaluate_row(
    row_id: str,
    row: dict,
    csv_dir: Path,
    face_evaluator,
    prompt_evaluator,
    quality_evaluator=None,
    artifact_evaluator=None,
    safety_evaluator=None,
    style_evaluator=None,
) -> dict:
    """
    Run both evaluators for a single CSV row.
    Never raises — all exceptions are captured and surfaced in the result dict.
    """
    evaluated_at = datetime.now(timezone.utc).isoformat()

    prompt = row.get("prompt", "").strip()
    identity_raw = row.get("identity_image", "").strip()
    generated_raw = row.get("generated_image", "").strip()

    # Extract optional passthrough columns (case-insensitive key match)
    row_lower = {k.strip().lower(): v for k, v in row.items()}
    llm_used = row_lower.get("llm used", "").strip() or None

    # Guard: generated_image must not be empty — happens when generation failed upstream
    if not generated_raw:
        error_msg = "generated_image is empty — image generation likely failed for this row"
        logger.error("[%s] %s", row_id, error_msg)
        return _error_result(row_id, prompt, identity_raw, generated_raw, error_msg, evaluated_at, llm_used)

    # Path resolution
    try:
        identity_path = _resolve_path(identity_raw, csv_dir)
        generated_path = _resolve_path(generated_raw, csv_dir)
    except Exception as exc:
        error_msg = f"Path resolution failed: {exc}"
        logger.error("[%s] %s", row_id, error_msg)
        return _error_result(row_id, prompt, identity_raw, generated_raw, error_msg, evaluated_at, llm_used)

    # Existence check
    missing = [str(p) for p in [identity_path, generated_path] if not p.exists()]
    if missing:
        error_msg = f"Image file(s) not found: {missing}"
        logger.error("[%s] %s", row_id, error_msg)
        return _error_result(row_id, prompt, identity_raw, generated_raw, error_msg, evaluated_at, llm_used)

    # Face similarity
    face_result = face_evaluator.evaluate(
        identity_image_path=identity_path,
        generated_image_path=generated_path,
    )

    # Prompt adherence
    prompt_result = prompt_evaluator.evaluate(
        prompt=prompt,
        generated_image_path=generated_path,
    )

    # Quality (blur + resolution)
    quality_result = quality_evaluator.evaluate(
        generated_image_path=generated_path,
    ) if quality_evaluator else None

    # Artifact detection
    artifact_result = artifact_evaluator.evaluate(
        generated_image_path=generated_path,
    ) if artifact_evaluator else None

    # Safety
    safety_result = safety_evaluator.evaluate(
        generated_image_path=generated_path,
    ) if safety_evaluator else None

    # Style / realism
    style_result = style_evaluator.evaluate(
        generated_image_path=generated_path,
        prompt=prompt,
    ) if style_evaluator else None

    return {
        "id": row_id,
        "prompt": prompt,
        "llm_used": llm_used,
        "identity_image": identity_raw,
        "generated_image": generated_raw,
        "face_similarity": face_result.to_dict(),
        "prompt_adherence": prompt_result.to_dict(),
        "quality": quality_result.to_dict() if quality_result else None,
        "artifact": artifact_result.to_dict() if artifact_result else None,
        "safety": safety_result.to_dict() if safety_result else None,
        "style": style_result.to_dict() if style_result else None,
        "evaluated_at": evaluated_at,
    }


def _error_result(
    row_id: str,
    prompt: str,
    identity_raw: str,
    generated_raw: str,
    error_msg: str,
    evaluated_at: str,
    llm_used: str | None = None,
) -> dict:
    """
    Return a result dict that records a row-level failure without scores.

    All six evaluator keys are populated with a shape compatible with the
    happy-path schema (status="error") so downstream consumers
    (run_eval._print_summary, generate_report.build_html) can index
    r["quality"], r["artifact"], etc. without KeyError-ing on error rows.
    """
    return {
        "id": row_id,
        "prompt": prompt,
        "llm_used": llm_used,
        "identity_image": identity_raw,
        "generated_image": generated_raw,
        "face_similarity": {
            "score": None,
            "status": "error",
            "model": None,
            "model_version": None,
            "faces_found_in_generated": 0,
            "error": error_msg,
        },
        "prompt_adherence": {
            "score": None,
            "status": "error",
            "model": None,
            "model_version": None,
            "clip_truncated": False,
            "token_count": None,
            "chunks_used": 0,
            "error": error_msg,
        },
        "quality": {
            "blur_score": None,
            "blur_status": "error",
            "resolution_width": None,
            "resolution_height": None,
            "resolution_status": "error",
            "overall_status": "error",
            "error": error_msg,
        },
        "artifact": {
            "overall_status": "error",
            "flagged_categories": [],
            "category_scores": {},
            "answers": {},
            "error": error_msg,
        },
        "safety": {
            "overall_status": "error",
            "flagged_categories": [],
            "category_scores": {},
            "answers": {},
            "error": error_msg,
        },
        "style": {
            "style_label": "unknown",
            "is_photorealistic": False,
            "style_match": None,
            "overall_status": "error",
            "answers": {},
            "error": error_msg,
        },
        "evaluated_at": evaluated_at,
    }
