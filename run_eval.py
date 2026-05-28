#!/usr/bin/env python3
# run_eval.py
#
# CLI entry point for the BLAST batch evaluator.
#
# Evaluate only (images already generated):
#   python run_eval.py --csv manifest.csv
#   python run_eval.py --csv manifest.csv --output-dir results/
#
# Generate via Gemini, then evaluate:
#   python run_eval.py --csv manifest.csv --generate --provider gemini --api-key YOUR_GEMINI_KEY
#   python run_eval.py --csv manifest.csv --generate --provider gemini  # reads GEMINI_API_KEY env var
#
# Generate via OpenAI, then evaluate:
#   python run_eval.py --csv manifest.csv --generate --provider openai --api-key YOUR_OPENAI_KEY
#   python run_eval.py --csv manifest.csv --generate --provider openai  # reads OPENAI_API_KEY env var
#
# CSV format when using --generate (generated_image column not needed):
#   id, prompt, identity_image, LLM used
#
# CSV format for evaluate-only (original):
#   id, prompt, identity_image, generated_image, LLM used

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup (before any imports that log at module level)
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Pre-generation step — fills in generated_image paths before batch_runner
# ---------------------------------------------------------------------------

def _generate_images(csv_path: Path, api_key: str, provider: str, logger) -> Path:
    """
    For every row in the CSV where generated_image is empty,
    call the chosen provider to generate one, save it to outputimage/, and write
    the path back into the generated_image column of the original CSV.

    The CSV is updated in place — it is the single source of truth.
    Rows that already have a generated_image path are left untouched.

    Parameters
    ----------
    provider : "gemini" | "openai"

    Returns csv_path (same file, now with generated_image filled in).
    """
    import config
    from pipeline.image_generator import GeminiImageGenerator, OpenAIImageGenerator

    if provider == "openai":
        logger.info("Loading OpenAI generator (%s)…", config.OPENAI_GENERATION_MODEL)
        generator = OpenAIImageGenerator(api_key=api_key)
    else:
        logger.info("Loading Gemini generator (%s)…", config.GENERATION_MODEL)
        generator = GeminiImageGenerator(api_key=api_key)

    csv_dir = csv_path.parent
    output_images_dir = csv_dir / config.GENERATED_IMAGES_DIR
    output_images_dir.mkdir(parents=True, exist_ok=True)

    updated_rows = []
    generated_count = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        # Ensure generated_image column exists
        if "generated_image" not in fieldnames:
            fieldnames.append("generated_image")

        for index, row in enumerate(reader):
            existing_gen = row.get("generated_image", "").strip()

            if existing_gen:
                # Path already filled — skip generation, keep as-is
                logger.info("Row %s — generated_image already set, skipping.", row.get("id", index))
                updated_rows.append(row)
                continue

            row_id = row.get("id", f"row_{index:04d}").strip()
            identity_raw = row.get("identity_image", "").strip()
            prompt = row.get("prompt", "").strip()

            if not identity_raw or not prompt:
                logger.warning("Row %s — missing identity_image or prompt, skipping.", row_id)
                updated_rows.append(row)
                continue

            # Resolve identity image path
            identity_path = Path(identity_raw)
            if not identity_path.is_absolute():
                identity_path = (csv_dir / identity_path).resolve()

            if not identity_path.exists():
                logger.error("Row %s — identity image not found: %s", row_id, identity_path)
                updated_rows.append(row)
                continue

            # Save generated image as: outputimage/{row_id}_generated.png
            out_path = output_images_dir / f"{row_id}_generated.png"

            logger.info("Generating image for row %s…", row_id)
            result = generator.generate(
                identity_image_path=identity_path,
                prompt=prompt,
                output_path=out_path,
            )

            if result.success:
                # Write absolute path back into the row
                row["generated_image"] = str(result.output_path.resolve())
                generated_count += 1
                logger.info("  → saved: %s", result.output_path)
            else:
                logger.error("  Generation failed for row %s: %s", row_id, result.error)

            updated_rows.append(row)

    # Write all rows (with generated_image now filled in) back to the SAME CSV.
    # Use a temp file + os.replace so the original CSV is never left truncated
    # if the process dies mid-write. os.replace is atomic on POSIX and Windows.
    # On the first successful write per session, also keep a .bak alongside
    # so the pre-generation manifest can be recovered.
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not backup_path.exists():
        try:
            shutil.copy2(csv_path, backup_path)
            logger.info("Backup of original CSV → %s", backup_path)
        except Exception as exc:
            logger.warning("Could not write CSV backup (%s): %s", backup_path, exc)

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated_rows)
    os.replace(tmp_path, csv_path)  # atomic swap

    logger.info(
        "Generation complete: %d image(s) generated. CSV updated: %s",
        generated_count, csv_path
    )
    return csv_path  # same file, now with generated_image paths filled in


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    if not results:
        print("\nNo results to summarise.")
        return

    print("\n" + "=" * 72)
    print(f"{'ID':<20} {'Face Score':>10} {'Face Status':<20} {'CLIP Score':>10} {'CLIP Status':<12}")
    print("-" * 72)

    face_scores = []
    clip_scores = []

    for r in results:
        row_id = r.get("id", "?")[:20]
        face = r.get("face_similarity", {})
        clip = r.get("prompt_adherence", {})

        f_score = face.get("score")
        c_score = clip.get("score")

        f_str = f"{f_score:.4f}" if f_score is not None else "  N/A  "
        c_str = f"{c_score:.4f}" if c_score is not None else "  N/A  "

        print(f"{row_id:<20} {f_str:>10} {face.get('status','?'):<20} {c_str:>10} {clip.get('status','?'):<12}")

        if f_score is not None:
            face_scores.append(f_score)
        if c_score is not None:
            clip_scores.append(c_score)

    print("-" * 72)

    avg_face = sum(face_scores) / len(face_scores) if face_scores else None
    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else None

    f_avg_str = f"{avg_face:.4f}" if avg_face is not None else "  N/A  "
    c_avg_str = f"{avg_clip:.4f}" if avg_clip is not None else "  N/A  "

    print(f"{'AVERAGE':<20} {f_avg_str:>10} {'':<20} {c_avg_str:>10}")
    print("=" * 72)
    print(f"\nTotal rows evaluated : {len(results)}")
    print(f"Face scores computed : {len(face_scores)}")
    print(f"CLIP scores computed : {len(clip_scores)}")

    errors = [r for r in results if r["face_similarity"]["status"] == "error" or r["prompt_adherence"]["status"] == "error"]
    if errors:
        print(f"Rows with errors     : {len(errors)}")
        for e in errors:
            err = e["face_similarity"].get("error") or e["prompt_adherence"].get("error")
            print(f"  [{e['id']}] {err}")

    print()


# ---------------------------------------------------------------------------
# CSV preview — printed at startup so stale/locked file issues are caught early
# ---------------------------------------------------------------------------

def _preview_csv(csv_path: Path, row_filter, logger) -> None:
    """Print a summary of every row loaded from the CSV before evaluation starts."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info("─" * 60)
        logger.info("CSV SNAPSHOT  (%d rows total)  %s", len(rows), csv_path.name)
        logger.info("─" * 60)

        for row in rows:
            row_id = str(row.get("id", "?")).strip()
            if row_filter and row_id not in row_filter:
                continue
            gen = row.get("generated_image", "").strip()
            ref = row.get("identity_image", "").strip()
            prompt = row.get("prompt", "").strip()[:60]
            gen_display = os.path.basename(gen) if gen else "⚠  EMPTY"
            ref_display = os.path.basename(ref) if ref else "⚠  EMPTY"
            logger.info(
                "  [%s]  ref=%-20s  gen=%-30s  prompt=%s…",
                row_id, ref_display, gen_display, prompt,
            )

        logger.info("─" * 60)
    except Exception as exc:
        logger.warning("Could not preview CSV: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BLAST — Batch evaluation of identity-preserving image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate only (images already generated):
  python run_eval.py --csv manifest.csv

  # Generate via OpenAI gpt-image-1, then evaluate:
  python run_eval.py --csv manifest.csv --generate --provider openai --api-key YOUR_OPENAI_KEY

  # Generate via Gemini, then evaluate:
  python run_eval.py --csv manifest.csv --generate --provider gemini --api-key YOUR_GEMINI_KEY

  # Evaluate specific rows only:
  python run_eval.py --csv manifest.csv --rows row_004,row_006

  # Generate + evaluate a single row:
  python run_eval.py --csv manifest.csv --generate --provider openai --api-key KEY --rows row_006

  # API key from environment variable:
  export OPENAI_API_KEY=YOUR_KEY
  python run_eval.py --csv manifest.csv --generate --provider openai

CSV columns when using --generate:
  id (optional), prompt, identity_image, LLM used (optional)

CSV columns for evaluate-only:
  id (optional), prompt, identity_image, generated_image, LLM used (optional)
        """,
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to the input CSV manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write JSON result files (default: ./results/).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="Generate images before evaluating. "
             "Rows that already have a generated_image path are skipped.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai"],
        help="Image generation provider to use with --generate (default: gemini).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for the chosen provider. "
            "Falls back to GEMINI_API_KEY (for --provider gemini) "
            "or OPENAI_API_KEY (for --provider openai) environment variables."
        ),
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help=(
            "Comma-separated list of row IDs to evaluate (default: all rows). "
            "Example: --rows row_004,row_006"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    if not args.csv.exists():
        logger.error("CSV file not found: %s", args.csv)
        sys.exit(1)

    # Create a timestamped subfolder under the base output dir so each run
    # is isolated and can be compared over time.
    # e.g. results/2026-05-25_11-30-38/
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse --rows into a set for O(1) lookup, or None to mean "all rows"
    row_filter = None
    if args.rows:
        row_filter = {r.strip() for r in args.rows.split(",") if r.strip()}

    logger.info("Starting BLAST evaluation")
    logger.info("  CSV        : %s", args.csv)
    logger.info("  Output dir : %s", output_dir)
    logger.info("  Run ID     : %s", run_timestamp)
    logger.info("  Generate   : %s", args.generate)
    if args.generate:
        logger.info("  Provider   : %s", args.provider)
    if row_filter:
        logger.info("  Row filter : %s", sorted(row_filter))

    # Print CSV contents at startup so you can verify what was loaded
    # before any evaluation begins. Catches stale/locked file issues early.
    _preview_csv(args.csv, row_filter, logger)

    # ------------------------------------------------------------------
    # Step 1 (optional): generate images via chosen provider
    # ------------------------------------------------------------------
    csv_to_evaluate = args.csv

    if args.generate:
        # Resolve API key: CLI flag → provider-specific env var
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
        api_key = args.api_key or os.environ.get(env_var)
        if not api_key:
            logger.error(
                "No API key provided. Use --api-key or set the %s environment variable.",
                env_var,
            )
            sys.exit(1)

        csv_to_evaluate = _generate_images(
            csv_path=args.csv,
            api_key=api_key,
            provider=args.provider,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Step 2: load evaluators and run batch
    # ------------------------------------------------------------------
    from evaluators.face_similarity import FaceSimilarityEvaluator
    from evaluators.prompt_adherence import PromptAdherenceEvaluator
    from evaluators.quality import QualityEvaluator
    from evaluators.artifact import ArtifactEvaluator
    from evaluators.safety import SafetyEvaluator
    from evaluators.style import StyleEvaluator
    from pipeline.vqa_loader import load_vqa

    logger.info("Loading evaluation models…")
    face_evaluator    = FaceSimilarityEvaluator()
    prompt_evaluator  = PromptAdherenceEvaluator()
    quality_evaluator = QualityEvaluator()

    # BLIP-VQA is loaded once and shared across artifact, safety, style.
    # The constructor kwarg is named `tokenizer=` for backwards compatibility,
    # but the value is actually a BlipProcessor.
    logger.info("Loading VQA model (shared across artifact / safety / style)…")
    vqa_model, vqa_processor = load_vqa()
    artifact_evaluator = ArtifactEvaluator(model=vqa_model, tokenizer=vqa_processor)
    safety_evaluator   = SafetyEvaluator(model=vqa_model, tokenizer=vqa_processor)
    style_evaluator    = StyleEvaluator(model=vqa_model, tokenizer=vqa_processor)
    logger.info("All models ready. Starting batch…")

    from pipeline.batch_runner import run_batch

    results = run_batch(
        csv_path=csv_to_evaluate,
        output_dir=output_dir,
        face_evaluator=face_evaluator,
        prompt_evaluator=prompt_evaluator,
        quality_evaluator=quality_evaluator,
        artifact_evaluator=artifact_evaluator,
        safety_evaluator=safety_evaluator,
        style_evaluator=style_evaluator,
        row_filter=row_filter,
    )

    _print_summary(results)
    logger.info("Results written to: %s", output_dir.resolve())

    # Auto-generate HTML report for this run
    try:
        from generate_report import load_results, build_html
        report_path = output_dir / "report.html"
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        html = build_html(results, generated_at)
        report_path.write_text(html, encoding="utf-8")
        logger.info("HTML report   : %s", report_path.resolve())
    except Exception as exc:
        logger.warning("Could not generate HTML report: %s", exc)


if __name__ == "__main__":
    main()
