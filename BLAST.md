# BLAST.md
# Multimodal Deep Evaluation Framework for Identity-Preserving Image Generation

<!-- DOCUMENT STATUS: This is the living design doc for the BLAST framework.
     Each section is annotated with its current implementation status:
       ✅ IMPLEMENTED   — built and working in the current codebase
       ⚠️  PARTIAL       — partially built; deviations or gaps noted
       ❌ NOT YET BUILT — planned but not started
       🔄 DEVIATED      — original approach changed; reason explained
     Read README.md for the current implementation reference.
     Read this file to understand the full vision and what remains to be built. -->

## Objective
Build a multimodal evaluation framework for AI image generation systems using:
- Text prompt
- Identity reference image
- Optional style/avatar reference image

<!-- STATUS: ✅ IMPLEMENTED (text prompt + identity reference image)
     STATUS: ❌ NOT YET BUILT (style/avatar reference image input)
     NOTE: The optional style_image input is not in the current CSV schema or
     any evaluator. Style is currently evaluated by comparing the generated
     image against style keywords in the text prompt — a simpler proxy.
     Full style reference image comparison (Phase 6) is still to be built. -->

The framework evaluates:
- Identity preservation
- Prompt adherence
- Style consistency
- Pose consistency
- Image quality
- Artifact detection
- Safety
- Regression stability

<!-- STATUS SUMMARY:
     ✅ Identity preservation   — implemented via InsightFace ArcFace
     ✅ Prompt adherence        — implemented via OpenCLIP ViT-B-32
     ⚠️  Style consistency       — partially implemented (prompt keyword proxy, not reference image comparison)
     ❌ Pose consistency        — not yet built
     ✅ Image quality           — implemented (blur + resolution)
     ✅ Artifact detection      — implemented via BLIP-VQA
     ✅ Safety                  — implemented via BLIP-VQA (deepfake misuse not done)
     ❌ Regression stability    — not yet built (each run is isolated, no cross-run tracking) -->

---

# High-Level Architecture

```text
Prompt + Identity Image + Style Image
                ↓
        Image Generation Model
                ↓
         Generated Image
                ↓
         Evaluation Engine
                ↓
 ┌────────────────────────────────┐
 │ Identity Similarity            │
 │ Prompt Adherence               │
 │ Style Similarity               │
 │ Pose Similarity                │
 │ Quality Evaluation             │
 │ Artifact Detection             │
 │ Safety Evaluation              │
 └────────────────────────────────┘
                ↓
        Composite Final Score
```

<!-- IMPLEMENTATION NOTE:
     The evaluation engine is implemented as a batch CSV pipeline (run_eval.py),
     not a real-time API. The engine runs all evaluators sequentially per row
     and writes one JSON result file per image. The composite final score is
     NOT yet computed — each evaluator outputs independently.

     ⚠️  CRITICAL DESIGN NOTE FOR VIBE CODERS:
     Not all evaluators compare reference vs generated. Only face similarity
     uses both images. All other evaluators (quality, artifact, safety, style)
     analyse the generated image in isolation. The reference image is NOT used
     for quality, artifact, safety, or style checks.

     Identity consistency gaps NOT covered by current evaluators:
     - Skin tone changes relative to reference
     - Age/gender changes relative to reference
     - Facial features appearing/disappearing (beard, glasses, scars)
     These require a dedicated reference-aware consistency evaluator — not yet built. -->

---

# Recommended Tech Stack

## Backend
- Python 3.11+
- FastAPI
- Pydantic

<!-- STATUS: ❌ NOT YET BUILT
     DEVIATION: No FastAPI or REST API built. Current implementation is a
     batch/offline CLI tool (run_eval.py) that reads a CSV manifest and writes
     JSON result files. This was a deliberate MVP decision — validate the
     evaluators first before adding serving infrastructure.
     Python 3.13 with conda on macOS is the tested environment. -->

## ML/CV
- PyTorch
- OpenCV
- Pillow
- NumPy

<!-- STATUS: ✅ IMPLEMENTED — all four are in requirements.txt and in use. -->

## Face Recognition
- InsightFace
- ArcFace

<!-- STATUS: ✅ IMPLEMENTED — InsightFace buffalo_l with ArcFace embeddings.
     Model: buffalo_l (~300MB, auto-downloaded to ~/.insightface/models/).
     Multi-face policy: configurable in config.py (largest / highest_confidence / fail).
     Handles no-face-detected gracefully — returns status "no_face" not an error. -->

## Vision Language
- OpenCLIP
- CLIP

<!-- STATUS: ✅ IMPLEMENTED for prompt adherence (OpenCLIP ViT-B-32/openai).
     NEW: BLIP-VQA (Salesforce/blip-vqa-base) was added for artifact, safety,
     and style evaluation. BLIP-VQA was chosen over CLIP for these tasks because:
     - CLIP uses global image embeddings and cannot localise specific body parts
       or features (e.g. "are the hands deformed?" on a headshot returns noise)
     - CLIP's training data biases caused false positives — dramatic lighting
       and anime styles were incorrectly flagged as NSFW or artifact-laden
     - BLIP-VQA answers binary yes/no questions about actual image content,
       making it far more reliable for artifact and safety detection
     - BLIP-VQA-base is ~450MB vs Moondream2's 1.8GB and has no Python 3.13
       compatibility issues -->

## Pose
- MediaPipe
- OpenPose

<!-- STATUS: ❌ NOT YET BUILT — Pose evaluator is in the roadmap (Phase 6). -->

## Storage
- PostgreSQL
- S3 / MinIO

<!-- STATUS: ❌ NOT YET BUILT
     DEVIATION: Current implementation uses local filesystem — one JSON file per
     evaluated image, written to a timestamped subfolder under results/.
     Each run is isolated. No cross-run comparison or persistence layer exists.
     This is intentional for MVP — add PostgreSQL when regression tracking is needed. -->

## Experiment Tracking
- Weights & Biases
- MLflow

<!-- STATUS: ❌ NOT YET BUILT
     DEVIATION: Experiment tracking is a per-run HTML report (generate_report.py)
     auto-generated at the end of every run. The report shows all evaluator scores,
     badges, per-row images, and a thresholds reference card.
     No cross-run trend tracking or W&B/MLflow integration yet. -->

---

# Recommended Directory Structure

```text
project-root/
├── app/
│   ├── api/
│   ├── evaluators/
│   ├── pipelines/
│   ├── services/
│   └── utils/
├── datasets/
├── models/
├── scripts/
├── tests/
├── docs/
└── BLAST.md
```

<!-- STATUS: 🔄 DEVIATED — Different structure implemented for batch/offline mode:

     Deepeval/
     ├── run_eval.py              # CLI entry point
     ├── generate_report.py       # HTML report generator
     ├── config.py                # All models, thresholds, paths — single source of truth
     ├── requirements.txt
     ├── sample_input.csv
     ├── evaluators/              # One file per evaluator
     │   ├── face_similarity.py
     │   ├── prompt_adherence.py
     │   ├── quality.py
     │   ├── artifact.py
     │   ├── safety.py
     │   └── style.py
     ├── pipeline/
     │   ├── batch_runner.py      # Orchestrates all evaluators per CSV row
     │   ├── image_generator.py   # Gemini + OpenAI image generators
     │   └── vqa_loader.py        # Loads BLIP-VQA once, shared across 3 evaluators
     ├── referenceimages/
     ├── outputimage/
     └── results/                 # Timestamped subfolder per run

     When the API layer is added (Phase 1), the evaluators/ and pipeline/ structure
     maps cleanly into app/evaluators/ and app/pipelines/. -->

---

# Core Evaluation Modules

# 1. Identity Preservation Evaluator

## Goal
Verify generated face matches identity image.

## Method
- Detect face
- Extract embeddings
- Compute cosine similarity

## Models
- InsightFace
- ArcFace

## Output
```json
{
  "identity_score": 0.91
}
```

## Thresholds
- >0.90 excellent
- 0.80–0.90 acceptable
- <0.70 failed

<!-- STATUS: ✅ IMPLEMENTED — evaluators/face_similarity.py

     DEVIATION — THRESHOLDS CALIBRATED DOWN FROM SPEC:
     Original thresholds (>0.90 excellent, 0.80–0.90 acceptable) were theoretical.
     After running against real AI-generated images, ArcFace scores are systematically
     lower — even a correct identity in a photorealistic style rarely exceeds 0.80,
     and stylized outputs (anime, cinematic) drop further.
     Calibrated thresholds based on real data (9 samples, 2 providers):
       Excellent  ≥ 0.75  (near-identical, photorealistic)
       Acceptable ≥ 0.55  (identity preserved, some stylization loss)
       Failed     ≥ 0.40  (significant identity loss)
       No face    —       (face detector returned no result)
     Revisit once dataset grows to 20–30+ samples across multiple identities.

     IMPROVEMENT — MULTI-FACE POLICY:
     The original spec did not address images with multiple detected faces.
     Implemented a configurable policy in config.py:
       FACE_MULTI_FACE_POLICY = "largest" | "highest_confidence" | "fail"
     Default is "largest" — picks the biggest face bounding box (usually the subject).

     KNOWN LIMITATION:
     ArcFace scores drop on stylized outputs (anime, painterly) even for the
     correct person. The face detector itself fails on heavily stylized outputs
     (returns no_face). This is an InsightFace limitation — it was trained on
     real photographs, not stylized images. -->

---

# 2. Prompt Adherence Evaluator

## Goal
Check whether image follows prompt.

## Method
- CLIP text embedding
- CLIP image embedding
- Similarity comparison

## Output
```json
{
  "prompt_score": 0.84
}
```

<!-- STATUS: ✅ IMPLEMENTED — evaluators/prompt_adherence.py

     IMPROVEMENT — PROMPT CHUNKING FOR LONG PROMPTS:
     The original spec did not address CLIP's 77-token hard limit. Prompts
     longer than ~55 words are silently truncated by the CLIP tokenizer,
     causing misleading scores — the score reflected only the first 55 words.
     Implemented overlapping word-level chunking:
       - Chunk size: 53 words (floor((77-2)/1.4) to account for tokenization overhead)
       - Overlap: 15 words between adjacent chunks
       - Score: max-pool across all chunk scores (most optimistic match)
       - Output: chunks_used field in JSON shows how many chunks were evaluated
     Tested on row_009 (471 tokens / 10 chunks) — score went from 0.2091
     (truncated first chunk only) to 0.2440 (correctly evaluated).
     LongCLIP-ready: changing CLIP_MAX_TOKENS to 248 in config.py automatically
     adjusts chunk sizing for the longer context model.

     NOTE ON SCORE SCALE:
     Raw CLIP cosine similarity is NOT on a 0–1 intuitive scale. Typical values
     for a well-matched image are 0.25–0.35, not 0.80+. The example output above
     (0.84) in the original spec is unrealistic for CLIP. Actual thresholds:
       Pass     ≥ 0.20
       Marginal   0.15–0.19
       Fail     < 0.15 -->

---

# 3. Style Similarity Evaluator

## Goal
Measure similarity to style reference image.

## Evaluate
- Color palette
- Composition
- Lighting
- Rendering style

## Methods
- CLIP embeddings
- DINO embeddings

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED — evaluators/style.py
     CONCEPT CHANGED — original vs what was built are different things:

     ORIGINAL SPEC: Compare generated image against a style REFERENCE IMAGE.
     Inputs: generated_image + style_image (optional third input).
     Goal: did the model reproduce the color palette, composition, lighting
     of the style reference?

     WHAT WAS BUILT: Compare generated image against style KEYWORDS IN THE PROMPT.
     Inputs: generated_image + text prompt (parsed for style keywords).
     Goal: did the model produce the visual style the prompt asked for?
     Examples: "ultra-realistic" prompt → photorealistic image expected;
               "anime-style" prompt → non-photorealistic image expected.

     WHY IT CHANGED:
     The style_image third input was never added to the CSV schema or generators.
     A simpler proxy was implemented first: detect the generated image's visual
     style via BLIP-VQA and compare against style keywords in the prompt text.
     This catches the most common failure mode — ultra-realistic prompt generating
     a cartoon/painting — without needing a reference image.
     Model used: BLIP-VQA-base (not CLIP/DINO as originally specified).

     REMAINING GAP:
     The original reference-image-based style comparison (CLIP/DINO embeddings
     comparing color palette, composition, lighting) is NOT built.
     That requires: (a) adding style_image field to CSV, (b) passing it through
     the generator API, (c) implementing CLIP/DINO style embedding comparison.
     This is still a valid and important feature — just not implemented yet. -->

---

# 4. Pose Evaluator

## Goal
Verify pose consistency.

## Methods
- MediaPipe
- OpenPose

## Process
- Extract keypoints
- Compare vectors

<!-- STATUS: ❌ NOT YET BUILT
     This evaluator is not started. MediaPipe and OpenPose are not in requirements.txt.
     Add to Phase 6 when style reference image comparison is also being built.
     Pose evaluation would compare keypoints between reference and generated image
     to detect cases where the model changed the person's body pose significantly. -->

---

# 5. Quality Evaluator

## Detect
- Blur
- Distortion
- Asymmetry
- Low resolution
- Poor lighting

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED — evaluators/quality.py

     IMPLEMENTED:
     - Blur: Laplacian variance on the generated image (sharp ≥100, acceptable ≥50)
     - Low resolution: PIL image dimensions (good ≥512px, acceptable ≥256px)
     Output: blur_score, blur_status, resolution_status, overall_status

     NOT IMPLEMENTED:
     - Distortion: no metric built (would need optical flow or geometry analysis)
     - Asymmetry: not implemented (was explored via BLIP-VQA but moved to artifact evaluator)
     - Poor lighting: no metric built (histogram analysis or VQA could work)

     NOTE: The artifact evaluator (BLIP-VQA) partially covers distortion and asymmetry
     through face_structure and eyes checks, but from a structural breakage angle
     rather than a photographic quality angle. -->

---

# 6. Artifact Detection

## Detect
- Extra fingers
- Broken eyes
- Duplicate limbs
- Warped faces

<!-- STATUS: ✅ IMPLEMENTED — evaluators/artifact.py

     IMPLEMENTATION APPROACH — BLIP-VQA (not CLIP as implied):
     CLIP-based paired-prompt scoring was initially used but produced 8/9 false
     positives on a 9-image test set. Root cause: CLIP uses global image embeddings
     and cannot localise. A headshot with no visible hands still gets a near-zero
     "are the hands deformed?" score from CLIP, which crossed the threshold.

     Switched to BLIP-VQA-base with targeted yes/no questions per category.
     Key design decision — "ask about breakage, not perfection":
     Questions are phrased to detect OBVIOUS structural breakage ("does this person
     have more than two eyes?") rather than confirm perfection ("do the eyes look
     natural?"). Asking for perfection caused BLIP to answer "no" for even
     acceptable AI images, producing false positives. Asking about obvious breakage
     only flags genuinely broken outputs. Result: 1/9 flagged vs 8/9 with the
     perfection-phrasing approach.

     CATEGORIES IMPLEMENTED:
     - hands_fingers: two-stage — presence check first, quality check only if hands
       visible (avoids false positives on headshots where hands aren't in frame)
     - face_structure: flags severe AI deformations (extra eyes, melted features)
     - eyes: flags structural eye errors (more than two eyes, fused, floating)

     NOT IMPLEMENTED:
     - Duplicate limbs: not yet a BLIP-VQA check
     - General body distortion beyond face/eyes/hands

     CALIBRATION STATUS:
     Questions and thresholds calibrated on 9 images. Revisit with larger dataset.
     Treat "flagged" as a signal for human review, not a definitive fail. -->

---

# 7. Safety Evaluator

## Detect
- NSFW
- Violence
- Harmful content
- Deepfake misuse

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED — evaluators/safety.py

     IMPLEMENTED (BLIP-VQA):
     - NSFW: explicit sexual content, nudity, adult-only material
     - Violence: graphic violence, blood, gore, disturbing imagery
     - Harmful: hate symbols, threatening weapons, extremist content
     Zero false positives on 9-image test set — improvement over original
     CLIP-based approach which flagged dramatic lighting as NSFW.

     NOT IMPLEMENTED:
     - Deepfake misuse: no deepfake detection classifier integrated.
       Would require a dedicated deepfake detection model (e.g. FaceForensics++
       trained detector) — out of scope for current BLIP-VQA approach.

     APPROACH NOTE:
     Original spec implied CLIP-based safety. Switched to BLIP-VQA for same
     reasons as artifact detection — CLIP's global embeddings cause false positives
     on dramatic lighting and anime art styles which push embeddings toward unsafe
     clusters even for clearly appropriate content. -->

---

# Composite Scoring

```text
Final Score =
(0.40 × Identity)
+ (0.25 × Prompt)
+ (0.15 × Style)
+ (0.10 × Pose)
+ (0.10 × Quality)
```

Artifact and safety checks should act as blockers.

<!-- STATUS: ❌ NOT YET BUILT
     Each evaluator currently outputs independently — there is no weighted composite
     score computed. The HTML report shows all individual scores but does not
     aggregate them into a final score.

     BLOCKING DEPENDENCIES before composite score can be built:
     1. Pose evaluator must be implemented (contributes 10%)
     2. Style evaluator must be upgraded to reference-image comparison (contributes 15%)
        — current prompt-keyword proxy is not equivalent to the designed method
     3. The weighting formula should be re-validated empirically once all evaluators
        are producing reliable scores on a larger dataset

     CURRENT WORKAROUND:
     The HTML report provides a pass/fail summary per evaluator and flags (artifact,
     safety, style mismatch) that can be used manually to make accept/reject decisions. -->

---

# Benchmark Dataset Requirements

## Include
- Male/female faces
- Different lighting
- Multiple angles
- Glasses/hats/beards
- Stylized prompts
- Anime/cinematic/3D references

## Regression Suite Example
- 100 prompts
- 50 identity images
- 20 style references

<!-- STATUS: ❌ NOT YET BUILT
     Current dataset: 9 rows in sample_input.csv (3 prompts × 3 providers, 2–3 identities).
     This is a development/calibration set, not a proper benchmark.

     WHAT IS NEEDED:
     - Diverse identity coverage: multiple ethnicities, ages, genders
     - Edge cases: glasses, hats, beards, heavy makeup, dark/bright lighting
     - Style coverage: photorealistic, anime, cinematic, painterly, 3D render
     - Provider coverage: Gemini, OpenAI, and any future providers
     - Ground truth labels: human-rated pass/fail for each evaluator dimension
     The 9-sample set was sufficient to calibrate thresholds but not to validate them.
     All current thresholds should be treated as starting points. -->

---

# API Design

# POST /generate

```json
{
  "prompt": "Create a royal warrior portrait",
  "identity_image": "base64",
  "style_image": "base64"
}
```

---

# POST /evaluate

```json
{
  "generated_image": "...",
  "identity_image": "...",
  "style_image": "...",
  "prompt": "..."
}
```

<!-- STATUS: ❌ NOT YET BUILT
     DEVIATION: No REST API exists. Current interface is a CLI batch runner:
       python run_eval.py --csv manifest.csv --generate --provider gemini

     The CSV manifest acts as a batch equivalent of repeated POST /evaluate calls.
     The image generator (pipeline/image_generator.py) handles generation per row,
     equivalent to POST /generate.

     MIGRATION PATH when API layer is added:
     - POST /generate → wraps GeminiImageGenerator or OpenAIImageGenerator
     - POST /evaluate → wraps batch_runner.run_batch() for a single row
     - The evaluator classes are already provider-agnostic and stateless —
       they can be imported and called directly from FastAPI route handlers.

     NOTE: style_image is not in the current CSV schema or generators.
     Adding it requires updating the CSV format, both generators, and the style evaluator. -->

---

# Database Schema

## Tables
- generations
- evaluations
- benchmark_runs

<!-- STATUS: ❌ NOT YET BUILT
     Current persistence: JSON files in results/YYYY-MM-DD_HH-MM-SS/*.json
     One JSON file per evaluated row per run. No cross-run querying possible.

     When PostgreSQL is added, the JSON schema maps naturally:
     - evaluations table: one row per image, all evaluator scores as JSONB columns
     - generations table: tracks which images were generated, by which provider
     - benchmark_runs table: groups evaluations into named benchmark runs for comparison -->

---

# Cost Optimization Strategy

## DO NOT evaluate every image fully.

## Tier 1 (Always Run)
- Identity similarity
- NSFW
- Blur detection

## Tier 2 (Sampled)
- CLIP prompt adherence
- Style scoring

## Tier 3 (Release Testing)
- Vision LLM judging
- Full benchmark suite

<!-- STATUS: ❌ NOT IMPLEMENTED
     All evaluators currently run on every row in every batch — no tiering.
     The tiered strategy makes sense for production at scale to control cost and
     latency. For the current batch/offline MVP it is not necessary since all
     evaluators run on CPU and complete in ~30–60s per image.

     When the API mode is built, implement tiering in batch_runner.py by accepting
     an evaluator_tier parameter (1, 2, or 3) and skipping evaluators accordingly. -->

---

# Human Evaluation Strategy

Build reviewer dashboard for:
- Realism
- Likeness
- Aesthetics
- Prompt adherence

Humans should rank:
- Best image
- Pass/fail
- Quality score

<!-- STATUS: ❌ NOT YET BUILT (Phase 10)
     The HTML report (generate_report.py) is a read-only evaluation report, not
     an interactive reviewer dashboard. It shows all scores and images side-by-side
     but has no input controls for human ratings or pass/fail decisions.

     CURRENT WORKAROUND:
     Reviewers view the HTML report and make decisions manually. The "flagged"
     status from artifact and safety evaluators, and "mismatch" from style, are
     intended to direct reviewer attention to specific rows that need human review. -->

---

# Security Requirements

This system processes biometric data.

## Requirements
- Encrypted storage
- Signed URLs
- Secure uploads
- Deletion workflows
- Audit logs

<!-- STATUS: ❌ NOT APPLICABLE for current batch/local mode
     The current implementation runs entirely locally — reference images and generated
     images are local file paths, results are local JSON files. No biometric data
     is transmitted or stored in a remote system.

     IMPORTANT: When the REST API and PostgreSQL/S3 layers are built, all security
     requirements become critical. Face embeddings (ArcFace vectors) are biometric
     data and must be treated accordingly under GDPR and similar regulations:
     - Embeddings must not be stored without consent
     - Deletion workflows must cascade from identity_image to all derived embeddings
     - Audit logs for all access to identity data
     - Signed URLs with expiry for any S3-stored images -->

---

# Major Risks

## Technical
- Identity mismatch
- Stylization reducing face similarity
- High GPU cost
- Evaluator inconsistency

## Product
- Users liking lower-scored images
- Subjective quality disagreement

<!-- RISK STATUS UPDATES:

     Identity mismatch: ⚠️ OBSERVED — face similarity scores are often in the
     "acceptable" (0.55–0.74) range rather than excellent. Stylization loss is
     a major contributor. Anime and cinematic prompts routinely score below 0.55.

     Stylization reducing face similarity: ✅ CONFIRMED AND DOCUMENTED — ArcFace
     thresholds were recalibrated down from the original spec after observing this.
     Anime outputs (row_003, row_006) scored 0.41 and 0.05 respectively.

     High GPU cost: ✅ MITIGATED — entire pipeline runs CPU-only. InsightFace
     uses ONNX CPU runtime, CLIP and BLIP run on CPU torch. Latency is ~30–60s
     per image on a MacBook M-series. GPU would reduce this to ~3–5s.

     Evaluator inconsistency: ⚠️ ONGOING — BLIP-VQA gives different answers to
     the same question on similar images. Treat evaluator outputs as signals for
     human review, not ground truth. Thresholds calibrated on only 9 samples.

     NEW RISK (discovered during implementation):
     Reference-image blind spots — artifact, safety, and style evaluators do not
     use the reference image. Changes introduced by the model relative to the
     reference (skin tone, age, gender, features) are invisible to the current
     evaluators. Only face similarity (ArcFace cosine) captures reference-relative
     identity drift, and only at the embedding level. -->

---

# Phase-by-Phase Implementation Plan

# Phase 1 — Foundation
- Setup FastAPI
- Docker
- PostgreSQL
- S3/MinIO
- Logging
- Config management

Deliverable:
- Running backend

<!-- STATUS: ⚠️ PARTIALLY DONE
     NOT BUILT: FastAPI, Docker, PostgreSQL, S3/MinIO
     BUILT: Config management (config.py — all model names, thresholds, paths in
     one file; values propagate to all output JSON records automatically)
     BUILT: Logging (Python logging with configurable level via --log-level CLI flag)
     DEVIATION: Infrastructure deferred in favour of batch/offline MVP. Add FastAPI
     and storage when the framework needs to serve real-time requests or persist
     results across runs for regression tracking. -->

---

# Phase 2 — Image Generation
- Integrate SDXL/Flux
- Prompt handling
- Reference image handling

Deliverable:
- Working generation API

<!-- STATUS: ✅ IMPLEMENTED (different providers than spec)
     DEVIATION: SDXL/Flux were not integrated. Instead:
     - Gemini (gemini-2.5-flash-image) via google-genai SDK
     - OpenAI (gpt-image-1) via images.edit() API
     Both use the reference identity image directly (OpenAI images.edit, Gemini
     multimodal) rather than LoRA or img2img approaches.

     IMPLEMENTATION: pipeline/image_generator.py
     - GeminiImageGenerator: sends reference image bytes + prefixed prompt
     - OpenAIImageGenerator: sends reference image + prompt via images.edit()
     The CSV is the source of truth — generated_image path is written back to the
     CSV after generation. Rows with existing generated_image are skipped.
     Selective generation via --rows flag allows regenerating specific rows only. -->

---

# Phase 3 — Identity Evaluator
- Integrate InsightFace
- Embedding extraction
- Cosine similarity

Deliverable:
- Identity scoring pipeline

<!-- STATUS: ✅ IMPLEMENTED — evaluators/face_similarity.py
     See notes in Evaluator section #1 above for calibration details and deviations. -->

---

# Phase 4 — Prompt Evaluator
- Integrate OpenCLIP
- Text/image embeddings
- Similarity scoring

Deliverable:
- Prompt adherence evaluator

<!-- STATUS: ✅ IMPLEMENTED — evaluators/prompt_adherence.py
     IMPROVEMENT: Prompt chunking added for prompts exceeding 77 CLIP tokens.
     See notes in Evaluator section #2 above for full details. -->

---

# Phase 5 — Quality & Artifact Detection
- Blur detection
- Anatomy checks
- Aesthetic scoring

Deliverable:
- Quality evaluator

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED
     BUILT (evaluators/quality.py):
     - Blur detection: Laplacian variance — sharp ≥100, acceptable ≥50
     - Resolution check: PIL dimensions — good ≥512px, acceptable ≥256px

     BUILT (evaluators/artifact.py — BLIP-VQA):
     - Anatomy checks: hands/fingers, face structure, eyes via targeted VQA questions
     - Design decision: questions detect obvious breakage ("more than two eyes?")
       not perfection — avoids false positives on acceptable AI images

     NOT BUILT:
     - Aesthetic scoring: no perceptual quality score (NIQE, BRISQUE, or similar)
     - Lighting quality: no histogram or VQA-based lighting assessment -->

---

# Phase 6 — Style & Pose Evaluation
- Pose extraction
- Style embedding comparison

Deliverable:
- Style + pose evaluators

<!-- STATUS: ⚠️ PARTIAL (style proxy built; pose not started)

     STYLE — prompt keyword proxy built, reference-image comparison not built:
     evaluators/style.py uses BLIP-VQA to detect the generated image's visual style
     (photorealistic / anime / cartoon / painting / 3d_render / other) and compares
     against style keywords in the prompt. This catches style mismatches like
     ultra-realistic prompt → painting output (caught correctly for row_009).
     The original design (CLIP/DINO comparing against a style reference image)
     is not implemented — requires style_image input field to be added first.

     POSE — not started:
     MediaPipe and OpenPose are not in requirements.txt. Pose evaluation would
     compare body keypoints between reference and generated image to flag cases
     where the model significantly changed the person's pose. -->

---

# Phase 7 — Evaluation Orchestration
- Async pipeline
- Weighted scoring
- Timeout handling

Deliverable:
- Unified evaluation engine

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED — pipeline/batch_runner.py
     BUILT: Unified evaluation engine that runs all 6 evaluators per row and
     writes JSON results. Handles per-row errors gracefully (one failed row
     does not stop the batch). Selective row filtering via --rows flag.
     Auto-generates HTML report at end of each run.

     NOT BUILT:
     - Async pipeline: evaluators run synchronously, sequentially per row
     - Weighted composite scoring: each evaluator outputs independently
     - Timeout handling: no per-evaluator timeout configured -->

---

# Phase 8 — Benchmark Dataset
- Collect prompts
- Collect identity references
- Create edge cases

Deliverable:
- Benchmark suite v1

<!-- STATUS: ❌ NOT YET BUILT
     Current: 9-row sample_input.csv for development and threshold calibration.
     See Benchmark Dataset Requirements section above for what needs to be collected. -->

---

# Phase 9 — Dashboard & Reporting
- Benchmark dashboards
- Regression tracking
- Score trends

Deliverable:
- Internal admin dashboard

<!-- STATUS: ⚠️ PARTIALLY IMPLEMENTED
     BUILT: Per-run HTML report (generate_report.py) — auto-generated at end of
     each run. Shows summary cards, per-evaluator breakdowns, per-row results
     with reference and generated images side-by-side, thresholds reference card.
     Each run writes to results/YYYY-MM-DD_HH-MM-SS/ — runs are isolated.

     NOT BUILT:
     - Regression tracking: no cross-run comparison (score trends over time)
     - Benchmark dashboard: no aggregated view across multiple benchmark runs
     - Score trend charts: would need PostgreSQL + time-series queries -->

---

# Phase 10 — Human Evaluation Platform
- Ranking UI
- Reviewer workflows
- Annotation storage

Deliverable:
- Human feedback system

<!-- STATUS: ❌ NOT YET BUILT
     The HTML report is read-only. No interactive reviewer interface exists.
     Human reviewers currently use the report to manually identify flagged rows. -->

---

# MVP Recommendation

Start ONLY with:
- Identity similarity
- CLIP prompt adherence
- Basic NSFW
- Basic blur detection

This gives:
- Regression testing
- Benchmarking
- Production validation

without massive infrastructure complexity.

<!-- STATUS: ✅ MVP COMPLETED AND EXCEEDED
     The MVP was completed (identity, prompt adherence, NSFW, blur) and then
     extended with:
     - Full artifact detection (BLIP-VQA with presence gating)
     - Full safety evaluation (BLIP-VQA — 3 categories)
     - Style evaluator (BLIP-VQA + prompt keyword matching)
     - Image generation via Gemini and OpenAI
     - HTML report auto-generation
     - Prompt chunking for long prompts
     - Selective row evaluation (--rows flag)
     - Timestamped output folders per run
     - Multi-face policy handling -->

---

# Recommended Success Metrics

## Technical
- Avg identity score
- Avg prompt score
- Evaluation latency
- Cost per evaluation

## Product
- User satisfaction
- Acceptance rate
- Regeneration frequency

<!-- CURRENT METRICS (from 9-sample test set, 2 providers):
     Avg identity score (Gemini rows 001–003): 0.54
     Avg identity score (OpenAI rows 004–006): 0.38
     Avg CLIP score (all rows): 0.26
     Evaluation latency: ~30–60s per image on CPU (MacBook M-series)
     Cost per evaluation: $0 (local CPU, no cloud inference)

     Latency breakdown per evaluator (approximate):
     - Face similarity (InsightFace): ~5–10s
     - Prompt adherence (CLIP): ~2–3s
     - Quality (OpenCV): <1s
     - Artifact (BLIP-VQA, 3–4 questions): ~15–20s
     - Safety (BLIP-VQA, 3 questions): ~10–15s
     - Style (BLIP-VQA, 2 questions): ~8–10s -->

---

# Suggested Timeline

## Week 1
- Backend setup
- Generation API

## Week 2
- Identity evaluator

## Week 3
- CLIP evaluator

## Week 4
- Quality checks
- Regression suite

## Week 5+
- Human evaluation
- Scaling
- Optimization

<!-- ACTUAL TIMELINE (for reference):
     Week 1: Identity + prompt adherence + quality + batch runner + CLI + HTML report
     Week 2: Artifact + safety (CLIP-based, then replaced with BLIP-VQA)
     Week 2: Style evaluator, Gemini + OpenAI image generators
     Week 2: Prompt chunking, threshold calibration, BLAST.md + README alignment

     Still remaining (next phases):
     - Style reference image comparison (original Phase 6 concept)
     - Pose evaluator (Phase 6)
     - Composite scoring (Phase 7)
     - FastAPI + PostgreSQL + S3 (Phase 1 infrastructure)
     - Benchmark dataset (Phase 8)
     - Regression tracking (Phase 9)
     - Human evaluation platform (Phase 10) -->

---

# Final Recommendation

Build incrementally.

Recommended order:
1. Identity similarity
2. Prompt adherence
3. Basic quality checks
4. Benchmark suite
5. Regression testing
6. Human preference system

<!-- UPDATED RECOMMENDED ORDER based on implementation learnings:
     ✅ 1. Identity similarity         — done
     ✅ 2. Prompt adherence            — done (+ chunking)
     ✅ 3. Basic quality checks        — done (blur + resolution)
     ✅ 4. Artifact + safety + style   — done (BLIP-VQA)
     ✅ 5. Image generation integration — done (Gemini + OpenAI)
        6. Style reference image comparison — next (Phase 6 original concept)
        7. Pose evaluator              — next (Phase 6)
        8. Composite scoring           — after pose is ready (Phase 7)
        9. FastAPI + storage layer     — when real-time serving is needed (Phase 1)
       10. Benchmark dataset           — needed to validate thresholds (Phase 8)
       11. Regression tracking         — after storage layer (Phase 9)
       12. Human evaluation platform  — after benchmark dataset (Phase 10) -->
