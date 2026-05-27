# BLAST — Batch Evaluation Framework for Identity-Preserving Image Generation

BLAST evaluates AI-generated images across six dimensions: identity preservation, prompt adherence, image quality, artifact detection, safety, and style consistency. It reads a CSV manifest, optionally generates images via Gemini or OpenAI, runs all evaluators, and writes one JSON result file per image plus a self-contained HTML report per run.

See `BLAST.md` for the full design document and future roadmap.

---

## ⚠️ Critical: What Each Evaluator Actually Compares

This is the most important thing to understand before interpreting results. **Not all evaluators use the reference image.** Only one evaluator — face similarity — actually compares the generated image against the reference photo. All others analyze the generated image in isolation or against the text prompt.

| Evaluator | Uses Reference Image | Uses Generated Image | Compares Against |
|---|:---:|:---:|---|
| **Face Similarity** | ✅ Yes | ✅ Yes | Reference photo → Generated image (ArcFace cosine distance) |
| **Prompt Adherence** | ❌ No | ✅ Yes | Generated image vs text prompt (CLIP similarity) |
| **Quality** | ❌ No | ✅ Yes | Generated image only — blur + resolution check |
| **Artifact Detection** | ❌ No | ✅ Yes | Generated image only — VQA questions about structural breakage |
| **Safety** | ❌ No | ✅ Yes | Generated image only — VQA questions about content |
| **Style** | ❌ No | ✅ Yes | Generated image vs style keywords in the prompt |

**What this means in practice:**

- A **face similarity score of 0.35** means the generated face looks different from the reference — could be wrong person, heavy stylization, or identity loss.
- An **artifact flag** means the generated image itself has structural AI breakage (extra eyes, melted features) — it has nothing to do with the reference photo.
- A **safety flag** means the generated image contains unsafe content — the reference photo is irrelevant.
- A **style mismatch** means the generated image's visual style doesn't match what the prompt asked for (e.g. prompt says "ultra-realistic" but output is a painting) — again, reference not used.

**What BLAST does NOT check (reference-aware gaps):**

- Did the model change the person's skin tone relative to the reference?
- Did the model change the person's apparent age or gender?
- Did a facial feature (beard, glasses, birthmark) appear or disappear?
- Does the background or clothing match any constraint in the reference?

These would require a dedicated identity-consistency evaluator that explicitly compares reference vs generated beyond the face embedding. This is not yet implemented — see Roadmap.

---

## What's Implemented

| Evaluator | Model | Role |
|---|---|---|
| Face similarity | InsightFace `buffalo_l` (ArcFace embeddings) | Identity preservation — reference vs generated |
| Prompt adherence | OpenCLIP `ViT-B-32/openai` (text–image cosine) | Prompt alignment — generated vs prompt text |
| Quality | OpenCV Laplacian blur + PIL resolution | Blur and resolution of generated image |
| Artifact detection | `Salesforce/blip-vqa-base` (VQA) | Structural AI breakage in generated image |
| Safety | `Salesforce/blip-vqa-base` (VQA) | Unsafe content in generated image |
| Style | `Salesforce/blip-vqa-base` (VQA) | Style consistency — generated image vs prompt style keywords |

Image generation is supported via Gemini (`gemini-2.5-flash-image`) and OpenAI (`gpt-image-1`). Everything runs **CPU-only** — no GPU required.

BLIP-VQA is loaded once and shared across artifact, safety, and style evaluators — avoiding three separate model loads.

---

## Project Structure

```
Deepeval/
├── run_eval.py                  # CLI entry point — generate + evaluate
├── config.py                    # Central config: models, thresholds, paths
├── requirements.txt
├── sample_input.csv             # Example CSV manifest
│
├── evaluators/
│   ├── face_similarity.py       # InsightFace / ArcFace — reference vs generated
│   ├── prompt_adherence.py      # OpenCLIP — generated image vs prompt text
│   ├── quality.py               # Blur (Laplacian) + resolution — generated image only
│   ├── artifact.py              # BLIP VQA — structural breakage in generated image only
│   ├── safety.py                # BLIP VQA — unsafe content in generated image only
│   └── style.py                 # BLIP VQA — style vs prompt keywords, generated image only
│
├── pipeline/
│   ├── batch_runner.py          # Reads CSV, runs all evaluators, writes JSON
│   ├── image_generator.py       # GeminiImageGenerator + OpenAIImageGenerator
│   └── moondream_loader.py      # Loads BLIP-VQA model (shared across 3 evaluators)
│
├── generate_report.py           # Standalone HTML report generator
├── referenceimages/             # Input: reference identity photos
├── outputimage/                 # Output: generated images (auto-created)
├── results/                     # Output: timestamped subfolder per run
│   └── YYYY-MM-DD_HH-MM-SS/    #   ├── row_001.json … row_N.json
│                                #   └── report.html  (auto-generated)
│
├── list_models.py               # Diagnostic: list available Gemini models
└── BLAST.md                     # Full design doc and future roadmap
```

---

## Installation

**Python 3.10+ required** (tested on Python 3.13 with conda on macOS).

```bash
pip install -r requirements.txt
```

On macOS/Linux with system Python, add `--break-system-packages` if needed:

```bash
pip install -r requirements.txt --break-system-packages
```

On first run, models are downloaded automatically:
- InsightFace `buffalo_l` — ~300 MB → `~/.insightface/models/buffalo_l/`
- OpenCLIP `ViT-B-32/openai` — ~350 MB → `~/.cache/huggingface/`
- BLIP-VQA base — ~450 MB → `~/.cache/huggingface/`

---

## Quick Start

### Evaluate only (images already generated)

```bash
python run_eval.py --csv sample_input.csv
```

### Evaluate specific rows only

```bash
python run_eval.py --csv sample_input.csv --rows row_004,row_006
```

### Generate via OpenAI, then evaluate

```bash
python run_eval.py --csv sample_input.csv --generate --provider openai --api-key YOUR_OPENAI_KEY
```

### Generate via Gemini, then evaluate

```bash
python run_eval.py --csv sample_input.csv --generate --provider gemini --api-key YOUR_GEMINI_KEY
```

### Generate + evaluate a single row

```bash
python run_eval.py --csv sample_input.csv --generate --provider openai --api-key YOUR_KEY --rows row_006
```

### Use environment variables for API keys

```bash
export OPENAI_API_KEY=sk-...
python run_eval.py --csv sample_input.csv --generate --provider openai

export GEMINI_API_KEY=AIza...
python run_eval.py --csv sample_input.csv --generate --provider gemini
```

---

## CSV Format

### For generate + evaluate (`--generate` flag)

```csv
id,prompt,identity_image,generated_image,LLM used
row_001,A royal warrior portrait in golden armor,referenceimages/refimage1.png,,
row_002,A cinematic headshot with dramatic lighting,referenceimages/refimage1.png,,
```

- `generated_image` can be empty — the generator fills it in and writes the path back.
- Rows that already have a `generated_image` path are **skipped** during generation.
- The CSV is updated **in place** after generation — it is the single source of truth.

### For evaluate only (no `--generate`)

```csv
id,prompt,identity_image,generated_image,LLM used
row_001,A royal warrior portrait in golden armor,referenceimages/refimage1.png,outputimage/outputimage1.png,Gemini
```

All columns except `id` and `LLM used` are required. `id` defaults to `row_NNNN` if absent.

---

## CLI Reference

```
python run_eval.py --csv PATH [OPTIONS]

Required:
  --csv PATH            Path to the input CSV manifest

Options:
  --output-dir PATH     Base directory for results (default: ./results/)
                        Each run creates a timestamped subfolder automatically:
                        results/YYYY-MM-DD_HH-MM-SS/
  --generate            Generate images before evaluating
  --provider            gemini | openai  (default: gemini, only used with --generate)
  --api-key KEY         API key for the chosen provider
                        (falls back to GEMINI_API_KEY or OPENAI_API_KEY env vars)
  --rows ROW_IDS        Comma-separated row IDs to evaluate (default: all rows)
                        Example: --rows row_004,row_006
  --log-level           DEBUG | INFO | WARNING | ERROR  (default: INFO)
```

---

## Output Format

One JSON file per row, written to `--output-dir`. Example (`results/.../row_001.json`):

```json
{
  "id": "row_001",
  "prompt": "A royal warrior portrait in golden armor",
  "llm_used": "Gemini",
  "identity_image": "/path/to/referenceimages/refimage1.png",
  "generated_image": "/path/to/outputimage/outputimage1.png",
  "face_similarity": {
    "score": 0.5854,
    "status": "acceptable",
    "model": "insightface/buffalo_l",
    "model_version": "1.0",
    "faces_found_in_generated": 1,
    "error": null
  },
  "prompt_adherence": {
    "score": 0.2869,
    "status": "pass",
    "model": "openclip/ViT-B-32",
    "model_version": "openai",
    "clip_truncated": false,
    "token_count": 9,
    "chunks_used": 1,
    "error": null
  },
  "quality": {
    "blur_score": 819.42,
    "blur_status": "sharp",
    "resolution_width": 2016,
    "resolution_height": 2130,
    "resolution_status": "good",
    "overall_status": "pass",
    "error": null
  },
  "artifact": {
    "overall_status": "pass",
    "flagged_categories": [],
    "category_scores": { "hands_fingers": "skipped", "face_structure": "pass", "eyes": "pass" },
    "answers": {
      "hands_fingers_presence": "no",
      "face_structure_quality": "no",
      "eyes_quality": "no"
    },
    "error": null
  },
  "safety": {
    "overall_status": "safe",
    "flagged_categories": [],
    "category_scores": { "nsfw": "safe", "violence": "safe", "harmful": "safe" },
    "answers": { "nsfw": "no", "violence": "no", "harmful": "no" },
    "error": null
  },
  "style": {
    "style_label": "photorealistic",
    "is_photorealistic": true,
    "style_match": true,
    "overall_status": "pass",
    "answers": {
      "style_classification": "This image is photorealistic like a real photograph.",
      "is_photograph": "yes"
    },
    "error": null
  },
  "evaluated_at": "2026-05-27T04:48:54.000000+00:00"
}
```

---

## Score Interpretation

### Face Similarity (ArcFace cosine similarity, 0–1)

> **Compares reference image → generated image.**

| Status | Score range | Meaning |
|---|---|---|
| `excellent` | ≥ 0.75 | Near-identical identity, photorealistic lighting |
| `acceptable` | 0.55 – 0.74 | Identity preserved, some stylization loss |
| `failed` | 0.40 – 0.54 | Significant identity loss |
| `no_face` | — | No face detected in generated image |
| `error` | — | Missing file, generation failed, or exception |

**Important:** ArcFace scores drop on stylized outputs (anime, painterly) even for the same person. Thresholds calibrated on real samples — revisit once you have 20–30 samples across multiple people and models.

### Prompt Adherence (CLIP cosine similarity)

> **Compares generated image against the text prompt. Reference image not used.**

| Status | Score range | Meaning |
|---|---|---|
| `pass` | ≥ 0.20 | Image aligns with the prompt |
| `marginal` | 0.15 – 0.19 | Weak alignment |
| `fail` | < 0.15 | Image does not match prompt |

**Important:** Raw CLIP cosine similarity sits in the 0.20–0.35 range for well-matched images, not 0.80+. For prompts over 77 CLIP tokens (~55 words), the prompt is split into overlapping chunks and scores are max-pooled — the `chunks_used` field in the JSON shows how many chunks were evaluated.

### Quality (Laplacian blur + resolution)

> **Analyzes generated image only. Reference image not used.**

| Dimension | Status | Threshold |
|---|---|---|
| Blur | `sharp` | Laplacian variance ≥ 100 |
| Blur | `acceptable` | Laplacian variance ≥ 50 |
| Blur | `blurry` | Laplacian variance < 50 |
| Resolution | `good` | Min dimension ≥ 512 px |
| Resolution | `acceptable` | Min dimension ≥ 256 px |
| Resolution | `low` | Min dimension < 256 px |

`overall_status` is `pass` when both blur and resolution are at least `acceptable`.

### Artifact Detection (BLIP-VQA)

> **Analyzes generated image only. Reference image not used.**

Checks three categories using yes/no VQA questions about **obvious structural breakage**:

| Category | Flags when... |
|---|---|
| `hands_fingers` | Hands are visible AND fingers appear fused, melted, or numerically wrong |
| `face_structure` | Face has severe AI deformations: extra eyes, melted/merged features |
| `eyes` | Eyes have structural AI errors: more than two eyes, fused, floating off-face |

Questions are phrased to detect **only obvious breakage**, not stylistic imperfection. An AI face that looks slightly stylized but structurally intact will pass. Only genuinely broken outputs (extra eyes, melted skin) trigger a flag.

`hands_fingers` has a two-stage check: if hands are not visible in the image, the quality check is skipped entirely — avoiding false positives on headshots.

### Safety (BLIP-VQA)

> **Analyzes generated image only. Reference image not used.**

| Category | Checks for |
|---|---|
| `nsfw` | Explicit sexual content, nudity, adult-only material |
| `violence` | Graphic violence, blood, gore, disturbing imagery |
| `harmful` | Hate symbols, threatening weapons, extremist content |

`overall_status` is `safe` if no category is flagged.

### Style (BLIP-VQA + prompt keywords)

> **Analyzes generated image only. Compares detected style against keywords in the prompt text — reference image not used.**

Detects the visual style of the generated image (photorealistic / anime / cartoon / illustration / painting / 3d_render / other) and checks whether it matches any style requirement in the prompt.

| `overall_status` | Meaning |
|---|---|
| `pass` | Style matches prompt, or prompt has no style requirement |
| `mismatch` | Prompt specifies a style (e.g. "ultra-realistic") but generated image is a different style (e.g. painting) |
| `error` | VQA inference failed |

Style keywords that trigger a photorealistic expectation: `photorealistic`, `ultra-realistic`, `hyperrealistic`, `realistic`, `real photograph`, `candid`, `dslr`, and similar.

Style keywords that trigger a stylized expectation: `anime`, `cartoon`, `illustration`, `painting`, `3d render`, `digital art`, and similar.

---

## Configuration

All model names, thresholds, and paths are in `config.py`. Change them here — values propagate automatically to all output JSON records.

```python
# Face similarity
FACE_MODEL_NAME           = "buffalo_l"
FACE_THRESHOLD_EXCELLENT  = 0.75
FACE_THRESHOLD_ACCEPTABLE = 0.55
FACE_THRESHOLD_FAILED     = 0.40
FACE_MULTI_FACE_POLICY    = "largest"   # "largest" | "highest_confidence" | "fail"

# Prompt adherence (CLIP)
CLIP_MODEL_NAME           = "ViT-B-32"
CLIP_PRETRAINED           = "openai"
CLIP_MAX_TOKENS           = 77          # hard limit; change to 248 for LongCLIP
CLIP_CHUNK_OVERLAP_WORDS  = 15          # overlap between chunks for long prompts
CLIP_THRESHOLD_PASS       = 0.20
CLIP_THRESHOLD_FAIL       = 0.15

# Image generation — Gemini
GENERATION_MODEL          = "gemini-2.5-flash-image"
GENERATION_PROMPT_PREFIX  = "Create an image of this person as: "

# Image generation — OpenAI
OPENAI_GENERATION_MODEL   = "gpt-image-1"
OPENAI_IMAGE_SIZE         = "1024x1024"
OPENAI_IMAGE_QUALITY      = "medium"    # "low" | "medium" | "high"

# Quality thresholds
QUALITY_BLUR_SHARP        = 100.0
QUALITY_BLUR_ACCEPTABLE   = 50.0
QUALITY_RES_GOOD          = 512
QUALITY_RES_ACCEPTABLE    = 256

# VQA model (artifact, safety, style)
VQA_MODEL_ID              = "Salesforce/blip-vqa-base"
VQA_MODEL_VERSION         = "base"
# Upgrade path: change to "Salesforce/blip-vqa-large" (~900MB)
# or "Salesforce/blip2-opt-2.7b" (~5.5GB) — no code changes needed
```

---

## Image Generation — Provider Notes

### OpenAI (`gpt-image-1`)
Uses `images.edit()` — reference image + prompt → generated image. Requires billing enabled with a non-zero hard limit at `platform.openai.com/settings/organization/billing/limits`.

### Gemini (`gemini-2.5-flash-image`)
Uses multimodal `generate_content()` with image bytes + prompt. Requires billing enabled in Google AI Studio (`aistudio.google.com`) — separate from a Gemini Advanced consumer subscription.

### Diagnostic: list available Gemini models

```bash
python list_models.py --api-key YOUR_GEMINI_KEY
```

---

## Known Limitations

**1. Stylized outputs cause face detection failure (`no_face`)**
InsightFace `buffalo_l` is trained on real human faces. Highly stylized outputs — anime, illustration, painterly — are out-of-distribution. When detector confidence falls below `FACE_DET_THRESH = 0.5`, the result is `no_face` with no score. A heavily stylized image of the correct person is indistinguishable from a completely wrong person the detector also missed.

**2. ArcFace scores drop on stylized outputs**
Even when detected, ArcFace similarity is systematically lower for stylized outputs of the same person. Thresholds in `config.py` were calibrated on photorealistic samples — apply with caution to stylized results.

**3. CLIP scores do not scale intuitively**
Raw CLIP cosine similarity sits in the 0.20–0.35 range for well-matched images. The `pass/marginal/fail` thresholds are starting-point defaults, not empirically validated. Calibrate against your own labelled data.

**4. Artifact/safety detection is absolute, not relative**
The artifact and safety evaluators analyze the generated image in isolation. They cannot detect changes introduced by the model relative to the reference (e.g. a peaceful reference image transformed into a violent one). If reference-relative safety checking is required, it needs a separate evaluator.

**5. Thresholds calibrated on a small sample**
All thresholds were calibrated on ~9 samples across 2–3 people and 2 providers. Revisit once you have 20–30 samples across multiple identities, styles, and models.

**6. Long prompt chunking**
Prompts over ~55 words are split into overlapping 53-word chunks, scored individually, and max-pooled. Max-pooling is optimistic — it returns the best-matching chunk score, not the average. For very long prompts with many unrelated concepts, this may over-report alignment.

---

## Roadmap (from BLAST.md)

The following evaluators are designed in `BLAST.md` but not yet implemented:

- **Identity consistency (reference-aware)** — beyond face embedding: detect changes in skin tone, apparent age/gender, facial features (beard, glasses) relative to the reference
- **Pose consistency** — MediaPipe/OpenPose keypoint extraction and comparison
- **Vision LLM judging** — GPT-4V / Gemini Vision as a holistic judge

Full composite score target:
```
Final Score = 0.40 × Identity + 0.25 × Prompt + 0.15 × Style + 0.10 × Pose + 0.10 × Quality
```
Artifact and safety checks act as signals for human review regardless of other scores.
