# config.py
# Central configuration for the BLAST evaluation framework.
# Pin model names and versions here — never inline them in evaluator code.
# Changing a model version here will propagate to all output JSON records,
# preserving regression integrity.

from pathlib import Path

# ---------------------------------------------------------------------------
# Face Similarity
# ---------------------------------------------------------------------------

FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_VERSION = "1.0"          # bump this if you swap model packs
FACE_DET_THRESH = 0.5               # InsightFace detection confidence threshold
FACE_MULTI_FACE_POLICY = "largest"  # "largest" | "highest_confidence" | "fail"

# Score thresholds (cosine similarity, 0–1)
# Calibrated on 3 samples (1 person, Gemini, 3 styles) — 2026-05-23
# Revisit once you have 20–30 samples across different people and LLMs.
FACE_THRESHOLD_EXCELLENT  = 0.75    # photorealistic, near-identical lighting
FACE_THRESHOLD_ACCEPTABLE = 0.55    # photorealistic or well-preserved stylized
FACE_THRESHOLD_FAILED     = 0.40    # wrong person or heavy identity loss

# ---------------------------------------------------------------------------
# Prompt Adherence (CLIP)
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"              # OpenAI weights, ~350MB vs 1.71GB for ViT-L-14
CLIP_MODEL_VERSION = "openai"
# To upgrade to the stronger model later, change to:
#   CLIP_MODEL_NAME = "ViT-L-14"
#   CLIP_PRETRAINED = "laion2b_s32b_b82k"
#   CLIP_MODEL_VERSION = "laion2b_s32b_b82k"
CLIP_MAX_TOKENS = 77                    # hard limit in CLIP tokenizer
# Chunking — long prompts are split into overlapping word-level chunks.
# Each chunk is sized so it safely fits within CLIP_MAX_TOKENS.
# Word budget per chunk: floor((CLIP_MAX_TOKENS - 2) / 1.35) ≈ 55 words for 77-token CLIP.
# The chunking logic uses these values directly, so upgrading to LongCLIP only
# requires changing CLIP_MAX_TOKENS to 248 and updating the model names above.
CLIP_CHUNK_OVERLAP_WORDS = 15          # words shared between adjacent chunks

# Raw cosine similarity between CLIP text/image embeddings is NOT on a 0–1
# scale in the intuitive sense. Typical "good" values are 0.25–0.35.
# Set thresholds empirically once you have baseline data.
# These are starting-point defaults.
CLIP_THRESHOLD_PASS = 0.20
CLIP_THRESHOLD_FAIL = 0.15

# ---------------------------------------------------------------------------
# Image Generation (Gemini)
# ---------------------------------------------------------------------------

GENERATION_MODEL = "gemini-2.5-flash-image"
# Prefix prepended to the CSV prompt when calling Gemini — mirrors the manual workflow
GENERATION_PROMPT_PREFIX = "Create an image of this person as: "
# Where generated images are saved (relative to the CSV, or absolute)
GENERATED_IMAGES_DIR = Path("outputimage")

# ---------------------------------------------------------------------------
# Quality Evaluation
# ---------------------------------------------------------------------------

# Blur detection via Laplacian variance (higher = sharper)
# Calibrate once you have real samples — these are reasonable starting points.
QUALITY_BLUR_SHARP     = 100.0   # >= this → sharp
QUALITY_BLUR_ACCEPTABLE = 50.0   # >= this → acceptable
# below QUALITY_BLUR_ACCEPTABLE → blurry

# Minimum resolution (shorter edge in pixels)
QUALITY_RES_GOOD       = 512     # >= this → good
QUALITY_RES_ACCEPTABLE = 256     # >= this → acceptable
# below QUALITY_RES_ACCEPTABLE → low_resolution

# ---------------------------------------------------------------------------
# Artifact Detection (CLIP-based paired prompts)
# ---------------------------------------------------------------------------

# Difference score = sim(image, normal_prompt) - sim(image, artifact_prompt)
# Positive = image looks more "normal"; negative = image looks more "artifact-like"
ARTIFACT_THRESHOLD_PASS = -0.05  # score > this → pass for a given category
# Flag is raised per-category; overall status = "flagged" if any category fails
# -0.05 buffer avoids false positives from CLIP's global embedding on headshots
# where the flagged body part (e.g. hands) isn't even visible in the frame.
# A genuine artifact scores -0.10 to -0.30; near-zero is noise.

# ---------------------------------------------------------------------------
# Safety Evaluation (CLIP-based paired prompts)
# ---------------------------------------------------------------------------

# Difference score = sim(image, safe_prompt) - sim(image, unsafe_prompt)
SAFETY_THRESHOLD_PASS = -0.05    # score > this → safe for a given category
# -0.05 buffer filters out near-zero false positives caused by CLIP training
# data bias: dramatic lighting and anime art styles push embeddings toward
# the unsafe cluster even for clearly safe images. A genuinely unsafe image
# scores -0.10 to -0.30.

# ---------------------------------------------------------------------------
# Image Generation (OpenAI)
# ---------------------------------------------------------------------------

OPENAI_GENERATION_MODEL = "gpt-image-1"
# images.edit() takes the reference image + prompt and generates a new image
# preserving the identity/content of the original.
# Size options: "1024x1024" | "1536x1024" | "1024x1536" | "auto"
OPENAI_IMAGE_SIZE = "1024x1024"
# Quality: "low" | "medium" | "high" | "auto"  (affects cost and latency)
OPENAI_IMAGE_QUALITY = "medium"

# ---------------------------------------------------------------------------
# BLIP-VQA (shared by artifact, safety, and style evaluators)
# ---------------------------------------------------------------------------

# BLIP-VQA base: ~450MB, standard transformers, no trust_remote_code.
# Designed for binary yes/no VQA — exactly what artifact/safety/style checks need.
# To swap in a larger model later, change this to e.g.:
#   "Salesforce/blip-vqa-large"          (~900MB, better accuracy)
#   "Salesforce/blip2-opt-2.7b"          (~5.5GB, much stronger but slower)
VQA_MODEL_ID      = "Salesforce/blip-vqa-base"
VQA_MODEL_VERSION = "base"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results")           # default, overridable from CLI
