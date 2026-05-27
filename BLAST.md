# BLAST.md
# Multimodal Deep Evaluation Framework for Identity-Preserving Image Generation

## Objective
Build a multimodal evaluation framework for AI image generation systems using:
- Text prompt
- Identity reference image
- Optional style/avatar reference image

The framework evaluates:
- Identity preservation
- Prompt adherence
- Style consistency
- Pose consistency
- Image quality
- Artifact detection
- Safety
- Regression stability

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

---

# Recommended Tech Stack

## Backend
- Python 3.11+
- FastAPI
- Pydantic

## ML/CV
- PyTorch
- OpenCV
- Pillow
- NumPy

## Face Recognition
- InsightFace
- ArcFace

## Vision Language
- OpenCLIP
- CLIP

## Pose
- MediaPipe
- OpenPose

## Storage
- PostgreSQL
- S3 / MinIO

## Experiment Tracking
- Weights & Biases
- MLflow

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

---

# 5. Quality Evaluator

## Detect
- Blur
- Distortion
- Asymmetry
- Low resolution
- Poor lighting

---

# 6. Artifact Detection

## Detect
- Extra fingers
- Broken eyes
- Duplicate limbs
- Warped faces

---

# 7. Safety Evaluator

## Detect
- NSFW
- Violence
- Harmful content
- Deepfake misuse

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

---

# Database Schema

## Tables
- generations
- evaluations
- benchmark_runs

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

---

# Security Requirements

This system processes biometric data.

## Requirements
- Encrypted storage
- Signed URLs
- Secure uploads
- Deletion workflows
- Audit logs

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

---

# Phase 2 — Image Generation
- Integrate SDXL/Flux
- Prompt handling
- Reference image handling

Deliverable:
- Working generation API

---

# Phase 3 — Identity Evaluator
- Integrate InsightFace
- Embedding extraction
- Cosine similarity

Deliverable:
- Identity scoring pipeline

---

# Phase 4 — Prompt Evaluator
- Integrate OpenCLIP
- Text/image embeddings
- Similarity scoring

Deliverable:
- Prompt adherence evaluator

---

# Phase 5 — Quality & Artifact Detection
- Blur detection
- Anatomy checks
- Aesthetic scoring

Deliverable:
- Quality evaluator

---

# Phase 6 — Style & Pose Evaluation
- Pose extraction
- Style embedding comparison

Deliverable:
- Style + pose evaluators

---

# Phase 7 — Evaluation Orchestration
- Async pipeline
- Weighted scoring
- Timeout handling

Deliverable:
- Unified evaluation engine

---

# Phase 8 — Benchmark Dataset
- Collect prompts
- Collect identity references
- Create edge cases

Deliverable:
- Benchmark suite v1

---

# Phase 9 — Dashboard & Reporting
- Benchmark dashboards
- Regression tracking
- Score trends

Deliverable:
- Internal admin dashboard

---

# Phase 10 — Human Evaluation Platform
- Ranking UI
- Reviewer workflows
- Annotation storage

Deliverable:
- Human feedback system

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
