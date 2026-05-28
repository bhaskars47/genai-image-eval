# BLAST — Principal Engineer Technical Review
**Project:** BLAST (Batch Evaluation Framework for Identity-Preserving Image Generation)  
**Reviewer:** Principal Engineer / System Architect (via Claude Opus)  
**Date:** 2026-05-27  
**Review Scope:** Full codebase — architecture, code quality, security, performance, production readiness  

---

## 1. Executive Summary

BLAST is a batch-offline CLI tool that evaluates AI-generated identity-preserving images across six dimensions: face similarity (ArcFace), prompt adherence (CLIP), image quality (OpenCV), artifact detection (BLIP-VQA), safety (BLIP-VQA), and style consistency (BLIP-VQA). It reads a CSV manifest, optionally generates images via Gemini or OpenAI, runs all six evaluators, writes one JSON result file per image, and generates a self-contained HTML report per run.

The codebase is **technically competent and well-structured for a research MVP**. The engineering decisions are sound, the architecture is clean, and the living design doc (BLAST.md) is unusually honest about gaps and deviations. However, BLAST is not production-ready. It has zero test coverage, a critical data-loss bug in its CSV mutation path, fragile string-based VQA answer parsing that is correct by luck rather than design, no concurrency, no retry logic, and no deployment infrastructure. The gap between what's built and what BLAST.md describes as the full vision (FastAPI, PostgreSQL, S3, human evaluation platform, regression tracking, pose evaluator) is large.

**If this stays a personal research tool: it works and works well.** If it needs to serve a team, ingest >50 rows reliably, or become a production evaluation service, it needs significant hardening before that transition.

---

## 2. Architecture Overview

```
CLI Entry (run_eval.py)
    │
    ├── [--generate] → pipeline/image_generator.py
    │       ├── GeminiImageGenerator   (google-genai SDK)
    │       └── OpenAIImageGenerator   (openai SDK images.edit)
    │
    └── pipeline/batch_runner.py
            ├── evaluators/face_similarity.py   [InsightFace/ArcFace]
            ├── evaluators/prompt_adherence.py  [OpenCLIP ViT-B-32]
            ├── evaluators/quality.py           [OpenCV Laplacian + PIL]
            ├── evaluators/artifact.py          [BLIP-VQA-base]
            ├── evaluators/safety.py            [BLIP-VQA-base]
            └── evaluators/style.py             [BLIP-VQA-base]
                    ↓
        results/YYYY-MM-DD_HH-MM-SS/
            ├── row_*.json      (one per evaluated image)
            └── report.html     (auto-generated, self-contained)
```

**Data flow:** CSV → batch_runner → evaluators → per-row JSON → HTML report. The CSV is also mutated in-place during generation (generated_image path written back). No database, no API layer, no message queue.

**Config centralization:** `config.py` is the single source of truth for all model names, versions, thresholds, and directory paths. This is genuinely well-designed — changing a threshold in one place propagates to all output JSON records.

**Model sharing:** BLIP-VQA is loaded once in `moondream_loader.py` and injected into all three VQA evaluators. InsightFace and OpenCLIP are each loaded once and reused across rows. This avoids multiple model loads and is the correct pattern.

---

## 3. Strengths

**Architecture and design decisions are strong:**

The evaluator-per-file structure with dataclass-based result objects and `to_dict()` serialization is clean. The boundary between pipeline orchestration and evaluation logic is clear. The decision to switch from CLIP-based paired prompts (which produced 8/9 false positives) to BLIP-VQA with carefully phrased questions is the right call and is well-documented. The "ask about obvious breakage, not perfection" principle in artifact detection is the correct design for VQA-based evaluation.

**Resilient batch processing:** Errors are captured per-row with full context and never kill the batch. The `_error_result()` pattern ensures the pipeline continues even when file paths are broken or models fail.

**Config-driven design:** `config.py` is exemplary — every threshold has a comment explaining how it was calibrated and why, with an explicit note to revisit once the dataset grows. This kind of empirical honesty is rare.

**Incremental resumability:** Writing `generated_image` back into the CSV after generation means interrupted runs can be resumed without regenerating images. The `--rows` flag for targeted evaluation is well thought out.

**Timestamped run isolation:** `results/YYYY-MM-DD_HH-MM-SS/` gives each run its own directory. Runs are non-destructive and can be compared.

**BLAST.md is a model living document:** The section-by-section status annotations (`✅ IMPLEMENTED`, `⚠️ PARTIAL`, `❌ NOT YET BUILT`, `🔄 DEVIATED`) give any new contributor an honest map of the codebase vs the vision. This is better documentation hygiene than most production codebases.

**Prompt chunking for CLIP's 77-token limit:** The overlapping word-level chunking with max-pooling is a genuine improvement over silent truncation. The LongCLIP upgrade path (change one constant in config.py) shows forward-thinking design.

---

## 4. Critical Issues

### CRIT-1: In-place CSV mutation without atomic write — data loss risk

**File:** `run_eval.py`, `_generate_images()`, lines 141–144

```python
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(...)
    writer.writeheader()
    writer.writerows(updated_rows)
```

The source CSV is opened for writing and truncated before any new content is written. If the process is killed, the machine loses power, or `writer.writerows()` throws, the CSV is left as a zero-byte or partially written file. There is no backup, no temp-file-then-rename pattern, no checksum. For a file that is the "single source of truth" for the entire pipeline, this is the highest-risk code path in the codebase.

**Fix:** Write to `csv_path.with_suffix('.tmp')`, then `os.replace(tmp, csv_path)`. `os.replace` is atomic on POSIX systems.

---

### CRIT-2: Zero test coverage

There are no test files anywhere in the project — no `tests/` directory, no `pytest`, no fixtures. For a framework built on:
- Threshold comparisons that were calibrated empirically
- String-based VQA answer parsing that must correctly detect "yes"/"no" variations
- Overlapping chunk logic with edge cases at chunk boundaries
- CSV header validation and path resolution

...the absence of tests means any refactor or threshold change is unverified. The VQA answer parsers alone (`_is_yes`, `_is_no`, `_is_flagged`) have at least 15 distinct string patterns — none validated against a known set of BLIP outputs.

---

### CRIT-3: VQA answer parsing is fragile and correct by luck

**Files:** `evaluators/artifact.py`, `evaluators/safety.py`, `evaluators/style.py`

The actual result JSON from the most recent run (`row_001.json`) reveals the problem directly:
```json
"answers": {
    "nsfw": "male",
    "violence": "no",
    "harmful": "copyright"
}
```

BLIP-VQA answered "male" to the NSFW question and "copyright" to the harmful content question. These answers are completely non-responsive to the questions asked. They parse as "safe" only because `_is_flagged()` doesn't match "male" or "copyright" — the system got the right answer for the wrong reason. This is not a rare edge case; BLIP-VQA-base is known to produce off-topic, hallucinated, or tangentially related answers on complex questions. The current parser has no "uncertain/non-responsive answer" category — every response maps to either flagged or safe/pass.

**Fix:** Add a "uncertain" category for answers that match neither affirmative nor negative patterns. Log these explicitly for human review rather than silently defaulting to "pass."

---

### CRIT-4: `_error_result()` produces inconsistent JSON schema

**File:** `pipeline/batch_runner.py`, lines 278–312

When a row-level error occurs (missing file, bad path), `_error_result()` returns a dict with only `face_similarity` and `prompt_adherence` keys. The keys `quality`, `artifact`, `safety`, and `style` are entirely absent. The `_evaluate_row()` normal path returns all six keys. Downstream consumers (`generate_report.py`, `_print_summary()`) defend against this via `.get()`, but any future consumer that does `r["quality"]` on an error row will get a `KeyError`. The schema contract is broken silently.

**Fix:** `_error_result()` should include all six keys with `None` or `{"status": "skipped", "error": error_msg}` values to maintain schema consistency.

---

### CRIT-5: `sample_input.csv` has absolute paths to a specific developer's machine

```
/Users/bhaskar.srivastava/Documents/Deepeval/referenceimages/refimage1.png
```

This makes the sample completely non-functional for any new contributor or CI environment. The "sample" input is not actually runnable by anyone other than the original developer.

**Fix:** Use relative paths (`referenceimages/refimage1.png`) in sample_input.csv.

---

### CRIT-6: No retry logic on API generation calls

**File:** `pipeline/image_generator.py`

Both `GeminiImageGenerator.generate()` and `OpenAIImageGenerator.generate()` make exactly one API attempt. Rate limit errors, transient 503s, and network blips all result in a permanent `GenerationResult(success=False)` for that row, requiring manual identification and re-run with `--rows`. For batch sizes above ~20, API rate limits are near-certain on first run.

**Fix:** Wrap the API call in exponential backoff with jitter. Libraries like `tenacity` make this two lines. At minimum, catch the specific rate-limit exception class and retry up to 3 times.

---

## 5. Security Risks

### SEC-1: No path traversal protection on CSV image paths

`_resolve_path()` in `batch_runner.py` resolves arbitrary strings relative to the CSV directory with no sanitization. A CSV with `identity_image = "../../../../etc/passwd"` would attempt to open `/etc/passwd` as an image. PIL would fail gracefully, but if this evolves into an API endpoint that accepts external CSV input, this becomes a directory traversal vulnerability.

### SEC-2: API key visible in process list

`--api-key YOUR_KEY` passes the API key as a CLI argument. On any shared machine, `ps aux | grep run_eval` exposes the key in plaintext. The code handles this correctly by also supporting env vars (which don't appear in process lists), and the README recommends env vars. But the CLI flag itself should log a warning or at minimum mask the key in log output.

### SEC-3: Biometric data (face images) stored unencrypted with absolute paths in JSON

Every result JSON stores the absolute path to the identity reference image:
```json
"identity_image": "/Users/bhaskar.srivastava/Documents/Deepeval/referenceimages/refimage4.jpg"
```

Face embeddings themselves are not stored (correctly — they're recomputed per run), but the source images are referenced by absolute path in every result file. BLAST.md correctly identifies this as a future risk when the API/storage layer is built. It's worth noting that even in batch/local mode, sharing result JSON files externally exposes the file system layout and confirms what biometric data exists.

### SEC-4: `patch_moondream_rope.py` mutates shared HuggingFace cache

This script writes to `~/.cache/huggingface/modules/transformers_modules/` — a user-shared model cache. On multi-user systems or containers with shared volumes, running this script modifies cached model files for all processes using that cache. There is no backup of the original file before patching. Additionally, since the project has migrated from Moondream2 to BLIP-VQA, this script is dead code that should not exist in the repository.

---

## 6. Performance Risks

### PERF-1: Sequential evaluation — O(n × 60s) per batch

With BLIP-VQA taking ~10–20s per question and each image requiring 8 questions (3 artifact + 3 safety + 2 style = 8), plus InsightFace (~5–10s) and CLIP (~2–3s), total wall time per image is ~60–80s on CPU. A 100-image batch runs for ~1.5–2.2 hours, sequentially. No parallelism exists in the batch runner.

### PERF-2: Reference image embedding recomputed per row

When the same reference image is used across multiple rows (as in rows 001–003 all using `refimage1.png`), InsightFace re-reads the image from disk, converts to numpy, and runs face detection + embedding extraction each time. For a typical evaluation scenario where each identity is tested against N prompts, this means N redundant model forward passes on the reference.

**Fix:** Cache reference embeddings in a `dict` keyed by resolved absolute path within the evaluator. Savings: (N−1) × ~5–10s per unique identity.

### PERF-3: No image loading cache across evaluators

Each of the four image-using evaluators (face_similarity, prompt_adherence, artifact, quality) independently opens and decodes the same generated image file. PIL is fast, but for large high-resolution images (e.g., row_001's 2016×2130 image), this is 4× redundant I/O and decode per row. 

### PERF-4: Memory ceiling with all models loaded simultaneously

InsightFace buffalo_l (~300MB), OpenCLIP ViT-B-32 (~350MB), BLIP-VQA-base (~450MB) = ~1.1GB model memory baseline, plus Python/PIL/NumPy overhead. On machines with ≤2GB RAM, OS swap pressure will significantly degrade throughput. There is no lazy-loading option or memory-constrained mode.

---

## 7. Scalability Concerns

### SCALE-1: CSV is a single-file, mutable coordination point

The source CSV is both the input manifest and the mutable state store for generation progress. Parallelizing generation across multiple workers is impossible without CSV locking. Adding rows or reprocessing specific rows while another evaluation is running risks CSV corruption. The design works for single-user, sequential use — it breaks under any form of concurrency.

### SCALE-2: Flat JSON result files — no cross-run queryability

Results are written as one JSON per row per run into timestamped directories. There is no way to query "what was the average face score for Gemini across the last 5 runs" without custom scripting. The `generate_report.py` re-reads JSON from a single run's directory — it has no cross-run aggregation. BLAST.md correctly identifies PostgreSQL as the fix; until then, any longitudinal analysis requires manual scripting.

### SCALE-3: No evaluator tiering

All six evaluators run on every row in every batch. For a benchmark of 1000 images, running the full suite (especially 8 BLIP-VQA questions per image) takes ~16–20 hours on CPU. BLAST.md describes a sensible tiering strategy (Tier 1: identity + safety + blur always; Tier 2: CLIP + style sampled; Tier 3: full suite for releases) that would reduce cost by 60–70% for large batches. None of this is implemented.

### SCALE-4: No rate limiting on generation API calls

Back-to-back API calls with no delay, retry budget, or backoff will trigger rate limiting from both Gemini and OpenAI for batches above ~20 images. The current implementation treats rate limit failures as permanent per-row errors.

---

## 8. Code Quality Review

### Good Patterns

The codebase shows consistent engineering discipline:

**Dataclass result objects** (`FaceSimilarityResult`, `PromptAdherenceResult`, etc.) with `to_dict()` create a clean serialization boundary. Adding a new field to a result requires changing exactly one place.

**Config-as-code**: Every magic number in `config.py` has a comment explaining its calibration basis and a note to revisit. This is better discipline than most production ML codebases.

**Type hints**: `str | Path`, `Optional[float]`, `set[str]` throughout — consistent and correct for Python 3.10+.

**Per-row error isolation**: The try/except pattern in every evaluator's `evaluate()` method means BLIP model failures, PIL decode errors, and unexpected exceptions are all caught and returned as structured error results rather than crashing the batch.

### Issues

**`moondream_loader.py` naming is misleading**: The function `load_moondream()` loads BLIP-VQA, not Moondream2. The returned variable `md_tokenizer` is actually a `BlipProcessor` (not a tokenizer in the HuggingFace sense). Throughout all three VQA evaluators, `self._tokenizer` is a `BlipProcessor`. This will confuse any ML engineer who knows that processors and tokenizers have different interfaces.

**`patch_moondream_rope.py` is dead code**: The project switched from Moondream2 to BLIP-VQA precisely to avoid the compatibility issues this script patches. The script should be removed or archived. Its presence at the project root implies it is still needed.

**`_score_to_status()` in `face_similarity.py` has a redundant branch**: 
```python
if score >= config.FACE_THRESHOLD_FAILED:   # >= 0.40 → "failed"
    return "failed"
return "failed"   # < 0.40 → also "failed"
```
Scores below 0.40 (wrong person, completely different identity) get the same "failed" label as scores between 0.40–0.54. A distinct label like `"identity_lost"` for very low scores would provide better signal for analysts.

**`list_models.py` imports at module level without try/except**: 
```python
from google import genai
```
This crashes with `ImportError` on import if `google-genai` is not installed, even for users who only want to use OpenAI. A lazy import with a helpful error message (as done in `image_generator.py`) would be consistent.

**`generate_report.py` uses `file://` URLs for images**: The HTML report renders images via `Path.as_uri()` (absolute `file://` paths). This means:
1. The report is non-portable — images are invisible if the report is moved or shared
2. The report fails completely on any web server or remote viewer
3. Images disappear if the `outputimage/` or `referenceimages/` directories move

Base64 data-URI embedding (`data:image/png;base64,...`) would make reports truly self-contained and shareable.

**`__pycache__` for both Python 3.10 and 3.13 are tracked in git** (no `.gitignore` present). `.DS_Store` files in multiple directories are also untracked.

**`style_label` parser mismatch with BLIP's actual outputs**: BLIP-VQA answers with brief, unexpected strings ("real", "yes", "no", "male", "copyright"). `_parse_style_label()` only matches multi-word phrases like `"real photo"` and `"realistic photo"` — it misses the single-word BLIP answers observed in actual runs. The result JSON shows `style_label = "other"` for an image BLIP described as "real", because "real" alone is not in the parser's phrase list (even though "real" IS in `PHOTOREALISTIC_LABELS` — but `style_label` is set to "other" before that check runs).

---

## 9. Production Readiness Score: **4 / 10**

| Dimension | Score | Notes |
|---|:---:|---|
| Architecture clarity | 8/10 | Clean module separation, good config centralization |
| Code quality | 6/10 | Solid patterns but fragile VQA parsing, dead code, naming issues |
| Test coverage | 0/10 | Zero tests |
| Error handling | 6/10 | Good per-row isolation; CSV mutation risk; incomplete `_error_result` schema |
| Security | 3/10 | Path traversal risk, no key masking, biometric data unencrypted |
| Performance | 3/10 | Sequential, no caching, no parallelism, no rate limiting |
| Scalability | 2/10 | CSV bottleneck, flat file storage, no tiering, no API |
| Observability | 4/10 | Good logging per run, but no metrics, no alerting, no cross-run tracking |
| Deployment | 1/10 | No Docker, no CI/CD, no environment pinning, machine-specific paths in CSV |
| Documentation | 8/10 | README and BLAST.md are honest, detailed, and well-maintained |

**Verdict**: This is production-ready as a personal research tool. It is not production-ready as a shared service or team infrastructure.

---

## 10. Technical Debt Assessment

### High-priority debt (blocks reliability/sharing)

| Item | Risk | Effort |
|---|---|---|
| CSV atomic write (temp → rename) | Data loss on crash | 15 min |
| Remove absolute paths from sample_input.csv | Contributor onboarding breaks | 5 min |
| Add `.gitignore` (pycache, .DS_Store, results/) | Repo pollution | 5 min |
| Fix `_error_result()` missing keys | Silent schema inconsistency | 30 min |
| Retry logic on API generation calls | Batch failures on large runs | 2 hrs |
| Base64-embed images in HTML report | Reports non-portable today | 1 hr |
| Remove / archive `patch_moondream_rope.py` | Dead code confuses contributors | 5 min |

### Medium-priority debt (blocks team use)

| Item | Risk | Effort |
|---|---|---|
| Test suite (unit + integration) | Regressions undetected | 3–5 days |
| VQA answer parser hardening (uncertain category) | False confidence in safety/artifact scores | 1 day |
| Reference embedding cache | 3× slowdown for multi-prompt per identity | 2 hrs |
| Rate limiting on API calls | Batch failures above 20 images | 2 hrs |
| Fix moondream_loader naming (processor, not tokenizer) | ML engineers confused | 30 min |

### Long-term debt (blocks production scaling)

| Item | Risk | Effort |
|---|---|---|
| Async/parallel batch processing | 100-image batch takes 2+ hrs | 1–2 days |
| FastAPI layer over evaluators | Can't serve real-time requests | 3–5 days |
| PostgreSQL for result persistence | No cross-run regression tracking | 3–5 days |
| Composite scoring | No single pass/fail signal per image | 1 day (after pose done) |
| Benchmark dataset (20–30+ labeled samples) | All thresholds uncalibrated | Ongoing |
| Pose evaluator | 10% of composite score missing | 2–3 days |
| Style reference image comparison | Style eval is a keyword proxy, not embedding comparison | 2–3 days |

---

## 11. Recommended Improvements (Priority Ordered)

### P0 — Fix before sharing with anyone (1–2 hours total)

**1. Atomic CSV write in `_generate_images()`**
```python
import tempfile, os
tmp = csv_path.with_suffix('.csv.tmp')
with tmp.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(updated_rows)
os.replace(tmp, csv_path)  # atomic on POSIX
```

**2. Relative paths in `sample_input.csv`**
Replace `/Users/bhaskar.srivastava/Documents/Deepeval/...` with `referenceimages/refimage1.png` etc.

**3. `.gitignore`**
```
__pycache__/
*.pyc
*.pyo
.DS_Store
results/
outputimage/
.env
*.tmp
```

**4. Complete `_error_result()` schema**
Add `quality`, `artifact`, `safety`, `style` with `None` or error-stub values so all result dicts have consistent keys.

**5. Remove `patch_moondream_rope.py`** (or move to `archive/`)

---

### P1 — Fix before team use (1–2 days)

**6. Add retry logic to API generators**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30),
       retry=retry_if_exception_type(APIError))
def _generate_with_retry(...):
    ...
```

**7. VQA answer parser: add `uncertain` category**
When `_is_yes()` and `_is_no()` both return False, classify as `uncertain` rather than `pass`. Log these for human review. The "copyright" and "male" answers from actual runs are `uncertain`, not `safe`.

**8. Reference embedding cache in `FaceSimilarityEvaluator`**
```python
self._embedding_cache: dict[str, Optional[np.ndarray]] = {}

def _get_embedding(self, image_path: Path):
    key = str(image_path.resolve())
    if key not in self._embedding_cache:
        self._embedding_cache[key] = self._compute_embedding(image_path)
    return self._embedding_cache[key]
```
This alone saves ~5–10s per repeated reference image.

**9. Base64-embed images in HTML report**
```python
import base64
def image_data_uri(path_str):
    p = Path(path_str)
    if not p.exists():
        return None
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{data}"
```

**10. Fix `list_models.py` import guard**
Wrap `from google import genai` in a try/except with a clear error message.

---

### P2 — Before scaling beyond 50 rows (3–5 days)

**11. Parallel batch processing**
Use `concurrent.futures.ProcessPoolExecutor` for embarrassingly parallel rows. Each evaluator is stateful (loaded model), so share the evaluator instances via a queue or use thread pool (BLIP/CLIP/InsightFace all release the GIL during inference).
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(_evaluate_row, ...): row_id for row_id in rows}
```

**12. Add evaluator tiering (`--tier 1|2|3` CLI flag)**
- Tier 1: face + safety + quality (always, fast)
- Tier 2: + CLIP + artifact (sampling / CI)
- Tier 3: + style (full benchmark)

**13. Write a test suite**
Minimum coverage targets:
- `test_chunking.py`: test `_split_into_chunks()` at boundary conditions (0 words, exactly _WORDS_PER_CHUNK, 2× limit)
- `test_answer_parsers.py`: test `_is_yes()`, `_is_no()`, `_is_flagged()` against a corpus of known BLIP outputs including edge cases
- `test_batch_runner.py`: mock evaluators, verify error isolation, schema completeness of result dicts
- `test_report.py`: test `compute_summary()` against fixture data, verify no KeyError on error-row dicts

**14. Rate limiting on generation**
Add `time.sleep(1.5)` between API calls at minimum, or use a token bucket. Log estimated completion time based on remaining rows.

---

### P3 — Before production service (weeks)

**15. FastAPI wrapper over batch_runner evaluators**
The evaluator classes are already stateless and injectable. A thin FastAPI layer over `_evaluate_row()` is straightforward.

**16. PostgreSQL result persistence**
Replace flat JSON files with a `evaluations` table. Use JSONB columns for per-evaluator scores. Add a `benchmark_runs` table for cross-run regression tracking.

**17. Composite scoring**
Implement the weighted formula from BLAST.md once pose and style-reference evaluators are complete. Until then, add a `composite_score: null` key to result JSON to signal it is pending.

**18. Pose evaluator (MediaPipe)**
Implement keypoint extraction on reference and generated images. Add to the batch runner as an optional evaluator (Tier 2 or 3).

---

## 12. Final Verdict

BLAST is a **well-reasoned, competently built research tool** that achieved its MVP goals and then some. The code reflects genuine engineering judgment: the switch from CLIP to BLIP-VQA for artifact/safety detection was the right call and was done thoughtfully; the config centralization is excellent; the incremental resumability of generation runs shows practical field-hardening; the BLAST.md living design doc is one of the best pieces of internal documentation I have seen in a project of this size.

**What will break first in production:**

The CSV mutation without atomic write is the most dangerous single line. A killed process during a 100-row generation job leaves a zero-byte CSV — the entire run history is gone. Fix this before anything else.

The second thing that will break is safety/artifact detection reliability. BLIP-VQA-base giving "male" and "copyright" as answers to safety questions and being classified as "safe" works today. It will silently fail for some image type in the future, and without tests, you won't know until a user reports it.

The third is scale. The first time someone tries to run 100 rows through this pipeline, they will wait 1.5–2 hours, hit a rate limit halfway through, get 40 rows with empty `generated_image`, and have no automatic retry. The fix is 2 hours of work.

**The path forward** is clear: BLAST.md already describes it. Fix the P0 items in an afternoon, add tests in a sprint, add parallelism and retry logic the sprint after, then graduate to FastAPI + PostgreSQL when real-time evaluation is needed. The evaluator architecture is solid enough to carry all of that without a rewrite.

**This project deserves the infrastructure investment it doesn't yet have.**

---

*Review based on complete analysis of: `config.py`, `run_eval.py`, `pipeline/batch_runner.py`, `pipeline/image_generator.py`, `pipeline/moondream_loader.py`, `evaluators/face_similarity.py`, `evaluators/prompt_adherence.py`, `evaluators/quality.py`, `evaluators/artifact.py`, `evaluators/safety.py`, `evaluators/style.py`, `generate_report.py`, `patch_moondream_rope.py`, `list_models.py`, `sample_input.csv`, `requirements.txt`, `README.md`, `BLAST.md`, and sample result JSON files from 12 evaluation runs.*
