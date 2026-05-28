# evaluators/face_similarity.py
#
# Evaluates whether the generated image contains the same person
# as the identity reference image.
#
# Method:
#   1. Detect faces in both images using InsightFace (buffalo_l).
#   2. Extract 512-d ArcFace embeddings.
#   3. Compute cosine similarity.
#
# Edge cases handled:
#   - No face detected in either image  → status: "no_face"
#   - Multiple faces in generated image → resolved via FACE_MULTI_FACE_POLICY
#   - Model load failure                → raises RuntimeError at init time (fail fast)

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class FaceSimilarityResult:
    score: Optional[float]          # cosine similarity, None if face not detected
    status: str                     # excellent | acceptable | failed | no_face | error
    model: str
    model_version: str
    faces_found_in_generated: int   # helps debug multi-face situations
    error: Optional[str]            # populated only on unexpected exceptions

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FaceSimilarityEvaluator:
    """
    Wraps InsightFace for face detection and ArcFace embedding extraction.

    Instantiate once and reuse across a batch — model loading is expensive (~1–2s).
    """

    def __init__(self) -> None:
        self._app = self._load_model()
        # Instance-level embedding cache, keyed on the resolved absolute path
        # string. Two motivations:
        #   1. A typical CSV reuses the same reference image across many rows
        #      (e.g. refimage1.png on rows 1-3); re-running ArcFace detection
        #      + embedding for each row is pure waste.
        #   2. Generated images are unique per row, so caching them has no
        #      cost penalty — the dict simply doesn't get a second hit for them.
        # We intentionally do NOT use @functools.lru_cache on the bound method
        # because that pattern holds a strong reference to `self` in a module-
        # level cache and leaks the evaluator (well-known anti-pattern). An
        # instance dict gives the same lookup semantics and dies with the
        # evaluator.
        # Caveat: keyed by resolved path, not content hash. If a file is
        # mutated between rows of the same batch the cache will be stale.
        # Realistic for a batch run? No — but worth knowing.
        self._embedding_cache: dict[str, tuple[Optional[np.ndarray], int]] = {}

    def _load_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=config.FACE_MODEL_NAME,
                providers=["CPUExecutionProvider"],   # CPU-only, explicit
            )
            # det_size: (640, 640) is the recommended default for buffalo_l
            app.prepare(ctx_id=-1, det_thresh=config.FACE_DET_THRESH, det_size=(640, 640))
            logger.info("InsightFace model loaded: %s v%s", config.FACE_MODEL_NAME, config.FACE_MODEL_VERSION)
            return app
        except ImportError as e:
            raise RuntimeError(
                "insightface is not installed. Run: pip install insightface onnxruntime"
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, image_path: Path) -> tuple[Optional[np.ndarray], int]:
        """
        Returns (embedding, num_faces_found).
        embedding is None if no face was detected.
        For multi-face images, applies FACE_MULTI_FACE_POLICY.

        Results are cached on the resolved absolute path. The cache is filled
        on both happy-path and no-face-detected outcomes so we don't redo the
        same detection work on a reference image that has no detectable face.
        """
        cache_key = str(Path(image_path).resolve())
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        # Use a context manager so the file descriptor is released as soon
        # as we have the pixel buffer — important for long batches where
        # hundreds of images would otherwise pile up open handles.
        with Image.open(image_path) as im:
            img = np.array(im.convert("RGB"))
        faces = self._app.get(img)

        # Compute the result, then cache it at a single exit. This makes the
        # cache write impossible to forget on any branch.
        if not faces:
            result: tuple[Optional[np.ndarray], int] = (None, 0)
        elif len(faces) == 1:
            result = (faces[0].normed_embedding, 1)
        else:
            # Multiple faces — apply policy
            policy = config.FACE_MULTI_FACE_POLICY
            logger.warning(
                "Multiple faces (%d) found in %s — applying policy: %s",
                len(faces), image_path.name, policy,
            )
            if policy == "largest":
                # bbox area = (x2-x1) * (y2-y1)
                selected = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                result = (selected.normed_embedding, len(faces))
            elif policy == "highest_confidence":
                selected = max(faces, key=lambda f: f.det_score)
                result = (selected.normed_embedding, len(faces))
            elif policy == "fail":
                result = (None, len(faces))
            else:
                raise ValueError(f"Unknown FACE_MULTI_FACE_POLICY: {policy!r}")

        self._embedding_cache[cache_key] = result
        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # InsightFace returns normed_embedding (already L2-normalised),
        # so dot product == cosine similarity.
        return float(np.dot(a, b))

    @staticmethod
    def _score_to_status(score: float) -> str:
        if score >= config.FACE_THRESHOLD_EXCELLENT:
            return "excellent"
        if score >= config.FACE_THRESHOLD_ACCEPTABLE:
            return "acceptable"
        if score >= config.FACE_THRESHOLD_FAILED:
            return "failed"
        return "failed"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        identity_image_path: str | Path,
        generated_image_path: str | Path,
    ) -> FaceSimilarityResult:
        """
        Compare the face in generated_image to the identity reference.

        Parameters
        ----------
        identity_image_path : path to the reference identity image
        generated_image_path : path to the generated image being evaluated

        Returns
        -------
        FaceSimilarityResult
        """
        identity_path = Path(identity_image_path)
        generated_path = Path(generated_image_path)

        base = dict(
            model=f"insightface/{config.FACE_MODEL_NAME}",
            model_version=config.FACE_MODEL_VERSION,
        )

        try:
            ref_embedding, ref_faces = self._get_embedding(identity_path)
            gen_embedding, gen_faces = self._get_embedding(generated_path)

            if ref_embedding is None:
                logger.warning("No face detected in identity image: %s", identity_path.name)
                return FaceSimilarityResult(
                    score=None,
                    status="no_face",
                    faces_found_in_generated=gen_faces,
                    error="No face detected in identity reference image",
                    **base,
                )

            if gen_embedding is None:
                logger.warning("No face detected in generated image: %s", generated_path.name)
                return FaceSimilarityResult(
                    score=None,
                    status="no_face",
                    faces_found_in_generated=gen_faces,
                    error="No face detected in generated image",
                    **base,
                )

            score = self._cosine_similarity(ref_embedding, gen_embedding)
            return FaceSimilarityResult(
                score=round(score, 4),
                status=self._score_to_status(score),
                faces_found_in_generated=gen_faces,
                error=None,
                **base,
            )

        except Exception as exc:
            logger.exception("Unexpected error in FaceSimilarityEvaluator: %s", exc)
            return FaceSimilarityResult(
                score=None,
                status="error",
                faces_found_in_generated=0,
                error=str(exc),
                **base,
            )
