# evaluators/quality.py
#
# Evaluates the technical quality of a generated image.
#
# Checks:
#   1. Blur      — Laplacian variance (higher = sharper)
#   2. Resolution — minimum edge length in pixels
#
# Both checks use only OpenCV + PIL — no new dependencies.
# Thresholds are in config.py; calibrate once you have a real sample set.

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
class QualityResult:
    blur_score: Optional[float]       # Laplacian variance — higher is sharper
    blur_status: str                  # sharp | acceptable | blurry
    resolution_width: Optional[int]
    resolution_height: Optional[int]
    resolution_status: str            # good | acceptable | low_resolution
    overall_status: str               # pass | fail | error
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class QualityEvaluator:
    """
    Checks blur and resolution of generated images.
    Stateless — no model loading required. Instantiate once and reuse.
    """

    # ------------------------------------------------------------------
    # Blur
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_blur(image_path: Path) -> float:
        """
        Laplacian variance method.
        Sharp images have high variance; blurry images have low variance.
        """
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            # Fallback: load via PIL and convert. Context-manager open so the
            # source file handle is released before we hand bytes to cv2.
            with Image.open(image_path) as pil_im:
                pil = pil_im.convert("RGB")
            img = np.array(pil)[:, :, ::-1]  # RGB → BGR for OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _blur_status(score: float) -> str:
        if score >= config.QUALITY_BLUR_SHARP:
            return "sharp"
        if score >= config.QUALITY_BLUR_ACCEPTABLE:
            return "acceptable"
        return "blurry"

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _check_resolution(image_path: Path) -> tuple[int, int, str]:
        """Returns (width, height, status)."""
        with Image.open(image_path) as img:
            w, h = img.size
        min_dim = min(w, h)
        if min_dim >= config.QUALITY_RES_GOOD:
            status = "good"
        elif min_dim >= config.QUALITY_RES_ACCEPTABLE:
            status = "acceptable"
        else:
            status = "low_resolution"
        return w, h, status

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------

    @staticmethod
    def _overall_status(blur_status: str, res_status: str) -> str:
        if blur_status == "blurry" or res_status == "low_resolution":
            return "fail"
        return "pass"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, generated_image_path: str | Path) -> QualityResult:
        """
        Run blur and resolution checks on the generated image.

        Parameters
        ----------
        generated_image_path : path to the generated image

        Returns
        -------
        QualityResult
        """
        path = Path(generated_image_path)

        try:
            blur_score = self._compute_blur(path)
            blur_st = self._blur_status(blur_score)
            logger.debug("Blur score for %s: %.2f (%s)", path.name, blur_score, blur_st)

            width, height, res_st = self._check_resolution(path)
            logger.debug("Resolution for %s: %dx%d (%s)", path.name, width, height, res_st)

            overall = self._overall_status(blur_st, res_st)

            return QualityResult(
                blur_score=round(blur_score, 2),
                blur_status=blur_st,
                resolution_width=width,
                resolution_height=height,
                resolution_status=res_st,
                overall_status=overall,
                error=None,
            )

        except Exception as exc:
            logger.exception("QualityEvaluator error for %s: %s", path.name, exc)
            return QualityResult(
                blur_score=None,
                blur_status="error",
                resolution_width=None,
                resolution_height=None,
                resolution_status="error",
                overall_status="error",
                error=str(exc),
            )
