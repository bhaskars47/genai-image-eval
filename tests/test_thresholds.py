# tests/test_thresholds.py
#
# Boundary tests for the two `_score_to_status` static methods. These pin
# the threshold-to-status mapping so any future threshold edit triggers a
# loud, line-numbered failure rather than silently shifting the meaning of
# every "pass" / "acceptable" label in production output JSON.
#
# Side benefit: surfaces the dead-branch issue in face_similarity
# (CRIT-8 in the original review) — FACE_THRESHOLD_FAILED currently has no
# effect on the returned status. We assert the *current behaviour* so the
# test passes today; if you fix CRIT-8 by adding a fourth bucket, this test
# will go red and tell you exactly which case changed.

from __future__ import annotations

import pytest

import config

# face_similarity has no heavy module-level imports (numpy, PIL only) so it
# imports cleanly even on a machine without InsightFace installed. We only
# touch the static `_score_to_status` method, never construct the class.
from evaluators.face_similarity import FaceSimilarityEvaluator

# prompt_adherence imports torch at module level — gate the import.
torch = pytest.importorskip("torch")
from evaluators.prompt_adherence import PromptAdherenceEvaluator   # noqa: E402


# ---------------------------------------------------------------------------
# FaceSimilarityEvaluator._score_to_status
# ---------------------------------------------------------------------------

class TestFaceScoreToStatus:
    """
    Buckets (per config.py):
      score >= 0.75              → "excellent"
      0.55 <= score < 0.75       → "acceptable"
      score < 0.55               → "failed"   (FACE_THRESHOLD_FAILED is unused)
    """

    @pytest.mark.parametrize("score, expected", [
        # ---- Excellent ----
        (1.00,  "excellent"),
        (0.90,  "excellent"),
        (0.75,  "excellent"),                                # exact boundary
        # ---- Acceptable ----
        (0.7499, "acceptable"),
        (0.65,   "acceptable"),
        (0.55,   "acceptable"),                              # exact boundary
        # ---- Failed ----
        (0.5499, "failed"),
        (0.50,   "failed"),
        (0.40,   "failed"),                                  # FAILED threshold — dead, still "failed"
        (0.3999, "failed"),
        (0.10,   "failed"),
        (0.00,   "failed"),
        (-0.10,  "failed"),                                  # ArcFace can return negatives
    ])
    def test_score_to_status_buckets(self, score: float, expected: str):
        assert FaceSimilarityEvaluator._score_to_status(score) == expected

    def test_thresholds_match_config(self):
        # If someone edits config.py, these tests should be re-derived.
        # This guards against silently bumping a threshold and getting away
        # with it — every threshold edit should also touch this file.
        assert config.FACE_THRESHOLD_EXCELLENT == 0.75
        assert config.FACE_THRESHOLD_ACCEPTABLE == 0.55
        # FACE_THRESHOLD_FAILED exists but doesn't carve a unique bucket —
        # see CRIT-8 in the principal-engineer review. Asserting the value so
        # a future fix to that issue triggers an explicit re-think here.
        assert config.FACE_THRESHOLD_FAILED == 0.40

    def test_failed_threshold_currently_has_no_effect(self):
        """
        Documents CRIT-8: scores at exactly FAILED threshold and below
        return the same status as scores just above it. If you add a
        fourth bucket (e.g. "wrong_person"), expect this test to break —
        that's the signal to update it.
        """
        at = FaceSimilarityEvaluator._score_to_status(config.FACE_THRESHOLD_FAILED)
        below = FaceSimilarityEvaluator._score_to_status(config.FACE_THRESHOLD_FAILED - 0.05)
        assert at == below == "failed"


# ---------------------------------------------------------------------------
# PromptAdherenceEvaluator._score_to_status
# ---------------------------------------------------------------------------

class TestClipScoreToStatus:
    """
    Buckets (per config.py):
      score >= 0.20             → "pass"
      0.15 <= score < 0.20      → "marginal"
      score < 0.15              → "fail"
    """

    @pytest.mark.parametrize("score, expected", [
        # ---- Pass ----
        (1.00,   "pass"),
        (0.35,   "pass"),
        (0.20,   "pass"),                                    # exact boundary
        # ---- Marginal ----
        (0.1999, "marginal"),
        (0.17,   "marginal"),
        (0.15,   "marginal"),                                # exact boundary
        # ---- Fail ----
        (0.1499, "fail"),
        (0.10,   "fail"),
        (0.00,   "fail"),
        (-0.05,  "fail"),                                    # CLIP cosines can dip slightly negative
    ])
    def test_score_to_status_buckets(self, score: float, expected: str):
        assert PromptAdherenceEvaluator._score_to_status(score) == expected

    def test_thresholds_match_config(self):
        assert config.CLIP_THRESHOLD_PASS == 0.20
        assert config.CLIP_THRESHOLD_FAIL == 0.15
