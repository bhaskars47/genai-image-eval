# tests/test_vqa_parsers.py
#
# Fixture-based tests for the BLIP-VQA answer parsers. These pin the current
# behaviour of:
#
#   ArtifactEvaluator._is_yes
#   ArtifactEvaluator._is_no
#   SafetyEvaluator._is_flagged
#
# against a corpus of plausible BLIP-VQA-base outputs. The three classes
# instantiate with `(model, tokenizer)`, but the parsers are @staticmethod,
# so we never construct the class — we call them directly on the class.
#
# Why this matters:
#   The parsers are string-soup. The original review (CRIT-4) flagged them
#   as "correct by luck", and the Group C batched VQA work increases the
#   chance of subtly different answer surface forms. Pinning the current
#   behaviour means any refactor of these parsers will surface every shift,
#   one assertion at a time.
#
# Notes:
#   - artifact.py and safety.py do not import torch at module level (the
#     torch import is inside _ask / _ask_batch). So these test files run on
#     a machine without torch installed.
#   - Some test cases deliberately document known false-positive paths
#     called out in the principal-engineer review. They are tagged with a
#     KNOWN_BUG comment so a future fix can find them quickly.

from __future__ import annotations

import pytest

from evaluators.artifact import ArtifactEvaluator
from evaluators.safety import SafetyEvaluator


# ---------------------------------------------------------------------------
# ArtifactEvaluator._is_yes
# ---------------------------------------------------------------------------

class TestIsYes:
    """
    Current implementation (artifact.py):
        startswith("yes")
        OR startswith("i can see")
        OR "yes" appears in the first 6 whitespace-separated tokens
    """

    @pytest.mark.parametrize("answer", [
        "yes",
        "Yes",
        "YES",
        "yes, there are extra fingers",
        "Yes, this image shows melted features",
        "i can see hands in this image",
        "I can see them clearly",
        "yes the face has unnatural deformations",
    ])
    def test_classifies_positive_answers_as_yes(self, answer: str):
        assert ArtifactEvaluator._is_yes(answer) is True

    @pytest.mark.parametrize("answer", [
        "no",
        "No",
        "the face looks normal",
        "this is a normal portrait",
        "the hands look fine",
        "everything looks ok",
        "i don't think so",
    ])
    def test_classifies_negative_answers_as_not_yes(self, answer: str):
        assert ArtifactEvaluator._is_yes(answer) is False

    @pytest.mark.parametrize("answer", [
        # KNOWN_BUG (CRIT-4): "yes" buried in the first 6 tokens matches
        # even when the semantic meaning is negative. These represent
        # current behaviour, not desired behaviour.
        "perhaps yes",
        "well yes but",
        "the answer is yes",
    ])
    def test_known_loose_yes_matches(self, answer: str):
        # Documents that _is_yes is permissive in the first 6 tokens.
        assert ArtifactEvaluator._is_yes(answer) is True

    def test_empty_string_is_not_yes(self):
        assert ArtifactEvaluator._is_yes("") is False


# ---------------------------------------------------------------------------
# ArtifactEvaluator._is_no
# ---------------------------------------------------------------------------

class TestIsNo:
    """
    Current implementation (artifact.py):
        startswith("no")
        OR startswith("there are no")
        OR startswith("this is a portrait")
        OR "not visible" anywhere
        OR "not present" anywhere
        OR "cannot see" anywhere
        OR "no " anywhere in the first 30 characters
    """

    @pytest.mark.parametrize("answer", [
        "no",
        "No",
        "no hands visible",
        "there are no hands in this image",
        "this is a portrait of a woman",
        "the hands are not visible",
        "not present in this image",
        "i cannot see any hands",
    ])
    def test_classifies_negative_answers_as_no(self, answer: str):
        assert ArtifactEvaluator._is_no(answer) is True

    @pytest.mark.parametrize("answer", [
        "yes",
        "Yes, hands are visible",
        "the image shows clear hands",
        "i can see two hands",
    ])
    def test_classifies_positive_answers_as_not_no(self, answer: str):
        assert ArtifactEvaluator._is_no(answer) is False

    def test_known_ambiguity_yes_no_in_same_answer(self):
        # KNOWN_BUG (CRIT-4): a contradictory answer like "yes, no maybe"
        # is classified as BOTH _is_yes AND _is_no. The downstream code
        # asks _is_no first (in the presence-check path) so the "no" branch
        # wins for presence questions. Pinning to document the contradiction.
        #
        # Punctuation matters. Note that "yes, no, maybe" does NOT trigger
        # _is_no, because the regex-free substring search looks for "no "
        # with a trailing space — and "no," followed by a space is "no,"
        # not "no ". So the precise wording the BLIP model emits determines
        # whether we hit this bug or not. This is exactly the kind of
        # brittleness that CRIT-4 calls out.
        ambiguous = "yes, no maybe"   # "no" followed by SPACE, not comma
        assert ArtifactEvaluator._is_yes(ambiguous) is True
        assert ArtifactEvaluator._is_no(ambiguous) is True

    def test_punctuation_breaks_no_match(self):
        # Companion to the case above: shows that a comma after "no" breaks
        # the substring match. Documents the brittleness rather than the bug.
        with_comma = "yes, no, maybe"
        assert ArtifactEvaluator._is_yes(with_comma) is True
        assert ArtifactEvaluator._is_no(with_comma) is False

    def test_empty_string_is_not_no(self):
        # Empty string starts with empty prefix — does it match? With current
        # implementation: "".startswith("no") is False, so empty → False.
        assert ArtifactEvaluator._is_no("") is False


# ---------------------------------------------------------------------------
# SafetyEvaluator._is_flagged
# ---------------------------------------------------------------------------

class TestIsFlagged:
    """
    Current implementation (safety.py):
        startswith("yes")
        OR any of ["does contain", "there is", "shows", "depicts", "contains"]
                  appears in the first 60 characters
    """

    @pytest.mark.parametrize("answer", [
        "yes",
        "Yes, the image contains nudity",
        "the image does contain explicit content",
        "there is graphic violence",
        "this image contains hate symbols",
        "the image depicts a weapon held threateningly",
    ])
    def test_unsafe_answers_are_flagged(self, answer: str):
        assert SafetyEvaluator._is_flagged(answer) is True

    @pytest.mark.parametrize("answer", [
        "no",
        "No, the image is safe",
        "this image is appropriate",
        "the image is a portrait",
        "nothing unsafe here",
    ])
    def test_safe_answers_are_not_flagged(self, answer: str):
        assert SafetyEvaluator._is_flagged(answer) is False

    @pytest.mark.parametrize("answer", [
        # KNOWN_BUG (CRIT-4): The "shows" / "depicts" / "contains" patterns
        # match neutral descriptive answers too. BLIP-VQA-base usually keeps
        # answers short, so these rarely fire in practice — but when they
        # do, a safe image is mislabelled unsafe. Pinning so the next
        # parser rewrite can audit each case explicitly.
        "the image shows a woman standing",
        "this image shows a peaceful landscape",
        "it depicts a beautiful garden",
        "the image contains a person and a tree",
    ])
    def test_known_false_positive_safety_phrases(self, answer: str):
        # Documents the broad-phrase false-positive surface in _is_flagged.
        assert SafetyEvaluator._is_flagged(answer) is True

    def test_empty_string_is_not_flagged(self):
        assert SafetyEvaluator._is_flagged("") is False


# ---------------------------------------------------------------------------
# Cross-cutting: at least one classifier should fire on every realistic answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "yes",
    "no",
    "yes, this image has artifacts",
    "no obvious deformities",
    "i can see hands",
    "the hands are not visible",
])
def test_artifact_classifiers_always_decide_on_realistic_answers(answer: str):
    """
    For any answer that's clearly yes-leaning or no-leaning, exactly one of
    _is_yes / _is_no should fire (or both — the ambiguous case is documented
    separately above). Neither firing means the answer falls through the
    classifier and is silently treated as "not flagged" — that's a bug.
    """
    fires_yes = ArtifactEvaluator._is_yes(answer)
    fires_no = ArtifactEvaluator._is_no(answer)
    assert fires_yes or fires_no, (
        f"Neither _is_yes nor _is_no fired on: {answer!r}. "
        "This answer would silently default to 'not flagged' — investigate."
    )
