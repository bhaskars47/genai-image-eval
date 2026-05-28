# tests/test_smoke.py
#
# Pure-logic smoke tests for the pipeline orchestration surface. These do
# NOT load any ML weights — they exercise CSV header validation, row-id
# fallback, path resolution, the error-result schema, and the atomic-write
# pattern that the --generate path relies on.
#
# Why these and not others:
#   - These are the helper functions most likely to silently regress on a
#     refactor (string parsing, path math, dict construction).
#   - They run in < 1 s on any developer machine.
#   - They don't require torch/transformers/insightface to be installed.

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import pytest

from pipeline.batch_runner import (
    _validate_csv_headers,
    _row_id,
    _resolve_path,
    _error_result,
    REQUIRED_COLUMNS,
)


# ---------------------------------------------------------------------------
# _validate_csv_headers
# ---------------------------------------------------------------------------

class TestValidateCsvHeaders:
    def test_accepts_exact_required_set(self):
        # Should not raise
        _validate_csv_headers(list(REQUIRED_COLUMNS))

    def test_accepts_required_plus_extras(self):
        headers = list(REQUIRED_COLUMNS) + ["id", "LLM used", "notes"]
        _validate_csv_headers(headers)

    def test_raises_on_missing_required_column(self):
        partial = list(REQUIRED_COLUMNS)[:-1]   # drop one
        with pytest.raises(ValueError) as exc:
            _validate_csv_headers(partial)
        # Error must mention what's missing — operators rely on this
        assert "missing required columns" in str(exc.value).lower()

    def test_raises_on_empty_headers(self):
        with pytest.raises(ValueError):
            _validate_csv_headers([])


# ---------------------------------------------------------------------------
# _row_id
# ---------------------------------------------------------------------------

class TestRowId:
    def test_uses_id_column_when_present(self):
        assert _row_id({"id": "row_042"}, index=0) == "row_042"

    def test_falls_back_to_zero_padded_index(self):
        assert _row_id({}, index=7) == "row_0007"

    def test_strips_whitespace(self):
        assert _row_id({"id": "  row_001  "}, index=0) == "row_001"

    def test_replaces_spaces_with_underscores(self):
        # If a user typed "row 001" in the CSV, the filename should not
        # contain a space (Windows + URLs hate it).
        assert _row_id({"id": "row 001"}, index=0) == "row_001"

    def test_returns_str_even_when_id_is_int(self):
        # CSV always reads strings; defensive in case a caller hands a dict
        # built programmatically with an int id.
        assert isinstance(_row_id({"id": 7}, index=0), str)


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_path_passes_through_unchanged(self, tmp_path):
        absolute = tmp_path / "ref.png"
        resolved = _resolve_path(str(absolute), csv_dir=Path("/some/other/dir"))
        assert resolved == absolute

    def test_relative_path_resolved_against_csv_dir(self, tmp_path):
        # csv lives in <tmp>/manifests, image is at <tmp>/manifests/refs/x.png
        csv_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        resolved = _resolve_path("refs/x.png", csv_dir=csv_dir)
        assert resolved == (csv_dir / "refs/x.png").resolve()

    def test_strips_leading_whitespace(self, tmp_path):
        resolved = _resolve_path("  refs/x.png  ", csv_dir=tmp_path)
        assert resolved.name == "x.png"


# ---------------------------------------------------------------------------
# _error_result schema (regression guard for the A4 fix)
# ---------------------------------------------------------------------------

class TestErrorResultSchema:
    """
    Group A's CRIT-5 fix added quality / artifact / safety / style keys to
    the row-level error path. These tests pin the schema so the next refactor
    can't silently regress it.
    """

    EXPECTED_EVALUATOR_KEYS = {
        "face_similarity",
        "prompt_adherence",
        "quality",
        "artifact",
        "safety",
        "style",
    }

    def _make(self) -> dict:
        return _error_result(
            row_id="row_001",
            prompt="A test prompt",
            identity_raw="ref.png",
            generated_raw="",
            error_msg="something broke",
            evaluated_at="2026-05-27T00:00:00Z",
            llm_used="Gemini",
        )

    def test_top_level_keys_present(self):
        r = self._make()
        for k in ("id", "prompt", "llm_used", "identity_image",
                  "generated_image", "evaluated_at"):
            assert k in r, f"Top-level key missing: {k}"

    def test_all_six_evaluator_keys_present(self):
        r = self._make()
        missing = self.EXPECTED_EVALUATOR_KEYS - set(r.keys())
        assert not missing, f"Missing evaluator keys: {missing}"

    def test_every_evaluator_slot_carries_error_message(self):
        r = self._make()
        for k in self.EXPECTED_EVALUATOR_KEYS:
            assert "error" in r[k], f"{k} slot missing 'error' field"
            assert r[k]["error"] == "something broke", (
                f"{k} error message was not propagated: {r[k]['error']!r}"
            )

    def test_face_and_prompt_status_are_error(self):
        # generate_report.compute_summary indexes these directly
        r = self._make()
        assert r["face_similarity"]["status"] == "error"
        assert r["prompt_adherence"]["status"] == "error"

    def test_quality_overall_status_is_error(self):
        r = self._make()
        assert r["quality"]["overall_status"] == "error"

    def test_artifact_safety_style_overall_status_is_error(self):
        r = self._make()
        assert r["artifact"]["overall_status"] == "error"
        assert r["safety"]["overall_status"] == "error"
        assert r["style"]["overall_status"] == "error"

    def test_prompt_adherence_carries_chunks_used_field(self):
        # generate_report.build_html reads `clip.get("chunks_used", 1)` —
        # if this key is missing the report still works, but if it's a non-int
        # the int() cast in Group B breaks. Pin it to int.
        r = self._make()
        assert "chunks_used" in r["prompt_adherence"]
        assert isinstance(r["prompt_adherence"]["chunks_used"], int)


# ---------------------------------------------------------------------------
# Atomic CSV write (regression guard for the A1 fix)
# ---------------------------------------------------------------------------

class TestAtomicCsvWrite:
    """
    Pins the temp+os.replace pattern used by run_eval._generate_images. We
    can't import _generate_images directly without triggering pipeline imports,
    so we replicate the write block here and assert the contract: original
    file is backed up, temp file is removed, new content lands atomically.
    """

    def test_temp_and_replace_pattern_preserves_backup(self, tmp_path: Path):
        csv_path = tmp_path / "manifest.csv"
        rows_before = [{
            "id": "row_001", "prompt": "p", "identity_image": "r.png",
            "generated_image": "", "LLM used": "Gemini",
        }]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_before[0]))
            w.writeheader()
            w.writerows(rows_before)
        original_text = csv_path.read_text()

        # --- Apply the pattern from run_eval._generate_images ---
        rows_after = list(rows_before)
        rows_after[0] = {**rows_after[0], "generated_image": "outputimage/x_gen.png"}
        fieldnames = list(rows_after[0].keys())

        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)

        tmp_path_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with tmp_path_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_after)
        os.replace(tmp_path_csv, csv_path)

        # Assertions
        assert backup_path.exists(), ".bak was not written"
        assert backup_path.read_text() == original_text, ".bak does not match original"
        assert not tmp_path_csv.exists(), ".tmp file was not cleaned up by os.replace"
        assert "x_gen.png" in csv_path.read_text(), "New content not written"

    def test_backup_does_not_overwrite_existing_backup(self, tmp_path: Path):
        # If a previous session already wrote a .bak, we keep it — preserves
        # the original-original state across multiple --generate invocations.
        csv_path = tmp_path / "manifest.csv"
        csv_path.write_text("id,prompt\nrow_001,p\n")
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        backup_path.write_text("ORIGINAL_BACKUP_CONTENT")

        # Simulate a second run
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)

        # Backup should still be untouched
        assert backup_path.read_text() == "ORIGINAL_BACKUP_CONTENT"
