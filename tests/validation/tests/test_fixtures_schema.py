"""Schema validation for committed fixtures.

These tests run by default (not gated by validation marker) because
they validate the schema of small committed fixture files, not the
full validation suite. They ensure the SCHEMA.md contract is honored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DICTATION_SAMPLES = FIXTURES_DIR / "rt_dictation_samples.jsonl"

ALLOWED_CATEGORIES = {
    "treatment_summary",
    "dose_prescription",
    "oar_constraints",
    "setup_instructions",
    "progress_note",
    "consent_discussion",
    "acronym_dense",
    "homograph_trap",
}

REQUIRED_FIELDS = {"id", "text", "category", "source", "license", "language"}


class TestDictationSamplesSchema:
    """Contract: rt_dictation_samples.jsonl follows SCHEMA.md."""

    def test_file_exists_or_skip(self) -> None:
        """If fixtures haven't been generated yet, skip — not an error."""
        if not DICTATION_SAMPLES.exists():
            pytest.skip("rt_dictation_samples.jsonl not yet generated")

    def test_each_line_is_valid_json(self) -> None:
        if not DICTATION_SAMPLES.exists():
            pytest.skip("fixture not generated")
        with DICTATION_SAMPLES.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {line_num} not valid JSON: {e}")

    def test_required_fields_present(self) -> None:
        if not DICTATION_SAMPLES.exists():
            pytest.skip("fixture not generated")
        with DICTATION_SAMPLES.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                missing = REQUIRED_FIELDS - obj.keys()
                if missing:
                    pytest.fail(f"Line {line_num} missing required fields: {missing}")

    def test_category_in_allowed_set(self) -> None:
        if not DICTATION_SAMPLES.exists():
            pytest.skip("fixture not generated")
        with DICTATION_SAMPLES.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if obj.get("category") not in ALLOWED_CATEGORIES:
                    pytest.fail(
                        f"Line {line_num} has invalid category: {obj.get('category')}. Allowed: {sorted(ALLOWED_CATEGORIES)}"
                    )

    def test_ids_are_unique(self) -> None:
        if not DICTATION_SAMPLES.exists():
            pytest.skip("fixture not generated")
        seen: set[str] = set()
        with DICTATION_SAMPLES.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                fixture_id = obj.get("id")
                if fixture_id in seen:
                    pytest.fail(f"Line {line_num} has duplicate id: {fixture_id}")
                seen.add(fixture_id)

    def test_text_is_non_empty(self) -> None:
        if not DICTATION_SAMPLES.exists():
            pytest.skip("fixture not generated")
        with DICTATION_SAMPLES.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not obj.get("text", "").strip():
                    pytest.fail(f"Line {line_num} has empty text")
