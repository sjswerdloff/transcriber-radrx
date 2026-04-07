"""Unit tests for the multi-backend bake-off runner.

Covers the pure helper functions that have no backend dependency:
backend spec parsing, per-term accuracy scoring, and summary aggregation.
Backend instantiation and piper synthesis are exercised by the script
itself during empirical bake-off runs, not by these unit tests.

Authors: silas-397300f6
"""

from __future__ import annotations

from tests.validation.scripts.run_multi_backend_e2e import (
    BackendSpec,
    _aggregate_backend,
    _parse_backend_arg,
    _summarize_backend_voice,
    _term_accuracy,
    _term_found,
)


class TestParseBackendArg:
    """Parsing the CLI backend argument."""

    def test_name_only(self) -> None:
        spec = _parse_backend_arg("mlx_whisper")
        assert spec == BackendSpec(name="mlx_whisper", model_id=None)

    def test_name_with_model_id(self) -> None:
        spec = _parse_backend_arg("mlx_whisper:mlx-community/whisper-large-v3-mlx")
        assert spec.name == "mlx_whisper"
        assert spec.model_id == "mlx-community/whisper-large-v3-mlx"

    def test_label_default_model(self) -> None:
        assert _parse_backend_arg("medasr").label == "medasr"

    def test_label_with_override_strips_org(self) -> None:
        spec = _parse_backend_arg("medasr:ainergiz/medasr-mlx-fp16")
        assert spec.label == "medasr:medasr-mlx-fp16"


class TestTermFound:
    """Whitespace-bounded term matching in normalized hypothesis text."""

    def test_exact_word(self) -> None:
        assert _term_found("gy", "prescribed dose of 54 gy in 30 fractions")

    def test_multi_word_phrase(self) -> None:
        assert _term_found("spinal cord", "maximum spinal cord dose less than 45 gy")

    def test_trailing_punctuation(self) -> None:
        # If caller normalized keeping periods, a trailing period must not
        # hide the term.
        assert _term_found("ptv", "boost to the ptv.")

    def test_not_substring_of_another_word(self) -> None:
        # "gy" must not match "foggy"
        assert not _term_found("gy", "the morning was foggy and cool")

    def test_missing_term(self) -> None:
        assert not _term_found("vmat", "intensity modulated radiation therapy")

    def test_empty_term_is_trivially_found(self) -> None:
        # Defensive: an empty term should not cause failures or match.
        assert _term_found("", "anything")


class TestTermAccuracy:
    """Per-fixture term accuracy scoring."""

    def test_all_terms_found(self) -> None:
        found, total, missing = _term_accuracy(
            ["gy", "ptv", "imrt"],
            "The patient received 54 gy to the ptv via imrt.",
        )
        assert (found, total, missing) == (3, 3, [])

    def test_partial_match(self) -> None:
        found, total, missing = _term_accuracy(
            ["gy", "ptv", "vmat"],
            "prescribed 54 gy to the ptv",
        )
        assert found == 2
        assert total == 3
        assert missing == ["vmat"]

    def test_empty_vocab_returns_zeroes(self) -> None:
        # No terms to check → (0, 0, []) and the caller should skip.
        assert _term_accuracy([], "anything") == (0, 0, [])

    def test_case_insensitive(self) -> None:
        found, total, _ = _term_accuracy(
            ["Gy", "PTV"],
            "PRESCRIBED 54 GY to the ptv",
        )
        assert found == 2
        assert total == 2


class TestSummarizeBackendVoice:
    """Summary metrics for one backend × voice cell."""

    def test_empty_results(self) -> None:
        assert _summarize_backend_voice([]) == {"sample_count": 0}

    def test_average_wer_and_recall(self) -> None:
        results: list[dict[str, object]] = [
            {
                "raw_wer": 0.1,
                "corrected_wer": 0.05,
                "terms_found": 3,
                "terms_total": 4,
                "error": False,
            },
            {
                "raw_wer": 0.3,
                "corrected_wer": 0.2,
                "terms_found": 2,
                "terms_total": 4,
                "error": False,
            },
        ]
        summary = _summarize_backend_voice(results)
        assert summary["sample_count"] == 2
        assert summary["avg_raw_wer"] == 0.2
        assert summary["avg_corrected_wer"] == 0.125
        assert summary["terms_found"] == 5
        assert summary["terms_total"] == 8
        assert summary["term_recall"] == 0.625
        assert summary["error_count"] == 0

    def test_term_recall_none_when_no_terms(self) -> None:
        results: list[dict[str, object]] = [
            {"raw_wer": 0.0, "corrected_wer": 0.0, "terms_found": 0, "terms_total": 0},
        ]
        summary = _summarize_backend_voice(results)
        assert summary["term_recall"] is None

    def test_error_count(self) -> None:
        results: list[dict[str, object]] = [
            {"raw_wer": 1.0, "corrected_wer": 1.0, "terms_found": 0, "terms_total": 3, "error": True},
            {"raw_wer": 0.1, "corrected_wer": 0.1, "terms_found": 3, "terms_total": 3, "error": False},
        ]
        summary = _summarize_backend_voice(results)
        assert summary["error_count"] == 1
        assert summary["sample_count"] == 2


class TestAggregateBackend:
    """Aggregating across voices for one backend."""

    def test_aggregates_samples_across_voices(self) -> None:
        voice_reports: list[dict[str, object]] = [
            {
                "voice": "alan",
                "summary": {"sample_count": 2},
                "samples": [
                    {"raw_wer": 0.1, "corrected_wer": 0.1, "terms_found": 3, "terms_total": 4},
                    {"raw_wer": 0.2, "corrected_wer": 0.2, "terms_found": 2, "terms_total": 4},
                ],
            },
            {
                "voice": "lessac",
                "summary": {"sample_count": 2},
                "samples": [
                    {"raw_wer": 0.3, "corrected_wer": 0.3, "terms_found": 4, "terms_total": 4},
                    {"raw_wer": 0.4, "corrected_wer": 0.4, "terms_found": 1, "terms_total": 4},
                ],
            },
        ]
        agg = _aggregate_backend("mlx_whisper", voice_reports)
        assert agg["backend"] == "mlx_whisper"
        by_voice = agg["by_voice"]
        assert isinstance(by_voice, list)
        assert [v["voice"] for v in by_voice] == ["alan", "lessac"]
        overall = agg["overall"]
        assert isinstance(overall, dict)
        assert overall["sample_count"] == 4
        assert overall["avg_raw_wer"] == 0.25
        assert overall["terms_found"] == 10
        assert overall["terms_total"] == 16
        assert overall["term_recall"] == 0.625
