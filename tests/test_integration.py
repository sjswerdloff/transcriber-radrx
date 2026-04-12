"""Integration tests for transcriber + corrector pipeline.

These tests verify that the post-processor is actually wired into the
transcribe() function (the dead-code bug Cora flagged in PR #1 review).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from transcriber_radrx.transcriber import (
    TranscriptionResult,
    apply_corrections,
    transcribe,
)


@pytest.fixture()
def rt_vocab(tmp_path: Path) -> Path:
    vocab = tmp_path / "rt_vocab.txt"
    vocab.write_text("IMRT\nVMAT\nSBRT\nGy\nGray\nPTV\nGTV\nbrachytherapy\n")
    return vocab


class TestApplyCorrections:
    """Contract tests for apply_corrections() helper."""

    def test_no_vocabulary_returns_unchanged(self) -> None:
        """Contract: missing vocabulary path returns text unchanged."""
        text, corrections = apply_corrections("any text here", None)
        assert text == "any text here"
        assert corrections == []

    def test_with_vocabulary_corrects_case(self, rt_vocab: Path) -> None:
        """Contract: case-insensitive correction works through helper."""
        text, corrections = apply_corrections("imrt plan", rt_vocab)
        assert "IMRT" in text
        assert len(corrections) == 1
        assert corrections[0].method == "case_insensitive"

    def test_safety_preserved_through_helper(self, rt_vocab: Path) -> None:
        """Contract: safety guards work when called through apply_corrections."""
        text, corrections = apply_corrections("our patient is supportive", rt_vocab)
        assert "OAR" not in text
        assert "SBRT" not in text


class TestTranscribePipeline:
    """End-to-end tests verifying corrector is wired into transcribe().

    All tests in this class patch mlx_whisper.transcribe to avoid real model
    loading. They are skipped on platforms where mlx_whisper is not installed
    (Linux / GitHub CI — Apple Silicon only extra).
    """

    @pytest.fixture(autouse=True)
    def _require_mlx_whisper(self) -> None:
        """Skip entire class when mlx_whisper is not installed."""
        pytest.importorskip(
            "mlx_whisper",
            reason="mlx_whisper not installed (Apple Silicon only — install with: uv sync --extra asr-whisper-mlx)",
        )

    def test_corrector_actually_runs_in_transcribe(self, rt_vocab: Path) -> None:
        """Contract: transcribe() applies correction dictionary, not dead code.

        This is the C-3 bug from Cora's review: the post-processor existed
        but transcribe() set corrected_text=raw_text and never called it.
        """
        # Mock mlx_whisper.transcribe to return a known string
        mock_result = {"text": "patient received imrt to the ptv"}

        with (
            patch("mlx_whisper.transcribe", return_value=mock_result),
            patch.object(Path, "exists", return_value=True),
        ):
            result = transcribe(
                Path("/fake/audio.wav"),
                vocabulary_path=rt_vocab,
            )

        assert isinstance(result, TranscriptionResult)
        # Raw text is the mock output
        assert result.text == "patient received imrt to the ptv"
        # Corrected text should have IMRT and PTV uppercased
        assert "IMRT" in result.corrected_text
        assert "PTV" in result.corrected_text
        # Corrections should be tracked with structured info
        assert len(result.corrections) == 2
        for c in result.corrections:
            assert c.method == "case_insensitive"
            assert c.score == 0.95
            assert c.offset >= 0

    def test_transcribe_safety_in_pipeline(self, rt_vocab: Path) -> None:
        """Contract: pipeline safety guards prevent corruption end-to-end.

        Even when called through transcribe(), 'our patient is supportive'
        must not become 'OAR patient is SBRT'.
        """
        unsafe_text = "Our patient is supportive of treatment"
        mock_result = {"text": unsafe_text}

        with (
            patch("mlx_whisper.transcribe", return_value=mock_result),
            patch.object(Path, "exists", return_value=True),
        ):
            result = transcribe(
                Path("/fake/audio.wav"),
                vocabulary_path=rt_vocab,
            )

        # The corrector should have left these common words alone
        assert "OAR" not in result.corrected_text
        assert "SBRT" not in result.corrected_text

    def test_no_vocabulary_no_corrections(self) -> None:
        """Contract: transcribe without vocab returns raw text only."""
        mock_result = {"text": "raw transcription text"}

        with (
            patch("mlx_whisper.transcribe", return_value=mock_result),
            patch.object(Path, "exists", return_value=True),
        ):
            result = transcribe(
                Path("/fake/audio.wav"),
                vocabulary_path=None,
            )

        assert result.text == "raw transcription text"
        assert result.corrected_text == "raw transcription text"
        assert result.corrections == []

    def test_phonetic_disabled_by_default_in_pipeline(self, rt_vocab: Path) -> None:
        """Contract: phonetic correction is OFF by default in transcribe()."""
        # 'guy' would be matched to 'Gy' if phonetic was enabled with permissive thresholds
        mock_result = {"text": "the guy is here"}

        with (
            patch("mlx_whisper.transcribe", return_value=mock_result),
            patch.object(Path, "exists", return_value=True),
        ):
            result = transcribe(
                Path("/fake/audio.wav"),
                vocabulary_path=rt_vocab,
            )

        # Default should preserve "guy" - no phonetic correction to "Gy"
        assert "guy" in result.corrected_text
        assert " Gy " not in result.corrected_text
