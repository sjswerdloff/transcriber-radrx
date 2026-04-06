"""Tests for transcription engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber_radrx.transcriber import build_initial_prompt


class TestBuildInitialPrompt:
    """Tests for vocabulary-to-prompt construction."""

    def test_builds_prompt_from_vocabulary_file(self, tmp_path: Path) -> None:
        """Contract: vocabulary terms appear in initial_prompt."""
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("IMRT\nVMAT\n3DCRT\n")

        prompt = build_initial_prompt(vocab)

        assert "IMRT" in prompt
        assert "VMAT" in prompt
        assert "3DCRT" in prompt
        assert "radiotherapy" in prompt.lower()

    def test_ignores_comments_and_blank_lines(self, tmp_path: Path) -> None:
        """Contract: lines starting with # and blank lines are excluded."""
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("# This is a comment\n\nIMRT\n# Another comment\nVMAT\n")

        prompt = build_initial_prompt(vocab)

        assert "comment" not in prompt.lower()
        assert "IMRT" in prompt
        assert "VMAT" in prompt

    def test_truncates_to_200_terms(self, tmp_path: Path) -> None:
        """Contract: prompt includes at most 200 terms to stay within token limits."""
        vocab = tmp_path / "vocab.txt"
        terms = [f"term_{i}" for i in range(300)]
        vocab.write_text("\n".join(terms))

        prompt = build_initial_prompt(vocab)

        assert "term_199" in prompt
        assert "term_200" not in prompt

    def test_empty_vocabulary_file(self, tmp_path: Path) -> None:
        """Contract: empty vocabulary produces valid prompt with no terms."""
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("# Only comments\n")

        prompt = build_initial_prompt(vocab)

        assert "radiotherapy" in prompt.lower()

    def test_file_not_found(self) -> None:
        """Contract: missing vocabulary file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_initial_prompt(Path("/nonexistent/vocab.txt"))


class TestTranscribe:
    """Tests for the transcribe function — requires mlx_whisper."""

    def test_audio_file_not_found(self) -> None:
        """Contract: missing audio file raises FileNotFoundError."""
        from transcriber_radrx.transcriber import transcribe

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe(Path("/nonexistent/audio.wav"))
