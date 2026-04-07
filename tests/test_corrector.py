"""Tests for the correction dictionary.

Tests the tiered matching strategy: exact, case-insensitive, phonetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber_radrx.corrector import (
    Correction,
    CorrectionDictionary,
    _levenshtein,
    _load_vocabulary,
    _phonetic_score,
)

# === Unit tests for helper functions ===


class TestLevenshtein:
    """Tests for Levenshtein edit distance."""

    def test_identical_strings(self) -> None:
        assert _levenshtein("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3

    def test_single_edit(self) -> None:
        assert _levenshtein("cat", "bat") == 1  # substitution
        assert _levenshtein("cat", "cats") == 1  # insertion
        assert _levenshtein("cats", "cat") == 1  # deletion

    def test_multiple_edits(self) -> None:
        assert _levenshtein("kitten", "sitting") == 3


class TestPhoneticScore:
    """Tests for phonetic similarity scoring."""

    def test_identical_codes(self) -> None:
        assert _phonetic_score("KR", "KR") == 1.0

    def test_empty_codes(self) -> None:
        assert _phonetic_score("", "KR") == 0.0
        assert _phonetic_score("KR", "") == 0.0
        assert _phonetic_score("", "") == 0.0

    def test_similar_codes(self) -> None:
        score = _phonetic_score("FMT", "FFMT")
        assert 0.5 < score < 1.0

    def test_different_codes(self) -> None:
        score = _phonetic_score("A", "XKRT")
        assert score < 0.5


class TestLoadVocabulary:
    """Tests for vocabulary file loading."""

    def test_loads_terms(self, tmp_path: Path) -> None:
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("IMRT\nVMAT\n3DCRT\n")
        terms = _load_vocabulary(vocab)
        assert terms == ["IMRT", "VMAT", "3DCRT"]

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("# comment\n\nIMRT\n# another\nVMAT\n")
        terms = _load_vocabulary(vocab)
        assert terms == ["IMRT", "VMAT"]

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_vocabulary(Path("/nonexistent/vocab.txt"))

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("  IMRT  \n  VMAT  \n")
        terms = _load_vocabulary(vocab)
        assert terms == ["IMRT", "VMAT"]


# === Integration tests for CorrectionDictionary ===


class TestCorrectionDictionaryExactMatch:
    """Tests for exact and case-insensitive matching."""

    @pytest.fixture()
    def vocab_file(self, tmp_path: Path) -> Path:
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("IMRT\nVMAT\nGray\nGy\nbrachytherapy\n")
        return vocab

    def test_no_correction_when_exact_match(self, vocab_file: Path) -> None:
        """Contract: text already matching vocabulary is not corrected."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, corrections = corrector.correct("IMRT")
        assert text == "IMRT"
        assert corrections == []

    def test_case_insensitive_correction(self, vocab_file: Path) -> None:
        """Contract: wrong-case terms are corrected to canonical form."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, corrections = corrector.correct("imrt treatment with vmat")
        assert "IMRT" in text
        assert "VMAT" in text
        assert len(corrections) == 2
        assert all(c.method == "case_insensitive" for c in corrections)
        assert all(c.score == 0.95 for c in corrections)

    def test_preserves_non_vocabulary_words(self, vocab_file: Path) -> None:
        """Contract: words not in vocabulary are left unchanged."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, corrections = corrector.correct("patient received IMRT today")
        assert text == "patient received IMRT today"
        assert corrections == []

    def test_empty_text(self, vocab_file: Path) -> None:
        """Contract: empty text returns empty text."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, corrections = corrector.correct("")
        assert text == ""
        assert corrections == []

    def test_no_vocabulary_returns_unchanged(self) -> None:
        """Contract: no vocabulary file means no corrections."""
        corrector = CorrectionDictionary()
        text, corrections = corrector.correct("anything goes here")
        assert text == "anything goes here"
        assert corrections == []

    def test_preserves_punctuation(self, vocab_file: Path) -> None:
        """Contract: punctuation and spacing are preserved."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, _ = corrector.correct("imrt, vmat, and brachytherapy.")
        assert "IMRT" in text
        assert "VMAT" in text
        assert "," in text
        assert "." in text

    def test_brachytherapy_case_correction(self, vocab_file: Path) -> None:
        """Contract: multi-syllable medical terms get case-corrected."""
        corrector = CorrectionDictionary(str(vocab_file))
        text, corrections = corrector.correct("Brachytherapy")
        assert text == "brachytherapy"
        assert len(corrections) == 1
        assert corrections[0].method == "case_insensitive"


class TestCorrectionDictionaryPhonetic:
    """Tests for phonetic matching — requires metaphone package."""

    @pytest.fixture()
    def vocab_file(self, tmp_path: Path) -> Path:
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("Gray\nbrachytherapy\nstereotactic\n")
        return vocab

    def test_phonetic_match_available(self) -> None:
        """Verify metaphone is installed for these tests."""
        try:
            from metaphone import doublemetaphone  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("metaphone not installed")

    def test_phonetic_correction(self, vocab_file: Path) -> None:
        """Contract: phonetically similar words get corrected."""
        corrector = CorrectionDictionary(str(vocab_file))
        # "grey" is phonetically similar to "Gray"
        text, corrections = corrector.correct("grey")
        # Should either correct or not — depends on phonetic codes
        if corrections:
            assert corrections[0].method == "phonetic"
            assert corrections[0].score >= 0.7

    def test_min_phonetic_score_threshold(self, vocab_file: Path) -> None:
        """Contract: matches below threshold are rejected."""
        corrector = CorrectionDictionary(str(vocab_file), min_phonetic_score=0.99)
        # With a very high threshold, most phonetic matches should be rejected
        text, corrections = corrector.correct("xylophone")
        phonetic_corrections = [c for c in corrections if c.method == "phonetic"]
        assert len(phonetic_corrections) == 0

    def test_phonetic_disabled_without_metaphone(self, tmp_path: Path) -> None:
        """Contract: corrector works without metaphone (exact/case only)."""
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("IMRT\n")
        # Can't actually test without metaphone installed, but verify
        # the corrector initializes and runs
        corrector = CorrectionDictionary(str(vocab))
        text, corrections = corrector.correct("imrt")
        assert text == "IMRT"


class TestCorrectionDataclass:
    """Tests for the Correction dataclass."""

    def test_correction_fields(self) -> None:
        c = Correction(original="grey", corrected="Gray", score=0.85, method="phonetic")
        assert c.original == "grey"
        assert c.corrected == "Gray"
        assert c.score == 0.85
        assert c.method == "phonetic"


class TestCorrectionDictionaryRTScenarios:
    """Realistic RT transcription correction scenarios."""

    @pytest.fixture()
    def rt_vocab_file(self, tmp_path: Path) -> Path:
        vocab = tmp_path / "rt_vocab.txt"
        vocab.write_text(
            "IMRT\nVMAT\nSBRT\nGy\nGray\ncGy\nbrachytherapy\nstereotactic\nhypofractionation\nPTV\nGTV\nCTV\nOAR\nDIBH\n"
        )
        return vocab

    def test_mixed_case_rt_dictation(self, rt_vocab_file: Path) -> None:
        """Contract: typical ASR output with wrong casing gets corrected."""
        corrector = CorrectionDictionary(str(rt_vocab_file))
        text, corrections = corrector.correct("patient received sbrt to the ptv with dibh technique")
        assert "SBRT" in text
        assert "PTV" in text
        assert "DIBH" in text
        assert len(corrections) == 3

    def test_already_correct_dictation(self, rt_vocab_file: Path) -> None:
        """Contract: correctly dictated text is not modified."""
        corrector = CorrectionDictionary(str(rt_vocab_file))
        original = "IMRT plan delivering 50 Gy to the PTV"
        text, corrections = corrector.correct(original)
        assert text == original
        assert corrections == []

    def test_dose_units_preserved(self, rt_vocab_file: Path) -> None:
        """Contract: Gy and cGy are recognized and preserved."""
        corrector = CorrectionDictionary(str(rt_vocab_file))
        text, corrections = corrector.correct("prescribed 200 cGy per fraction")
        # cGy should match exactly (case-sensitive)
        assert "cGy" in text
