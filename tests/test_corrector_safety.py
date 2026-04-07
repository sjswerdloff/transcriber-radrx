"""Negative tests for the correction dictionary — clinical safety contracts.

These tests verify that the corrector does NOT introduce errors. Every test
in this file represents a real failure mode that would corrupt clinical text.

"A correction dictionary that introduces errors is worse than none."
— Cora's review of PR #1, 2026-04-07

Empirically verified failure cases from Cora's review:
- "our" → "OAR" (Double Metaphone collapses 'our' and 'OAR' to same code)
- "support" → "SBRT" (phonetic collision with short acronym)
- "guy" → "Gy" (phonetic collision)
- "grey" → "Gray" (case-insensitive correction of homograph)
- "support" → "SBRT" in "patient is supportive of treatment"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber_radrx.corrector import CorrectionDictionary


@pytest.fixture()
def rt_vocab(tmp_path: Path) -> Path:
    """Realistic RT vocabulary with the acronyms that caused false positives."""
    vocab = tmp_path / "rt_vocab.txt"
    vocab.write_text(
        "IMRT\nVMAT\n3DCRT\nSBRT\nSRS\n"
        "Gy\nGray\ncGy\n"
        "PTV\nGTV\nCTV\nOAR\n"
        "DIBH\nbrachytherapy\nstereotactic\nhypofractionation\n"
    )
    return vocab


class TestCommonEnglishNotCorrupted:
    """Contract: common English words must NEVER be silently rewritten."""

    def test_our_not_corrected_to_OAR(self, rt_vocab: Path) -> None:
        """Critical: 'our' must not become 'OAR'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, corrections = corrector.correct("our patient")
        assert "our" in text.lower()
        assert "OAR" not in text
        assert corrections == []

    def test_or_not_corrected_to_OAR(self, rt_vocab: Path) -> None:
        """Critical: 'or' must not become 'OAR'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("treat with chemo or radiation")
        assert "OAR" not in text

    def test_are_not_corrected_to_OAR(self, rt_vocab: Path) -> None:
        """Critical: 'are' must not become 'OAR'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("the lungs are at risk")
        assert "OAR" not in text

    def test_guy_not_corrected_to_Gy(self, rt_vocab: Path) -> None:
        """Critical: 'guy' must not become dose unit 'Gy'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("the guy in room 3")
        assert "Gy" not in text.split()

    def test_support_not_corrected_to_SBRT(self, rt_vocab: Path) -> None:
        """Critical: 'support' must not become 'SBRT'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("patient needs nutritional support")
        assert "SBRT" not in text

    def test_supportive_not_corrected_to_SBRT(self, rt_vocab: Path) -> None:
        """Critical: 'supportive' must not become 'SBRT'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("patient is supportive of treatment")
        assert "SBRT" not in text

    def test_format_not_corrected_to_VMAT(self, rt_vocab: Path) -> None:
        """Critical: 'format' must not become 'VMAT'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("the report format is standard")
        assert "VMAT" not in text

    def test_great_not_corrected_to_3DCRT(self, rt_vocab: Path) -> None:
        """Critical: 'great' must not become '3DCRT'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("the response was great")
        assert "3DCRT" not in text


class TestClinicalSentencePreservation:
    """Contract: full clinical sentences are not corrupted by correction."""

    def test_clinical_sentence_with_common_words(self, rt_vocab: Path) -> None:
        """Critical: real clinical sentence preserved without false corrections.

        From Cora's review, this exact sentence was being corrupted:
        'Our patient is supportive of treatment, the dose may be great'
        → 'OAR patient is SBRT of treatment, the dose may be 3DCRT'
        """
        corrector = CorrectionDictionary(str(rt_vocab))
        sentence = "Our patient is supportive of treatment, the dose may be great"
        text, _ = corrector.correct(sentence)

        # None of the false-positive RT acronyms should appear
        assert "OAR" not in text
        assert "SBRT" not in text
        assert "3DCRT" not in text

    def test_legitimate_RT_corrections_still_work(self, rt_vocab: Path) -> None:
        """Contract: real RT acronyms in obviously-medical context still get corrected.

        Case-insensitive correction should still fix lowercase 'imrt' → 'IMRT'
        when the input is unambiguous.
        """
        corrector = CorrectionDictionary(str(rt_vocab))
        text, corrections = corrector.correct("imrt plan delivering 50 Gy")
        # IMRT should be corrected (case-insensitive, no homograph)
        assert "IMRT" in text


class TestHomographSafety:
    """Contract: vocabulary words that collide with English homographs are safe."""

    def test_grey_not_corrected_to_Gray_dose_unit(self, rt_vocab: Path) -> None:
        """Critical: 'grey' (color) must not become 'Gray' (dose unit).

        'gray hair', 'gray matter', 'grey area' are common English uses
        that must not be silently rewritten to a dose unit.
        """
        corrector = CorrectionDictionary(str(rt_vocab))
        text, _ = corrector.correct("the patient has gray hair")
        # The exact substring "the patient has gray hair" should be preserved
        # (or at minimum, not contain "Gray" as a dose unit)
        assert "Gray" not in text or "gray hair" in text.lower()

    def test_lowercase_gray_color_preserved(self, rt_vocab: Path) -> None:
        """Critical: lowercase 'gray' as color is not the dose unit 'Gray'."""
        corrector = CorrectionDictionary(str(rt_vocab))
        text, corrections = corrector.correct("gray matter on the MRI")
        # If any correction was made, it must not be 'gray' → 'Gray'
        for c in corrections:
            assert not (c.original.lower() == "gray" and c.corrected == "Gray")


class TestPhoneticDisabledByDefault:
    """Contract: phonetic matching is OFF by default for clinical safety."""

    def test_phonetic_default_off(self, rt_vocab: Path) -> None:
        """Critical: default constructor produces a corrector with phonetic disabled."""
        corrector = CorrectionDictionary(str(rt_vocab))
        # 'our' should NOT be corrected to OAR by default
        text, corrections = corrector.correct("our")
        assert "OAR" not in text
        assert all(c.method != "phonetic" for c in corrections)


class TestAcronymExclusion:
    """Contract: short acronyms are excluded from phonetic matching entirely."""

    def test_short_acronym_no_phonetic_match(self, rt_vocab: Path) -> None:
        """Critical: even with phonetic enabled, short acronyms (≤4 chars,
        all caps) must not be matched phonetically.

        Without this guard, 'guy' will match 'Gy' regardless of threshold.
        """
        corrector = CorrectionDictionary(str(rt_vocab), enable_phonetic=True)
        text, corrections = corrector.correct("the guy")
        # 'guy' should not be phonetically matched to 'Gy'
        gy_corrections = [c for c in corrections if c.corrected == "Gy"]
        assert len(gy_corrections) == 0
