"""Tests for the phrase-level domain corrector.

Covers:
- Each pattern fires on known ASR failure modes.
- Safety tests: clean text passes through unchanged.
- Multi-correction sentences.
- Audit fields on returned PhraseCorrection objects.
- Integration with CorrectionDictionary.correct_full().
"""

from __future__ import annotations

import pytest

from transcriber_radrx.phrase_corrector import (
    PhraseCorrection,
    PhraseCorrectorPipeline,
    apply_phrase_corrections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _correct(text: str) -> tuple[str, list[PhraseCorrection]]:
    """Apply default phrase corrections."""
    return PhraseCorrectorPipeline().correct(text)


# ---------------------------------------------------------------------------
# Dose-unit pattern: gy_after_number and gy_bare_g_after_number
# ---------------------------------------------------------------------------


class TestGyAfterNumber:
    """Pattern: dose unit Gy misrecognized as j/gi/etc after a numeric dose."""

    def test_ji_after_decimal_dose(self) -> None:
        result, corrections = _correct("50.4 ji")
        assert result == "50.4 Gy"
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "gy_after_number"

    def test_j_after_integer_dose(self) -> None:
        result, _ = _correct("60 j")
        assert result == "60 Gy"

    def test_gi_after_dose(self) -> None:
        result, _ = _correct("45 gi")
        assert result == "45 Gy"

    def test_chi_after_dose(self) -> None:
        result, _ = _correct("70 chi")
        assert result == "70 Gy"

    def test_die_after_dose(self) -> None:
        result, _ = _correct("2 die")
        assert result == "2 Gy"

    def test_jai_after_dose(self) -> None:
        result, _ = _correct("20 jai")
        assert result == "20 Gy"

    def test_bare_g_after_dose(self) -> None:
        result, _ = _correct("2 g")
        assert result == "2 Gy"

    def test_bare_G_uppercase_after_dose(self) -> None:
        result, _ = _correct("50 G")
        assert result == "50 Gy"

    def test_embedded_in_sentence(self) -> None:
        result, corrections = _correct("a dose of 50.4 ji was prescribed")
        assert "50.4 Gy" in result
        assert len(corrections) >= 1

    def test_multiple_dose_values_in_sentence(self) -> None:
        result, corrections = _correct("prescribed 45 gi in 25 fractions then 20 ji boost")
        assert "45 Gy" in result
        assert "20 Gy" in result
        assert len(corrections) >= 2

    def test_preserves_g_minor_music(self) -> None:
        """'g minor' should not become 'Gy minor'."""
        result, corrections = _correct("the piece is in g minor")
        assert result == "the piece is in g minor"
        assert len(corrections) == 0

    def test_preserves_gi_bill(self) -> None:
        """'gi bill' should not become 'Gy bill'."""
        result, corrections = _correct("the gi bill was passed in 1944")
        assert result == "the gi bill was passed in 1944"
        assert len(corrections) == 0

    def test_preserves_grams(self) -> None:
        """'50 grams' should not become '50 Gy rams'."""
        result, corrections = _correct("patient weighs 50 grams")
        # 'grams' is not just 'g' so the bare_g pattern won't fire
        assert "grams" in result
        assert len(corrections) == 0

    def test_preserves_g_with_hyphen_context(self) -> None:
        """'2 g/dL' should not fire — hyphen/slash context excluded."""
        result, corrections = _correct("haemoglobin 12 g/dL")
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Compound word joins
# ---------------------------------------------------------------------------


class TestChemoradiationJoin:
    """Pattern: ASR splits chemoradiation into two words."""

    def test_space_separated(self) -> None:
        result, corrections = _correct("chemo radiation was planned")
        assert "chemoradiation" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "chemoradiation_join"

    def test_hyphen_separated(self) -> None:
        result, _ = _correct("chemo-radiation therapy")
        assert "chemoradiation" in result

    def test_already_joined(self) -> None:
        result, corrections = _correct("concurrent chemoradiation")
        assert result == "concurrent chemoradiation"
        assert len(corrections) == 0


class TestNeoadjuvantJoin:
    """Pattern: ASR splits neoadjuvant or substitutes 'new'."""

    def test_hyphen_separated(self) -> None:
        result, corrections = _correct("neo-adjuvant chemotherapy")
        assert "neoadjuvant" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "neoadjuvant_join"

    def test_space_separated(self) -> None:
        result, _ = _correct("neo adjuvant setting")
        assert "neoadjuvant" in result

    def test_new_adjuvant_substitution(self) -> None:
        result, corrections = _correct("new adjuvant chemotherapy")
        assert "neoadjuvant" in result
        assert len(corrections) == 1

    def test_already_correct(self) -> None:
        result, corrections = _correct("neoadjuvant chemoradiation")
        assert result == "neoadjuvant chemoradiation"
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Multi-word term corrections
# ---------------------------------------------------------------------------


class TestDosePainting:
    """Pattern: 'dose pending/bending' -> 'dose painting'."""

    def test_dose_pending(self) -> None:
        result, corrections = _correct("dose pending was applied")
        assert "dose painting" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "dose_painting"

    def test_dose_bending(self) -> None:
        result, _ = _correct("dose bending technique")
        assert "dose painting" in result

    def test_dose_bent_in(self) -> None:
        result, _ = _correct("dose bent in to the tumor")
        assert "dose painting" in result

    def test_already_correct(self) -> None:
        result, corrections = _correct("dose painting by numbers")
        assert result == "dose painting by numbers"
        assert len(corrections) == 0


class TestFiducialMarker:
    """Pattern: 'physical/fusion/situational marker' -> 'fiducial marker'."""

    def test_physical_marker(self) -> None:
        result, corrections = _correct("three physical markers were placed")
        assert "fiducial marker" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "fiducial_marker"

    def test_fusion_marker(self) -> None:
        result, _ = _correct("fusion marker placement confirmed")
        assert "fiducial marker" in result

    def test_situational_marker(self) -> None:
        result, _ = _correct("situational marker identified on CT")
        assert "fiducial marker" in result

    def test_already_correct(self) -> None:
        result, corrections = _correct("fiducial marker inserted under ultrasound")
        assert result == "fiducial marker inserted under ultrasound"
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Single-word substitutions for brachytherapy
# ---------------------------------------------------------------------------


class TestBrachytherapyVariants:
    """Patterns: 'bracket therapy', 'bradytherapy', 'practice therapy' -> 'brachytherapy'."""

    def test_bracket_therapy(self) -> None:
        result, corrections = _correct("bracket therapy seeds were implanted")
        assert "brachytherapy" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "brachytherapy_bracket"

    def test_bradytherapy(self) -> None:
        result, corrections = _correct("bradytherapy was recommended")
        assert "brachytherapy" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "brachytherapy_brady"

    def test_practice_therapy(self) -> None:
        result, corrections = _correct("practice therapy for prostate cancer")
        assert "brachytherapy" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "brachytherapy_practice"

    def test_practic_therapy(self) -> None:
        """Variant without trailing 'e'."""
        result, _ = _correct("practic therapy indicated")
        assert "brachytherapy" in result

    def test_practise_therapy(self) -> None:
        """British spelling variant."""
        result, _ = _correct("practise therapy approach")
        assert "brachytherapy" in result

    def test_already_correct(self) -> None:
        result, corrections = _correct("brachytherapy boost after external beam")
        assert result == "brachytherapy boost after external beam"
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Vulvar context pattern
# ---------------------------------------------------------------------------


class TestVulvarContext:
    """Pattern: 'vulva squamous/carcinoma/...' -> 'vulvar squamous/...'."""

    def test_vulva_squamous(self) -> None:
        result, corrections = _correct("vulva squamous cell carcinoma")
        assert result.startswith("vulvar squamous")
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "vulvar_to_vulva"

    def test_vulva_carcinoma(self) -> None:
        result, _ = _correct("vulva carcinoma stage II")
        assert "vulvar carcinoma" in result

    def test_vulva_cancer(self) -> None:
        result, _ = _correct("vulva cancer was confirmed")
        assert "vulvar cancer" in result

    def test_vulva_lesion(self) -> None:
        result, _ = _correct("vulva lesion biopsy")
        assert "vulvar lesion" in result

    def test_vulva_tumor(self) -> None:
        result, _ = _correct("vulva tumor measuring 2 cm")
        assert "vulvar tumor" in result

    def test_vulva_tumour(self) -> None:
        result, _ = _correct("vulva tumour British spelling")
        assert "vulvar tumour" in result

    def test_vulva_alone_unchanged(self) -> None:
        """'vulva' without medical context should not be changed."""
        result, corrections = _correct("examination of the vulva")
        assert result == "examination of the vulva"
        assert len(corrections) == 0

    def test_vulva_end_of_sentence_unchanged(self) -> None:
        result, corrections = _correct("treatment targeted the vulva")
        assert "vulva" in result
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Lumpectomy
# ---------------------------------------------------------------------------


class TestLumpectomy:
    """Patterns: 'lymphectomy'/'lymptectomy' -> 'lumpectomy'."""

    def test_lymphectomy(self) -> None:
        result, corrections = _correct("lymphectomy with sentinel node biopsy")
        assert "lumpectomy" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "lumpectomy_lymph"

    def test_lymptectomy(self) -> None:
        result, _ = _correct("lymptectomy was performed")
        assert "lumpectomy" in result

    def test_already_correct(self) -> None:
        result, corrections = _correct("lumpectomy followed by whole-breast radiotherapy")
        assert result == "lumpectomy followed by whole-breast radiotherapy"
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Oropharyngeal
# ---------------------------------------------------------------------------


class TestOropharyngeal:
    """Pattern: 'all pharyngeal' -> 'oropharyngeal'."""

    def test_all_pharyngeal(self) -> None:
        result, corrections = _correct("all pharyngeal squamous cell carcinoma")
        assert "oropharyngeal" in result
        assert len(corrections) == 1
        assert corrections[0].pattern_name == "oropharyngeal_all_pharyngeal"

    def test_already_correct(self) -> None:
        result, corrections = _correct("oropharyngeal cancer p16 positive")
        assert result == "oropharyngeal cancer p16 positive"
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Safety tests
# ---------------------------------------------------------------------------


class TestSafety:
    """Ensure clean RT sentences pass through unchanged and no false positives."""

    def test_no_false_positive_on_clean_rt_sentence(self) -> None:
        text = "The patient received 45 Gy in 25 fractions with concurrent chemoradiation."
        result, corrections = _correct(text)
        assert result == text
        assert len(corrections) == 0

    def test_empty_text(self) -> None:
        result, corrections = _correct("")
        assert result == ""
        assert corrections == []

    def test_whitespace_only(self) -> None:
        result, corrections = _correct("   ")
        assert result == "   "
        assert corrections == []

    def test_no_corrections_on_standard_clinical_note(self) -> None:
        text = (
            "Patient underwent lumpectomy followed by adjuvant whole-breast "
            "radiotherapy to 50 Gy in 25 fractions with a 10 Gy boost to the "
            "tumour bed using brachytherapy."
        )
        result, corrections = _correct(text)
        assert result == text
        assert len(corrections) == 0

    def test_multiple_corrections_in_one_sentence(self) -> None:
        result, corrections = _correct("50.4 ji chemo radiation neo adjuvant setting")
        assert "50.4 Gy" in result
        assert "chemoradiation" in result
        assert "neoadjuvant" in result
        assert len(corrections) == 3

    def test_case_insensitive_patterns(self) -> None:
        """ASR output may have unexpected capitalisation."""
        result, _ = _correct("Chemo Radiation was planned")
        assert "chemoradiation" in result.lower()

    def test_case_insensitive_gy(self) -> None:
        result, _ = _correct("45 JI prescribed")
        assert "45 Gy" in result

    def test_canonical_replacement_is_title_case_Gy(self) -> None:
        """The replacement should always be canonical 'Gy', not the matched case."""
        result, _ = _correct("60 JI total")
        assert "Gy" in result


# ---------------------------------------------------------------------------
# Audit field tests
# ---------------------------------------------------------------------------


class TestAuditFields:
    """PhraseCorrection objects must carry all required fields."""

    def test_corrections_list_populated(self) -> None:
        _, corrections = _correct("bracket therapy was used")
        assert len(corrections) == 1
        c = corrections[0]
        assert isinstance(c, PhraseCorrection)
        assert c.original != ""
        assert c.corrected != ""
        assert c.pattern_name != ""
        assert isinstance(c.start, int)
        assert isinstance(c.end, int)

    def test_correction_start_end_order(self) -> None:
        _, corrections = _correct("bradytherapy seeds")
        assert len(corrections) == 1
        c = corrections[0]
        assert c.start < c.end

    def test_correction_original_is_matched_span(self) -> None:
        text = "bracket therapy seeds"
        _, corrections = _correct(text)
        assert len(corrections) == 1
        c = corrections[0]
        # original should be what was matched in the input text
        assert c.original.lower() == "bracket therapy"

    def test_correction_corrected_field(self) -> None:
        _, corrections = _correct("50.4 ji dose prescribed")
        assert len(corrections) == 1
        assert corrections[0].corrected == "50.4 Gy"

    def test_phrase_correction_is_frozen(self) -> None:
        c = PhraseCorrection(
            original="ji",
            corrected="Gy",
            pattern_name="gy_after_number",
            start=5,
            end=7,
        )
        with pytest.raises((AttributeError, TypeError)):
            c.original = "something else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class TestApplyPhraseCorrectionsWrapper:
    """apply_phrase_corrections() should behave identically to the pipeline."""

    def test_wrapper_returns_same_as_pipeline(self) -> None:
        text = "bradytherapy and 50 ji"
        pipeline_result, pipeline_corrections = PhraseCorrectorPipeline().correct(text)
        wrapper_result, wrapper_corrections = apply_phrase_corrections(text)
        assert pipeline_result == wrapper_result
        assert len(pipeline_corrections) == len(wrapper_corrections)

    def test_wrapper_empty_string(self) -> None:
        result, corrections = apply_phrase_corrections("")
        assert result == ""
        assert corrections == []


# ---------------------------------------------------------------------------
# Integration: CorrectionDictionary.correct_full()
# ---------------------------------------------------------------------------


class TestCorrectFullIntegration:
    """correct_full() chains phrase corrections then word corrections."""

    def test_correct_full_chains_phrase_then_word(self, tmp_path: object) -> None:
        """Phrase corrections fire first; word corrector sees the phrase-corrected text."""
        from pathlib import Path

        from transcriber_radrx.corrector import CorrectionDictionary

        assert isinstance(tmp_path, Path)
        vocab_file = tmp_path / "vocab.txt"
        # 'brachytherapy' is a vocabulary term; word corrector knows it
        vocab_file.write_text("brachytherapy\n")

        cd = CorrectionDictionary(str(vocab_file))
        # Input: 'bracket therapy' should become 'brachytherapy' via phrase correction.
        # 'brachytherapy' is already correct so the word corrector adds nothing.
        final_text, word_corrections, phrase_corrections = cd.correct_full("bracket therapy was used")

        assert "brachytherapy" in final_text
        assert len(phrase_corrections) == 1
        assert phrase_corrections[0].pattern_name == "brachytherapy_bracket"
        # Word corrections should be empty — 'brachytherapy' is an exact match
        assert len(word_corrections) == 0

    def test_correct_full_returns_three_tuple(self, tmp_path: object) -> None:
        from pathlib import Path

        from transcriber_radrx.corrector import CorrectionDictionary

        assert isinstance(tmp_path, Path)
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("IMRT\n")

        cd = CorrectionDictionary(str(vocab_file))
        result = cd.correct_full("IMRT planning")
        assert len(result) == 3
        final_text, word_corrections, phrase_corrections = result
        assert isinstance(final_text, str)
        assert isinstance(word_corrections, list)
        assert isinstance(phrase_corrections, list)

    def test_correct_full_phrase_fires_before_word_corrector(self, tmp_path: object) -> None:
        """Verify ordering: phrase pass runs on raw text, word pass runs on phrase-corrected text."""
        from pathlib import Path

        from transcriber_radrx.corrector import CorrectionDictionary

        assert isinstance(tmp_path, Path)
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("chemoradiation\n")

        cd = CorrectionDictionary(str(vocab_file))
        final_text, word_corrections, phrase_corrections = cd.correct_full("chemo radiation concurrently")

        # phrase pass joined 'chemo radiation' → 'chemoradiation'
        assert "chemoradiation" in final_text
        # word corrector sees 'chemoradiation' which is exact match → no correction
        assert len(phrase_corrections) == 1
        assert len(word_corrections) == 0

    def test_correct_full_no_vocab_still_applies_phrases(self, tmp_path: object) -> None:
        """Even with an empty vocabulary, phrase corrections should still fire."""
        from pathlib import Path

        from transcriber_radrx.corrector import CorrectionDictionary

        assert isinstance(tmp_path, Path)
        # CorrectionDictionary with no vocabulary_path
        cd = CorrectionDictionary()
        final_text, word_corrections, phrase_corrections = cd.correct_full("bradytherapy was planned")

        assert "brachytherapy" in final_text
        assert len(phrase_corrections) == 1
        assert len(word_corrections) == 0
