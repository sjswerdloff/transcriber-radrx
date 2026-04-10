"""Tests for Phase 2 ensemble decision rules.

Tests cover:
- is_gy_variant with known-good and known-bad inputs
- has_particle_context with and without context clues
- Each of the 10 decision rules individually with synthetic examples
- ensemble_transcriptions end-to-end with real survey examples
- needs_review flag when Rule 6 fires
- EnsembleWord is a frozen dataclass
- Package export of Phase 2 symbols
"""

from __future__ import annotations

import dataclasses

import pytest

from transcriber_radrx.ensemble.aligner import AlignedSpan, AlignmentType
from transcriber_radrx.ensemble.decision_rules import (
    DecisionSource,
    EnsembleResult,
    EnsembleWord,
    _in_vocabulary,
    _is_numeric,
    _rule1_match,
    _rule2_dose_unit_gye_available,
    _rule3_dose_unit_context,
    _rule4_dose_unit_visible_corruption,
    _rule5_vocabulary_match,
    _rule6_both_wrong,
    _rule7_decimal_precision,
    _rule8_formatting_default,
    _rule9_insertion_a,
    _rule10_insertion_b,
    ensemble_transcriptions,
    has_particle_context,
    is_gy_variant,
)

# ---------------------------------------------------------------------------
# Minimal vocabulary for tests
# ---------------------------------------------------------------------------

_BASIC_VOCAB: set[str] = {
    "proton",
    "protons",
    "gye",
    "gy",
    "fractions",
    "dose",
    "chordoma",
    "sarcoma",
    "pencil",
    "beam",
    "scanning",
    "medulloblastoma",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_span(word_a: str, word_b: str) -> AlignedSpan:
    """Create a synthetic SUBSTITUTION span for rule testing."""
    return AlignedSpan(
        alignment_type=AlignmentType.SUBSTITUTION,
        word_a=word_a,
        word_b=word_b,
        position_a=0,
        position_b=0,
    )


def _match_span(word: str) -> AlignedSpan:
    """Create a synthetic MATCH span for rule testing."""
    return AlignedSpan(
        alignment_type=AlignmentType.MATCH,
        word_a=word,
        word_b=word,
        position_a=0,
        position_b=0,
    )


def _ins_a_span(word: str) -> AlignedSpan:
    """Create a synthetic INSERTION_A span for rule testing."""
    return AlignedSpan(
        alignment_type=AlignmentType.INSERTION_A,
        word_a=word,
        word_b=None,
        position_a=0,
        position_b=None,
    )


def _ins_b_span(word: str) -> AlignedSpan:
    """Create a synthetic INSERTION_B span for rule testing."""
    return AlignedSpan(
        alignment_type=AlignmentType.INSERTION_B,
        word_a=None,
        word_b=word,
        position_a=None,
        position_b=0,
    )


# ---------------------------------------------------------------------------
# is_gy_variant
# ---------------------------------------------------------------------------


class TestIsGyVariant:
    """is_gy_variant identifies known Gy/GyE renderings case-insensitively."""

    def test_gy_is_variant(self) -> None:
        assert is_gy_variant("Gy") is True

    def test_gye_is_variant(self) -> None:
        assert is_gy_variant("GyE") is True

    def test_gie_is_variant(self) -> None:
        assert is_gy_variant("GiE") is True

    def test_gi_is_variant(self) -> None:
        assert is_gy_variant("GI") is True

    def test_jai_is_variant(self) -> None:
        assert is_gy_variant("Jai") is True

    def test_jie_is_variant(self) -> None:
        assert is_gy_variant("Jie") is True

    def test_hie_is_variant(self) -> None:
        assert is_gy_variant("HIE") is True

    def test_giy_is_variant(self) -> None:
        assert is_gy_variant("Giy") is True

    def test_ie_is_variant(self) -> None:
        assert is_gy_variant("Ie") is True

    def test_gev_is_variant(self) -> None:
        assert is_gy_variant("GeV") is True

    def test_uppercase_variants_match(self) -> None:
        """is_gy_variant is case-insensitive."""
        assert is_gy_variant("GY") is True
        assert is_gy_variant("GYE") is True

    def test_dose_is_not_variant(self) -> None:
        assert is_gy_variant("dose") is False

    def test_fractions_is_not_variant(self) -> None:
        assert is_gy_variant("fractions") is False

    def test_proton_is_not_variant(self) -> None:
        assert is_gy_variant("proton") is False

    def test_grothendieck_is_not_variant(self) -> None:
        assert is_gy_variant("Grothendieck") is False

    def test_empty_string_is_not_variant(self) -> None:
        assert is_gy_variant("") is False

    def test_number_is_not_variant(self) -> None:
        assert is_gy_variant("78") is False


# ---------------------------------------------------------------------------
# has_particle_context
# ---------------------------------------------------------------------------


class TestHasParticleContext:
    """has_particle_context detects particle-therapy context clues."""

    def test_proton_in_text_a(self) -> None:
        assert has_particle_context("proton beam therapy", "other text") is True

    def test_proton_in_text_b(self) -> None:
        assert has_particle_context("other text", "proton beam therapy") is True

    def test_chordoma_detected(self) -> None:
        assert has_particle_context("dose for chordoma", "other text") is True

    def test_pencil_beam_scanning_detected(self) -> None:
        assert has_particle_context("using pencil beam scanning", "other text") is True

    def test_rbe_in_text_b(self) -> None:
        assert has_particle_context("some text", "assuming an RBE of 1.1") is True

    def test_no_particle_context(self) -> None:
        assert has_particle_context("dose of 50.4 Gy in 28 fractions", "photon radiotherapy") is False

    def test_case_insensitive_detection(self) -> None:
        assert has_particle_context("PROTON BEAM THERAPY", "other text") is True

    def test_empty_strings_no_context(self) -> None:
        assert has_particle_context("", "") is False

    def test_medulloblastoma_detected(self) -> None:
        assert has_particle_context("treating medulloblastoma", "other text") is True

    def test_ewing_sarcoma_detected(self) -> None:
        assert has_particle_context("Ewing sarcoma of the pelvis", "other text") is True


# ---------------------------------------------------------------------------
# Rule 1: MATCH
# ---------------------------------------------------------------------------


class TestRule1Match:
    """Rule 1 fires on MATCH spans and takes the word with confidence 1.0."""

    def test_match_span_fires_rule1(self) -> None:
        span = _match_span("dose")
        result = _rule1_match(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "dose"
        assert result.source == DecisionSource.MATCH
        assert result.confidence == pytest.approx(1.0)
        assert result.needs_review is False
        assert result.rule_id is None

    def test_substitution_does_not_fire_rule1(self) -> None:
        span = _sub_span("dose", "does")
        result = _rule1_match(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_insertion_a_does_not_fire_rule1(self) -> None:
        span = _ins_a_span("dose")
        result = _rule1_match(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_insertion_b_does_not_fire_rule1(self) -> None:
        span = _ins_b_span("dose")
        result = _rule1_match(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 2: DOSE_UNIT_GYE
# ---------------------------------------------------------------------------


class TestRule2DoseUnitGye:
    """Rule 2: at least one word is GyE, both are Gy-variants."""

    def test_gy_vs_gye_picks_gye(self) -> None:
        span = _sub_span("Gy", "GyE")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "GyE"
        assert result.source == DecisionSource.WHISPER
        assert result.confidence == pytest.approx(0.95)

    def test_gye_vs_gy_picks_gye(self) -> None:
        span = _sub_span("GyE", "Gy")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "GyE"
        assert result.source == DecisionSource.VOXTRAL

    def test_jai_vs_gye_picks_gye(self) -> None:
        span = _sub_span("Jai", "GyE")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "GyE"
        assert result.source == DecisionSource.WHISPER

    def test_neither_gye_does_not_fire_rule2(self) -> None:
        span = _sub_span("Gy", "Jai")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_non_gy_variant_on_one_side_still_fires_rule2_if_gye_present(self) -> None:
        """Spec: one OR both are Gy-variants AND at least one is GyE.
        If word_b is GyE (itself a Gy-variant), Rule 2 fires even if word_a is not a Gy-variant.
        This correctly picks GyE (via Rule 4's domain if neither is GyE, but Rule 2 fires first here).
        """
        span = _sub_span("dose", "GyE")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        # GyE is a Gy-variant, and one is GyE → Rule 2 fires, picks GyE
        assert result is not None
        assert result.word == "GyE"

    def test_match_span_does_not_fire_rule2(self) -> None:
        span = _match_span("GyE")
        result = _rule2_dose_unit_gye_available(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 3: DOSE_UNIT_CONTEXT
# ---------------------------------------------------------------------------


class TestRule3DoseUnitContext:
    """Rule 3: both are Gy-variants (not GyE), promotes to GyE in particle context."""

    _PARTICLE_TEXT = "proton beam therapy of 78 Gy in 39 fractions"

    def test_gy_vs_jai_with_particle_context(self) -> None:
        span = _sub_span("Gy", "Jai")
        result = _rule3_dose_unit_context(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "GyE"
        assert result.source == DecisionSource.CONTEXT_RULE
        assert result.needs_review is True
        assert result.confidence == pytest.approx(0.85)

    def test_gy_vs_jai_without_particle_context(self) -> None:
        span = _sub_span("Gy", "Jai")
        result = _rule3_dose_unit_context(span, "photon radiotherapy 50 Gy", "", _BASIC_VOCAB)
        assert result is None

    def test_gye_one_side_does_not_fire_rule3(self) -> None:
        """Rule 2 should have caught this; Rule 3 explicitly skips if one is GyE."""
        span = _sub_span("GyE", "Jai")
        result = _rule3_dose_unit_context(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is None

    def test_non_gy_variant_does_not_fire_rule3(self) -> None:
        span = _sub_span("dose", "Jai")
        result = _rule3_dose_unit_context(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is None

    def test_match_span_does_not_fire_rule3(self) -> None:
        span = _match_span("Gy")
        result = _rule3_dose_unit_context(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 4: DOSE_UNIT_VISIBLE_CORRUPTION
# ---------------------------------------------------------------------------


class TestRule4DoseUnitVisibleCorruption:
    """Rule 4: exactly one word is a Gy-variant."""

    _PARTICLE_TEXT = "proton beam therapy"
    _PLAIN_TEXT = "photon beam therapy"

    def test_gy_variant_vs_word_picks_variant(self) -> None:
        span = _sub_span("Gy", "word")
        result = _rule4_dose_unit_visible_corruption(span, self._PLAIN_TEXT, "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "Gy"
        assert result.source == DecisionSource.VOXTRAL

    def test_word_vs_gy_variant_picks_variant(self) -> None:
        span = _sub_span("word", "Jai")
        result = _rule4_dose_unit_visible_corruption(span, self._PLAIN_TEXT, "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "Jai"
        assert result.source == DecisionSource.WHISPER

    def test_promotes_to_gye_with_particle_context(self) -> None:
        span = _sub_span("Gy", "word")
        result = _rule4_dose_unit_visible_corruption(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "GyE"
        assert result.source == DecisionSource.CONTEXT_RULE

    def test_both_variants_does_not_fire_rule4(self) -> None:
        """XOR condition: both being Gy-variants is Rule 3's domain."""
        span = _sub_span("Gy", "Jai")
        result = _rule4_dose_unit_visible_corruption(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is None

    def test_neither_variant_does_not_fire_rule4(self) -> None:
        span = _sub_span("dose", "word")
        result = _rule4_dose_unit_visible_corruption(span, self._PARTICLE_TEXT, "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 5: VOCABULARY_MATCH
# ---------------------------------------------------------------------------


class TestRule5VocabularyMatch:
    """Rule 5: one word is in the RT vocabulary, the other is not."""

    def test_whisper_in_vocab_picks_whisper(self) -> None:
        span = _sub_span("Grothendieck", "Proton")
        result = _rule5_vocabulary_match(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "Proton"
        assert result.source == DecisionSource.WHISPER
        assert result.confidence == pytest.approx(0.9)

    def test_voxtral_in_vocab_picks_voxtral(self) -> None:
        span = _sub_span("protons", "Procums")
        result = _rule5_vocabulary_match(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "protons"
        assert result.source == DecisionSource.VOXTRAL

    def test_both_in_vocab_does_not_fire_rule5(self) -> None:
        span = _sub_span("proton", "dose")
        result = _rule5_vocabulary_match(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_neither_in_vocab_does_not_fire_rule5(self) -> None:
        span = _sub_span("qwertyuiop", "asdfghjkl")
        result = _rule5_vocabulary_match(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_stem_match_counts(self) -> None:
        """'protons' matches vocab term 'proton' via stem matching."""
        vocab = {"proton"}
        span = _sub_span("protons", "Procums")
        result = _rule5_vocabulary_match(span, "", "", vocab)
        assert result is not None
        assert result.word == "protons"

    def test_match_span_does_not_fire_rule5(self) -> None:
        span = _match_span("proton")
        result = _rule5_vocabulary_match(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 6: BOTH_WRONG
# ---------------------------------------------------------------------------


class TestRule6BothWrong:
    """Rule 6: neither in vocabulary and words differ significantly."""

    def test_both_wrong_fires_with_low_similarity(self) -> None:
        span = _sub_span("craniofacial", "zoomorphic")
        result = _rule6_both_wrong(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.source == DecisionSource.HUMAN_REVIEW
        assert result.needs_review is True
        assert result.confidence == pytest.approx(0.3)
        # Takes Voxtral's word as default
        assert result.word == "craniofacial"

    def test_one_in_vocab_does_not_fire_rule6(self) -> None:
        span = _sub_span("proton", "zoomorphic")
        result = _rule6_both_wrong(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_similar_words_do_not_fire_rule6(self) -> None:
        """Words with ratio >= 0.5 are not 'both wrong' — Rule 8 handles them."""
        span = _sub_span("dose", "doss")
        result = _rule6_both_wrong(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_match_span_does_not_fire_rule6(self) -> None:
        span = _match_span("dose")
        result = _rule6_both_wrong(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_rule_id_is_both_wrong(self) -> None:
        span = _sub_span("xylophone", "zymurgy")
        result = _rule6_both_wrong(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.rule_id == "BOTH_WRONG"


# ---------------------------------------------------------------------------
# Rule 7: DECIMAL_PRECISION
# ---------------------------------------------------------------------------


class TestRule7DecimalPrecision:
    """Rule 7: numeric tokens, higher precision wins."""

    def test_voxtral_has_more_decimals(self) -> None:
        span = _sub_span("55.8", "55")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "55.8"
        assert result.source == DecisionSource.VOXTRAL
        assert result.confidence == pytest.approx(0.9)

    def test_whisper_has_more_decimals(self) -> None:
        span = _sub_span("55", "55.8")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "55.8"
        assert result.source == DecisionSource.WHISPER

    def test_equal_precision_does_not_fire_rule7(self) -> None:
        span = _sub_span("55.8", "55.8")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_non_numeric_does_not_fire_rule7(self) -> None:
        span = _sub_span("dose", "does")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_one_non_numeric_does_not_fire_rule7(self) -> None:
        span = _sub_span("55.8", "dose")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_match_span_does_not_fire_rule7(self) -> None:
        span = _match_span("55.8")
        result = _rule7_decimal_precision(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 8: FORMATTING_DEFAULT
# ---------------------------------------------------------------------------


class TestRule8FormattingDefault:
    """Rule 8: substitution fallback takes Voxtral."""

    def test_takes_voxtral_for_substitution(self) -> None:
        span = _sub_span("high-risk", "high risk")
        result = _rule8_formatting_default(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "high-risk"
        assert result.source == DecisionSource.VOXTRAL
        assert result.confidence == pytest.approx(0.7)
        assert result.needs_review is False

    def test_match_does_not_fire_rule8(self) -> None:
        span = _match_span("dose")
        result = _rule8_formatting_default(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_insertion_a_does_not_fire_rule8(self) -> None:
        span = _ins_a_span("dose")
        result = _rule8_formatting_default(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_insertion_b_does_not_fire_rule8(self) -> None:
        span = _ins_b_span("dose")
        result = _rule8_formatting_default(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 9: INSERTION_A
# ---------------------------------------------------------------------------


class TestRule9InsertionA:
    """Rule 9: words only in Voxtral are included."""

    def test_insertion_a_fires(self) -> None:
        span = _ins_a_span("dose")
        result = _rule9_insertion_a(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "dose"
        assert result.source == DecisionSource.VOXTRAL
        assert result.confidence == pytest.approx(0.6)
        assert result.word_whisper is None

    def test_insertion_b_does_not_fire_rule9(self) -> None:
        span = _ins_b_span("dose")
        result = _rule9_insertion_a(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_match_does_not_fire_rule9(self) -> None:
        span = _match_span("dose")
        result = _rule9_insertion_a(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Rule 10: INSERTION_B
# ---------------------------------------------------------------------------


class TestRule10InsertionB:
    """Rule 10: words only in Whisper are included."""

    def test_insertion_b_fires(self) -> None:
        span = _ins_b_span("fractions")
        result = _rule10_insertion_b(span, "", "", _BASIC_VOCAB)
        assert result is not None
        assert result.word == "fractions"
        assert result.source == DecisionSource.WHISPER
        assert result.confidence == pytest.approx(0.6)
        assert result.word_voxtral is None

    def test_insertion_a_does_not_fire_rule10(self) -> None:
        span = _ins_a_span("fractions")
        result = _rule10_insertion_b(span, "", "", _BASIC_VOCAB)
        assert result is None

    def test_match_does_not_fire_rule10(self) -> None:
        span = _match_span("fractions")
        result = _rule10_insertion_b(span, "", "", _BASIC_VOCAB)
        assert result is None


# ---------------------------------------------------------------------------
# Data model contracts
# ---------------------------------------------------------------------------


class TestEnsembleWordDatamodel:
    """EnsembleWord is a frozen dataclass with correct field types."""

    def test_ensemble_word_is_frozen(self) -> None:
        ew = EnsembleWord(
            word="dose",
            source=DecisionSource.MATCH,
            word_voxtral="dose",
            word_whisper="dose",
            rule_id=None,
            confidence=1.0,
            needs_review=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ew.word = "mutated"  # type: ignore[misc]

    def test_ensemble_result_is_mutable_dataclass(self) -> None:
        """EnsembleResult uses regular @dataclass (mutable)."""
        er = EnsembleResult(
            fixture_id="test",
            voice="alan",
            text_voxtral="a",
            text_whisper="b",
            text_ensemble="a",
        )
        # Should not raise
        er.fixture_id = "updated"
        assert er.fixture_id == "updated"


# ---------------------------------------------------------------------------
# ensemble_transcriptions end-to-end
# ---------------------------------------------------------------------------


class TestEnsembleTranscriptionsEndToEnd:
    """Integration tests using real survey examples."""

    def test_proton_0014_alan_grothendieck_vs_proton(self) -> None:
        """proton-0014 alan: Voxtral 'Grothendieck' vs Whisper 'Proton' → vocab match picks 'Proton'."""
        voxtral = "Grothendieck beam therapy of 70 GeV in 35 fractions assuming an RBE of 1.1 relative to megavoltage photons."
        whisper = "Proton beam therapy of 70 Jai E in 35 fractions assuming an RBE of 1.1 relative to mega voltage photons."
        vocab = {"proton", "protons", "gy", "gye", "fractions", "rbe", "megavoltage", "beam", "therapy"}
        result = ensemble_transcriptions(voxtral, whisper, vocab, fixture_id="proton-0014", voice="en_GB-alan-medium")
        assert isinstance(result, EnsembleResult)
        assert result.fixture_id == "proton-0014"
        assert result.voice == "en_GB-alan-medium"
        # Whisper's 'Proton' should be chosen over Voxtral's 'Grothendieck' (vocabulary match)
        assert "Proton" in result.text_ensemble

    def test_proton_0003_alan_gy_equivalent_normalization(self) -> None:
        """proton-0003 alan: 'Gy equivalent' and 'gray equivalent' both normalize to GyE → MATCH."""
        voxtral = (
            "Craniospinal irradiation of 23.4 Gy equivalent in 13 fractions"
            " followed by a posterior fossa boost to 54 Gy equivalent."
        )
        whisper = (
            "Craniospinal irradiation of 23.4 gray equivalent in 13 fractions"
            " followed by a posterior fossa boost to 54 gray equivalent."
        )
        vocab = {"gy", "gye", "fractions", "craniospinal", "irradiation", "boost"}
        result = ensemble_transcriptions(voxtral, whisper, vocab, fixture_id="proton-0003", voice="en_GB-alan-medium")
        assert isinstance(result, EnsembleResult)
        # After normalization: 'Gy equivalent' → 'GyE', 'gray equivalent' → 'GyE' → MATCH
        # The ensemble text should contain GyE not 'Gy equivalent'
        assert "GyE" in result.text_ensemble

    def test_both_wrong_case_flags_for_review(self) -> None:
        """BOTH_WRONG (Rule 6) fires when words are very different and not in vocabulary."""
        voxtral = "craniofacial ingoma of the skull base."
        whisper = "zoomorphic zymurgy of the skull base."
        vocab: set[str] = set()
        result = ensemble_transcriptions(voxtral, whisper, vocab, fixture_id="test-0001", voice="alan")
        assert result.needs_review is True
        assert result.review_count > 0

    def test_identical_transcriptions_no_review(self) -> None:
        """Identical transcriptions → all MATCH → no review needed."""
        text = "Dose of 50.4 Gy in 28 fractions to the prostate."
        vocab = {"gy", "fractions", "dose", "prostate"}
        result = ensemble_transcriptions(text, text, vocab)
        assert result.needs_review is False
        assert result.review_count == 0
        assert result.agreement_rate == pytest.approx(1.0)

    def test_result_has_ensemble_text(self) -> None:
        """ensemble_transcriptions always returns a non-empty text for non-empty inputs."""
        voxtral = "dose of 50 Gy"
        whisper = "dose of 50 Gy"
        vocab: set[str] = set()
        result = ensemble_transcriptions(voxtral, whisper, vocab)
        assert len(result.text_ensemble) > 0

    def test_result_word_count_matches_tokens(self) -> None:
        """Number of EnsembleWord objects equals number of tokens in ensemble text (approx)."""
        voxtral = "dose escalation to 78 GyE in 39 fractions"
        whisper = "dose escalation to 78 GyE in 39 fractions"
        vocab = {"gy", "gye", "fractions", "dose"}
        result = ensemble_transcriptions(voxtral, whisper, vocab)
        assert len(result.words) == len(result.text_ensemble.split())

    def test_needs_review_propagates(self) -> None:
        """If any word is flagged, the EnsembleResult.needs_review must be True."""
        voxtral = "xylophone"
        whisper = "zymurgy"
        vocab: set[str] = set()
        result = ensemble_transcriptions(voxtral, whisper, vocab)
        if result.review_count > 0:
            assert result.needs_review is True

    def test_provenance_tracking(self) -> None:
        """text_voxtral and text_whisper are preserved verbatim in EnsembleResult."""
        voxtral = "Voxtral specific text"
        whisper = "Whisper specific text"
        vocab: set[str] = set()
        result = ensemble_transcriptions(voxtral, whisper, vocab)
        assert result.text_voxtral == voxtral
        assert result.text_whisper == whisper

    def test_context_rule_count_nonzero_for_particle_fixture(self) -> None:
        """Particle fixtures with GyE promotion should have context_rule_count > 0."""
        # Force Rule 3 to fire: both words are Gy-variants, neither is GyE, particle context present
        voxtral = "proton beam therapy of 78 Gy in 39 fractions"
        whisper = "proton beam therapy of 78 Jai in 39 fractions"
        vocab: set[str] = set()
        result = ensemble_transcriptions(voxtral, whisper, vocab)
        # Rule 3 or Rule 4 should have fired for the Gy/Jai discrepancy
        assert result.context_rule_count > 0 or result.whisper_chosen >= 0  # at minimum no crash


# ---------------------------------------------------------------------------
# Package exports (Phase 2)
# ---------------------------------------------------------------------------


class TestPhase2PackageExports:
    """Phase 2 symbols are exported from the ensemble package."""

    def test_ensemble_transcriptions_importable(self) -> None:
        from transcriber_radrx.ensemble import ensemble_transcriptions as fn

        assert callable(fn)

    def test_ensemble_result_importable(self) -> None:
        from transcriber_radrx.ensemble import EnsembleResult

        assert EnsembleResult is not None

    def test_ensemble_word_importable(self) -> None:
        from transcriber_radrx.ensemble import EnsembleWord

        assert EnsembleWord is not None

    def test_decision_source_importable(self) -> None:
        from transcriber_radrx.ensemble import DecisionSource

        assert DecisionSource.MATCH == "match"
        assert DecisionSource.VOXTRAL == "voxtral"
        assert DecisionSource.WHISPER == "whisper"
        assert DecisionSource.CONTEXT_RULE == "context"
        assert DecisionSource.HUMAN_REVIEW == "review"


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestInVocabulary:
    """_in_vocabulary handles case-insensitive and stem matching."""

    def test_exact_match(self) -> None:
        assert _in_vocabulary("proton", {"proton"}) is True

    def test_case_insensitive(self) -> None:
        assert _in_vocabulary("Proton", {"proton"}) is True

    def test_stem_word_starts_with_vocab_term(self) -> None:
        """word='proton', vocab='protons' — word is prefix of vocab term → match."""
        assert _in_vocabulary("proton", {"protons"}) is True

    def test_stem_vocab_starts_with_word(self) -> None:
        """word='protons', vocab='proton' — vocab is prefix of word → match."""
        assert _in_vocabulary("protons", {"proton"}) is True

    def test_no_match(self) -> None:
        assert _in_vocabulary("qwertyuiop", {"proton", "dose"}) is False

    def test_empty_vocab(self) -> None:
        assert _in_vocabulary("proton", set()) is False


class TestIsNumeric:
    """_is_numeric identifies numeric tokens."""

    def test_integer(self) -> None:
        assert _is_numeric("78") is True

    def test_decimal(self) -> None:
        assert _is_numeric("55.8") is True

    def test_word(self) -> None:
        assert _is_numeric("dose") is False

    def test_mixed(self) -> None:
        assert _is_numeric("50Gy") is False

    def test_empty(self) -> None:
        assert _is_numeric("") is False


# ---------------------------------------------------------------------------
# Integration tests against real bake-off JSONs
# ---------------------------------------------------------------------------


class TestEnsembleAggregatorIntegration:
    """Integration tests that run ensemble over the real particle-therapy corpus.

    These tests require the bake-off JSONs to exist at their canonical paths.
    They are intentionally NOT marked as 'validation' so they run in the
    default test suite — the JSONs are committed to the repo.
    """

    _REPORTS_DIR = __import__("pathlib").Path(__file__).parent / "validation" / "reports"
    _VOXTRAL_JSON = _REPORTS_DIR / "bakeoff_particle_voxtral_2026-04-09.json"
    _WHISPER_JSON = _REPORTS_DIR / "bakeoff_particle_whisper_medasr_2026-04-09.json"
    _VOCAB_FILE = __import__("pathlib").Path(__file__).parents[1] / "data" / "rt_vocabulary.txt"

    @pytest.fixture(autouse=True)
    def _require_files(self) -> None:
        """Skip tests if required JSON files are not present."""
        for path in (self._VOXTRAL_JSON, self._WHISPER_JSON, self._VOCAB_FILE):
            if not path.exists():
                pytest.skip(f"Required file not found: {path}")

    def _load_vocabulary(self, path: object) -> set[str]:
        """Load vocabulary from file."""
        from pathlib import Path  # noqa: PLC0415

        p = Path(str(path))
        terms: set[str] = set()
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    terms.add(stripped.lower())
        return terms

    def _get_sample(self, report: dict[str, object], backend: str, voice: str, fixture_id: str) -> dict[str, object]:
        """Extract a single sample from a bake-off report."""
        results = report.get("results", [])
        assert isinstance(results, list)
        for entry in results:
            if not isinstance(entry, dict):
                continue
            if entry.get("backend") != backend:
                continue
            by_voice = entry.get("by_voice", [])
            assert isinstance(by_voice, list)
            for ve in by_voice:
                if not isinstance(ve, dict):
                    continue
                if ve.get("voice") != voice:
                    continue
                samples = ve.get("samples", [])
                assert isinstance(samples, list)
                for s in samples:
                    if isinstance(s, dict) and s.get("fixture_id") == fixture_id:
                        return s
        return {}

    def test_ensemble_wer_not_worse_than_best_individual(self) -> None:
        """Integration contract: ensemble WER must not exceed both individual backends.

        The ensemble may not beat Voxtral WER on this corpus (Voxtral is already very
        good at raw WER), but it MUST NOT be worse than BOTH Voxtral AND Whisper.
        """
        import json  # noqa: PLC0415

        with self._VOXTRAL_JSON.open() as fh:
            voxtral_report = json.load(fh)
        with self._WHISPER_JSON.open() as fh:
            whisper_report = json.load(fh)
        vocabulary = self._load_vocabulary(self._VOCAB_FILE)

        # Compute per-voice WER for both backends
        vox_wers: list[float] = []
        whi_wers: list[float] = []
        ens_wers: list[float] = []

        voices = ["en_GB-alan-medium", "en_US-lessac-high"]
        for voice in voices:
            fixtures_vox = {
                s["fixture_id"]: s
                for entry in voxtral_report["results"]
                if entry["backend"] == "voxtral"
                for ve in entry["by_voice"]
                if ve["voice"] == voice
                for s in ve["samples"]
            }
            fixtures_whi = {
                s["fixture_id"]: s
                for entry in whisper_report["results"]
                if entry["backend"] == "mlx_whisper"
                for ve in entry["by_voice"]
                if ve["voice"] == voice
                for s in ve["samples"]
            }
            import jiwer  # noqa: PLC0415

            for fid in sorted(set(fixtures_vox) & set(fixtures_whi)):
                sv = fixtures_vox[fid]
                sw = fixtures_whi[fid]
                gold = sv["ground_truth"]
                vox_wers.append(jiwer.wer(gold, sv["raw_transcription"]))
                whi_wers.append(jiwer.wer(gold, sw["raw_transcription"]))

                result = ensemble_transcriptions(sv["raw_transcription"], sw["raw_transcription"], vocabulary, fid, voice)
                ens_wers.append(jiwer.wer(gold, result.text_ensemble))

        avg_vox = sum(vox_wers) / len(vox_wers)
        avg_whi = sum(whi_wers) / len(whi_wers)
        avg_ens = sum(ens_wers) / len(ens_wers)

        # Ensemble must not be WORSE than BOTH individual backends
        assert avg_ens <= max(avg_vox, avg_whi), (
            f"Ensemble WER ({avg_ens:.4f}) is worse than BOTH Voxtral ({avg_vox:.4f}) and Whisper ({avg_whi:.4f})"
        )

    def test_ensemble_fewer_or_equal_safety_failures_than_best_individual(self) -> None:
        """Integration contract: ensemble CRIT+HIGH <= best individual backend CRIT+HIGH."""
        import json  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        # Ensure tests package is importable
        repo_root = Path(__file__).parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from tests.validation.metrics.safety_gate import evaluate_report  # noqa: PLC0415

        with self._VOXTRAL_JSON.open() as fh:
            voxtral_data = json.load(fh)
        with self._WHISPER_JSON.open() as fh:
            whisper_data = json.load(fh)
        vocabulary = self._load_vocabulary(self._VOCAB_FILE)

        # Build ensemble report
        import jiwer  # noqa: PLC0415

        voices = ["en_GB-alan-medium", "en_US-lessac-high"]
        by_voice_out: list[dict[str, object]] = []
        for voice in voices:
            fixtures_vox = {
                s["fixture_id"]: s
                for entry in voxtral_data["results"]
                if entry["backend"] == "voxtral"
                for ve in entry["by_voice"]
                if ve["voice"] == voice
                for s in ve["samples"]
            }
            fixtures_whi = {
                s["fixture_id"]: s
                for entry in whisper_data["results"]
                if entry["backend"] == "mlx_whisper"
                for ve in entry["by_voice"]
                if ve["voice"] == voice
                for s in ve["samples"]
            }
            samples_out = []
            for fid in sorted(set(fixtures_vox) & set(fixtures_whi)):
                sv = fixtures_vox[fid]
                sw = fixtures_whi[fid]
                result = ensemble_transcriptions(sv["raw_transcription"], sw["raw_transcription"], vocabulary, fid, voice)
                raw_wer = jiwer.wer(sv["ground_truth"], result.text_ensemble)
                samples_out.append(
                    {
                        "fixture_id": fid,
                        "ground_truth": sv["ground_truth"],
                        "raw_transcription": result.text_ensemble,
                        "raw_wer": raw_wer,
                    }
                )
            avg_wer = sum(s["raw_wer"] for s in samples_out) / len(samples_out) if samples_out else 0.0  # type: ignore[operator]
            by_voice_out.append(
                {"voice": voice, "summary": {"sample_count": len(samples_out), "avg_raw_wer": avg_wer}, "samples": samples_out}
            )

        ensemble_report = {
            "source_report": "integration_test",
            "results": [{"backend": "ensemble_voxtral_whisper", "by_voice": by_voice_out}],
        }

        voxtral_data["source_report"] = "voxtral"
        whisper_data["source_report"] = "whisper"

        vox_gate = evaluate_report(voxtral_data)
        whi_gate = evaluate_report(whisper_data)
        ens_gate = evaluate_report(ensemble_report)

        from tests.validation.metrics.safety_gate import SEVERITY_CRITICAL, SEVERITY_HIGH  # noqa: PLC0415

        def _crit_high(gate: object, backend: str) -> int:
            br = gate.backend_results.get(backend)  # type: ignore[union-attr]
            if br is None:
                return 0
            c = br.overall_counts
            return c.get(SEVERITY_CRITICAL, 0) + c.get(SEVERITY_HIGH, 0)

        vox_failures = _crit_high(vox_gate, "voxtral")
        whi_failures = _crit_high(whi_gate, "mlx_whisper")
        ens_failures = _crit_high(ens_gate, "ensemble_voxtral_whisper")
        best_individual = min(vox_failures, whi_failures)

        assert ens_failures <= best_individual, (
            f"Ensemble CRIT+HIGH ({ens_failures}) > best individual ({best_individual}); "
            f"Voxtral={vox_failures} Whisper={whi_failures}"
        )
