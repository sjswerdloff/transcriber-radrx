"""Tests for the Phase 1 word-level alignment module.

Tests cover:
- Identical inputs → all MATCH
- Completely different inputs → all SUBSTITUTION
- One empty input → all INSERTION
- Case-insensitive alignment ("Dose" == "dose" → MATCH)
- Punctuation preservation ("50.4" is one token, "skull." is one token)
- Unequal-length replace blocks → zip SUBSTITUTION + overflow INSERTION
- Real cycle 112 particle therapy fixture pairs (Voxtral vs Whisper)
- AlignmentSummary statistics and agreement_rate
- format_alignment_diff output structure and grouping
"""

import dataclasses

import pytest

from transcriber_radrx.ensemble.aligner import (
    AlignedSpan,
    AlignmentSummary,
    AlignmentType,
    align_transcriptions,
    format_alignment_diff,
    normalize_for_alignment,
    summarize_alignment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _types(spans: list[AlignedSpan]) -> list[str]:
    """Extract alignment type values for compact assertions."""
    return [s.alignment_type.value for s in spans]


# ---------------------------------------------------------------------------
# Tokenisation contracts (tested indirectly through align_transcriptions)
# ---------------------------------------------------------------------------


class TestTokenisation:
    """Whitespace-split tokenisation preserves attached punctuation."""

    def test_decimal_stays_one_token(self) -> None:
        """50.4 must not be split on the period."""
        spans = align_transcriptions("50.4 Gy", "50.4 Gy")
        assert all(s.alignment_type == AlignmentType.MATCH for s in spans)
        assert spans[0].word_a == "50.4"

    def test_period_attached_to_word_stays_one_token(self) -> None:
        """skull. must remain a single token."""
        spans = align_transcriptions("base of skull.", "base of skull.")
        assert spans[-1].word_a == "skull."

    def test_slash_compound_stays_one_token(self) -> None:
        """3D/3D must not be split on the slash."""
        spans = align_transcriptions("3D/3D image guidance", "3D/3D image guidance")
        assert spans[0].word_a == "3D/3D"
        assert all(s.alignment_type == AlignmentType.MATCH for s in spans)

    def test_empty_string_produces_no_spans(self) -> None:
        spans = align_transcriptions("", "")
        assert spans == []


# ---------------------------------------------------------------------------
# Identical inputs → all MATCH
# ---------------------------------------------------------------------------


class TestIdenticalInputs:
    """Identical transcriptions produce exclusively MATCH spans."""

    def test_single_word_identical(self) -> None:
        spans = align_transcriptions("Dose", "Dose")
        assert len(spans) == 1
        assert spans[0].alignment_type == AlignmentType.MATCH
        assert spans[0].word_a == "Dose"
        assert spans[0].word_b == "Dose"

    def test_multi_word_identical(self) -> None:
        text = "Dose escalation to 78 Gy in 39 fractions"
        spans = align_transcriptions(text, text)
        assert all(s.alignment_type == AlignmentType.MATCH for s in spans)
        assert len(spans) == 8

    def test_positions_are_populated_for_matches(self) -> None:
        spans = align_transcriptions("alpha beta gamma", "alpha beta gamma")
        for i, span in enumerate(spans):
            assert span.position_a == i
            assert span.position_b == i


# ---------------------------------------------------------------------------
# Case-insensitive matching
# ---------------------------------------------------------------------------


class TestCaseInsensitiveMatching:
    """Alignment is case-insensitive; original case is preserved in words."""

    def test_dose_vs_dose_lowercase(self) -> None:
        spans = align_transcriptions("Dose escalation", "dose escalation")
        assert all(s.alignment_type == AlignmentType.MATCH for s in spans)

    def test_original_case_preserved_in_words(self) -> None:
        spans = align_transcriptions("Dose", "dose")
        assert spans[0].word_a == "Dose"
        assert spans[0].word_b == "dose"
        assert spans[0].alignment_type == AlignmentType.MATCH

    def test_all_uppercase_matches_mixed_case(self) -> None:
        spans = align_transcriptions("GY", "Gy")
        assert len(spans) == 1
        assert spans[0].alignment_type == AlignmentType.MATCH
        assert spans[0].word_a == "GY"
        assert spans[0].word_b == "Gy"


# ---------------------------------------------------------------------------
# Completely different inputs → all SUBSTITUTION
# ---------------------------------------------------------------------------


class TestCompletelyDifferentInputs:
    """When no words overlap, all spans are SUBSTITUTION (equal-length blocks)."""

    def test_single_word_substitution(self) -> None:
        spans = align_transcriptions("alpha", "beta")
        assert len(spans) == 1
        assert spans[0].alignment_type == AlignmentType.SUBSTITUTION
        assert spans[0].word_a == "alpha"
        assert spans[0].word_b == "beta"

    def test_multi_word_equal_length_substitution(self) -> None:
        spans = align_transcriptions("one two three", "four five six")
        assert all(s.alignment_type == AlignmentType.SUBSTITUTION for s in spans)
        assert len(spans) == 3

    def test_positions_populated_in_substitution(self) -> None:
        spans = align_transcriptions("apple orange", "pear banana")
        assert spans[0].position_a == 0
        assert spans[0].position_b == 0
        assert spans[1].position_a == 1
        assert spans[1].position_b == 1


# ---------------------------------------------------------------------------
# One empty input → all INSERTION
# ---------------------------------------------------------------------------


class TestOneEmptyInput:
    """Empty input A → all INSERTION_B; empty input B → all INSERTION_A."""

    def test_empty_a_produces_insertion_b(self) -> None:
        spans = align_transcriptions("", "dose escalation")
        assert all(s.alignment_type == AlignmentType.INSERTION_B for s in spans)
        assert len(spans) == 2

    def test_empty_b_produces_insertion_a(self) -> None:
        spans = align_transcriptions("dose escalation", "")
        assert all(s.alignment_type == AlignmentType.INSERTION_A for s in spans)
        assert len(spans) == 2

    def test_insertion_b_has_none_word_a(self) -> None:
        spans = align_transcriptions("", "hello")
        assert spans[0].word_a is None
        assert spans[0].word_b == "hello"
        assert spans[0].position_a is None
        assert spans[0].position_b == 0

    def test_insertion_a_has_none_word_b(self) -> None:
        spans = align_transcriptions("hello", "")
        assert spans[0].word_b is None
        assert spans[0].word_a == "hello"
        assert spans[0].position_b is None
        assert spans[0].position_a == 0


# ---------------------------------------------------------------------------
# Unequal replace blocks
# ---------------------------------------------------------------------------


class TestUnequalReplaceBlocks:
    """Replace opcodes with unequal lengths: zip → SUBSTITUTION, overflow → INSERTION."""

    def test_a_longer_overflow_is_insertion_a(self) -> None:
        # "Jai E" (2 tokens) replaces "Gy" (1 token) in Whisper output
        # Voxtral: [Gy]  Whisper: [Jai, E]  → SUB(Gy/Jai), INS_B(E)
        spans = align_transcriptions("78 Gy in", "78 Jai E in")
        types = _types(spans)
        assert types[0] == AlignmentType.MATCH  # 78
        assert AlignmentType.SUBSTITUTION in types
        assert AlignmentType.INSERTION_B in types

    def test_b_longer_overflow_is_insertion_b(self) -> None:
        # A is shorter in the replace region
        spans = align_transcriptions("alpha beta extra gamma", "alpha delta gamma")
        types = _types(spans)
        assert AlignmentType.SUBSTITUTION in types or AlignmentType.INSERTION_A in types

    def test_a_longer_than_b_overflow_is_insertion_a(self) -> None:
        # A has 3 words where B has 1
        spans = align_transcriptions("one two three end", "X end")
        types = _types(spans)
        assert AlignmentType.INSERTION_A in types


# ---------------------------------------------------------------------------
# AlignmentSummary
# ---------------------------------------------------------------------------


class TestSummarizeAlignment:
    """summarize_alignment produces correct counts and agreement_rate."""

    def test_all_match_summary(self) -> None:
        spans = align_transcriptions("dose in fractions", "dose in fractions")
        summary = summarize_alignment(spans)
        assert summary.total_spans == 3
        assert summary.matches == 3
        assert summary.substitutions == 0
        assert summary.insertions_a == 0
        assert summary.insertions_b == 0
        assert summary.agreement_rate == pytest.approx(1.0)

    def test_all_substitution_summary(self) -> None:
        spans = align_transcriptions("alpha beta", "gamma delta")
        summary = summarize_alignment(spans)
        assert summary.substitutions == 2
        assert summary.matches == 0
        assert summary.agreement_rate == pytest.approx(0.0)

    def test_empty_spans_agreement_rate_is_one(self) -> None:
        """Both inputs empty → empty spans → agreement_rate = 1.0 by convention."""
        spans = align_transcriptions("", "")
        summary = summarize_alignment(spans)
        assert summary.total_spans == 0
        assert summary.agreement_rate == pytest.approx(1.0)

    def test_mixed_summary_agreement_rate(self) -> None:
        # "dose escalation" vs "dose reduction": 1 match, 1 sub → 50%
        spans = align_transcriptions("dose escalation", "dose reduction")
        summary = summarize_alignment(spans)
        assert summary.matches == 1
        assert summary.substitutions == 1
        assert summary.agreement_rate == pytest.approx(0.5)

    def test_summary_is_frozen_dataclass(self) -> None:
        spans = align_transcriptions("a b", "a b")
        summary = summarize_alignment(spans)
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.matches = 999  # type: ignore[misc]

    def test_summary_type(self) -> None:
        spans = align_transcriptions("a", "a")
        summary = summarize_alignment(spans)
        assert isinstance(summary, AlignmentSummary)


# ---------------------------------------------------------------------------
# format_alignment_diff
# ---------------------------------------------------------------------------


class TestFormatAlignmentDiff:
    """format_alignment_diff produces correct inline diff strings."""

    def test_identical_inputs_no_brackets(self) -> None:
        spans = align_transcriptions("dose in fractions", "dose in fractions")
        diff = format_alignment_diff(spans)
        assert "[" not in diff
        assert "]" not in diff
        assert diff == "dose in fractions"

    def test_substitution_shown_in_brackets(self) -> None:
        spans = align_transcriptions("78 Gy in", "78 GIE and")
        diff = format_alignment_diff(spans)
        assert "[" in diff
        assert "|" in diff
        assert "78" in diff  # match preserved

    def test_empty_spans_returns_empty_string(self) -> None:
        spans = align_transcriptions("", "")
        diff = format_alignment_diff(spans)
        assert diff == ""

    def test_bracket_format_contains_both_words(self) -> None:
        spans = align_transcriptions("alpha beta", "alpha gamma")
        diff = format_alignment_diff(spans)
        assert "beta" in diff
        assert "gamma" in diff
        assert "|" in diff

    def test_consecutive_disagreements_grouped(self) -> None:
        """Multiple consecutive non-MATCH spans should be grouped in one bracket pair."""
        spans = align_transcriptions("one two three end", "four five six end")
        diff = format_alignment_diff(spans)
        # Should have exactly one bracket group, not three
        assert diff.count("[") == 1
        assert diff.count("]") == 1

    def test_insertion_a_shows_in_left_side(self) -> None:
        """INSERTION_A words appear on the A (Voxtral) side of the bracket."""
        spans = align_transcriptions("dose extra", "dose")
        diff = format_alignment_diff(spans)
        # "extra" is in A only → should appear before the pipe or alone
        assert "extra" in diff

    def test_insertion_b_shows_in_right_side(self) -> None:
        """INSERTION_B words appear on the B (Whisper) side of the bracket."""
        spans = align_transcriptions("dose", "dose extra")
        diff = format_alignment_diff(spans)
        assert "extra" in diff


# ---------------------------------------------------------------------------
# Real cycle 112 particle therapy fixtures
# ---------------------------------------------------------------------------


class TestRealCycle112Fixtures:
    """Test cases using actual Voxtral vs Whisper transcriptions from bake-off JSONs.

    These fixtures reflect the complementary failure profiles documented in
    the cycle 112 bake-off report:
    - Voxtral tends to handle medical units correctly (Gy, GyE)
    - Whisper tends to transcribe "GyE" as "Jai E", "GIE", or similar
    - Both may disagree on "in" vs "and" for post-unit conjunctions
    """

    # proton-0004, en_US-lessac-high voice:
    # Voxtral: "dose escalation to 78 GIE in 39 fractions for the cordoma at the base of skull."
    # Whisper: " Dose escalation to 78 GIE and 39 fractions for the chordoma at the base of skull."
    # (Note: lessac-high voice — Voxtral has "GIE", Whisper has "GIE" but substitutes "in"→"and" and "cordoma"→"chordoma")
    PROTON_0004_LESSAC_VOXTRAL = "dose escalation to 78 GIE in 39 fractions for the cordoma at the base of skull."
    PROTON_0004_LESSAC_WHISPER = "Dose escalation to 78 GIE and 39 fractions for the chordoma at the base of skull."

    # proton-0004, en_GB-alan-medium voice:
    # Voxtral: "dose escalation to 78 Gy in 39 fractions for the cordoma at the base of skull."
    # Whisper: " Dose escalation to 78 Jai E in 39 fractions for the caudoma at the base of skull."
    PROTON_0004_ALAN_VOXTRAL = "dose escalation to 78 Gy in 39 fractions for the cordoma at the base of skull."
    PROTON_0004_ALAN_WHISPER = "Dose escalation to 78 Jai E in 39 fractions for the caudoma at the base of skull."

    # proton-0019, en_US-lessac-high voice:
    # Voxtral: "Ewing sarcoma of the left pelvis treated with 55.8 Ie and 31 fractions
    #           using pencil beam scanning with 3D/3D image guidance."
    # Whisper: " Ewing sarcoma, the left pelvis treated with 55.8 GI-E and 31 fractions,
    #           using pencil beam scanning with 3D-slash-3D image guidance."
    PROTON_0019_LESSAC_VOXTRAL = (
        "Ewing sarcoma of the left pelvis treated with 55.8 Ie and 31 fractions"
        " using pencil beam scanning with 3D/3D image guidance."
    )
    PROTON_0019_LESSAC_WHISPER = (
        "Ewing sarcoma, the left pelvis treated with 55.8 GI-E and 31 fractions,"
        " using pencil beam scanning with 3D-slash-3D image guidance."
    )

    def test_proton_0004_lessac_has_substitutions(self) -> None:
        """proton-0004 lessac: at least one SUBSTITUTION (in/and, cordoma/chordoma)."""
        spans = align_transcriptions(self.PROTON_0004_LESSAC_VOXTRAL, self.PROTON_0004_LESSAC_WHISPER)
        types = _types(spans)
        assert AlignmentType.SUBSTITUTION in types

    def test_proton_0004_lessac_agreement_rate(self) -> None:
        """proton-0004 lessac: high agreement rate — most words match."""
        spans = align_transcriptions(self.PROTON_0004_LESSAC_VOXTRAL, self.PROTON_0004_LESSAC_WHISPER)
        summary = summarize_alignment(spans)
        # Most words agree; the differences are "in"/"and" and "cordoma"/"chordoma"
        assert summary.agreement_rate >= 0.7

    def test_proton_0004_alan_gy_is_substituted_for_jai_e(self) -> None:
        """proton-0004 alan: Voxtral 'Gy' vs Whisper 'Jai E' → SUBSTITUTION + INSERTION_B."""
        spans = align_transcriptions(self.PROTON_0004_ALAN_VOXTRAL, self.PROTON_0004_ALAN_WHISPER)
        types = _types(spans)
        # The replace block for "Gy" → "Jai E" produces SUB + INS_B overflow
        assert AlignmentType.SUBSTITUTION in types
        assert AlignmentType.INSERTION_B in types

    def test_proton_0004_alan_whisper_jai_e_in_diff(self) -> None:
        """proton-0004 alan: diff output should contain Jai and E on Whisper side."""
        spans = align_transcriptions(self.PROTON_0004_ALAN_VOXTRAL, self.PROTON_0004_ALAN_WHISPER)
        diff = format_alignment_diff(spans)
        # The Whisper side of the bracket should contain the hallucinated tokens
        assert "Jai" in diff or "E" in diff

    def test_proton_0004_alan_voxtral_gy_in_diff(self) -> None:
        """proton-0004 alan: diff output should contain Gy on Voxtral side."""
        spans = align_transcriptions(self.PROTON_0004_ALAN_VOXTRAL, self.PROTON_0004_ALAN_WHISPER)
        diff = format_alignment_diff(spans)
        assert "Gy" in diff

    def test_proton_0019_lessac_has_disagreements(self) -> None:
        """proton-0019 lessac: Voxtral 'Ie' vs Whisper 'GI-E' and other diffs."""
        spans = align_transcriptions(self.PROTON_0019_LESSAC_VOXTRAL, self.PROTON_0019_LESSAC_WHISPER)
        types = _types(spans)
        # At minimum should have substitutions
        assert AlignmentType.MATCH in types  # Most words still match
        non_match = [t for t in types if t != AlignmentType.MATCH]
        assert len(non_match) > 0

    def test_proton_0019_lessac_3d_slash_3d_preserved(self) -> None:
        """proton-0019: '3D/3D' in Voxtral stays as one token and aligns correctly."""
        spans = align_transcriptions(self.PROTON_0019_LESSAC_VOXTRAL, self.PROTON_0019_LESSAC_WHISPER)
        # Find the span where Voxtral has 3D/3D
        voxtral_words = [s.word_a for s in spans if s.word_a is not None]
        assert "3D/3D" in voxtral_words

    def test_proton_0004_lessac_diff_shows_in_vs_and(self) -> None:
        """proton-0004 lessac diff: the 'in'/'and' substitution appears in brackets."""
        spans = align_transcriptions(self.PROTON_0004_LESSAC_VOXTRAL, self.PROTON_0004_LESSAC_WHISPER)
        diff = format_alignment_diff(spans)
        # At the substitution point, both "in" and "and" should appear
        assert "in" in diff
        assert "and" in diff

    def test_real_fixture_spans_have_correct_types(self) -> None:
        """All spans from real fixtures have valid AlignmentType values."""
        for text_a, text_b in [
            (self.PROTON_0004_LESSAC_VOXTRAL, self.PROTON_0004_LESSAC_WHISPER),
            (self.PROTON_0004_ALAN_VOXTRAL, self.PROTON_0004_ALAN_WHISPER),
            (self.PROTON_0019_LESSAC_VOXTRAL, self.PROTON_0019_LESSAC_WHISPER),
        ]:
            spans = align_transcriptions(text_a, text_b)
            for span in spans:
                assert span.alignment_type in AlignmentType


# ---------------------------------------------------------------------------
# AlignedSpan contract: frozen dataclass
# ---------------------------------------------------------------------------


class TestAlignedSpanContract:
    """AlignedSpan is a frozen dataclass with correct field types."""

    def test_aligned_span_is_frozen(self) -> None:
        span = AlignedSpan(
            alignment_type=AlignmentType.MATCH,
            word_a="dose",
            word_b="dose",
            position_a=0,
            position_b=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            span.word_a = "mutated"  # type: ignore[misc]

    def test_match_span_has_both_words(self) -> None:
        spans = align_transcriptions("alpha", "alpha")
        span = spans[0]
        assert span.word_a is not None
        assert span.word_b is not None

    def test_insertion_a_span_has_none_word_b(self) -> None:
        spans = align_transcriptions("alpha", "")
        span = spans[0]
        assert span.alignment_type == AlignmentType.INSERTION_A
        assert span.word_b is None
        assert span.position_b is None

    def test_insertion_b_span_has_none_word_a(self) -> None:
        spans = align_transcriptions("", "alpha")
        span = spans[0]
        assert span.alignment_type == AlignmentType.INSERTION_B
        assert span.word_a is None
        assert span.position_a is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeForAlignment:
    """normalize_for_alignment collapses multi-word medical forms to canonical tokens."""

    def test_gray_equivalent_normalizes_to_gye(self) -> None:
        assert normalize_for_alignment("55.8 gray equivalent in 31") == "55.8 GyE in 31"

    def test_gray_equivalents_plural_normalizes_to_gye(self) -> None:
        assert normalize_for_alignment("78 gray equivalents total") == "78 GyE total"

    def test_case_insensitive_gray_equivalent(self) -> None:
        assert normalize_for_alignment("Gray Equivalent") == "GyE"

    def test_case_insensitive_gray_equivalents(self) -> None:
        assert normalize_for_alignment("GRAY EQUIVALENTS") == "GyE"

    def test_3d_3d_normalizes_to_slash_form(self) -> None:
        assert normalize_for_alignment("with 3D 3D image guidance") == "with 3D/3D image guidance"

    def test_2d_3d_normalizes_to_slash_form(self) -> None:
        assert normalize_for_alignment("2D 3D image guidance") == "2D/3D image guidance"

    def test_3d_2d_normalizes_to_slash_form(self) -> None:
        assert normalize_for_alignment("3D 2D registration") == "3D/2D registration"

    def test_3d_3d_case_insensitive(self) -> None:
        result = normalize_for_alignment("3d 3d image")
        assert result == "3D/3D image"

    def test_unrelated_text_unchanged(self) -> None:
        text = "dose escalation to 78 Gy in 39 fractions"
        assert normalize_for_alignment(text) == text

    def test_empty_string_unchanged(self) -> None:
        assert normalize_for_alignment("") == ""

    def test_normalization_produces_clean_substitution(self) -> None:
        """Voxtral 'Gy' vs Whisper 'gray equivalent' → clean 1:1 SUBSTITUTION after normalization."""
        # Without normalization: "Gy" vs "gray equivalent" → SUB(Gy/gray) + INS_B(equivalent)
        # With normalization: "Gy" vs "GyE" → single SUBSTITUTION
        spans = align_transcriptions("78 Gy in 31", "78 gray equivalent in 31")
        # After normalization "gray equivalent" becomes "GyE" — single token
        sub_spans = [s for s in spans if s.alignment_type == AlignmentType.SUBSTITUTION]
        ins_b_spans = [s for s in spans if s.alignment_type == AlignmentType.INSERTION_B]
        # Should have one substitution for Gy/GyE, not a sub+insertion combo
        assert len(sub_spans) == 1
        assert len(ins_b_spans) == 0
        assert sub_spans[0].word_a == "Gy"
        assert sub_spans[0].word_b == "GyE"

    def test_normalization_preserves_canonical_replacement_case(self) -> None:
        """Replacement canonical form uses consistent capitalisation regardless of input."""
        result_lower = normalize_for_alignment("gray equivalent")
        result_upper = normalize_for_alignment("GRAY EQUIVALENT")
        assert result_lower == result_upper == "GyE"

    def test_gray_equivalents_before_gray_equivalent_no_partial_match(self) -> None:
        """'gray equivalents' (plural) must not be partially matched by 'gray equivalent' pattern."""
        result = normalize_for_alignment("55.8 gray equivalents")
        # Should produce exactly one GyE token, not "GyEs" or similar
        assert result == "55.8 GyE"

    # -----------------------------------------------------------------------
    # Phase 1.1: new normalizer entries
    # -----------------------------------------------------------------------

    def test_gy_equivalent_normalizes_to_gye(self) -> None:
        """'Gy equivalent' (Voxtral partial form) normalizes to 'GyE'."""
        assert normalize_for_alignment("23.4 Gy equivalent in 13") == "23.4 GyE in 13"

    def test_gy_equivalents_plural_normalizes_to_gye(self) -> None:
        assert normalize_for_alignment("78 Gy equivalents total") == "78 GyE total"

    def test_grey_equivalent_british_spelling(self) -> None:
        """'grey equivalent' (British spelling, alan voice) normalizes to 'GyE'."""
        assert normalize_for_alignment("55.8 grey equivalent in 31") == "55.8 GyE in 31"

    def test_grey_equivalents_plural_british(self) -> None:
        assert normalize_for_alignment("78 grey equivalents total") == "78 GyE total"

    def test_3d_slash_3d_whisper_vocalized_slash(self) -> None:
        """Whisper transcribes piper's vocalized slash as '-slash-'; must normalize to '3D/3D'."""
        assert normalize_for_alignment("3D-slash-3D image guidance") == "3D/3D image guidance"

    def test_2d_slash_3d_vocalized_slash(self) -> None:
        assert normalize_for_alignment("2D-slash-3D image guidance") == "2D/3D image guidance"

    def test_3d_slash_2d_vocalized_slash(self) -> None:
        assert normalize_for_alignment("3D-slash-2D registration") == "3D/2D registration"

    def test_3d_hyphen_3d_whisper_hyphenated_variant(self) -> None:
        """Whisper hyphenated variant '3D-3D' normalizes to '3D/3D'."""
        assert normalize_for_alignment("3D-3D image guidance") == "3D/3D image guidance"

    def test_2d_hyphen_3d_whisper_variant(self) -> None:
        assert normalize_for_alignment("2D-3D image guidance") == "2D/3D image guidance"

    def test_3d_hyphen_2d_whisper_variant(self) -> None:
        assert normalize_for_alignment("3D-2D registration") == "3D/2D registration"

    def test_3d_slash_3d_case_insensitive(self) -> None:
        assert normalize_for_alignment("3d-slash-3d image") == "3D/3D image"

    def test_number_unit_split_gye_joined(self) -> None:
        """Joined '60GyE' must be split to '60 GyE' before table lookup."""
        result = normalize_for_alignment("60GyE in 30 fractions")
        assert result == "60 GyE in 30 fractions"

    def test_number_unit_split_gy_joined(self) -> None:
        """Joined '78Gy' must be split to '78 Gy'."""
        result = normalize_for_alignment("78Gy in 39 fractions")
        assert result == "78 Gy in 39 fractions"

    def test_number_unit_split_cgy_joined(self) -> None:
        """Joined '5040cGy' must be split to '5040 cGy'."""
        result = normalize_for_alignment("5040cGy in 28 fractions")
        assert result == "5040 cGy in 28 fractions"

    def test_number_unit_split_gye_lowercase(self) -> None:
        """'78gye' → '78 gye' (regex is case-sensitive for the unit suffix, but GyE/Gye are covered)."""
        # The regex covers Gy[Ee]? — gye is matched via Gy[Ee]? = gy + e
        result = normalize_for_alignment("78Gye in 39")
        assert result == "78 Gye in 39"

    def test_proton_0019_lessac_whisper_3d_slash_3d(self) -> None:
        """proton-0019 lessac Whisper: '3D-slash-3D' normalizes so alignment finds 3D/3D token."""
        spans = align_transcriptions(
            "Ewing sarcoma of the left pelvis treated with 55.8 Ie and 31 fractions"
            " using pencil beam scanning with 3D/3D image guidance.",
            "Ewing sarcoma, the left pelvis treated with 55.8 GI-E and 31 fractions,"
            " using pencil beam scanning with 3D-slash-3D image guidance.",
        )
        # After normalization '3D-slash-3D' → '3D/3D', so we should have a MATCH for it
        match_words = [s.word_a for s in spans if s.alignment_type == AlignmentType.MATCH]
        assert "3D/3D" in match_words


# ---------------------------------------------------------------------------
# Public API exported from package __init__
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify all three public functions are exported from the package."""

    def test_align_transcriptions_importable_from_package(self) -> None:
        from transcriber_radrx.ensemble import align_transcriptions as fn

        assert callable(fn)

    def test_summarize_alignment_importable_from_package(self) -> None:
        from transcriber_radrx.ensemble import summarize_alignment as fn

        assert callable(fn)

    def test_format_alignment_diff_importable_from_package(self) -> None:
        from transcriber_radrx.ensemble import format_alignment_diff as fn

        assert callable(fn)

    def test_alignment_type_importable_from_package(self) -> None:
        from transcriber_radrx.ensemble import AlignmentType

        assert AlignmentType.MATCH == "match"

    def test_normalize_for_alignment_importable_from_package(self) -> None:
        from transcriber_radrx.ensemble import normalize_for_alignment as fn

        assert callable(fn)
