"""Token-class decision rules for 2-backend ASR ensemble.

Phase 2 of the ensemble pipeline: applies per-word decision rules to
AlignedSpan objects and produces EnsembleWord objects with full provenance.

Rules are evaluated in priority order (Rule 1 → Rule 10). First match wins.
See ENSEMBLE_SPEC.md for the full design specification.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from transcriber_radrx.ensemble.aligner import (
    AlignedSpan,
    AlignmentType,
    align_transcriptions,
    summarize_alignment,
)

# ---------------------------------------------------------------------------
# Gy-variant detection
# ---------------------------------------------------------------------------

#: Known-bad single-token renderings of Gy / GyE (case-insensitive matching).
#: Multi-word variants like "Jai E" are handled after tokenisation.
_GY_VARIANT_SET: frozenset[str] = frozenset(
    [
        "gy",
        "gye",
        "gie",
        "gi",
        "gi-e",
        "giy",
        "jai",
        "jie",
        "jee",
        "hie",
        "jy",
        "ji",
        "j",
        "ge",
        "gae",
        "gi.e.",
        "gie",
        "j,e",
        "gje",
        "jae",
        "ie",
        "die",
        "gev",
    ]
)

# ---------------------------------------------------------------------------
# Particle-therapy context clues
# ---------------------------------------------------------------------------

#: Context clues that indicate a particle-therapy fixture (subset of safety_gate.py).
_PARTICLE_THERAPY_CLUES: frozenset[str] = frozenset(
    [
        "proton",
        "protons",
        "pencil beam scanning",
        "pbs",
        "carbon ion",
        "particle therapy",
        "craniospinal",
        "craniospinal irradiation",
        "csi",
        "chordoma",
        "medulloblastoma",
        "ewing sarcoma",
        "rhabdomyosarcoma",
        "ependymoma",
        "craniopharyngioma",
        "neuroblastoma",
        "germinoma",
        "rbe",
        "relative biological effectiveness",
    ]
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DecisionSource(StrEnum):
    """How an ensemble word was chosen.

    Attributes:
        MATCH: Both backends produced the same word.
        VOXTRAL: Voxtral's word was chosen.
        WHISPER: Whisper's word was chosen.
        CONTEXT_RULE: Word was derived from context (e.g. GyE promotion).
        HUMAN_REVIEW: Unresolvable; flagged for human review.
    """

    MATCH = "match"
    VOXTRAL = "voxtral"
    WHISPER = "whisper"
    CONTEXT_RULE = "context"
    HUMAN_REVIEW = "review"


@dataclass(frozen=True)
class EnsembleWord:
    """One word in the ensemble output with full provenance.

    Attributes:
        word: The chosen word.
        source: How it was chosen (DecisionSource).
        word_voxtral: What Voxtral said (None if INSERTION_B).
        word_whisper: What Whisper said (None if INSERTION_A).
        rule_id: Which rule fired (None for MATCH).
        confidence: Confidence in the choice (1.0 for MATCH).
        needs_review: True if flagged for human review.
    """

    word: str
    source: DecisionSource
    word_voxtral: str | None
    word_whisper: str | None
    rule_id: str | None
    confidence: float
    needs_review: bool


@dataclass
class EnsembleResult:
    """The ensemble output for one fixture × voice pair.

    Attributes:
        fixture_id: Fixture identifier.
        voice: Voice identifier.
        text_voxtral: Original Voxtral transcription.
        text_whisper: Original Whisper transcription.
        text_ensemble: The ensemble-chosen transcription.
        words: Per-word provenance list.
        needs_review: True if ANY word is flagged.
        review_count: Number of words flagged for review.
        agreement_rate: From alignment summary (matches / total_spans).
        voxtral_chosen: Count of words from Voxtral.
        whisper_chosen: Count of words from Whisper.
        context_rule_count: Count of words from context rules.
    """

    fixture_id: str
    voice: str
    text_voxtral: str
    text_whisper: str
    text_ensemble: str
    words: list[EnsembleWord] = field(default_factory=list)
    needs_review: bool = False
    review_count: int = 0
    agreement_rate: float = 1.0
    voxtral_chosen: int = 0
    whisper_chosen: int = 0
    context_rule_count: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def is_gy_variant(word: str) -> bool:
    """Return True if word is a known rendering of Gy or GyE.

    The check is case-insensitive and covers all known failure modes from
    the bake-off corpus (cycle 110 + 112).

    Args:
        word: A single token from a transcription.

    Returns:
        True if the word is a known Gy/GyE variant.
    """
    return word.lower() in _GY_VARIANT_SET


def has_particle_context(text_a: str, text_b: str) -> bool:
    """Return True if either transcription contains particle-therapy context clues.

    Checks both the Voxtral and Whisper full transcription strings for context
    clues (not just the local disagreement span).

    Args:
        text_a: Full Voxtral transcription.
        text_b: Full Whisper transcription.

    Returns:
        True if any particle-therapy context clue is found in either text.
    """
    combined_lower = (text_a + " " + text_b).lower()
    return any(clue in combined_lower for clue in _PARTICLE_THERAPY_CLUES)


def _is_numeric(word: str) -> bool:
    """Return True if word looks like a number (int or decimal).

    Args:
        word: Token to check.

    Returns:
        True if the token matches \\d+\\.?\\d*.
    """
    return bool(re.fullmatch(r"\d+\.?\d*", word))


def _decimal_digits(word: str) -> int:
    """Return the number of decimal digits in a numeric string.

    Args:
        word: Numeric token.

    Returns:
        Number of digits after the decimal point (0 if integer).
    """
    if "." in word:
        return len(word.split(".")[1])
    return 0


def _in_vocabulary(word: str, vocabulary: set[str]) -> bool:
    """Return True if word (or a vocabulary term starting with word) is in the vocabulary.

    Vocabulary matching is case-insensitive. Also handles stems: if the word
    lowercased starts any vocabulary entry (e.g. 'proton' in 'protons'), it counts.

    Args:
        word: Token to look up.
        vocabulary: Set of lowercase vocabulary terms.

    Returns:
        True if word matches the vocabulary.
    """
    lower = word.lower()
    if lower in vocabulary:
        return True
    # Stem match: check if any vocabulary term starts with this word
    # (handles "proton" matching entry "protons" or vice-versa)
    return any(term.startswith(lower) or lower.startswith(term) for term in vocabulary if term)


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------


def _rule1_match(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 1: MATCH — both backends agree.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.MATCH:
        return None
    return EnsembleWord(
        word=span.word_a or "",
        source=DecisionSource.MATCH,
        word_voxtral=span.word_a,
        word_whisper=span.word_b,
        rule_id=None,
        confidence=1.0,
        needs_review=False,
    )


def _rule2_dose_unit_gye_available(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 2: DOSE_UNIT_GYE — dose unit with GyE available.

    Condition: one or both words are a Gy-variant AND at least one word is 'GyE'.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused — no context check needed).
        _text_b: Full Whisper transcription (unused — no context check needed).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type not in (AlignmentType.SUBSTITUTION,):
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    a_is_gy = is_gy_variant(wa)
    b_is_gy = is_gy_variant(wb)

    if not (a_is_gy or b_is_gy):
        return None
    a_is_gye = wa.lower() == "gye"
    b_is_gye = wb.lower() == "gye"
    if not (a_is_gye or b_is_gye):
        return None

    # At least one is GyE — pick GyE
    if a_is_gye and b_is_gye:
        source = DecisionSource.MATCH
    elif a_is_gye:
        source = DecisionSource.VOXTRAL
    else:
        source = DecisionSource.WHISPER

    return EnsembleWord(
        word="GyE",
        source=source,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="DOSE_UNIT_GYE",
        confidence=0.95,
        needs_review=False,
    )


def _rule3_dose_unit_context(span: AlignedSpan, text_a: str, text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 3: DOSE_UNIT_CONTEXT — both produce Gy-variant, neither is GyE.

    Promotes to GyE when particle-therapy context is present.

    Args:
        span: Aligned span to evaluate.
        text_a: Full Voxtral transcription.
        text_b: Full Whisper transcription.
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    if not (is_gy_variant(wa) and is_gy_variant(wb)):
        return None
    # Neither should be GyE (Rule 2 would have caught it)
    if wa.lower() == "gye" or wb.lower() == "gye":
        return None
    if not has_particle_context(text_a, text_b):
        return None
    return EnsembleWord(
        word="GyE",
        source=DecisionSource.CONTEXT_RULE,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="DOSE_UNIT_CONTEXT",
        confidence=0.85,
        needs_review=True,
    )


def _rule4_dose_unit_visible_corruption(span: AlignedSpan, text_a: str, text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 4: DOSE_UNIT_VISIBLE_CORRUPTION — exactly one word is a Gy-variant.

    Args:
        span: Aligned span to evaluate.
        text_a: Full Voxtral transcription.
        text_b: Full Whisper transcription.
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    a_is_gy = is_gy_variant(wa)
    b_is_gy = is_gy_variant(wb)
    if not (a_is_gy ^ b_is_gy):
        return None

    # The one that is a Gy-variant wins
    if a_is_gy:
        chosen = wa
        source = DecisionSource.VOXTRAL
    else:
        chosen = wb
        source = DecisionSource.WHISPER

    # Promote to GyE if particle context is present
    if has_particle_context(text_a, text_b):
        chosen = "GyE"
        source = DecisionSource.CONTEXT_RULE

    return EnsembleWord(
        word=chosen,
        source=source,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="DOSE_UNIT_VISIBLE_CORRUPTION",
        confidence=0.88,
        needs_review=False,
    )


def _rule5_vocabulary_match(span: AlignedSpan, _text_a: str, _text_b: str, vocab: set[str]) -> EnsembleWord | None:
    """Rule 5: VOCABULARY_MATCH — one word is in the RT vocabulary, the other is not.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        vocab: RT vocabulary set.

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    a_in_vocab = _in_vocabulary(wa, vocab)
    b_in_vocab = _in_vocabulary(wb, vocab)
    if a_in_vocab == b_in_vocab:
        # Both or neither — rule doesn't resolve
        return None
    if a_in_vocab:
        return EnsembleWord(
            word=wa,
            source=DecisionSource.VOXTRAL,
            word_voxtral=wa,
            word_whisper=wb,
            rule_id="VOCABULARY_MATCH",
            confidence=0.9,
            needs_review=False,
        )
    return EnsembleWord(
        word=wb,
        source=DecisionSource.WHISPER,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="VOCABULARY_MATCH",
        confidence=0.9,
        needs_review=False,
    )


def _rule6_both_wrong(span: AlignedSpan, _text_a: str, _text_b: str, vocab: set[str]) -> EnsembleWord | None:
    """Rule 6: BOTH_WRONG — neither word matches vocabulary and words differ significantly.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        vocab: RT vocabulary set.

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    if _in_vocabulary(wa, vocab) or _in_vocabulary(wb, vocab):
        return None
    similarity = difflib.SequenceMatcher(None, wa.lower(), wb.lower()).ratio()
    if similarity >= 0.5:
        return None
    # Both likely wrong — take Voxtral, flag for review
    return EnsembleWord(
        word=wa,
        source=DecisionSource.HUMAN_REVIEW,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="BOTH_WRONG",
        confidence=0.3,
        needs_review=True,
    )


def _rule7_decimal_precision(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 7: DECIMAL_PRECISION — one word has more decimal digits.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    if not (_is_numeric(wa) and _is_numeric(wb)):
        return None
    digits_a = _decimal_digits(wa)
    digits_b = _decimal_digits(wb)
    if digits_a == digits_b:
        return None
    if digits_a > digits_b:
        return EnsembleWord(
            word=wa,
            source=DecisionSource.VOXTRAL,
            word_voxtral=wa,
            word_whisper=wb,
            rule_id="DECIMAL_PRECISION",
            confidence=0.9,
            needs_review=False,
        )
    return EnsembleWord(
        word=wb,
        source=DecisionSource.WHISPER,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="DECIMAL_PRECISION",
        confidence=0.9,
        needs_review=False,
    )


def _rule8_formatting_default(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 8: FORMATTING_DEFAULT — all other substitutions, take Voxtral.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies (always applies to SUBSTITUTION spans).
    """
    if span.alignment_type != AlignmentType.SUBSTITUTION:
        return None
    wa = span.word_a or ""
    wb = span.word_b or ""
    return EnsembleWord(
        word=wa,
        source=DecisionSource.VOXTRAL,
        word_voxtral=wa,
        word_whisper=wb,
        rule_id="FORMATTING_DEFAULT",
        confidence=0.7,
        needs_review=False,
    )


def _rule9_insertion_a(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 9: INSERTION_A — word only in Voxtral.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.INSERTION_A:
        return None
    wa = span.word_a or ""
    return EnsembleWord(
        word=wa,
        source=DecisionSource.VOXTRAL,
        word_voxtral=wa,
        word_whisper=None,
        rule_id="INSERTION_A",
        confidence=0.6,
        needs_review=False,
    )


def _rule10_insertion_b(span: AlignedSpan, _text_a: str, _text_b: str, _vocab: set[str]) -> EnsembleWord | None:
    """Rule 10: INSERTION_B — word only in Whisper.

    Args:
        span: Aligned span to evaluate.
        _text_a: Full Voxtral transcription (unused).
        _text_b: Full Whisper transcription (unused).
        _vocab: RT vocabulary set (unused).

    Returns:
        EnsembleWord if rule applies, else None.
    """
    if span.alignment_type != AlignmentType.INSERTION_B:
        return None
    wb = span.word_b or ""
    return EnsembleWord(
        word=wb,
        source=DecisionSource.WHISPER,
        word_voxtral=None,
        word_whisper=wb,
        rule_id="INSERTION_B",
        confidence=0.6,
        needs_review=False,
    )


# Priority-ordered list of rule functions
_RULES = [
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
]


def _apply_rules(
    span: AlignedSpan,
    text_a: str,
    text_b: str,
    vocab: set[str],
) -> EnsembleWord:
    """Apply decision rules in priority order and return the first match.

    Args:
        span: Aligned span to decide on.
        text_a: Full Voxtral transcription.
        text_b: Full Whisper transcription.
        vocab: RT vocabulary set.

    Returns:
        EnsembleWord from the first matching rule.

    Raises:
        RuntimeError: If no rule fires (should not happen — Rule 8 or 9/10 always fire).
    """
    for rule_fn in _RULES:
        result = rule_fn(span, text_a, text_b, vocab)
        if result is not None:
            return result
    # Defensive fallback — should never reach here
    msg = f"No rule matched span: {span}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensemble_transcriptions(
    text_voxtral: str,
    text_whisper: str,
    vocabulary: set[str],
    fixture_id: str = "",
    voice: str = "",
) -> EnsembleResult:
    """Produce an ensemble transcription from two backend outputs.

    Applies word-level alignment followed by token-class decision rules
    (Rules 1–10, priority order) to resolve disagreements between Voxtral
    and Whisper transcriptions.

    Args:
        text_voxtral: Raw Voxtral transcription.
        text_whisper: Raw Whisper transcription.
        vocabulary: Set of known RT vocabulary terms (lowercase) for tiebreaking.
        fixture_id: For provenance tracking.
        voice: For provenance tracking.

    Returns:
        EnsembleResult with per-word provenance and aggregate statistics.
    """
    spans = align_transcriptions(text_voxtral, text_whisper)
    summary = summarize_alignment(spans)

    words: list[EnsembleWord] = []
    for span in spans:
        try:
            ew = _apply_rules(span, text_voxtral, text_whisper, vocabulary)
        except RuntimeError:
            logger.exception(
                "Rule application failed for span fixture_id={} voice={} span={}",
                fixture_id,
                voice,
                span,
            )
            # Emergency fallback: take Voxtral's word or Whisper's
            fallback_word = span.word_a or span.word_b or ""
            ew = EnsembleWord(
                word=fallback_word,
                source=DecisionSource.HUMAN_REVIEW,
                word_voxtral=span.word_a,
                word_whisper=span.word_b,
                rule_id=None,
                confidence=0.0,
                needs_review=True,
            )
        words.append(ew)

    text_ensemble = " ".join(w.word for w in words if w.word)
    needs_review = any(w.needs_review for w in words)
    review_count = sum(1 for w in words if w.needs_review)
    voxtral_chosen = sum(1 for w in words if w.source in (DecisionSource.VOXTRAL, DecisionSource.MATCH))
    whisper_chosen = sum(1 for w in words if w.source == DecisionSource.WHISPER)
    context_rule_count = sum(1 for w in words if w.source == DecisionSource.CONTEXT_RULE)

    return EnsembleResult(
        fixture_id=fixture_id,
        voice=voice,
        text_voxtral=text_voxtral,
        text_whisper=text_whisper,
        text_ensemble=text_ensemble,
        words=words,
        needs_review=needs_review,
        review_count=review_count,
        agreement_rate=summary.agreement_rate,
        voxtral_chosen=voxtral_chosen,
        whisper_chosen=whisper_chosen,
        context_rule_count=context_rule_count,
    )
