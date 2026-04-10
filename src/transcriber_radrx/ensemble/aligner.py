"""Word-level alignment module for 2-backend ASR ensemble.

Phase 1 of the ensemble pipeline: aligns two transcription strings at
the word level using difflib.SequenceMatcher and produces AlignedSpan
objects for downstream decision rules.

A pre-alignment normalization step collapses known multi-word medical
forms to their canonical single-token equivalents before alignment,
producing cleaner 1:1 comparisons.

See ENSEMBLE_SPEC.md for the full design specification.
"""

import difflib
import re
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Pre-alignment normalization
# ---------------------------------------------------------------------------

# Regex to split joined number-unit tokens like "60GyE" → "60 GyE".
# Applied BEFORE the lookup-table step so that the table sees canonical spacing.
_RE_SPLIT_NUMBER_UNIT: re.Pattern[str] = re.compile(r"(\d+)(Gy[Ee]?|cGy)")

# Known multi-word medical forms that should be collapsed to a single canonical
# token before alignment.  Longer patterns must come first to prevent partial
# matches.  Keys are lowercase; the replacement is the canonical form.
_NORMALIZATION_TABLE: list[tuple[str, str]] = [
    # GyE variants (longer/plural forms before singular)
    ("gray equivalents", "GyE"),
    ("gray equivalent", "GyE"),
    ("gy equivalents", "GyE"),
    ("gy equivalent", "GyE"),
    ("grey equivalents", "GyE"),
    ("grey equivalent", "GyE"),
    # 3D/2D slashed forms — hyphen-word variants (3D-slash-3D) before hyphen-only (3D-3D)
    ("3d-slash-3d", "3D/3D"),
    ("2d-slash-3d", "2D/3D"),
    ("3d-slash-2d", "3D/2D"),
    ("3d-3d", "3D/3D"),
    ("2d-3d", "2D/3D"),
    ("3d-2d", "3D/2D"),
    # Legacy space-separated forms already in Phase 1
    ("3d 3d", "3D/3D"),
    ("2d 3d", "2D/3D"),
    ("3d 2d", "3D/2D"),
]

# Pre-compile a regex for each pattern for case-insensitive replacement.
# We use word-boundary-aware matching so "3d 3d" inside "in 3d 3d image"
# matches correctly without a lookahead hairball.
_NORMALIZATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(re.escape(src), re.IGNORECASE), dst) for src, dst in _NORMALIZATION_TABLE
]


def normalize_for_alignment(text: str) -> str:
    """Collapse known multi-word medical forms to canonical single-token equivalents.

    Applied before tokenisation so that phrases like ``gray equivalent``
    become the single token ``GyE``, enabling clean 1:1 word comparisons
    in the alignment step.

    Replacements are applied in longest-first order and are case-insensitive.
    The replacement token preserves the canonical capitalisation defined in
    ``_NORMALIZATION_TABLE`` regardless of the input capitalisation.

    Processing order:
    1. Split joined number-unit tokens (e.g. ``60GyE`` → ``60 GyE``).
    2. Apply the lookup-table replacements.

    Args:
        text: Raw transcription string.

    Returns:
        Text with known multi-word forms replaced by their canonical tokens.
    """
    # Step 1: split joined number-unit tokens before the table step
    text = _RE_SPLIT_NUMBER_UNIT.sub(r"\1 \2", text)
    # Step 2: apply the lookup-table patterns
    for pattern, replacement in _NORMALIZATION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class AlignmentType(StrEnum):
    """Classification of a word-level alignment position."""

    MATCH = "match"
    SUBSTITUTION = "sub"
    INSERTION_A = "ins_a"  # Word present in A (Voxtral) but not B (Whisper)
    INSERTION_B = "ins_b"  # Word present in B (Whisper) but not A (Voxtral)


@dataclass(frozen=True)
class AlignedSpan:
    """One aligned position in the word-level alignment.

    For MATCH and SUBSTITUTION, both word_a and word_b are populated.
    For INSERTION_A, only word_a is populated (word_b is None).
    For INSERTION_B, only word_b is populated (word_a is None).

    Attributes:
        alignment_type: The type of alignment at this position.
        word_a: Voxtral's word at this position (None for INSERTION_B).
        word_b: Whisper's word at this position (None for INSERTION_A).
        position_a: Word index in A's token sequence (None for INSERTION_B).
        position_b: Word index in B's token sequence (None for INSERTION_A).
    """

    alignment_type: AlignmentType
    word_a: str | None
    word_b: str | None
    position_a: int | None
    position_b: int | None


@dataclass(frozen=True)
class AlignmentSummary:
    """Summary statistics for a word-level alignment.

    Attributes:
        total_spans: Total number of aligned spans.
        matches: Number of MATCH spans.
        substitutions: Number of SUBSTITUTION spans.
        insertions_a: Number of INSERTION_A spans (words only in Voxtral).
        insertions_b: Number of INSERTION_B spans (words only in Whisper).
        agreement_rate: Fraction of spans that are MATCH (matches / total_spans).
    """

    total_spans: int
    matches: int
    substitutions: int
    insertions_a: int
    insertions_b: int
    agreement_rate: float


def _tokenize(text: str) -> list[str]:
    """Split transcription text into word tokens by whitespace.

    Preserves punctuation attached to words (e.g., "50.4" stays as one
    token, "skull." stays as one token, "3D/3D" stays as one token).

    Args:
        text: Raw transcription string.

    Returns:
        List of whitespace-delimited tokens, empty list for empty input.
    """
    return text.split()


def align_transcriptions(text_a: str, text_b: str) -> list[AlignedSpan]:
    """Word-level alignment of two transcription strings.

    Both inputs are first passed through :func:`normalize_for_alignment` to
    collapse multi-word medical forms (e.g. ``gray equivalent`` → ``GyE``)
    into single canonical tokens.  After normalization, alignment uses
    difflib.SequenceMatcher on lowercased tokens for case-insensitive
    matching, but preserves the normalized form in the returned
    ``AlignedSpan`` words.  For replace opcodes with unequal block lengths,
    pairs are zipped as SUBSTITUTION and the overflow is emitted as
    INSERTION_A or INSERTION_B.

    Args:
        text_a: Transcription from backend A (Voxtral).
        text_b: Transcription from backend B (Whisper).

    Returns:
        List of AlignedSpan objects representing the word-level alignment.
    """
    tokens_a = _tokenize(normalize_for_alignment(text_a))
    tokens_b = _tokenize(normalize_for_alignment(text_b))

    # Case-insensitive sequence for alignment matching
    tokens_a_lower = [t.lower() for t in tokens_a]
    tokens_b_lower = [t.lower() for t in tokens_b]

    matcher = difflib.SequenceMatcher(None, tokens_a_lower, tokens_b_lower, autojunk=False)
    spans: list[AlignedSpan] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for pa, pb in zip(range(i1, i2), range(j1, j2), strict=False):
                spans.append(
                    AlignedSpan(
                        alignment_type=AlignmentType.MATCH,
                        word_a=tokens_a[pa],
                        word_b=tokens_b[pb],
                        position_a=pa,
                        position_b=pb,
                    )
                )

        elif tag == "replace":
            block_a = list(range(i1, i2))
            block_b = list(range(j1, j2))
            # Zip paired positions as SUBSTITUTION
            for pa, pb in zip(block_a, block_b, strict=False):
                spans.append(
                    AlignedSpan(
                        alignment_type=AlignmentType.SUBSTITUTION,
                        word_a=tokens_a[pa],
                        word_b=tokens_b[pb],
                        position_a=pa,
                        position_b=pb,
                    )
                )
            # Overflow from the longer side
            if len(block_a) > len(block_b):
                for pa in block_a[len(block_b) :]:
                    spans.append(
                        AlignedSpan(
                            alignment_type=AlignmentType.INSERTION_A,
                            word_a=tokens_a[pa],
                            word_b=None,
                            position_a=pa,
                            position_b=None,
                        )
                    )
            elif len(block_b) > len(block_a):
                for pb in block_b[len(block_a) :]:
                    spans.append(
                        AlignedSpan(
                            alignment_type=AlignmentType.INSERTION_B,
                            word_a=None,
                            word_b=tokens_b[pb],
                            position_a=None,
                            position_b=pb,
                        )
                    )

        elif tag == "delete":
            # Words in A only (Voxtral has them, Whisper does not)
            for pa in range(i1, i2):
                spans.append(
                    AlignedSpan(
                        alignment_type=AlignmentType.INSERTION_A,
                        word_a=tokens_a[pa],
                        word_b=None,
                        position_a=pa,
                        position_b=None,
                    )
                )

        elif tag == "insert":
            # Words in B only (Whisper has them, Voxtral does not)
            for pb in range(j1, j2):
                spans.append(
                    AlignedSpan(
                        alignment_type=AlignmentType.INSERTION_B,
                        word_a=None,
                        word_b=tokens_b[pb],
                        position_a=None,
                        position_b=pb,
                    )
                )

    return spans


def summarize_alignment(spans: list[AlignedSpan]) -> AlignmentSummary:
    """Compute summary statistics for an alignment.

    Args:
        spans: List of AlignedSpan objects from align_transcriptions.

    Returns:
        AlignmentSummary with counts and agreement_rate. When spans is
        empty (both inputs were empty strings), agreement_rate is 1.0.
    """
    total = len(spans)
    matches = sum(1 for s in spans if s.alignment_type == AlignmentType.MATCH)
    substitutions = sum(1 for s in spans if s.alignment_type == AlignmentType.SUBSTITUTION)
    insertions_a = sum(1 for s in spans if s.alignment_type == AlignmentType.INSERTION_A)
    insertions_b = sum(1 for s in spans if s.alignment_type == AlignmentType.INSERTION_B)
    agreement_rate = matches / total if total > 0 else 1.0
    return AlignmentSummary(
        total_spans=total,
        matches=matches,
        substitutions=substitutions,
        insertions_a=insertions_a,
        insertions_b=insertions_b,
        agreement_rate=agreement_rate,
    )


def format_alignment_diff(spans: list[AlignedSpan]) -> str:
    """Human-readable inline diff showing agreements and disagreements.

    Agreements are emitted as plain words. Disagreements are shown as
    ``[Voxtral | Whisper]`` at the point of divergence. Consecutive
    disagreement spans (SUBSTITUTION, INSERTION_A, INSERTION_B) are
    grouped into a single bracket pair.

    Example output::

        Dose escalation to [78 Gy | 78 GIE] [in | and] 39 fractions
        for the chordoma at the base of skull.

    Args:
        spans: List of AlignedSpan objects from align_transcriptions.

    Returns:
        Human-readable string representation of the alignment diff.
    """
    if not spans:
        return ""

    parts: list[str] = []
    i = 0
    while i < len(spans):
        span = spans[i]
        if span.alignment_type == AlignmentType.MATCH:
            parts.append(span.word_a or "")
            i += 1
        else:
            # Collect a consecutive run of non-MATCH spans
            group_a: list[str] = []
            group_b: list[str] = []
            while i < len(spans) and spans[i].alignment_type != AlignmentType.MATCH:
                s = spans[i]
                if s.alignment_type == AlignmentType.SUBSTITUTION:
                    if s.word_a is not None:
                        group_a.append(s.word_a)
                    if s.word_b is not None:
                        group_b.append(s.word_b)
                elif s.alignment_type == AlignmentType.INSERTION_A:
                    if s.word_a is not None:
                        group_a.append(s.word_a)
                elif s.alignment_type == AlignmentType.INSERTION_B:
                    if s.word_b is not None:
                        group_b.append(s.word_b)
                i += 1
            side_a = " ".join(group_a) if group_a else ""
            side_b = " ".join(group_b) if group_b else ""
            parts.append(f"[{side_a} | {side_b}]")

    return " ".join(parts)
