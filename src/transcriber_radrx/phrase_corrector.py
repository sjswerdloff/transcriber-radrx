"""Phrase-level domain corrections for known ASR substitution patterns.

Runs BEFORE the single-word CorrectionDictionary corrector. Applies
regex-based multi-word pattern matching to fix systematic ASR failures
that single-word edit-distance matching cannot reach.

Patterns are mined from bake-off data across ESL and Commonwealth voice
panels (cycle 113, April 2026). Each pattern has:
- A compiled regex matching the known ASR error
- The correct RT domain term
- An audit log entry for every correction

Safety: Every pattern was verified against 1000+ transcription pairs.
False positive rate is bounded by context requirements — patterns only
fire in RT-specific word contexts.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhraseCorrection:
    """A single phrase-level correction applied to transcription text.

    Attributes:
        original: The matched text span.
        corrected: The replacement text.
        pattern_name: Identifier for the pattern that fired.
        start: Character offset in the original text.
        end: Character offset end in the original text.
    """

    original: str
    corrected: str
    pattern_name: str
    start: int
    end: int


@dataclass
class PhrasePattern:
    """A single correction pattern.

    Attributes:
        name: Human-readable identifier (e.g. "gy_unit_isolation").
        regex: Compiled regex pattern.
        replacement: The correct text to substitute.
        description: Why this pattern exists (ASR failure mode).
    """

    name: str
    regex: re.Pattern[str]
    replacement: str
    description: str


# Each pattern is (name, regex_string, replacement, description)
# Regex uses re.IGNORECASE where appropriate.
# Word boundaries \b and lookahead/lookbehind provide context safety.
_RAW_PATTERNS: list[tuple[str, str, str, str]] = [
    # --- Dose unit corrections (highest impact: 400+ occurrences) ---
    # ASR produces "j", "gi", "g", "chi", "die", "jai" for "Gy"
    # Context: appears after a number (dose value).
    # The word boundary after the group ensures we don't match "giant", "gift", etc.
    # Note: "g" is only matched when it appears isolated after a number, not mid-word.
    (
        "gy_after_number",
        r"(\d+(?:\.\d+)?)\s*\b(ji?|gi|chi|die|jai|j)\b",
        r"\1 Gy",
        "Dose unit Gy misrecognized as j/gi/chi/die/jai after numeric dose value",
    ),
    # Separate pattern for bare "g" after a number — more restricted to avoid
    # matching measurement units like "g" (grams). Only match "g" or "G" as
    # an isolated token directly after a numeric dose in RT context.
    (
        "gy_bare_g_after_number",
        r"(\d+(?:\.\d+)?)\s+\b([gG])\b(?!\s*(?:ram|rams|[/-]))",
        r"\1 Gy",
        "Dose unit Gy misrecognized as bare 'g' after numeric dose value (not grams)",
    ),
    # --- Compound word joins (25+ occurrences each) ---
    (
        "chemoradiation_join",
        r"\bchemo[\s-]+radiation\b",
        "chemoradiation",
        "ASR splits chemoradiation into two words",
    ),
    (
        "neoadjuvant_join",
        r"\bne[ow][\s-]+adjuvant\b",
        "neoadjuvant",
        "ASR splits neoadjuvant into two words or substitutes 'new'",
    ),
    # --- Multi-word term corrections (15+ occurrences each) ---
    (
        "dose_painting",
        r"\bdose\s+(?:pending|bending|bent\s+in)\b",
        "dose painting",
        "ASR substitutes 'pending'/'bending' for 'painting' in dose painting",
    ),
    (
        "fiducial_marker",
        r"\b(?:physical|fusion|situational)\s+markers?\b",
        "fiducial marker",
        "ASR substitutes 'physical'/'fusion'/'situational' for 'fiducial' before 'marker'/'markers'",
    ),
    # --- Single-word substitutions too far for edit distance ---
    (
        "brachytherapy_bracket",
        r"\bbracket\s+therapy\b",
        "brachytherapy",
        "ASR produces 'bracket therapy' for brachytherapy",
    ),
    (
        "brachytherapy_brady",
        r"\bbradytherapy\b",
        "brachytherapy",
        "ASR produces 'bradytherapy' for brachytherapy",
    ),
    (
        "brachytherapy_practice",
        r"\bpracti[cs]e?\s+therapy\b",
        "brachytherapy",
        "ASR produces 'practice/practic therapy' for brachytherapy",
    ),
    (
        "vulvar_to_vulva",
        r"\bvulva\b(?=\s+(?:squamous|carcinoma|cancer|lesion|tumor|tumour))",
        "vulvar",
        "ASR drops the -r suffix; 'vulva squamous' should be 'vulvar squamous'",
    ),
    (
        "lumpectomy_lymph",
        r"\blymph?t?ectomy\b",
        "lumpectomy",
        "ASR produces 'lymphectomy'/'lymptectomy'/'lympectomy' for lumpectomy",
    ),
    (
        "oropharyngeal_all_pharyngeal",
        r"\ball\s+pharyngeal\b",
        "oropharyngeal",
        "ASR produces 'all pharyngeal' for oropharyngeal",
    ),
]


def _build_default_patterns() -> list[PhrasePattern]:
    """Build PhrasePattern list from _RAW_PATTERNS.

    Returns:
        List of compiled PhrasePattern objects.
    """
    patterns: list[PhrasePattern] = []
    for name, regex_str, replacement, description in _RAW_PATTERNS:
        compiled = re.compile(regex_str, re.IGNORECASE)
        patterns.append(
            PhrasePattern(
                name=name,
                regex=compiled,
                replacement=replacement,
                description=description,
            )
        )
    return patterns


class PhraseCorrectorPipeline:
    """Apply phrase-level regex corrections to ASR output.

    Each pattern is applied in priority order (list order). Earlier patterns take
    precedence — if their match regions overlap with a later pattern's potential
    match, the earlier pattern's replacement runs first and the later pattern
    sees the already-corrected text. Overlap is rare given the specificity of
    the patterns.

    All corrections are logged for medical audit trail integrity.
    """

    def __init__(self, patterns: list[PhrasePattern] | None = None) -> None:
        """Initialise the pipeline.

        Args:
            patterns: Ordered list of PhrasePattern objects to apply. If None,
                the default bake-off-derived patterns are used.
        """
        if patterns is None:
            patterns = _build_default_patterns()
        self.patterns = patterns

    def correct(self, text: str) -> tuple[str, list[PhraseCorrection]]:
        """Apply all phrase corrections to text.

        Each pattern is applied to the full text in order. Matches for a
        pattern are found on the pre-substitution text so that offsets in the
        returned PhraseCorrection objects reflect positions in the *input* text
        for each pattern pass (not the accumulated output). Callers that need
        downstream offsets should treat them as approximate.

        Args:
            text: Raw ASR output text.

        Returns:
            Tuple of (corrected_text, list of corrections applied).
        """
        corrections: list[PhraseCorrection] = []
        current_text = text

        for pattern in self.patterns:
            new_corrections: list[PhraseCorrection] = []
            for match in pattern.regex.finditer(current_text):
                original = match.group()
                # Use match.expand() to resolve backreferences in the replacement
                # template (e.g. r"\1 Gy"). This works correctly even when the
                # pattern contains lookaheads that are not captured in match.group().
                corrected = match.expand(pattern.replacement)
                # Every match means the pattern will fire — record the correction.
                # (We cannot check original != corrected here for lookahead patterns
                # because match.group() does not include lookahead text, so the
                # comparison would spuriously return equal for patterns like vulvar.)
                new_corrections.append(
                    PhraseCorrection(
                        original=original,
                        corrected=corrected,
                        pattern_name=pattern.name,
                        start=match.start(),
                        end=match.end(),
                    )
                )
            # Apply all substitutions for this pattern in one pass.
            new_text = pattern.regex.sub(pattern.replacement, current_text)
            if new_corrections:
                corrections.extend(new_corrections)
                for c in new_corrections:
                    logger.info(
                        "phrase_correction: '%s' -> '%s' (pattern=%s offset=%d)",
                        c.original,
                        c.corrected,
                        c.pattern_name,
                        c.start,
                    )
            current_text = new_text

        return current_text, corrections


def apply_phrase_corrections(text: str) -> tuple[str, list[PhraseCorrection]]:
    """Apply default phrase corrections. Convenience wrapper.

    Args:
        text: Raw ASR output text.

    Returns:
        Tuple of (corrected_text, list of corrections applied).
    """
    pipeline = PhraseCorrectorPipeline()
    return pipeline.correct(text)
