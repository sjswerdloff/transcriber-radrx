"""Post-processing correction for ASR output.

Applies domain-specific corrections to raw ASR transcription text using
a tiered matching strategy with multiple safety guards. Designed for
medical/clinical use where false corrections corrupt patient data.

Safety principles (per Cora's PR #1 review, 2026-04-07):
- "A correction dictionary that introduces errors is worse than none."
- Phonetic matching is OFF by default. Must be explicitly enabled.
- Short acronyms (≤4 chars, all caps) are matched ONLY by exact match.
- Common English stop words are never corrected.
- Case-insensitive correction respects homograph stop list.
- Every correction is logged with full provenance.

Tiered matching (when phonetic enabled):
  1. Exact match (score 1.0)
  2. Case-insensitive (score 0.95) — guarded against homographs
  3. Bounded orthographic edit distance for long tokens (≥0.9 similarity)
  4. Double Metaphone phonetic (≥6 char terms only, ≥0.85 similarity)

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber_radrx.phrase_corrector import PhraseCorrection

logger = logging.getLogger(__name__)

# Common English stop words that must never be silently rewritten.
# Curated from cases that caused false positives in Cora's empirical review:
# 'our' → 'OAR', 'support' → 'SBRT', 'guy' → 'Gy', etc.
DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    [
        # Articles, pronouns, prepositions, common verbs
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "and",
        "or",
        "but",
        "nor",
        "for",
        "yet",
        "so",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "by",
        "of",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "may",
        "might",
        "must",
        "shall",
        "would",
        "could",
        # Words that caused false positives in PR #1 review (Cora)
        "guy",
        "guys",
        "support",
        "supports",
        "supportive",
        "supporting",
        "supported",
        "format",
        "formats",
        "formatted",
        "formatting",
        "great",
        "greater",
        "greatest",
        "grey",
        "gray",  # color, not the dose unit
        "emerald",
        "emeralds",
        # Common medical narrative words that resemble RT acronyms
        "patient",
        "patients",
        "treatment",
        "treatments",
        "report",
        "reports",
        "reported",
        "find",
        "finds",
        "finding",
        "findings",
        "show",
        "shows",
        "showed",
        "shown",
        "see",
        "sees",
        "saw",
        "seen",
        "good",
        "well",
        "better",
        "best",
        "bad",
        "worse",
        "worst",
    ]
)

# Word boundary pattern: splits on whitespace and common punctuation
_WORD_PATTERN = re.compile(r"[\w'-]+|[^\w\s]", re.UNICODE)

# Minimum word length for phonetic/edit-distance matching
_MIN_NONEXACT_LENGTH = 6
# Minimum vocabulary term length for non-exact matching
_MIN_NONEXACT_VOCAB_LENGTH = 6
# Default phonetic similarity threshold (high to prevent false positives)
_DEFAULT_PHONETIC_THRESHOLD = 0.85
# Default edit distance similarity threshold
_DEFAULT_EDIT_DISTANCE_THRESHOLD = 0.9
# Case-insensitive correction confidence
_CASE_INSENSITIVE_SCORE = 0.95


@dataclass(frozen=True)
class Correction:
    """A single correction applied to transcription text.

    Immutable for audit trail integrity.

    Attributes:
        original: The text as transcribed by ASR.
        corrected: The domain-correct replacement.
        score: Confidence score (1.0 = exact match, <1.0 = scored).
        method: How the match was found ("case_insensitive",
                "edit_distance", or "phonetic").
        offset: Character offset in original text where correction was applied.
    """

    original: str
    corrected: str
    score: float
    method: str
    offset: int = 0


@dataclass
class VocabularyEntry:
    """A vocabulary term with precomputed phonetic codes.

    Attributes:
        canonical: The correct spelling of the term.
        lower: Lowercase form for case-insensitive matching.
        is_acronym: True if term is ≤4 chars and all uppercase.
        phonetic_primary: Primary Double Metaphone code (empty if disabled).
        phonetic_secondary: Secondary Double Metaphone code (may be empty).
    """

    canonical: str
    lower: str
    is_acronym: bool
    phonetic_primary: str = ""
    phonetic_secondary: str = ""


def _is_acronym(term: str) -> bool:
    """Check if a term qualifies as a short acronym for safety exclusion.

    Acronyms (≤4 chars, all uppercase) are only matched by exact match.
    They are never matched case-insensitively or non-exactly because they
    collide with common English words.

    Args:
        term: The term to check.

    Returns:
        True if term is a short acronym requiring strict matching.
    """
    return len(term) <= 4 and term.isupper() and term.isalpha()


def _load_vocabulary(path: Path) -> list[str]:
    """Load vocabulary terms from file.

    Args:
        path: Path to vocabulary file (one term per line, # comments).

    Returns:
        List of vocabulary terms (deduplicated, order preserved).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If vocabulary file is empty after filtering.
    """
    if not path.exists():
        msg = f"Vocabulary file not found: {path}"
        raise FileNotFoundError(msg)

    seen: set[str] = set()
    terms: list[str] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and stripped not in seen:
                seen.add(stripped)
                terms.append(stripped)

    if not terms:
        msg = f"Vocabulary file contains no terms: {path}"
        raise ValueError(msg)

    return terms


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _orthographic_similarity(s1: str, s2: str) -> float:
    """Compute normalized orthographic similarity between two strings.

    Uses Levenshtein distance normalized by max length.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    return 1.0 - (_levenshtein(s1, s2) / max_len)


def _phonetic_similarity(code1: str, code2: str) -> float:
    """Score similarity between two phonetic codes.

    Args:
        code1: First phonetic code.
        code2: Second phonetic code.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not code1 or not code2:
        return 0.0
    if code1 == code2:
        return 1.0
    max_len = max(len(code1), len(code2))
    return 1.0 - (_levenshtein(code1, code2) / max_len)


# Backward compatibility alias for tests
_phonetic_score = _phonetic_similarity


class CorrectionDictionary:
    """Post-processing corrector for RT domain vocabulary.

    Tiered matching strategy with safety guards:
    1. Exact match (score 1.0) — always enabled
    2. Case-insensitive (score 0.95) — homograph stop list applied
    3. Bounded edit distance (≥0.9) — phonetic_enabled only
    4. Double Metaphone (≥0.85, ≥6 chars) — phonetic_enabled only

    Default behavior: only tiers 1 and 2 are active. Phonetic matching
    must be explicitly enabled because it caused unsafe corrections in
    Cora's empirical review of PR #1.
    """

    def __init__(
        self,
        vocabulary_path: str | None = None,
        *,
        enable_phonetic: bool = False,
        min_phonetic_score: float = _DEFAULT_PHONETIC_THRESHOLD,
        min_edit_distance_score: float = _DEFAULT_EDIT_DISTANCE_THRESHOLD,
        stop_words: frozenset[str] | None = None,
    ) -> None:
        """Initialize with vocabulary file and safety options.

        Args:
            vocabulary_path: Path to RT vocabulary terms file.
            enable_phonetic: Enable phonetic and edit-distance matching tiers
                (DEFAULT: False). Phonetic matching can produce false positives
                — only enable after verifying with negative tests.
            min_phonetic_score: Minimum phonetic similarity to accept (0.0-1.0).
            min_edit_distance_score: Minimum edit distance similarity to accept.
            stop_words: Custom stop word set. If None, DEFAULT_STOP_WORDS used.
        """
        self._entries: list[VocabularyEntry] = []
        self._exact_map: dict[str, str] = {}
        self._lower_map: dict[str, str] = {}
        self._enable_phonetic = enable_phonetic
        self._min_phonetic_score = min_phonetic_score
        self._min_edit_distance_score = min_edit_distance_score
        self._stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS

        self._metaphone_available = False
        if enable_phonetic:
            try:
                from metaphone import doublemetaphone  # noqa: F401

                self._metaphone_available = True
            except ImportError:
                logger.warning(
                    "metaphone not installed but enable_phonetic=True; "
                    "phonetic matching will be silently disabled. "
                    "Install with: uv sync --extra phonetic"
                )

        if vocabulary_path is not None:
            self._load(Path(vocabulary_path))

        logger.info(
            "CorrectionDictionary initialized: %d terms, phonetic_enabled=%s, metaphone_available=%s",
            len(self._entries),
            enable_phonetic,
            self._metaphone_available,
        )

    def _load(self, path: Path) -> None:
        """Load and index vocabulary terms.

        Args:
            path: Path to vocabulary file.
        """
        terms = _load_vocabulary(path)

        for term in terms:
            primary = ""
            secondary = ""
            if self._metaphone_available and len(term) >= _MIN_NONEXACT_VOCAB_LENGTH:
                from metaphone import doublemetaphone

                codes = doublemetaphone(term)
                primary = codes[0]
                secondary = codes[1]

            entry = VocabularyEntry(
                canonical=term,
                lower=term.lower(),
                is_acronym=_is_acronym(term),
                phonetic_primary=primary,
                phonetic_secondary=secondary,
            )
            self._entries.append(entry)
            self._exact_map[term] = term

            # Case-insensitive map: only register if not a homograph collision
            # with a stop word, AND only first occurrence wins.
            lower = term.lower()
            if lower not in self._stop_words and lower not in self._lower_map:
                self._lower_map[lower] = term

    def _match_word(self, word: str, offset: int) -> Correction | None:
        """Try to match a single word against the vocabulary.

        Args:
            word: Word from ASR output.
            offset: Character offset in original text.

        Returns:
            Correction if a safe match is found, None otherwise.
        """
        # Tier 1: exact match (no correction needed)
        if word in self._exact_map:
            return None

        lower = word.lower()

        # Stop word guard: never correct common English
        if lower in self._stop_words:
            return None

        # Tier 2: case-insensitive — only for vocab terms registered in lower map
        # (the load step already excluded homograph collisions with stop words)
        if lower in self._lower_map:
            canonical = self._lower_map[lower]
            if canonical != word:
                return Correction(
                    original=word,
                    corrected=canonical,
                    score=_CASE_INSENSITIVE_SCORE,
                    method="case_insensitive",
                    offset=offset,
                )
            return None

        # Tiers 3 and 4 require phonetic matching enabled
        if not self._enable_phonetic:
            return None

        # Length guard: short words can't be safely matched non-exactly
        if len(word) < _MIN_NONEXACT_LENGTH:
            return None

        # Tier 3: bounded edit distance for non-acronym vocabulary terms
        best_edit: tuple[VocabularyEntry, float] | None = None
        for entry in self._entries:
            if entry.is_acronym:
                continue
            if len(entry.canonical) < _MIN_NONEXACT_VOCAB_LENGTH:
                continue
            score = _orthographic_similarity(lower, entry.lower)
            if score >= self._min_edit_distance_score and (best_edit is None or score > best_edit[1]):
                best_edit = (entry, score)

        if best_edit is not None:
            entry, score = best_edit
            return Correction(
                original=word,
                corrected=entry.canonical,
                score=score,
                method="edit_distance",
                offset=offset,
            )

        # Tier 4: phonetic matching for multisyllabic terms only
        if not self._metaphone_available:
            return None

        from metaphone import doublemetaphone

        word_codes = doublemetaphone(word)
        word_primary = word_codes[0]
        word_secondary = word_codes[1]

        # Phonetic codes must be substantial (≥3 chars) to be reliable
        if len(word_primary) < 3:
            return None

        best_phonetic: tuple[VocabularyEntry, float] | None = None
        for entry in self._entries:
            if entry.is_acronym:
                continue
            if not entry.phonetic_primary or len(entry.phonetic_primary) < 3:
                continue

            score = _phonetic_similarity(word_primary, entry.phonetic_primary)
            if word_secondary:
                score = max(
                    score,
                    _phonetic_similarity(word_secondary, entry.phonetic_primary),
                )
            if entry.phonetic_secondary:
                score = max(
                    score,
                    _phonetic_similarity(word_primary, entry.phonetic_secondary),
                )

            if score >= self._min_phonetic_score and (
                best_phonetic is None
                or score > best_phonetic[1]
                or (score == best_phonetic[1] and len(entry.canonical) > len(best_phonetic[0].canonical))
            ):
                best_phonetic = (entry, score)

        if best_phonetic is not None:
            entry, score = best_phonetic
            return Correction(
                original=word,
                corrected=entry.canonical,
                score=score,
                method="phonetic",
                offset=offset,
            )

        return None

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        """Apply corrections to transcribed text.

        Processes text word-by-word, applying tiered matching with safety
        guards. Multi-word vocabulary terms are not yet supported.

        Args:
            text: Raw transcription text from ASR.

        Returns:
            Tuple of (corrected_text, list of corrections applied).
        """
        if not self._entries:
            return text, []

        corrections: list[Correction] = []
        result_parts: list[str] = []
        last_end = 0

        for match in _WORD_PATTERN.finditer(text):
            result_parts.append(text[last_end : match.start()])
            word = match.group()

            correction = self._match_word(word, offset=match.start())
            if correction is not None:
                result_parts.append(correction.corrected)
                corrections.append(correction)
                # Per-correction audit logging for medical traceability
                logger.info(
                    "correction_applied: %s -> %s (method=%s score=%.2f offset=%d)",
                    correction.original,
                    correction.corrected,
                    correction.method,
                    correction.score,
                    correction.offset,
                )
            else:
                result_parts.append(word)

            last_end = match.end()

        result_parts.append(text[last_end:])

        return "".join(result_parts), corrections

    def correct_full(self, text: str) -> tuple[str, list[Correction], list[PhraseCorrection]]:
        """Apply phrase-level then word-level corrections.

        Runs PhraseCorrectorPipeline first (multi-word patterns), then
        the existing single-word CorrectionDictionary. This ordering ensures
        that multi-word fixes (e.g., 'bracket therapy' -> 'brachytherapy')
        are applied before single-word matching tries to correct individual
        tokens that are part of a larger pattern.

        Args:
            text: Raw transcription text from ASR.

        Returns:
            Tuple of (corrected_text, word_corrections, phrase_corrections).
        """
        from transcriber_radrx.phrase_corrector import PhraseCorrectorPipeline

        phrase_pipeline = PhraseCorrectorPipeline()
        text_after_phrases, phrase_corrections = phrase_pipeline.correct(text)
        final_text, word_corrections = self.correct(text_after_phrases)
        return final_text, word_corrections, phrase_corrections
