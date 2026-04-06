"""Post-processing correction for ASR output.

Applies domain-specific corrections to raw ASR transcription text using
a tiered matching strategy: exact match, then case-insensitive, then
Double Metaphone phonetic matching with edit-distance scoring.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single correction applied to transcription text.

    Attributes:
        original: The text as transcribed by ASR.
        corrected: The domain-correct replacement.
        score: Confidence score (1.0 = exact match, <1.0 = phonetic).
        method: How the match was found ("exact", "case_insensitive", or "phonetic").
    """

    original: str
    corrected: str
    score: float
    method: str


@dataclass
class VocabularyEntry:
    """A vocabulary term with precomputed phonetic codes.

    Attributes:
        canonical: The correct spelling of the term.
        lower: Lowercase form for case-insensitive matching.
        phonetic_primary: Primary Double Metaphone code.
        phonetic_secondary: Secondary Double Metaphone code (may be empty).
    """

    canonical: str
    lower: str
    phonetic_primary: str
    phonetic_secondary: str


def _load_vocabulary(path: Path) -> list[str]:
    """Load vocabulary terms from file.

    Args:
        path: Path to vocabulary file (one term per line, # comments).

    Returns:
        List of vocabulary terms.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        msg = f"Vocabulary file not found: {path}"
        raise FileNotFoundError(msg)

    terms: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
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


def _phonetic_score(code1: str, code2: str) -> float:
    """Score similarity between two phonetic codes.

    Uses normalized edit distance: 1.0 = identical, 0.0 = completely different.

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
    distance = _levenshtein(code1, code2)
    return 1.0 - (distance / max_len)


# Word boundary pattern: splits on whitespace and common punctuation
_WORD_PATTERN = re.compile(r"[\w'-]+|[^\w\s]", re.UNICODE)


@dataclass
class CorrectionDictionary:
    """Post-processing corrector for RT domain vocabulary.

    Tiered matching strategy:
    1. Exact match (score 1.0)
    2. Case-insensitive match (score 0.95)
    3. Double Metaphone phonetic match (score based on code similarity)

    Phonetic matching requires the 'metaphone' package (install with
    ``pip install transcriber-radrx[phonetic]``).
    """

    _entries: list[VocabularyEntry] = field(default_factory=list)
    _exact_map: dict[str, str] = field(default_factory=dict)
    _lower_map: dict[str, str] = field(default_factory=dict)
    _phonetic_enabled: bool = False
    _min_phonetic_score: float = 0.7

    def __init__(
        self,
        vocabulary_path: str | None = None,
        *,
        min_phonetic_score: float = 0.7,
    ) -> None:
        """Initialize with optional vocabulary file.

        Args:
            vocabulary_path: Path to RT vocabulary terms file.
            min_phonetic_score: Minimum phonetic similarity to accept (0.0-1.0).
        """
        self._entries = []
        self._exact_map = {}
        self._lower_map = {}
        self._min_phonetic_score = min_phonetic_score
        self._phonetic_enabled = False

        try:
            from metaphone import doublemetaphone  # noqa: F401

            self._phonetic_enabled = True
        except ImportError:
            logger.info("metaphone not installed — phonetic matching disabled")

        if vocabulary_path is not None:
            self._load(Path(vocabulary_path))

        logger.info(
            "CorrectionDictionary: %d terms, phonetic=%s",
            len(self._entries),
            self._phonetic_enabled,
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
            if self._phonetic_enabled:
                from metaphone import doublemetaphone

                codes = doublemetaphone(term)
                primary = codes[0]
                secondary = codes[1]

            entry = VocabularyEntry(
                canonical=term,
                lower=term.lower(),
                phonetic_primary=primary,
                phonetic_secondary=secondary,
            )
            self._entries.append(entry)
            self._exact_map[term] = term
            # First canonical form wins for case-insensitive
            lower = term.lower()
            if lower not in self._lower_map:
                self._lower_map[lower] = term

    def _match_word(self, word: str) -> Correction | None:
        """Try to match a single word against the vocabulary.

        Args:
            word: Word from ASR output.

        Returns:
            Correction if a match is found, None otherwise.
        """
        # Tier 1: exact match
        if word in self._exact_map:
            return None  # Already correct

        # Tier 2: case-insensitive
        lower = word.lower()
        if lower in self._lower_map:
            canonical = self._lower_map[lower]
            if canonical != word:
                return Correction(
                    original=word,
                    corrected=canonical,
                    score=0.95,
                    method="case_insensitive",
                )
            return None  # Already correct

        # Tier 3: phonetic matching
        if not self._phonetic_enabled:
            return None

        from metaphone import doublemetaphone

        word_codes = doublemetaphone(word)
        word_primary = word_codes[0]
        word_secondary = word_codes[1]

        best_score = 0.0
        best_entry: VocabularyEntry | None = None

        for entry in self._entries:
            # Compare primary-to-primary
            score = _phonetic_score(word_primary, entry.phonetic_primary)

            # Also try cross-comparisons for better coverage
            if word_secondary:
                score = max(score, _phonetic_score(word_secondary, entry.phonetic_primary))
            if entry.phonetic_secondary:
                score = max(score, _phonetic_score(word_primary, entry.phonetic_secondary))
            if word_secondary and entry.phonetic_secondary:
                score = max(score, _phonetic_score(word_secondary, entry.phonetic_secondary))

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self._min_phonetic_score:
            # Don't correct if already matches the canonical form
            if word == best_entry.canonical:
                return None
            return Correction(
                original=word,
                corrected=best_entry.canonical,
                score=best_score,
                method="phonetic",
            )

        return None

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        """Apply corrections to transcribed text.

        Processes text word-by-word, applying tiered matching. Multi-word
        vocabulary terms are not yet supported (future: n-gram matching).

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
            # Preserve whitespace/punctuation between words
            result_parts.append(text[last_end : match.start()])
            word = match.group()

            correction = self._match_word(word)
            if correction is not None:
                result_parts.append(correction.corrected)
                corrections.append(correction)
            else:
                result_parts.append(word)

            last_end = match.end()

        # Trailing text after last match
        result_parts.append(text[last_end:])

        corrected_text = "".join(result_parts)
        if corrections:
            logger.info("Applied %d corrections to text", len(corrections))
        return corrected_text, corrections
