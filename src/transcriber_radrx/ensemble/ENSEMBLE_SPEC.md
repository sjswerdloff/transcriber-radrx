# 2-Backend Ensemble — Design Spec (task #119, pivoted)

**Author:** silas-397300f6
**Status:** Phase 1 spec (word-level alignment)
**Driving insight:** Voxtral Mini 3B and Whisper large-v3 have complementary
failure profiles on the cycle 112 particle therapy corpus. An ensemble that
aligns both outputs and applies token-class decision rules at disagreement
points can leverage each model's strengths.

## Architecture overview

```
Audio file
    ├── Voxtral Mini 3B ──► raw transcription A
    └── Whisper large-v3 ──► raw transcription B
                                    │
                          Word-level alignment
                                    │
                          Token-class decision rules
                                    │
                          Ensemble transcription + provenance
                                    │
                          Safety-gate metric
                                    │
                          Clinician review rendering
```

## Phase 1: Word-level alignment module

**Location:** `src/transcriber_radrx/ensemble/aligner.py`

**Input:** Two transcription strings (one from each backend).

**Output:** A list of `AlignedSpan` objects representing the word-level
alignment between the two transcriptions.

### Data model

```python
class AlignmentType(StrEnum):
    MATCH = "match"          # Both backends produced the same word
    SUBSTITUTION = "sub"     # Different words at the same position
    INSERTION_A = "ins_a"    # Word present in A (Voxtral) but not B (Whisper)
    INSERTION_B = "ins_b"    # Word present in B (Whisper) but not A (Voxtral)

@dataclass(frozen=True)
class AlignedSpan:
    """One aligned position in the word-level alignment.

    For MATCH and SUBSTITUTION, both word_a and word_b are populated.
    For INSERTION_A, only word_a is populated (word_b is None).
    For INSERTION_B, only word_b is populated (word_a is None).
    """
    alignment_type: AlignmentType
    word_a: str | None       # Voxtral's word at this position
    word_b: str | None       # Whisper's word at this position
    position_a: int | None   # Word index in A's token sequence
    position_b: int | None   # Word index in B's token sequence
```

### Algorithm

1. Tokenize both transcriptions into word lists by splitting on whitespace.
   Preserve punctuation attached to words (e.g., "50.4" stays as one token,
   "skull." stays as one token).
2. Use `difflib.SequenceMatcher` on the two word lists to produce an
   alignment via `get_opcodes()`.
3. Convert opcodes to `AlignedSpan` objects:
   - `"equal"` → one `MATCH` span per word pair
   - `"replace"` → one `SUBSTITUTION` span per word pair (zip, with overflow
     going to INSERTION_A or INSERTION_B if the replace block has unequal
     lengths)
   - `"insert"` → `INSERTION_B` spans
   - `"delete"` → `INSERTION_A` spans
4. Return the list of spans.

### Comparison: case-sensitive or case-insensitive?

Case-insensitive for alignment matching (so "Dose" matches "dose"), but
preserve the original case in `word_a` and `word_b` for downstream
decision rules and provenance. The alignment step determines WHAT matches;
the decision rules determine WHICH version to keep.

### Summary statistics

The module should also provide a summary function:

```python
@dataclass(frozen=True)
class AlignmentSummary:
    total_spans: int
    matches: int
    substitutions: int
    insertions_a: int    # Words only in Voxtral
    insertions_b: int    # Words only in Whisper
    agreement_rate: float  # matches / total_spans
```

### Public API

```python
def align_transcriptions(text_a: str, text_b: str) -> list[AlignedSpan]:
    """Word-level alignment of two transcription strings."""

def summarize_alignment(spans: list[AlignedSpan]) -> AlignmentSummary:
    """Compute summary statistics for an alignment."""

def format_alignment_diff(spans: list[AlignedSpan]) -> str:
    """Human-readable inline diff showing agreements and disagreements.

    Example output:
      Dose escalation to [78 GyE | 78 Jai E] in 39 fractions for the
      [chordoma | Kodoma] at the base of skull.

    Square brackets show disagreement points with [Voxtral | Whisper].
    """
```

### Tests

- Identical inputs → all MATCH
- Completely different inputs → all SUBSTITUTION
- One empty input → all INSERTION
- Real examples from cycle 112 data:
  - proton-0004 en_US-lessac-high: Voxtral vs Whisper on the 78 GyE chordoma fixture
  - proton-0019: Voxtral vs Whisper on 55.8 GyE Ewing fixture
  - proton-0008: the mixed Gy/GyE disambiguation fixture
- Case-insensitive matching: "Dose" == "dose" should be MATCH
- Punctuation preservation: "50.4" stays as one token, not split on the period

### Package setup

- `src/transcriber_radrx/ensemble/__init__.py` — package marker, exports
  `align_transcriptions`, `summarize_alignment`, `format_alignment_diff`
- `src/transcriber_radrx/ensemble/aligner.py` — implementation
- `tests/unit/test_ensemble_aligner.py` or
  `tests/validation/tests/test_ensemble_aligner.py` — tests

Check the existing test directory structure and place the tests where
they fit the project convention.

## Phase 1.1: Normalizer fixes (discovered from survey data)

The alignment survey surfaced additional forms that the normalizer must handle.

**Additional normalization entries:**

| Raw form (case-insensitive) | Normalized | Source |
|---|---|---|
| `Gy equivalent` | `GyE` | Voxtral partial form (proton-0003, 0020) |
| `Gy equivalents` | `GyE` | Plural of above |
| `grey equivalent` | `GyE` | British spelling (proton-0013 alan) |
| `grey equivalents` | `GyE` | Plural of above |
| `3D-slash-3D` | `3D/3D` | Whisper transcribes piper's vocalized slash (proton-0019 lessac) |
| `2D-slash-3D` | `2D/3D` | Same pattern |
| `3D-slash-2D` | `3D/2D` | Same pattern |
| `3D-3D` | `3D/3D` | Whisper hyphenated variant (proton-0020 alan) |
| `2D-3D` | `2D/3D` | Same pattern |
| `3D-2D` | `3D/2D` | Same pattern |

**Additional normalizer behavior: split joined number-unit tokens.**

The survey found Voxtral producing `60GyE` (number joined to unit without
space) where the gold has `60 GyE`. Add a regex step that splits
`(\d+)(Gy[Ee]?|cGy)` → `\1 \2` before the lookup-table step. This ensures
dose values and units are always separate tokens for alignment.

## Phase 2: Token-class decision rules

Designed from the Phase 1 alignment survey data (56 Voxtral×Whisper pairs
on the particle therapy corpus). The survey showed ~86% average word-level
agreement, with disagreements concentrated at dose units, medical terminology,
and formatting — exactly the token classes predicted.

### Data model

```python
class DecisionSource(StrEnum):
    MATCH = "match"            # Both backends agreed
    VOXTRAL = "voxtral"        # Voxtral's word chosen
    WHISPER = "whisper"         # Whisper's word chosen
    CONTEXT_RULE = "context"   # Neither backend's word; derived from context
    HUMAN_REVIEW = "review"    # Unresolved; flagged for human review

@dataclass(frozen=True)
class EnsembleWord:
    """One word in the ensemble output with full provenance."""
    word: str                    # The chosen word
    source: DecisionSource       # How it was chosen
    word_voxtral: str | None     # What Voxtral said (None if insertion)
    word_whisper: str | None     # What Whisper said (None if insertion)
    rule_id: str | None          # Which rule fired (None for MATCH)
    confidence: float            # 1.0 for MATCH, varies for rules
    needs_review: bool           # True if flagged for human review
```

### Decision rules (priority order)

Rules are evaluated in order. First match wins.

**Rule 1: MATCH — both backends agree.**
- Condition: `alignment_type == MATCH`
- Action: take the word, source=MATCH, confidence=1.0
- This covers ~86% of all spans.

**Rule 2: DOSE_UNIT_GYE — dose unit with GyE available.**
- Condition: one or both words are a Gy-variant (`Gy`, `GyE`, `GiE`, `Jai`,
  `JIE`, `JEE`, `HIE`, `J`, `GJE`, `JAE`, `Jie`, `ji`, etc.) AND at least
  one word is `GyE`.
- Action: take `GyE`, source=CONTEXT_RULE if promoted from non-GyE, or
  VOXTRAL/WHISPER if one already had it. confidence=0.95.
- Example: Voxtral `Gy` vs Whisper `GyE` → take `GyE` (source=WHISPER).

**Rule 3: DOSE_UNIT_CONTEXT — both produce Gy-variant, neither is GyE.**
- Condition: both words are Gy-variants (from the known-bad list + `Gy`) AND
  particle-therapy context clues exist in the fixture.
- Action: take `GyE`, source=CONTEXT_RULE, confidence=0.85. Needs_review=True
  (the promotion is context-inferred, not directly observed).
- Example: Voxtral `Gy` vs Whisper `Jai E` in a fixture mentioning `proton`
  → take `GyE`, flag for review.

**Rule 4: DOSE_UNIT_VISIBLE_CORRUPTION — one is Gy-variant, other is not.**
- Condition: exactly one word is a Gy-variant.
- Action: take the Gy-variant (it's at least partially right). If the fixture
  has particle context, promote to `GyE`.
- This handles cases where one backend completely missed the unit.

**Rule 5: VOCABULARY_MATCH — one word is in the RT vocabulary.**
- Condition: one word (case-insensitive) is found in `data/rt_vocabulary.txt`
  and the other is not.
- Action: take the vocabulary match, source=VOXTRAL or WHISPER.
  confidence=0.9.
- Example: Voxtral `Grothendieck` vs Whisper `Proton` → `Proton` is in
  the vocabulary, take Whisper.
- Example: Voxtral `protons` vs Whisper `Procums` → `protons` matches
  vocabulary (or a stem of a vocab entry), take Voxtral.

**Rule 6: BOTH_WRONG — neither word matches vocabulary, AND the words
differ significantly from each other.**
- Condition: neither word is in the RT vocabulary, and `difflib.SequenceMatcher
  .ratio()` between them is < 0.5 (very different words = likely both wrong).
- Action: take Voxtral's word (lower WER default), BUT set
  needs_review=True, source=HUMAN_REVIEW, confidence=0.3.
- Example: Voxtral `craniofacial ingoma` vs Whisper `craniophore inguia` →
  take Voxtral's version, flag for human review.

**Rule 7: DECIMAL_PRECISION — one word has more decimal digits.**
- Condition: both words look numeric (`\d+\.?\d*`), and one has more decimal
  precision than the other.
- Action: take the higher-precision word. confidence=0.9.
- Example: Voxtral `55.8` vs Whisper `55` → take `55.8`.

**Rule 8: FORMATTING_DEFAULT — all other substitutions.**
- Condition: none of the above rules matched.
- Action: take Voxtral's word (lower WER, better formatting).
  source=VOXTRAL, confidence=0.7.
- Example: `high-risk` vs `high risk` → take Voxtral.

**Rule 9: INSERTION_A — word only in Voxtral.**
- Condition: `alignment_type == INSERTION_A`.
- Action: include the word, source=VOXTRAL, confidence=0.6.

**Rule 10: INSERTION_B — word only in Whisper.**
- Condition: `alignment_type == INSERTION_B`.
- Action: include the word, source=WHISPER, confidence=0.6.

### Gy-variant detection

A utility function `is_gy_variant(word: str) -> bool` that returns True if
the word is any known rendering of Gy or GyE. The list (case-insensitive):

```
Gy, GyE, GiE, GI, GI-E, GIE, Gie, Gi, Giy, Jai, JIE, JEE, HIE, Jy,
Ji, Jie, J, GE, GAE, gi.e., giE, J,E, GJE, JAE, Ie, DiE, GeV
```

This list grows as new failure modes are discovered. It is maintained as a
module constant.

### Particle-therapy context detection

Reuse the context-clue vocabulary from `safety_gate.py`:

```
proton, protons, pencil beam scanning, PBS, carbon ion, particle therapy,
craniospinal, craniospinal irradiation, CSI, chordoma, medulloblastoma,
Ewing sarcoma, rhabdomyosarcoma, ependymoma, craniopharyngioma,
neuroblastoma, germinoma, RBE, relative biological effectiveness
```

Check both the Voxtral and Whisper full transcription strings for these clues
(not just the local disagreement span). If any clue is present in either
transcription, the fixture has particle context.

### Ensemble output

```python
@dataclass
class EnsembleResult:
    """The ensemble output for one fixture × voice pair."""
    fixture_id: str
    voice: str
    text_voxtral: str          # Original Voxtral transcription
    text_whisper: str          # Original Whisper transcription
    text_ensemble: str         # The ensemble-chosen transcription
    words: list[EnsembleWord]  # Per-word provenance
    needs_review: bool         # True if ANY word is flagged
    review_count: int          # Number of words flagged for review
    agreement_rate: float      # From alignment summary
    voxtral_chosen: int        # Count of words from Voxtral
    whisper_chosen: int        # Count of words from Whisper
    context_rule_count: int    # Count of words from context rules
```

### Public API

```python
def ensemble_transcriptions(
    text_voxtral: str,
    text_whisper: str,
    vocabulary: set[str],
    fixture_id: str = "",
    voice: str = "",
) -> EnsembleResult:
    """Produce an ensemble transcription from two backend outputs.

    Args:
        text_voxtral: Raw Voxtral transcription.
        text_whisper: Raw Whisper transcription.
        vocabulary: Set of known RT vocabulary terms for tiebreaking.
        fixture_id: For provenance tracking.
        voice: For provenance tracking.

    Returns:
        EnsembleResult with per-word provenance.
    """
```

### Ensemble aggregator script

`tests/validation/scripts/run_ensemble_aggregator.py`:

1. Load both bake-off JSONs (Voxtral and Whisper).
2. Load the RT vocabulary from `data/rt_vocabulary.txt`.
3. For each fixture × voice pair, run `ensemble_transcriptions`.
4. Compute WER of the ensemble output against gold.
5. Run the safety-gate metric on the ensemble transcriptions.
6. Output an ensemble JSON report in the same schema as the bake-off reports
   (so the safety gate can consume it directly).
7. Print a console summary: ensemble WER, safety-gate decisions, count of
   words flagged for review, comparison table (Voxtral-alone vs Whisper-alone
   vs Ensemble).

## Phase 3+: deferred until Phase 2 is verified
