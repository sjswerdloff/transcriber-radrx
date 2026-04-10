"""Ensemble package for 2-backend ASR transcription fusion.

Phase 1 exports: word-level alignment utilities.
Phase 2 exports: token-class decision rules and ensemble aggregation.
"""

from transcriber_radrx.ensemble.aligner import (
    AlignedSpan,
    AlignmentSummary,
    AlignmentType,
    align_transcriptions,
    format_alignment_diff,
    normalize_for_alignment,
    summarize_alignment,
)
from transcriber_radrx.ensemble.decision_rules import (
    DecisionSource,
    EnsembleResult,
    EnsembleWord,
    ensemble_transcriptions,
)

__all__ = [
    # Phase 1: aligner
    "AlignedSpan",
    "AlignmentSummary",
    "AlignmentType",
    "align_transcriptions",
    "format_alignment_diff",
    "normalize_for_alignment",
    "summarize_alignment",
    # Phase 2: decision rules
    "DecisionSource",
    "EnsembleResult",
    "EnsembleWord",
    "ensemble_transcriptions",
]
