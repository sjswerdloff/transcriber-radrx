"""Safety-gate metric for radiation oncology ASR bake-off reports.

Turns the set of known safety-critical failure classes into a formal
deployment gate that can be applied to any bake-off run's per-sample
output JSON (produced by run_multi_backend_e2e.py).

Five failure classes (v1.1):
  DECIMAL_DROP          — CRITICAL (weight 30) — UNRECOVERABLE
  DOSE_VALUE_MISSING    — CRITICAL (weight 30) — UNRECOVERABLE
  SILENT_UNIT_SUBSTITUTION — HIGH (weight 5)   — CONTEXT_RULE
  SLASHED_FORM_LOSS     — MEDIUM (weight 2)    — ADJACENCY_RULE
  DOSE_UNIT_CORRUPTION  — HIGH (weight 5)      — PHONETIC_MAP

Usage (CLI):
    python -m tests.validation.metrics.safety_gate <report.json>

Output: writes <report.json>.safety_gate.json next to the input file.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "1.1"

#: Particle-therapy context clues (positive signal for GyE presence)
PARTICLE_THERAPY_CLUES: frozenset[str] = frozenset(
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

#: Known-bad renderings of "Gy" from bake-off data (cycle 110 + 112)
KNOWN_BAD_GY_RENDERINGS: tuple[str, ...] = (
    "GI",
    "GI-E",
    "GIE",
    "Gie",
    "GiE",
    "Giy",
    "Jai E",
    "JIE",
    "JEE",
    "HIE",
    "Jy",
    "Ji",
    "Jie",
    "GE",
    "GAE",
    "gi.e.",
    "giE",
    "J E",
    "J to",
    "J in",
)

#: Severity levels
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"

#: Severity weights for safety-adjusted WER penalty per failure (v1.1: CRITICAL bumped 0.05 → 0.15)
SEVERITY_WEIGHTS: dict[str, float] = {
    SEVERITY_CRITICAL: 0.15,
    SEVERITY_HIGH: 0.025,
    SEVERITY_MEDIUM: 0.01,
}

#: Gate thresholds — deployment decisions, not credentials
GATE_PASS = "PASS"  # noqa: S105
GATE_CONDITIONAL = "CONDITIONAL"
GATE_FAIL = "FAIL"

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_DOSE_UNIT = r"(?:Gy|GyE|gray(?:\s+equivalent)?|cGy)"

# Class 1: decimal dose detection in gold
_RE_DECIMAL_DOSE = re.compile(
    r"\b(\d+)\.(\d+)\b\s*(?:Gy|GyE|gray|cGy|gray\s+equivalent)",
    re.IGNORECASE,
)

# Class 2: any dose value (int or decimal) followed by a dose unit
_RE_ANY_DOSE_VALUE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:Gy|GyE|gray|cGy|gray\s+equivalent)",
    re.IGNORECASE,
)

# Class 3: GyE dose expressions in gold
_RE_GYE_DOSE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(GyE|gray\s+equivalent)",
    re.IGNORECASE,
)

# Class 3: prediction has Gy (without E) at the same number position
_RE_GY_NOT_GYE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*Gy\b(?!\s*E\b)(?!E)",
    re.IGNORECASE,
)

# Class 4: slashed IGRT form in gold
_RE_SLASHED_FORM = re.compile(r"\b([23]D)/([23]D)\b")

# Class 5: dose expression in gold (Gy or GyE)
_RE_GY_EXPR = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(Gy|GyE)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Correctability taxonomy (v1.1)
# ---------------------------------------------------------------------------


class Correctability(StrEnum):
    """Correctability class for a Failure — describes how a downstream corrector can handle it.

    Attributes:
        UNRECOVERABLE: Information is lost at the signal level. Corrector cannot recover.
        CONTEXT_RULE: Fixable by a rule that uses surrounding context.
        PHONETIC_MAP: Fixable by a known-bad-list phonetic mapping.
        ADJACENCY_RULE: Fixable by a simple adjacency rule (e.g., slash restoration).
    """

    UNRECOVERABLE = "UNRECOVERABLE"
    CONTEXT_RULE = "CONTEXT_RULE"
    PHONETIC_MAP = "PHONETIC_MAP"
    ADJACENCY_RULE = "ADJACENCY_RULE"


#: Fixed mapping from failure class → correctability tag (per v1.1 spec)
FAILURE_CLASS_CORRECTABILITY: dict[str, Correctability] = {
    "DECIMAL_DROP": Correctability.UNRECOVERABLE,
    "DOSE_VALUE_MISSING": Correctability.UNRECOVERABLE,
    "SILENT_UNIT_SUBSTITUTION": Correctability.CONTEXT_RULE,
    "DOSE_UNIT_CORRUPTION": Correctability.PHONETIC_MAP,
    "SLASHED_FORM_LOSS": Correctability.ADJACENCY_RULE,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """A single detected safety-critical failure on one sample.

    Attributes:
        fixture_id: Identifier of the failing sample (e.g. "proton-0001").
        class_: Failure class name (e.g. "DECIMAL_DROP").
        severity: One of CRITICAL / HIGH / MEDIUM.
        correctability: How a downstream corrector pipeline can handle this failure.
        detail: Human-readable description of what went wrong.
        gold_value: The value as it appeared in ground truth, or None.
        pred_value: The value as it appeared in the prediction, or None.
    """

    fixture_id: str
    class_: str
    severity: str
    correctability: Correctability
    detail: str
    gold_value: str | None
    pred_value: str | None


@dataclass
class VoiceResult:
    """Per-voice safety-gate results for one backend.

    Attributes:
        voice: Voice identifier string.
        n_samples: Number of samples evaluated.
        failures: All detected failures for this voice.
        raw_wer_sum: Sum of raw WER across all samples (for penalty calculation).
        safety_adjusted_wer: Safety-adjusted WER (raw_wer + per-failure penalties).
        gate: Deployment gate decision (v1.0 alias for raw_gate): PASS / CONDITIONAL / FAIL.
        raw_gate: v1.1 gate considering ALL failures. Same value as gate.
        voice_post_correction_gate: v1.1 gate considering only UNRECOVERABLE failures.
    """

    voice: str
    n_samples: int
    failures: list[Failure] = field(default_factory=list)
    raw_wer_sum: float = 0.0
    safety_adjusted_wer: float = 0.0
    gate: str = GATE_PASS
    raw_gate: str = GATE_PASS
    voice_post_correction_gate: str = GATE_PASS

    def failures_by_severity(self) -> dict[str, list[Failure]]:
        """Group failures by severity level.

        Returns:
            Dict mapping severity level to list of Failure objects.
        """
        result: dict[str, list[Failure]] = {
            SEVERITY_CRITICAL: [],
            SEVERITY_HIGH: [],
            SEVERITY_MEDIUM: [],
        }
        for f in self.failures:
            result[f.severity].append(f)
        return result

    def counts(self) -> dict[str, int]:
        """Count failures per severity level.

        Returns:
            Dict mapping severity to failure count.
        """
        by_sev = self.failures_by_severity()
        return {sev: len(lst) for sev, lst in by_sev.items()}

    def counts_by_correctability(self) -> dict[str, int]:
        """Count failures per correctability class.

        Returns:
            Dict mapping Correctability value to failure count.
        """
        result: dict[str, int] = {c.value: 0 for c in Correctability}
        for f in self.failures:
            result[f.correctability.value] += 1
        return result

    def residual_unrecoverable(self) -> dict[str, int]:
        """Count UNRECOVERABLE failures per severity level.

        Returns:
            Dict mapping severity level to count of UNRECOVERABLE failures at that severity.
        """
        unrec = [f for f in self.failures if f.correctability == Correctability.UNRECOVERABLE]
        return {
            SEVERITY_CRITICAL: sum(1 for f in unrec if f.severity == SEVERITY_CRITICAL),
            SEVERITY_HIGH: sum(1 for f in unrec if f.severity == SEVERITY_HIGH),
            SEVERITY_MEDIUM: sum(1 for f in unrec if f.severity == SEVERITY_MEDIUM),
        }


@dataclass
class BackendResult:
    """Aggregated safety-gate results for one backend across all voices.

    Attributes:
        backend: Backend identifier string.
        voice_results: List of per-voice results.
        overall_gate: Aggregate gate decision across all voices (v1.0 alias for overall_raw_gate).
        overall_raw_gate: v1.1 raw gate considering ALL failures across all voices.
        overall_post_correction_gate: v1.1 gate considering only UNRECOVERABLE failures.
        overall_safety_adjusted_wer: Mean safety-adjusted WER across voices.
        overall_counts: Summed failure counts across all voices.
        overall_counts_by_correctability: Summed correctability counts across all voices.
        overall_residual_unrecoverable: Summed UNRECOVERABLE failure counts by severity.
    """

    backend: str
    voice_results: list[VoiceResult] = field(default_factory=list)
    overall_gate: str = GATE_PASS
    overall_raw_gate: str = GATE_PASS
    overall_post_correction_gate: str = GATE_PASS
    overall_safety_adjusted_wer: float = 0.0
    overall_counts: dict[str, int] = field(default_factory=dict)
    overall_counts_by_correctability: dict[str, int] = field(default_factory=dict)
    overall_residual_unrecoverable: dict[str, int] = field(default_factory=dict)


@dataclass
class SafetyGateReport:
    """Full safety-gate report for one bake-off JSON.

    Attributes:
        source_report: Filename of the source bake-off JSON.
        timestamp: ISO timestamp of when the report was generated.
        backend_results: Dict mapping backend name to BackendResult.
    """

    source_report: str
    timestamp: str
    backend_results: dict[str, BackendResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector functions
# ---------------------------------------------------------------------------


def _detect_decimal_drop(fixture_id: str, gold: str, pred: str) -> list[Failure]:
    """Detect DECIMAL_DROP: gold has X.Y dose, prediction has only X.

    Args:
        fixture_id: Sample identifier.
        gold: Ground-truth text.
        pred: Predicted transcription text.

    Returns:
        List of Failure objects (may be empty).
    """
    failures: list[Failure] = []
    for m in _RE_DECIMAL_DOSE.finditer(gold):
        integer_part = m.group(1)
        decimal_part = m.group(2)
        full_value = f"{integer_part}.{decimal_part}"
        if full_value not in pred and re.search(r"\b" + re.escape(integer_part) + r"\b", pred):
            failures.append(
                Failure(
                    fixture_id=fixture_id,
                    class_="DECIMAL_DROP",
                    severity=SEVERITY_CRITICAL,
                    correctability=FAILURE_CLASS_CORRECTABILITY["DECIMAL_DROP"],
                    detail=f"Gold has '{full_value}' but prediction contains only integer '{integer_part}'",
                    gold_value=full_value,
                    pred_value=integer_part,
                )
            )
    return failures


def _detect_dose_value_missing(fixture_id: str, gold: str, pred: str) -> list[Failure]:
    """Detect DOSE_VALUE_MISSING: gold dose number is absent from prediction.

    Args:
        fixture_id: Sample identifier.
        gold: Ground-truth text.
        pred: Predicted transcription text.

    Returns:
        List of Failure objects (may be empty).
    """
    failures: list[Failure] = []
    seen_values: set[str] = set()
    for m in _RE_ANY_DOSE_VALUE.finditer(gold):
        value = m.group(1)
        if value in seen_values:
            continue
        seen_values.add(value)
        # Check if the number (or its decimal representation) appears anywhere in pred
        if not re.search(r"\b" + re.escape(value) + r"\b", pred):
            # Also not there as plain number in case of decimal
            failures.append(
                Failure(
                    fixture_id=fixture_id,
                    class_="DOSE_VALUE_MISSING",
                    severity=SEVERITY_CRITICAL,
                    correctability=FAILURE_CLASS_CORRECTABILITY["DOSE_VALUE_MISSING"],
                    detail=f"Gold dose value '{value}' does not appear anywhere in prediction",
                    gold_value=value,
                    pred_value=None,
                )
            )
    return failures


def _has_particle_therapy_context(text: str) -> bool:
    """Check if text contains any particle-therapy context clue.

    Args:
        text: Text to search (case-insensitive).

    Returns:
        True if any context clue is found.
    """
    lower = text.lower()
    return any(clue in lower for clue in PARTICLE_THERAPY_CLUES)


def _detect_silent_unit_substitution(fixture_id: str, gold: str, pred: str) -> list[Failure]:
    """Detect SILENT_UNIT_SUBSTITUTION: GyE silently replaced with Gy.

    Requires that the gold contains GyE or 'gray equivalent', that
    particle-therapy context clues are present in the gold, and that
    the prediction uses Gy (without E) at the same numeric position.

    Args:
        fixture_id: Sample identifier.
        gold: Ground-truth text.
        pred: Predicted transcription text.

    Returns:
        List of Failure objects (may be empty).
    """
    failures: list[Failure] = []

    # Step 1: gold must contain GyE or gray equivalent
    if not _RE_GYE_DOSE.search(gold):
        return failures

    # Step 2: particle-therapy context must be present in gold
    if not _has_particle_therapy_context(gold):
        return failures

    # Step 3: for each GyE / gray equivalent dose in gold, check if pred has same
    # number + Gy (without E) or + gray (without "equivalent")
    for m in _RE_GYE_DOSE.finditer(gold):
        number = m.group(1)
        gold_unit = m.group(2)

        # Look for this number + Gy (not GyE) OR + gray (not gray equivalent) in the prediction.
        # Two sub-patterns:
        #   1. "number Gy" — does not contain the E suffix
        #   2. "number gray" — does not contain " equivalent" after it
        gy_pattern = re.compile(
            r"\b" + re.escape(number) + r"\s*Gy\b(?!E)(?!\s*E\b)",
            re.IGNORECASE,
        )
        gray_pattern = re.compile(
            r"\b" + re.escape(number) + r"\s*gray\b(?!\s+equivalent)",
            re.IGNORECASE,
        )
        if gy_pattern.search(pred) or gray_pattern.search(pred):
            failures.append(
                Failure(
                    fixture_id=fixture_id,
                    class_="SILENT_UNIT_SUBSTITUTION",
                    severity=SEVERITY_HIGH,
                    correctability=FAILURE_CLASS_CORRECTABILITY["SILENT_UNIT_SUBSTITUTION"],
                    detail=(
                        f"Gold has '{number} {gold_unit}' but prediction silently uses '{number} Gy' "
                        f"(particle-therapy context present; RBE correction lost)"
                    ),
                    gold_value=f"{number} {gold_unit}",
                    pred_value=f"{number} Gy",
                )
            )
    return failures


def _detect_slashed_form_loss(fixture_id: str, gold: str, pred: str) -> list[Failure]:
    """Detect SLASHED_FORM_LOSS: IGRT slashed form absent from prediction.

    Args:
        fixture_id: Sample identifier.
        gold: Ground-truth text.
        pred: Predicted transcription text.

    Returns:
        List of Failure objects (may be empty).
    """
    failures: list[Failure] = []
    for m in _RE_SLASHED_FORM.finditer(gold):
        slashed = m.group(0)  # e.g. "3D/3D"
        if slashed not in pred:
            # Check if unslashed variant is present (recoverable)
            unslashed = f"{m.group(1)} {m.group(2)}"
            has_unslashed = unslashed in pred
            detail = f"Gold has slashed IGRT form '{slashed}' which is absent from prediction" + (
                f" (unslashed '{unslashed}' found — recoverable by corrector)" if has_unslashed else ""
            )
            failures.append(
                Failure(
                    fixture_id=fixture_id,
                    class_="SLASHED_FORM_LOSS",
                    severity=SEVERITY_MEDIUM,
                    correctability=FAILURE_CLASS_CORRECTABILITY["SLASHED_FORM_LOSS"],
                    detail=detail,
                    gold_value=slashed,
                    pred_value=unslashed if has_unslashed else None,
                )
            )
    return failures


def _detect_dose_unit_corruption(fixture_id: str, gold: str, pred: str) -> list[Failure]:
    """Detect DOSE_UNIT_CORRUPTION: Gy/GyE rendered as a known-bad form.

    Args:
        fixture_id: Sample identifier.
        gold: Ground-truth text.
        pred: Predicted transcription text.

    Returns:
        List of Failure objects (may be empty).
    """
    failures: list[Failure] = []
    if not _RE_GY_EXPR.search(gold):
        return failures

    # Build a pattern for any known-bad rendering appearing near a number
    for bad in KNOWN_BAD_GY_RENDERINGS:
        bad_pattern = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*" + re.escape(bad) + r"\b",
            re.IGNORECASE,
        )
        for m in bad_pattern.finditer(pred):
            number = m.group(1)
            failures.append(
                Failure(
                    fixture_id=fixture_id,
                    class_="DOSE_UNIT_CORRUPTION",
                    severity=SEVERITY_HIGH,
                    correctability=FAILURE_CLASS_CORRECTABILITY["DOSE_UNIT_CORRUPTION"],
                    detail=f"Prediction contains known-bad Gy rendering '{number} {bad}' for gold dose near '{number} Gy/GyE'",
                    gold_value=None,
                    pred_value=f"{number} {bad}",
                )
            )
    return failures


# ---------------------------------------------------------------------------
# Core evaluation API
# ---------------------------------------------------------------------------


def evaluate_sample(sample: dict[str, object]) -> list[Failure]:
    """Run all five failure-class detectors on a single bake-off sample.

    Args:
        sample: One element of ``results[].by_voice[].samples[]`` from the
            bake-off JSON. Must have ``fixture_id``, ``ground_truth``, and
            ``raw_transcription`` keys.

    Returns:
        List of Failure objects, empty if the sample passes all checks.
    """
    fixture_id = str(sample.get("fixture_id", "unknown"))
    gold = str(sample.get("ground_truth", ""))
    pred = str(sample.get("raw_transcription", ""))

    failures: list[Failure] = []
    failures.extend(_detect_decimal_drop(fixture_id, gold, pred))
    failures.extend(_detect_dose_value_missing(fixture_id, gold, pred))
    failures.extend(_detect_silent_unit_substitution(fixture_id, gold, pred))
    failures.extend(_detect_slashed_form_loss(fixture_id, gold, pred))
    failures.extend(_detect_dose_unit_corruption(fixture_id, gold, pred))
    return failures


def _gate_decision(counts: dict[str, int]) -> str:
    """Determine gate decision from failure counts.

    This implements the raw gate logic: all failures are considered.

    Args:
        counts: Dict with CRITICAL / HIGH / MEDIUM counts.

    Returns:
        One of PASS / CONDITIONAL / FAIL.
    """
    if counts.get(SEVERITY_CRITICAL, 0) > 0 or counts.get(SEVERITY_HIGH, 0) > 0:
        return GATE_FAIL
    if counts.get(SEVERITY_MEDIUM, 0) > 0:
        return GATE_CONDITIONAL
    return GATE_PASS


def post_correction_gate(failures: list[Failure]) -> str:
    """v1.1 gate: considers only UNRECOVERABLE failures.

    Assumes an ideal corrector pipeline has applied all rule-based
    classes (CONTEXT_RULE, PHONETIC_MAP, ADJACENCY_RULE) and eliminated
    those failures. Only UNRECOVERABLE failures remain.

    Args:
        failures: All failures detected for this voice or backend.

    Returns:
        One of PASS / CONDITIONAL / FAIL based on residual unrecoverable failures.
    """
    residual = [f for f in failures if f.correctability == Correctability.UNRECOVERABLE]
    residual_counts = {
        SEVERITY_CRITICAL: sum(1 for f in residual if f.severity == SEVERITY_CRITICAL),
        SEVERITY_HIGH: sum(1 for f in residual if f.severity == SEVERITY_HIGH),
        SEVERITY_MEDIUM: sum(1 for f in residual if f.severity == SEVERITY_MEDIUM),
    }
    return _gate_decision(residual_counts)


def _compute_safety_adjusted_wer(raw_wer: float, failures: list[Failure], n_samples: int) -> float:
    """Compute safety-adjusted WER for one voice/backend combination.

    Formula (v1):
        safety_adjusted_wer = raw_wer + sum(weight[sev] per failure) / n_samples

    Args:
        raw_wer: Average raw WER across all samples for this voice.
        failures: All failures detected for this voice.
        n_samples: Number of samples evaluated.

    Returns:
        Safety-adjusted WER (unbounded above raw_wer).
    """
    if n_samples == 0:
        return raw_wer
    penalty = sum(SEVERITY_WEIGHTS[f.severity] for f in failures)
    return raw_wer + penalty / n_samples


def evaluate_report(report: dict[str, object]) -> SafetyGateReport:
    """Run the safety gate over a complete bake-off report JSON.

    Args:
        report: Parsed bake-off JSON dict (full file contents).

    Returns:
        SafetyGateReport with per-backend and per-voice results.
    """
    source = str(report.get("source_report", "unknown"))
    timestamp = datetime.now(UTC).isoformat()
    gate_report = SafetyGateReport(source_report=source, timestamp=timestamp)

    results = report.get("results", [])
    if not isinstance(results, list):
        logger.warning("'results' key missing or not a list in bake-off report")
        return gate_report

    for backend_entry in results:
        if not isinstance(backend_entry, dict):
            continue
        backend_name = str(backend_entry.get("backend", "unknown"))
        by_voice = backend_entry.get("by_voice", [])
        if not isinstance(by_voice, list):
            continue

        backend_result = BackendResult(backend=backend_name)
        all_backend_failures: list[Failure] = []

        for voice_entry in by_voice:
            if not isinstance(voice_entry, dict):
                continue
            voice_name = str(voice_entry.get("voice", "unknown"))
            samples = voice_entry.get("samples", [])
            summary = voice_entry.get("summary", {})
            if not isinstance(samples, list):
                continue
            if not isinstance(summary, dict):
                summary = {}

            n_samples = int(summary.get("sample_count", len(samples)))
            avg_raw_wer = float(summary.get("avg_raw_wer", 0.0))

            voice_failures: list[Failure] = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                sample_failures = evaluate_sample(sample)
                voice_failures.extend(sample_failures)
                if sample_failures:
                    logger.debug(
                        "backend={} voice={} fixture={} failures={}",
                        backend_name,
                        voice_name,
                        sample.get("fixture_id"),
                        len(sample_failures),
                    )

            safety_adj = _compute_safety_adjusted_wer(avg_raw_wer, voice_failures, n_samples)
            counts = {
                SEVERITY_CRITICAL: sum(1 for f in voice_failures if f.severity == SEVERITY_CRITICAL),
                SEVERITY_HIGH: sum(1 for f in voice_failures if f.severity == SEVERITY_HIGH),
                SEVERITY_MEDIUM: sum(1 for f in voice_failures if f.severity == SEVERITY_MEDIUM),
            }
            gate = _gate_decision(counts)
            pcg = post_correction_gate(voice_failures)

            voice_result = VoiceResult(
                voice=voice_name,
                n_samples=n_samples,
                failures=voice_failures,
                raw_wer_sum=avg_raw_wer,
                safety_adjusted_wer=safety_adj,
                gate=gate,
                raw_gate=gate,
                voice_post_correction_gate=pcg,
            )
            backend_result.voice_results.append(voice_result)
            all_backend_failures.extend(voice_failures)

        # Aggregate across voices
        overall_counts = {
            SEVERITY_CRITICAL: sum(1 for f in all_backend_failures if f.severity == SEVERITY_CRITICAL),
            SEVERITY_HIGH: sum(1 for f in all_backend_failures if f.severity == SEVERITY_HIGH),
            SEVERITY_MEDIUM: sum(1 for f in all_backend_failures if f.severity == SEVERITY_MEDIUM),
        }
        overall_corr: dict[str, int] = {c.value: 0 for c in Correctability}
        for f in all_backend_failures:
            overall_corr[f.correctability.value] += 1
        unrec_all = [f for f in all_backend_failures if f.correctability == Correctability.UNRECOVERABLE]
        overall_residual = {
            SEVERITY_CRITICAL: sum(1 for f in unrec_all if f.severity == SEVERITY_CRITICAL),
            SEVERITY_HIGH: sum(1 for f in unrec_all if f.severity == SEVERITY_HIGH),
            SEVERITY_MEDIUM: sum(1 for f in unrec_all if f.severity == SEVERITY_MEDIUM),
        }
        overall_raw_gate = _gate_decision(overall_counts)
        overall_pcg = post_correction_gate(all_backend_failures)
        backend_result.overall_gate = overall_raw_gate
        backend_result.overall_raw_gate = overall_raw_gate
        backend_result.overall_post_correction_gate = overall_pcg
        backend_result.overall_counts = overall_counts
        backend_result.overall_counts_by_correctability = overall_corr
        backend_result.overall_residual_unrecoverable = overall_residual
        if backend_result.voice_results:
            backend_result.overall_safety_adjusted_wer = sum(
                vr.safety_adjusted_wer for vr in backend_result.voice_results
            ) / len(backend_result.voice_results)

        gate_report.backend_results[backend_name] = backend_result
        logger.info(
            "backend={} gate={} CRITICAL={} HIGH={} MEDIUM={}",
            backend_name,
            backend_result.overall_gate,
            overall_counts[SEVERITY_CRITICAL],
            overall_counts[SEVERITY_HIGH],
            overall_counts[SEVERITY_MEDIUM],
        )

    return gate_report


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _failure_to_dict(f: Failure) -> dict[str, object]:
    """Serialize a Failure to a JSON-serializable dict.

    Args:
        f: Failure dataclass instance.

    Returns:
        Dict representation of the failure.
    """
    return {
        "fixture_id": f.fixture_id,
        "class": f.class_,
        "severity": f.severity,
        "correctability": f.correctability.value,
        "detail": f.detail,
        "gold_value": f.gold_value,
        "pred_value": f.pred_value,
    }


def _voice_result_to_dict(vr: VoiceResult) -> dict[str, object]:
    """Serialize a VoiceResult to a JSON-serializable dict.

    Includes both v1.0 backwards-compat fields (gate) and v1.1 additions
    (counts_by_correctability, residual_unrecoverable, raw_gate, post_correction_gate).

    Args:
        vr: VoiceResult dataclass instance.

    Returns:
        Dict representation of the voice result.
    """
    by_sev = vr.failures_by_severity()
    return {
        "n_samples": vr.n_samples,
        "failures_by_severity": {sev: [_failure_to_dict(f) for f in lst] for sev, lst in by_sev.items()},
        "counts": vr.counts(),
        "counts_by_correctability": vr.counts_by_correctability(),
        "residual_unrecoverable": vr.residual_unrecoverable(),
        "gate": vr.gate,  # v1.0 backwards-compat alias for raw_gate
        "raw_gate": vr.raw_gate,
        "post_correction_gate": vr.voice_post_correction_gate,
        "safety_adjusted_wer": round(vr.safety_adjusted_wer, 6),
    }


def report_to_dict(report: SafetyGateReport) -> dict[str, object]:
    """Serialize a SafetyGateReport to a JSON-serializable dict.

    Args:
        report: SafetyGateReport dataclass instance.

    Returns:
        Dict following the output schema defined in SAFETY_GATE_SPEC.md.
    """
    backends_dict: dict[str, object] = {}
    for backend_name, br in report.backend_results.items():
        voices_dict: dict[str, object] = {}
        for vr in br.voice_results:
            voices_dict[vr.voice] = _voice_result_to_dict(vr)
        backends_dict[backend_name] = {
            "voices": voices_dict,
            "overall": {
                "counts": br.overall_counts,
                "counts_by_correctability": br.overall_counts_by_correctability,
                "residual_unrecoverable": br.overall_residual_unrecoverable,
                "gate": br.overall_gate,  # v1.0 backwards-compat alias for raw_gate
                "raw_gate": br.overall_raw_gate,
                "post_correction_gate": br.overall_post_correction_gate,
                "safety_adjusted_wer": round(br.overall_safety_adjusted_wer, 6),
            },
        }
    return {
        "metric_version": VERSION,
        "source_report": report.source_report,
        "timestamp": report.timestamp,
        "backends": backends_dict,
    }


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def _print_console_table(gate_report: SafetyGateReport) -> None:
    """Print a ranked summary table of backends to stdout.

    Args:
        gate_report: Completed safety-gate report.
    """
    # Rank by gate status (FAIL < CONDITIONAL < PASS) then safety-adjusted WER
    gate_rank = {GATE_FAIL: 0, GATE_CONDITIONAL: 1, GATE_PASS: 2}
    ranked = sorted(
        gate_report.backend_results.values(),
        key=lambda br: (gate_rank[br.overall_gate], br.overall_safety_adjusted_wer),
    )

    header = (
        f"{'Backend':<20} {'Raw Gate':<12} {'Post Gate':<12} {'Adj.WER':>8}"
        f" {'CRIT':>6} {'HIGH':>6} {'MED':>5} {'UnrecovCRIT':>12}"
    )
    print("\n=== Safety Gate Results ===")
    print(header)
    print("-" * len(header))
    for br in ranked:
        c = br.overall_counts
        unrec_crit = br.overall_residual_unrecoverable.get(SEVERITY_CRITICAL, 0)
        print(
            f"{br.backend:<20} {br.overall_raw_gate:<12} {br.overall_post_correction_gate:<12}"
            f" {br.overall_safety_adjusted_wer:>8.4f}"
            f" {c.get(SEVERITY_CRITICAL, 0):>6} {c.get(SEVERITY_HIGH, 0):>6}"
            f" {c.get(SEVERITY_MEDIUM, 0):>5} {unrec_crit:>12}"
        )

    print()
    for br in ranked:
        print(f"  {br.backend} [raw={br.overall_raw_gate}] [post={br.overall_post_correction_gate}]")
        for vr in br.voice_results:
            vr_unrec_crit = vr.residual_unrecoverable().get(SEVERITY_CRITICAL, 0)
            print(
                f"    {vr.voice}: raw_gate={vr.raw_gate}  post_gate={vr.voice_post_correction_gate}  "
                f"CRITICAL={vr.counts().get(SEVERITY_CRITICAL, 0)}  "
                f"HIGH={vr.counts().get(SEVERITY_HIGH, 0)}  "
                f"MEDIUM={vr.counts().get(SEVERITY_MEDIUM, 0)}  "
                f"UnrecovCRIT={vr_unrec_crit}  "
                f"adj_wer={vr.safety_adjusted_wer:.4f}"
            )
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    """CLI entry point for the safety-gate metric.

    Args:
        argv: Command-line arguments (not including the module invocation).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    if len(argv) < 1:
        print("Usage: python -m tests.validation.metrics.safety_gate <report.json>", file=sys.stderr)
        return 1

    for arg in argv:
        report_path = Path(arg)
        if not report_path.exists():
            logger.error("Report file not found: {}", report_path)
            return 1

        logger.info("Loading bake-off report: {}", report_path)
        with report_path.open() as fh:
            raw = json.load(fh)

        # Inject source filename into the report so evaluate_report can surface it
        raw["source_report"] = report_path.name

        gate_report = evaluate_report(raw)
        output_dict = report_to_dict(gate_report)

        output_path = report_path.with_suffix(report_path.suffix + ".safety_gate.json")
        with output_path.open("w") as fh:
            json.dump(output_dict, fh, indent=2)
        logger.info("Safety-gate report written to: {}", output_path)

        _print_console_table(gate_report)

    return 0


if __name__ == "__main__":
    sys.exit(_run_cli(sys.argv[1:]))
