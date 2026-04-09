"""Unit and integration tests for the safety-gate metric (task #115).

Test organisation:
  - One class per failure class (five total)
  - TestSafetyGateIntegration: runs against both cycle 112 bake-off JSONs
  - TestCounterfactual: verifies proton-0015 produces no false positives
  - TestAggregation: gate-decision and safety-adjusted-WER formulas
  - TestCLI: __main__ path writes output file correctly

All test fixtures are synthetic sentences derived from the cycle 112 proton
bake-off findings documented in bakeoff_proton_findings_2026-04-09.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.validation.metrics.safety_gate import (
    GATE_CONDITIONAL,
    GATE_FAIL,
    GATE_PASS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    Correctability,
    Failure,
    VoiceResult,
    _compute_safety_adjusted_wer,
    _detect_decimal_drop,
    _detect_dose_unit_corruption,
    _detect_dose_value_missing,
    _detect_silent_unit_substitution,
    _detect_slashed_form_loss,
    _gate_decision,
    _has_particle_therapy_context,
    _run_cli,
    evaluate_report,
    evaluate_sample,
    post_correction_gate,
    report_to_dict,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path(__file__).parent.parent / "reports"
_DRAFT_JSON = _REPORTS_DIR / "bakeoff_proton_draft_2026-04-09.json"
_VOXTRAL_JSON = _REPORTS_DIR / "bakeoff_proton_voxtral_2026-04-09.json"


# ---------------------------------------------------------------------------
# Helper to build a minimal sample dict
# ---------------------------------------------------------------------------


def _make_sample(fixture_id: str, gold: str, pred: str) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "ground_truth": gold,
        "raw_transcription": pred,
        "raw_wer": 0.1,
    }


# ===========================================================================
# Class 1: DECIMAL_DROP (CRITICAL)
# ===========================================================================


class TestDecimalDrop:
    """Tests for _detect_decimal_drop (Class 1: DECIMAL_DROP, CRITICAL)."""

    def test_detects_decimal_drop_in_dose(self) -> None:
        """Contract: gold has '50.4 GyE', prediction has only '50'."""
        # proton-0006 derived fixture
        gold = "Prescribed dose of 50.4 GyE in 28 fractions to the rectal PTV."
        pred = "Prescribed dose of 50 GyE in 28 fractions to the rectal PTV."
        failures = _detect_decimal_drop("test-0001", gold, pred)
        assert len(failures) == 1
        assert failures[0].class_ == "DECIMAL_DROP"
        assert failures[0].severity == SEVERITY_CRITICAL
        assert failures[0].gold_value == "50.4"
        assert failures[0].pred_value == "50"

    def test_detects_multiple_decimal_drops(self) -> None:
        """Contract: gold has '23.4' and '55.8', prediction drops both."""
        # proton-0027 derived fixture
        gold = "CSI dose of 23.4 GyE followed by boost to 55.8 GyE."
        pred = "CSI dose of 23 GyE followed by boost to 55 GyE."
        failures = _detect_decimal_drop("test-0002", gold, pred)
        assert len(failures) == 2
        gold_values = {f.gold_value for f in failures}
        assert "23.4" in gold_values
        assert "55.8" in gold_values

    def test_no_false_positive_when_decimal_preserved(self) -> None:
        """Contract: full decimal value present — no DECIMAL_DROP."""
        gold = "Prescribed dose of 50.4 GyE in 28 fractions."
        pred = "Prescribed dose of 50.4 GyE in 28 fractions."
        failures = _detect_decimal_drop("test-0003", gold, pred)
        assert failures == []

    def test_no_false_positive_when_no_decimal_dose(self) -> None:
        """Contract: gold has integer-only dose — no DECIMAL_DROP."""
        gold = "Prescribed dose of 60 GyE in 30 fractions."
        pred = "Prescribed dose of 60 GyE in 30 fractions."
        failures = _detect_decimal_drop("test-0004", gold, pred)
        assert failures == []

    def test_case_insensitive_unit_matching(self) -> None:
        """Contract: detector is case-insensitive on the dose unit."""
        gold = "Dose of 59.4 gray in 33 fractions."
        pred = "Dose of 59 gray in 33 fractions."
        failures = _detect_decimal_drop("test-0005", gold, pred)
        assert len(failures) == 1
        assert failures[0].gold_value == "59.4"

    def test_decimal_drop_with_cgy(self) -> None:
        """Contract: cGy unit is included in the dose regex."""
        gold = "Boost dose of 14.4 cGy to the posterior fossa."
        pred = "Boost dose of 14 cGy to the posterior fossa."
        failures = _detect_decimal_drop("test-0006", gold, pred)
        assert len(failures) == 1


# ===========================================================================
# Class 2: DOSE_VALUE_MISSING (CRITICAL)
# ===========================================================================


class TestDoseValueMissing:
    """Tests for _detect_dose_value_missing (Class 2: DOSE_VALUE_MISSING, CRITICAL)."""

    def test_detects_whole_number_dose_missing(self) -> None:
        """Contract: gold has '54 GyE in 30 fx', prediction lacks '54'."""
        # proton-0002 derived fixture
        gold = "Prescribed 54 GyE in 30 fractions via RBE-weighted proton dose."
        pred = "Prescribed GyE in 30 fractions via RBE-weighted proton dose."
        failures = _detect_dose_value_missing("test-0010", gold, pred)
        assert any(f.class_ == "DOSE_VALUE_MISSING" for f in failures)
        missing = [f for f in failures if f.class_ == "DOSE_VALUE_MISSING"]
        assert any(f.gold_value == "54" for f in missing)
        assert all(f.severity == SEVERITY_CRITICAL for f in missing)

    def test_detects_dose_missing_when_only_fractions_remain(self) -> None:
        """Contract: gold '60 GyE in 30 fx', pred has only '30 fractions'."""
        # proton-0011 derived fixture
        gold = "Reirradiation dose of 60 GyE in 30 fractions to the tumor bed."
        pred = "Reirradiation dose in 30 fractions to the tumor bed."
        failures = _detect_dose_value_missing("test-0011", gold, pred)
        assert any(f.gold_value == "60" for f in failures)

    def test_no_false_positive_when_value_present(self) -> None:
        """Contract: dose value present in prediction — no DOSE_VALUE_MISSING."""
        gold = "Prescribed 60 GyE in 30 fractions."
        pred = "Prescribed 60 Gy in 30 fractions."
        failures = _detect_dose_value_missing("test-0012", gold, pred)
        assert failures == []

    def test_detects_multiple_missing_values(self) -> None:
        """Contract: SIB dual-dose — gold has '70 GyE' and '56 GyE', pred drops '56'."""
        # proton-0005 derived fixture
        gold = "SIB: 70 GyE to GTV and 56 GyE to CTV in 35 fractions."
        pred = "SIB: 70 GyE to GTV and in 35 fractions."
        failures = _detect_dose_value_missing("test-0013", gold, pred)
        missing_vals = {f.gold_value for f in failures if f.class_ == "DOSE_VALUE_MISSING"}
        assert "56" in missing_vals
        assert "70" not in missing_vals  # 70 is present

    def test_no_false_positive_for_non_dose_numbers(self) -> None:
        """Contract: numbers not followed by dose units are not flagged."""
        gold = "Prescribed 60 GyE to the prostate in 30 fractions over 6 weeks."
        pred = "Prescribed 60 GyE to the prostate in 30 fractions over 6 weeks."
        failures = _detect_dose_value_missing("test-0014", gold, pred)
        assert failures == []

    def test_case_insensitive_unit_matching(self) -> None:
        """Contract: 'gray' (lowercase) triggers the detector."""
        gold = "Prescribed 54 gray in 30 fractions."
        pred = "Prescribed in 30 fractions."
        failures = _detect_dose_value_missing("test-0015", gold, pred)
        assert any(f.gold_value == "54" for f in failures)


# ===========================================================================
# Class 3: SILENT_UNIT_SUBSTITUTION (HIGH)
# ===========================================================================


class TestSilentUnitSubstitution:
    """Tests for _detect_silent_unit_substitution (Class 3: SILENT_UNIT_SUBSTITUTION, HIGH)."""

    def test_detects_gye_replaced_with_gy_in_proton_context(self) -> None:
        """Contract: gold '79.2 GyE' + proton context clue → pred '79.2 Gy' flagged."""
        # proton-0001 derived fixture
        gold = "Prescribed dose of 79.2 GyE in 44 fractions to the prostate PTV using pencil beam scanning."
        pred = "Prescribed dose of 79.2 Gy in 44 fractions to the prostate PTV using pencil beam scanning."
        failures = _detect_silent_unit_substitution("test-0020", gold, pred)
        assert len(failures) == 1
        assert failures[0].class_ == "SILENT_UNIT_SUBSTITUTION"
        assert failures[0].severity == SEVERITY_HIGH
        assert "79.2" in (failures[0].gold_value or "")

    def test_detects_substitution_with_proton_clue(self) -> None:
        """Contract: 'proton' alone is a sufficient context clue."""
        gold = "The proton plan delivers 60 GyE in 30 fractions."
        pred = "The proton plan delivers 60 Gy in 30 fractions."
        failures = _detect_silent_unit_substitution("test-0021", gold, pred)
        assert any(f.class_ == "SILENT_UNIT_SUBSTITUTION" for f in failures)

    def test_no_flag_without_particle_therapy_context(self) -> None:
        """Contract: no context clue → SILENT_UNIT_SUBSTITUTION not raised."""
        gold = "Prescribed dose of 50 GyE in 25 fractions."
        pred = "Prescribed dose of 50 Gy in 25 fractions."
        failures = _detect_silent_unit_substitution("test-0022", gold, pred)
        assert failures == []

    def test_counterfactual_proton_0015(self) -> None:
        """Contract: proton-0015 — gold has 'Gy' (not 'GyE'), so no SILENT_UNIT_SUBSTITUTION.

        Sentence: "The patient declined protons and proceeded with IMRT to 79.2 Gy in
        44 fractions to the prostate."
        The word 'protons' appears but the gold unit is Gy, not GyE. The detector must
        not fire because step 1 (gold contains GyE) fails.
        """
        gold = "The patient declined protons and proceeded with IMRT to 79.2 Gy in 44 fractions to the prostate."
        pred = "the patient declined protons and proceeded with IMMET to 79.2 Gy in 44 fractions to the prostate."
        failures = _detect_silent_unit_substitution("proton-0015", gold, pred)
        assert failures == []

    def test_no_flag_when_prediction_preserves_gye(self) -> None:
        """Contract: prediction correctly has GyE — no substitution flagged."""
        gold = "Prescribed dose of 79.2 GyE using pencil beam scanning."
        pred = "Prescribed dose of 79.2 GyE using pencil beam scanning."
        failures = _detect_silent_unit_substitution("test-0024", gold, pred)
        assert failures == []

    def test_context_clue_detection_case_insensitive(self) -> None:
        """Contract: context clue matching is case-insensitive."""
        gold = "PROTON beam to 54 GyE in 30 fractions."
        pred = "PROTON beam to 54 Gy in 30 fractions."
        failures = _detect_silent_unit_substitution("test-0025", gold, pred)
        assert any(f.class_ == "SILENT_UNIT_SUBSTITUTION" for f in failures)

    def test_gray_equivalent_form_in_gold(self) -> None:
        """Contract: 'gray equivalent' (spelled out) triggers the detector."""
        gold = "Dose of 54 gray equivalent in 30 fractions via proton therapy."
        pred = "Dose of 54 gray in 30 fractions via proton therapy."
        failures = _detect_silent_unit_substitution("test-0026", gold, pred)
        assert any(f.class_ == "SILENT_UNIT_SUBSTITUTION" for f in failures)

    def test_multiple_substitutions_in_one_sample(self) -> None:
        """Contract: multiple GyE values all silently replaced — all flagged."""
        gold = "CSI 23.4 GyE boost 55.8 GyE medulloblastoma proton."
        pred = "CSI 23.4 Gy boost 55.8 Gy medulloblastoma proton."
        failures = _detect_silent_unit_substitution("test-0027", gold, pred)
        assert len(failures) == 2


# ===========================================================================
# Class 4: SLASHED_FORM_LOSS (MEDIUM)
# ===========================================================================


class TestSlashedFormLoss:
    """Tests for _detect_slashed_form_loss (Class 4: SLASHED_FORM_LOSS, MEDIUM)."""

    def test_detects_3d_3d_loss(self) -> None:
        """Contract: gold '3D/3D' is absent from prediction."""
        # proton-0019 derived fixture
        gold = "Image guidance using 3D/3D CBCT matching prior to each fraction."
        pred = "Image guidance using 3D 3D CBCT matching prior to each fraction."
        failures = _detect_slashed_form_loss("test-0030", gold, pred)
        assert len(failures) == 1
        assert failures[0].class_ == "SLASHED_FORM_LOSS"
        assert failures[0].severity == SEVERITY_MEDIUM
        assert failures[0].gold_value == "3D/3D"

    def test_detects_2d_3d_loss(self) -> None:
        """Contract: gold '2D/3D' is absent from prediction."""
        gold = "Portal imaging with 2D/3D matching protocol."
        pred = "Portal imaging with 2D 3D matching protocol."
        failures = _detect_slashed_form_loss("test-0031", gold, pred)
        assert len(failures) == 1
        assert failures[0].gold_value == "2D/3D"

    def test_detects_3d_2d_loss(self) -> None:
        """Contract: gold '3D/2D' is absent from prediction."""
        gold = "Weekly 3D/2D image guidance."
        pred = "Weekly 3D 2D image guidance."
        failures = _detect_slashed_form_loss("test-0032", gold, pred)
        assert len(failures) == 1
        assert failures[0].gold_value == "3D/2D"

    def test_no_false_positive_when_slashed_form_preserved(self) -> None:
        """Contract: prediction contains the slashed form — no flag."""
        gold = "Image guidance using 3D/3D CBCT."
        pred = "Image guidance using 3D/3D CBCT."
        failures = _detect_slashed_form_loss("test-0033", gold, pred)
        assert failures == []

    def test_unslashed_variant_noted_in_detail(self) -> None:
        """Contract: when '3D 3D' is in prediction, detail says recoverable."""
        gold = "Verification with 3D/3D protocol."
        pred = "Verification with 3D 3D protocol."
        failures = _detect_slashed_form_loss("test-0034", gold, pred)
        assert len(failures) == 1
        assert "recoverable" in failures[0].detail.lower()

    def test_detects_multiple_slashed_forms(self) -> None:
        """Contract: multiple slashed forms in gold, all absent from pred."""
        gold = "Initial 3D/3D then weekly 2D/3D imaging."
        pred = "Initial 3D 3D then weekly 2D 3D imaging."
        failures = _detect_slashed_form_loss("test-0035", gold, pred)
        assert len(failures) == 2


# ===========================================================================
# Class 5: DOSE_UNIT_CORRUPTION (HIGH)
# ===========================================================================


class TestDoseUnitCorruption:
    """Tests for _detect_dose_unit_corruption (Class 5: DOSE_UNIT_CORRUPTION, HIGH)."""

    def test_detects_gi_corruption(self) -> None:
        """Contract: gold '79.2 GyE', pred '79.2 GI E' → DOSE_UNIT_CORRUPTION."""
        # Whisper proton-0001 derived fixture
        gold = "Prescribed dose of 79.2 GyE in 44 fractions."
        pred = "Prescribed dose of 79.2 GI E in 44 fractions."
        failures = _detect_dose_unit_corruption("test-0040", gold, pred)
        assert any(f.class_ == "DOSE_UNIT_CORRUPTION" for f in failures)
        assert all(f.severity == SEVERITY_HIGH for f in failures if f.class_ == "DOSE_UNIT_CORRUPTION")

    def test_detects_jai_e_corruption(self) -> None:
        """Contract: 'Jai E' is a known-bad rendering."""
        gold = "Prescribed 54 GyE in 30 fractions."
        pred = "Prescribed 54 Jai E in 30 fractions."
        failures = _detect_dose_unit_corruption("test-0041", gold, pred)
        assert any(f.class_ == "DOSE_UNIT_CORRUPTION" for f in failures)

    def test_detects_jie_corruption(self) -> None:
        """Contract: 'JIE' is a known-bad rendering."""
        gold = "Prescribed 60 GyE in 30 fractions."
        pred = "Prescribed 60 JIE in 30 fractions."
        failures = _detect_dose_unit_corruption("test-0042", gold, pred)
        assert any(f.class_ == "DOSE_UNIT_CORRUPTION" for f in failures)

    def test_detects_gie_corruption(self) -> None:
        """Contract: 'GiE' is a known-bad rendering from MedASR."""
        gold = "Prescribed 70 GyE in 35 fractions."
        pred = "Prescribed 70 GiE in 35 fractions."
        failures = _detect_dose_unit_corruption("test-0043", gold, pred)
        assert any(f.class_ == "DOSE_UNIT_CORRUPTION" for f in failures)

    def test_no_false_positive_with_correct_gy(self) -> None:
        """Contract: correct 'Gy' or 'GyE' in prediction — no DOSE_UNIT_CORRUPTION."""
        gold = "Prescribed 60 Gy in 30 fractions."
        pred = "Prescribed 60 Gy in 30 fractions."
        failures = _detect_dose_unit_corruption("test-0044", gold, pred)
        assert failures == []

    def test_no_flag_when_gold_has_no_dose(self) -> None:
        """Contract: gold without a Gy/GyE expression — detector returns empty."""
        gold = "Patient tolerated treatment well."
        pred = "Patient tolerated treatment GI well."  # GI here is irrelevant
        failures = _detect_dose_unit_corruption("test-0045", gold, pred)
        assert failures == []

    def test_corruption_case_insensitive(self) -> None:
        """Contract: known-bad rendering detection is case-insensitive."""
        gold = "Prescribed 54 GyE."
        pred = "Prescribed 54 gi."
        failures = _detect_dose_unit_corruption("test-0046", gold, pred)
        # 'gi' matches known-bad 'GI' case-insensitively
        # 'gi' lower → needs to match against known-bad 'GI' via re.IGNORECASE
        assert any(f.class_ == "DOSE_UNIT_CORRUPTION" for f in failures)


# ===========================================================================
# Aggregation and gate logic
# ===========================================================================


class TestAggregation:
    """Tests for gate decision and safety-adjusted WER calculations."""

    def test_gate_pass_with_zero_failures(self) -> None:
        """Contract: no failures → PASS."""
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        assert _gate_decision(counts) == GATE_PASS

    def test_gate_fail_with_critical_failures(self) -> None:
        """Contract: any CRITICAL → FAIL."""
        counts = {"CRITICAL": 1, "HIGH": 0, "MEDIUM": 0}
        assert _gate_decision(counts) == GATE_FAIL

    def test_gate_fail_with_high_failures(self) -> None:
        """Contract: any HIGH → FAIL."""
        counts = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0}
        assert _gate_decision(counts) == GATE_FAIL

    def test_gate_conditional_with_only_medium_failures(self) -> None:
        """Contract: MEDIUM only → CONDITIONAL."""
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 3}
        assert _gate_decision(counts) == GATE_CONDITIONAL

    def test_safety_adjusted_wer_increases_with_critical(self) -> None:
        """Contract: CRITICAL failures increase safety-adjusted WER (v1.1 weight 0.15)."""
        raw_wer = 0.10
        failures = [
            Failure("x", "DECIMAL_DROP", SEVERITY_CRITICAL, Correctability.UNRECOVERABLE, "d", "50.4", "50"),
        ]
        adj = _compute_safety_adjusted_wer(raw_wer, failures, n_samples=28)
        # v1.1 penalty = 0.15 / 28 ≈ 0.00536
        assert adj > raw_wer
        assert abs(adj - (raw_wer + 0.15 / 28)) < 1e-9

    def test_safety_adjusted_wer_with_no_failures(self) -> None:
        """Contract: no failures → safety-adjusted WER equals raw WER."""
        adj = _compute_safety_adjusted_wer(0.15, [], n_samples=28)
        assert adj == pytest.approx(0.15)

    def test_safety_adjusted_wer_zero_samples_returns_raw(self) -> None:
        """Contract: n_samples=0 returns raw_wer to avoid ZeroDivisionError."""
        adj = _compute_safety_adjusted_wer(0.10, [], n_samples=0)
        assert adj == pytest.approx(0.10)

    def test_voice_result_counts(self) -> None:
        """Contract: VoiceResult.counts() returns correct per-severity counts."""
        vr = VoiceResult(voice="en_GB-alan-medium", n_samples=28)
        vr.failures.append(Failure("a", "DECIMAL_DROP", SEVERITY_CRITICAL, Correctability.UNRECOVERABLE, "d", "50.4", "50"))
        vr.failures.append(
            Failure("b", "DOSE_UNIT_CORRUPTION", SEVERITY_HIGH, Correctability.PHONETIC_MAP, "d", None, "79.2 GI")
        )
        vr.failures.append(
            Failure("c", "SLASHED_FORM_LOSS", SEVERITY_MEDIUM, Correctability.ADJACENCY_RULE, "d", "3D/3D", None)
        )
        counts = vr.counts()
        assert counts[SEVERITY_CRITICAL] == 1
        assert counts[SEVERITY_HIGH] == 1
        assert counts[SEVERITY_MEDIUM] == 1


# ===========================================================================
# Particle-therapy context detection
# ===========================================================================


class TestParticleTherapyContextDetection:
    """Tests for _has_particle_therapy_context helper."""

    def test_detects_proton(self) -> None:
        assert _has_particle_therapy_context("Proton beam scanning.") is True

    def test_detects_pencil_beam_scanning(self) -> None:
        assert _has_particle_therapy_context("using pencil beam scanning") is True

    def test_detects_craniospinal(self) -> None:
        assert _has_particle_therapy_context("craniospinal irradiation") is True

    def test_detects_rbe(self) -> None:
        assert _has_particle_therapy_context("RBE of 1.1 applied") is True

    def test_no_detection_in_imrt_context(self) -> None:
        """Contract: IMRT context without any particle clue → False."""
        text = "The patient proceeded with IMRT to 79.2 Gy in 44 fractions to the prostate."
        assert _has_particle_therapy_context(text) is False

    def test_case_insensitive(self) -> None:
        assert _has_particle_therapy_context("PROTON therapy prescribed.") is True


# ===========================================================================
# Integration: run against both cycle 112 bake-off JSON files
# ===========================================================================


@pytest.mark.skipif(not _DRAFT_JSON.exists(), reason="bakeoff_proton_draft JSON not found")
class TestIntegrationDraftJson:
    """Integration tests against bakeoff_proton_draft_2026-04-09.json (Whisper + MedASR)."""

    @pytest.fixture(scope="class")
    def draft_report(self) -> dict[str, object]:
        with _DRAFT_JSON.open() as fh:
            data: dict[str, object] = json.load(fh)
        data["source_report"] = _DRAFT_JSON.name
        return data

    @pytest.fixture(scope="class")
    def gate_report(self, draft_report: dict[str, object]) -> object:
        return evaluate_report(draft_report)

    def test_whisper_fails_gate(self, gate_report: object) -> None:
        """Acceptance criterion 2: Whisper FAIL with DOSE_UNIT_CORRUPTION dominant."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        whisper = gate_report.backend_results.get("mlx_whisper")
        assert whisper is not None, "mlx_whisper backend not found in report"
        assert whisper.overall_gate == GATE_FAIL, f"Expected FAIL, got {whisper.overall_gate}"
        # DOSE_UNIT_CORRUPTION (HIGH) should be dominant — ensure HIGH > 0
        assert whisper.overall_counts.get(SEVERITY_HIGH, 0) > 0, "Expected HIGH failures for Whisper"

    def test_whisper_dose_unit_corruption_dominates(self, gate_report: object) -> None:
        """Acceptance criterion 2: Whisper dominant failure class is DOSE_UNIT_CORRUPTION."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        whisper = gate_report.backend_results["mlx_whisper"]
        all_failures = [f for vr in whisper.voice_results for f in vr.failures]
        duc_count = sum(1 for f in all_failures if f.class_ == "DOSE_UNIT_CORRUPTION")
        other_high = sum(
            1 for f in all_failures if f.severity in (SEVERITY_HIGH, SEVERITY_CRITICAL) and f.class_ != "DOSE_UNIT_CORRUPTION"
        )
        assert duc_count > 0, "Expected DOSE_UNIT_CORRUPTION failures for Whisper"
        assert duc_count >= other_high, "DOSE_UNIT_CORRUPTION should be the dominant HIGH class for Whisper"

    def test_medasr_fails_gate(self, gate_report: object) -> None:
        """Acceptance criterion 2: MedASR FAIL with CRITICAL dominant."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        medasr = gate_report.backend_results.get("medasr")
        assert medasr is not None, "medasr backend not found in report"
        assert medasr.overall_gate == GATE_FAIL, f"Expected FAIL, got {medasr.overall_gate}"

    def test_medasr_critical_failures_dominant(self, gate_report: object) -> None:
        """Acceptance criterion 2: MedASR has DECIMAL_DROP + DOSE_VALUE_MISSING as dominant."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        medasr = gate_report.backend_results["medasr"]
        assert medasr.overall_counts.get(SEVERITY_CRITICAL, 0) > 0, (
            "Expected CRITICAL failures (DECIMAL_DROP + DOSE_VALUE_MISSING) for MedASR"
        )
        # Findings: MedASR had 7 decimal drops + ~5 dose-value-missing
        # We require at least 5 CRITICAL failures total
        assert medasr.overall_counts[SEVERITY_CRITICAL] >= 5, (
            f"Expected ≥5 CRITICAL failures for MedASR, got {medasr.overall_counts[SEVERITY_CRITICAL]}"
        )

    def test_draft_json_no_false_positive_on_proton_0015(self, gate_report: object) -> None:
        """Acceptance criterion 3: proton-0015 (IMRT, 79.2 Gy) produces no SILENT_UNIT_SUBSTITUTION."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        for br in gate_report.backend_results.values():
            for vr in br.voice_results:
                sus_on_0015 = [
                    f for f in vr.failures if f.fixture_id == "proton-0015" and f.class_ == "SILENT_UNIT_SUBSTITUTION"
                ]
                assert sus_on_0015 == [], (
                    f"False positive SILENT_UNIT_SUBSTITUTION on proton-0015 for backend={br.backend} voice={vr.voice}"
                )

    def test_report_is_serialisable(self, gate_report: object) -> None:
        """Contract: report_to_dict produces JSON-serialisable output."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        d = report_to_dict(gate_report)
        json_str = json.dumps(d)
        assert len(json_str) > 0


@pytest.mark.skipif(not _VOXTRAL_JSON.exists(), reason="bakeoff_proton_voxtral JSON not found")
class TestIntegrationVoxtralJson:
    """Integration tests against bakeoff_proton_voxtral_2026-04-09.json (Voxtral)."""

    @pytest.fixture(scope="class")
    def voxtral_report(self) -> dict[str, object]:
        with _VOXTRAL_JSON.open() as fh:
            data: dict[str, object] = json.load(fh)
        data["source_report"] = _VOXTRAL_JSON.name
        return data

    @pytest.fixture(scope="class")
    def gate_report(self, voxtral_report: dict[str, object]) -> object:
        return evaluate_report(voxtral_report)

    def test_voxtral_fails_gate(self, gate_report: object) -> None:
        """Acceptance criterion 2: Voxtral FAIL with SILENT_UNIT_SUBSTITUTION dominant."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        voxtral = gate_report.backend_results.get("voxtral")
        assert voxtral is not None, "voxtral backend not found in report"
        assert voxtral.overall_gate == GATE_FAIL, f"Expected FAIL, got {voxtral.overall_gate}"

    def test_voxtral_silent_substitution_is_dominant(self, gate_report: object) -> None:
        """Acceptance criterion 2: Voxtral has many SILENT_UNIT_SUBSTITUTION (HIGH), expect ≥20."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        voxtral = gate_report.backend_results["voxtral"]
        all_failures = [f for vr in voxtral.voice_results for f in vr.failures]
        sus_count = sum(1 for f in all_failures if f.class_ == "SILENT_UNIT_SUBSTITUTION")
        assert sus_count >= 20, f"Expected ≥20 SILENT_UNIT_SUBSTITUTION for Voxtral, got {sus_count}"

    def test_voxtral_no_false_positive_on_proton_0015(self, gate_report: object) -> None:
        """Acceptance criterion 3: proton-0015 (IMRT, 79.2 Gy) — no SILENT_UNIT_SUBSTITUTION.

        Gold is '... proceeded with IMRT to 79.2 Gy in 44 fractions ...'.
        The word 'protons' appears but gold uses Gy not GyE — detector step 1 fails.
        """
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        voxtral = gate_report.backend_results["voxtral"]
        for vr in voxtral.voice_results:
            sus_on_0015 = [f for f in vr.failures if f.fixture_id == "proton-0015" and f.class_ == "SILENT_UNIT_SUBSTITUTION"]
            assert sus_on_0015 == [], f"False positive SILENT_UNIT_SUBSTITUTION on proton-0015 voice={vr.voice}"

    def test_voxtral_report_serialisable(self, gate_report: object) -> None:
        """Contract: report_to_dict produces JSON-serialisable output for Voxtral."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        assert isinstance(gate_report, SafetyGateReport)
        d = report_to_dict(gate_report)
        json.dumps(d)  # raises on non-serialisable values


# ===========================================================================
# CLI integration test
# ===========================================================================


@pytest.mark.skipif(not _VOXTRAL_JSON.exists(), reason="bakeoff_proton_voxtral JSON not found")
class TestCLI:
    """Tests for the __main__ CLI path."""

    def test_cli_writes_output_file(self, tmp_path: Path) -> None:
        """Contract: CLI writes <report>.safety_gate.json next to the input file."""
        import shutil

        # Copy the voxtral JSON to a temp directory so we don't pollute the repo
        src = _VOXTRAL_JSON
        dst = tmp_path / src.name
        shutil.copy(src, dst)

        exit_code = _run_cli([str(dst)])
        assert exit_code == 0

        output = dst.with_suffix(dst.suffix + ".safety_gate.json")
        assert output.exists(), f"Expected output file at {output}"

        with output.open() as fh:
            result = json.load(fh)
        assert result["metric_version"] == "1.1"
        assert "backends" in result

    def test_cli_missing_file_returns_error(self) -> None:
        """Contract: CLI exits with non-zero when file does not exist."""
        exit_code = _run_cli(["/nonexistent/path/report.json"])
        assert exit_code != 0

    def test_cli_no_args_returns_error(self) -> None:
        """Contract: CLI with no arguments exits with non-zero."""
        exit_code = _run_cli([])
        assert exit_code != 0


# ===========================================================================
# evaluate_sample public API
# ===========================================================================


class TestEvaluateSampleAPI:
    """Tests for the evaluate_sample() public entry point."""

    def test_clean_sample_returns_no_failures(self) -> None:
        """Contract: a sample with no safety issues returns empty list."""
        sample = _make_sample(
            "clean-001",
            "Prescribed dose of 60 Gy in 30 fractions to the prostate.",
            "Prescribed dose of 60 Gy in 30 fractions to the prostate.",
        )
        assert evaluate_sample(sample) == []

    def test_sample_with_all_failure_types(self) -> None:
        """Contract: a pathological sample triggers multiple failure classes.

        Note: when a decimal value is dropped (50.4 → 50) AND the unit is silently
        substituted (GyE → Gy), the SILENT_UNIT_SUBSTITUTION detector does NOT fire
        because it looks for the exact number from gold ("50.4 Gy") in the pred.
        The pred has "50 Gy" — so only DECIMAL_DROP fires for that expression.
        To get both classes in one sample, use separate dose expressions.
        """
        # gold: 3D/3D slashed form + decimal drop (50.4→50) + integer silent sub (60 GyE→60 Gy)
        gold = "3D/3D CBCT verification. Proton dose 50.4 GyE and 60 GyE in 28 fractions."
        pred = "3D 3D CBCT verification. Proton dose 50 GyE and 60 Gy in 28 fractions."
        sample = _make_sample("bad-001", gold, pred)
        failures = evaluate_sample(sample)
        classes = {f.class_ for f in failures}
        # Should have: DECIMAL_DROP (50.4 → 50 in pred), SILENT_UNIT_SUBSTITUTION (60 GyE→60 Gy),
        # SLASHED_FORM_LOSS (3D/3D absent)
        assert "DECIMAL_DROP" in classes
        assert "SILENT_UNIT_SUBSTITUTION" in classes
        assert "SLASHED_FORM_LOSS" in classes

    def test_sample_without_fixture_id_handled_gracefully(self) -> None:
        """Contract: missing fixture_id key does not raise — defaults to 'unknown'."""
        sample: dict[str, object] = {
            "ground_truth": "Prescribed 60 Gy in 30 fractions.",
            "raw_transcription": "Prescribed 60 Gy in 30 fractions.",
        }
        failures = evaluate_sample(sample)
        assert isinstance(failures, list)


# ===========================================================================
# v1.1 tests: Correctability tags per failure class (A11)
# ===========================================================================


class TestCorrectabilityTags:
    """Tests for Correctability enum assignment per failure class (v1.1 change 1)."""

    def test_decimal_drop_is_unrecoverable(self) -> None:
        """Contract: DECIMAL_DROP failure has UNRECOVERABLE correctability."""
        gold = "Prescribed dose of 50.4 GyE in 28 fractions to the rectal PTV."
        pred = "Prescribed dose of 50 GyE in 28 fractions to the rectal PTV."
        failures = _detect_decimal_drop("test-c001", gold, pred)
        assert len(failures) == 1
        assert failures[0].correctability == Correctability.UNRECOVERABLE

    def test_dose_value_missing_is_unrecoverable(self) -> None:
        """Contract: DOSE_VALUE_MISSING failure has UNRECOVERABLE correctability."""
        gold = "Prescribed 54 GyE in 30 fractions via RBE-weighted proton dose."
        pred = "Prescribed GyE in 30 fractions via RBE-weighted proton dose."
        failures = _detect_dose_value_missing("test-c002", gold, pred)
        assert len(failures) >= 1
        assert all(f.correctability == Correctability.UNRECOVERABLE for f in failures)

    def test_silent_unit_substitution_is_context_rule(self) -> None:
        """Contract: SILENT_UNIT_SUBSTITUTION failure has CONTEXT_RULE correctability."""
        gold = "Prescribed dose of 79.2 GyE in 44 fractions using pencil beam scanning."
        pred = "Prescribed dose of 79.2 Gy in 44 fractions using pencil beam scanning."
        failures = _detect_silent_unit_substitution("test-c003", gold, pred)
        assert len(failures) == 1
        assert failures[0].correctability == Correctability.CONTEXT_RULE

    def test_dose_unit_corruption_is_phonetic_map(self) -> None:
        """Contract: DOSE_UNIT_CORRUPTION failure has PHONETIC_MAP correctability."""
        gold = "Prescribed 54 GyE in 30 fractions."
        pred = "Prescribed 54 GiE in 30 fractions."
        failures = _detect_dose_unit_corruption("test-c004", gold, pred)
        assert len(failures) >= 1
        assert all(f.correctability == Correctability.PHONETIC_MAP for f in failures)

    def test_slashed_form_loss_is_adjacency_rule(self) -> None:
        """Contract: SLASHED_FORM_LOSS failure has ADJACENCY_RULE correctability."""
        gold = "Image guidance using 3D/3D CBCT matching prior to each fraction."
        pred = "Image guidance using 3D 3D CBCT matching prior to each fraction."
        failures = _detect_slashed_form_loss("test-c005", gold, pred)
        assert len(failures) == 1
        assert failures[0].correctability == Correctability.ADJACENCY_RULE


# ===========================================================================
# v1.1 tests: post_correction_gate logic (A12)
# ===========================================================================


class TestPostCorrectionGate:
    """Tests for the post_correction_gate function (v1.1 change 3)."""

    def test_pass_when_all_failures_are_correctable(self) -> None:
        """Contract: post_correction_gate = PASS when all failures are correctable (no UNRECOVERABLE)."""
        failures = [
            Failure(
                "a",
                "SILENT_UNIT_SUBSTITUTION",
                SEVERITY_HIGH,
                Correctability.CONTEXT_RULE,
                "d",
                "79.2 GyE",
                "79.2 Gy",
            ),
            Failure(
                "b",
                "DOSE_UNIT_CORRUPTION",
                SEVERITY_HIGH,
                Correctability.PHONETIC_MAP,
                "d",
                None,
                "54 GiE",
            ),
            Failure(
                "c",
                "SLASHED_FORM_LOSS",
                SEVERITY_MEDIUM,
                Correctability.ADJACENCY_RULE,
                "d",
                "3D/3D",
                "3D 3D",
            ),
        ]
        # raw_gate should FAIL (has HIGH), but post_correction_gate should PASS (no UNRECOVERABLE)
        counts = {"CRITICAL": 0, "HIGH": 2, "MEDIUM": 1}
        assert _gate_decision(counts) == GATE_FAIL
        assert post_correction_gate(failures) == GATE_PASS

    def test_fail_when_unrecoverable_critical_exists(self) -> None:
        """Contract: post_correction_gate = FAIL when at least one UNRECOVERABLE CRITICAL failure exists."""
        failures = [
            Failure(
                "a",
                "DECIMAL_DROP",
                SEVERITY_CRITICAL,
                Correctability.UNRECOVERABLE,
                "d",
                "50.4",
                "50",
            ),
            Failure(
                "b",
                "SILENT_UNIT_SUBSTITUTION",
                SEVERITY_HIGH,
                Correctability.CONTEXT_RULE,
                "d",
                "60 GyE",
                "60 Gy",
            ),
        ]
        assert post_correction_gate(failures) == GATE_FAIL

    def test_pass_when_empty_failures(self) -> None:
        """Contract: post_correction_gate = PASS when no failures at all."""
        assert post_correction_gate([]) == GATE_PASS

    def test_conditional_when_only_unrecoverable_medium_remains(self) -> None:
        """Contract: CONDITIONAL if only UNRECOVERABLE MEDIUM failures remain.

        Note: per the current correctability mapping, no failure class is both
        MEDIUM severity AND UNRECOVERABLE (MEDIUM=SLASHED_FORM_LOSS=ADJACENCY_RULE).
        This test uses an artificial failure to verify the gate logic is correct.
        """
        failures = [
            Failure(
                "a",
                "HYPOTHETICAL_MEDIUM",
                SEVERITY_MEDIUM,
                Correctability.UNRECOVERABLE,
                "d",
                None,
                None,
            ),
        ]
        assert post_correction_gate(failures) == GATE_CONDITIONAL

    def test_raw_gate_fails_but_post_gate_passes_in_voxtral_like_scenario(self) -> None:
        """Contract: Voxtral-like scenario — many HIGH correctable failures pass post-gate."""
        # Voxtral's dominant class is SILENT_UNIT_SUBSTITUTION (CONTEXT_RULE)
        failures = [
            Failure(
                f"proton-{i:04d}", "SILENT_UNIT_SUBSTITUTION", SEVERITY_HIGH, Correctability.CONTEXT_RULE, "d", "X GyE", "X Gy"
            )
            for i in range(47)
        ]
        raw_counts = {"CRITICAL": 0, "HIGH": 47, "MEDIUM": 0}
        assert _gate_decision(raw_counts) == GATE_FAIL  # raw gate fails
        assert post_correction_gate(failures) == GATE_PASS  # post-correction passes


# ===========================================================================
# v1.1 integration tests: cycle 112 unrecoverable CRITICAL counts (A10)
# ===========================================================================


@pytest.mark.skipif(
    not (_DRAFT_JSON.exists() and _VOXTRAL_JSON.exists()),
    reason="cycle 112 bake-off JSONs not found",
)
class TestIntegrationCycle112V11:
    """Integration tests verifying v1.1 gate results and unrecoverable CRITICAL counts."""

    @pytest.fixture(scope="class")
    def all_gate_reports(self) -> dict[str, object]:
        """Load and evaluate both cycle 112 JSONs, return mapping backend->BackendResult."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        results: dict[str, object] = {}
        for json_path in [_DRAFT_JSON, _VOXTRAL_JSON]:
            with json_path.open() as fh:
                data: dict[str, object] = json.load(fh)
            data["source_report"] = json_path.name
            report = evaluate_report(data)
            assert isinstance(report, SafetyGateReport)
            for backend_name, br in report.backend_results.items():
                results[backend_name] = br
        return results

    def test_all_backends_raw_gate_fail(self, all_gate_reports: dict[str, object]) -> None:
        """A10: all three backends have raw_gate = FAIL."""
        from tests.validation.metrics.safety_gate import BackendResult

        for backend_name in ("voxtral", "mlx_whisper", "medasr"):
            br = all_gate_reports.get(backend_name)
            assert br is not None, f"{backend_name} not found in reports"
            assert isinstance(br, BackendResult)
            assert br.overall_raw_gate == GATE_FAIL, f"{backend_name}: expected raw_gate=FAIL, got {br.overall_raw_gate}"

    def test_all_backends_post_correction_gate_fail(self, all_gate_reports: dict[str, object]) -> None:
        """A10: all three backends have post_correction_gate = FAIL."""
        from tests.validation.metrics.safety_gate import BackendResult

        for backend_name in ("voxtral", "mlx_whisper", "medasr"):
            br = all_gate_reports.get(backend_name)
            assert br is not None, f"{backend_name} not found in reports"
            assert isinstance(br, BackendResult)
            assert br.overall_post_correction_gate == GATE_FAIL, (
                f"{backend_name}: expected post_correction_gate=FAIL, got {br.overall_post_correction_gate}"
            )

    def test_voxtral_unrecoverable_critical_count(self, all_gate_reports: dict[str, object]) -> None:
        """A10: Voxtral has exactly 2 unrecoverable CRITICAL failures."""
        from tests.validation.metrics.safety_gate import BackendResult

        br = all_gate_reports["voxtral"]
        assert isinstance(br, BackendResult)
        unrec_crit = br.overall_residual_unrecoverable.get(SEVERITY_CRITICAL, 0)
        assert unrec_crit == 2, f"Voxtral: expected 2 unrecoverable CRITICAL, got {unrec_crit}"

    def test_whisper_unrecoverable_critical_count(self, all_gate_reports: dict[str, object]) -> None:
        """A10: Whisper (mlx_whisper) has exactly 3 unrecoverable CRITICAL failures."""
        from tests.validation.metrics.safety_gate import BackendResult

        br = all_gate_reports["mlx_whisper"]
        assert isinstance(br, BackendResult)
        unrec_crit = br.overall_residual_unrecoverable.get(SEVERITY_CRITICAL, 0)
        assert unrec_crit == 3, f"Whisper: expected 3 unrecoverable CRITICAL, got {unrec_crit}"

    def test_medasr_unrecoverable_critical_count(self, all_gate_reports: dict[str, object]) -> None:
        """A10: MedASR has exactly 13 unrecoverable CRITICAL failures."""
        from tests.validation.metrics.safety_gate import BackendResult

        br = all_gate_reports["medasr"]
        assert isinstance(br, BackendResult)
        unrec_crit = br.overall_residual_unrecoverable.get(SEVERITY_CRITICAL, 0)
        assert unrec_crit == 13, f"MedASR: expected 13 unrecoverable CRITICAL, got {unrec_crit}"

    def test_safety_adjusted_wer_ranking_preserved(self, all_gate_reports: dict[str, object]) -> None:
        """A10: safety-adjusted WER ranking preserved (Voxtral < Whisper < MedASR)."""
        from tests.validation.metrics.safety_gate import BackendResult

        voxtral = all_gate_reports["voxtral"]
        whisper = all_gate_reports["mlx_whisper"]
        medasr = all_gate_reports["medasr"]
        assert isinstance(voxtral, BackendResult)
        assert isinstance(whisper, BackendResult)
        assert isinstance(medasr, BackendResult)
        assert voxtral.overall_safety_adjusted_wer < whisper.overall_safety_adjusted_wer, (
            f"Expected Voxtral ({voxtral.overall_safety_adjusted_wer:.4f}) < "
            f"Whisper ({whisper.overall_safety_adjusted_wer:.4f})"
        )
        assert whisper.overall_safety_adjusted_wer < medasr.overall_safety_adjusted_wer, (
            f"Expected Whisper ({whisper.overall_safety_adjusted_wer:.4f}) < MedASR ({medasr.overall_safety_adjusted_wer:.4f})"
        )

    def test_output_json_has_v11_fields(self) -> None:
        """A8: output dict includes counts_by_correctability, residual_unrecoverable, raw_gate, post_correction_gate."""
        from tests.validation.metrics.safety_gate import SafetyGateReport

        # Re-run the voxtral report to get a full SafetyGateReport
        with _VOXTRAL_JSON.open() as fh:
            data: dict[str, object] = json.load(fh)
        data["source_report"] = _VOXTRAL_JSON.name
        gate_report = evaluate_report(data)
        assert isinstance(gate_report, SafetyGateReport)
        d = report_to_dict(gate_report)
        backends = d.get("backends", {})
        assert isinstance(backends, dict)
        voxtral_dict = backends.get("voxtral", {})
        assert isinstance(voxtral_dict, dict)
        overall = voxtral_dict.get("overall", {})
        assert isinstance(overall, dict)
        # Backend-level v1.1 fields
        assert "counts_by_correctability" in overall, "overall missing counts_by_correctability"
        assert "residual_unrecoverable" in overall, "overall missing residual_unrecoverable"
        assert "raw_gate" in overall, "overall missing raw_gate"
        assert "post_correction_gate" in overall, "overall missing post_correction_gate"
        # v1.0 backwards compat
        assert "gate" in overall, "overall missing gate (v1.0 alias)"
        assert overall["gate"] == overall["raw_gate"], "gate must equal raw_gate (backwards compat)"
        # Check voice level too
        voices = voxtral_dict.get("voices", {})
        assert isinstance(voices, dict)
        for voice_name, vdata in voices.items():
            assert isinstance(vdata, dict), f"voice {voice_name} is not a dict"
            assert "counts_by_correctability" in vdata, f"voice {voice_name} missing counts_by_correctability"
            assert "residual_unrecoverable" in vdata, f"voice {voice_name} missing residual_unrecoverable"
            assert "raw_gate" in vdata, f"voice {voice_name} missing raw_gate"
            assert "post_correction_gate" in vdata, f"voice {voice_name} missing post_correction_gate"
            assert "gate" in vdata, f"voice {voice_name} missing gate (v1.0 alias)"
            assert vdata["gate"] == vdata["raw_gate"], f"voice {voice_name}: gate must equal raw_gate"

    def test_correctability_counts_in_output_json_are_positive_for_voxtral(self, all_gate_reports: dict[str, object]) -> None:
        """Contract: Voxtral should have non-zero CONTEXT_RULE count (dominant class is SILENT_UNIT_SUBSTITUTION)."""
        from tests.validation.metrics.safety_gate import BackendResult

        br = all_gate_reports["voxtral"]
        assert isinstance(br, BackendResult)
        context_rule_count = br.overall_counts_by_correctability.get("CONTEXT_RULE", 0)
        assert context_rule_count >= 20, f"Voxtral: expected ≥20 CONTEXT_RULE failures, got {context_rule_count}"
