"""Ensemble aggregator script for 2-backend (Voxtral + Whisper) particle-therapy corpus.

Loads the two bake-off JSONs, runs ``ensemble_transcriptions`` for each
fixture × voice pair, computes per-sample and aggregate WER, emits an
ensemble bake-off JSON in the same schema as the input reports, runs the
safety-gate metric, and prints a console comparison table.

Usage::

    uv run python tests/validation/scripts/run_ensemble_aggregator.py

Output:
    tests/validation/reports/bakeoff_particle_ensemble_2026-04-09.json
    (+ .safety_gate.json written by safety_gate.evaluate_report)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import jiwer
from loguru import logger

# ---------------------------------------------------------------------------
# Path bootstrap: ensure repo root is on sys.path so both 'src' package and
# 'tests' package (for safety_gate) are importable when the script is run
# directly via: uv run python tests/validation/scripts/run_ensemble_aggregator.py
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT_CANDIDATE = _SCRIPT_DIR.parents[2]  # tests/validation/scripts → repo root
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPO_ROOT = _REPO_ROOT_CANDIDATE
_REPORTS_DIR = _REPO_ROOT / "tests" / "validation" / "reports"
_VOXTRAL_JSON = _REPORTS_DIR / "bakeoff_particle_voxtral_2026-04-09.json"
_WHISPER_JSON = _REPORTS_DIR / "bakeoff_particle_whisper_medasr_2026-04-09.json"
_ENSEMBLE_JSON = _REPORTS_DIR / "bakeoff_particle_ensemble_2026-04-09.json"
_VOCAB_FILE = _REPO_ROOT / "data" / "rt_vocabulary.txt"


# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------


def _load_vocabulary(path: Path) -> set[str]:
    """Load RT vocabulary from a text file.

    Lines beginning with '#' and empty lines are ignored.
    Returns a set of lowercase terms.

    Args:
        path: Path to the vocabulary text file.

    Returns:
        Set of lowercase vocabulary terms.
    """
    terms: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                terms.add(stripped.lower())
    return terms


# ---------------------------------------------------------------------------
# Report-parsing helpers
# ---------------------------------------------------------------------------


def _extract_samples_by_fixture(report: dict[str, object], backend_name: str) -> dict[str, dict[str, dict[str, object]]]:
    """Extract samples indexed by voice → fixture_id from a bake-off report.

    Args:
        report: Parsed bake-off JSON.
        backend_name: The backend name to extract (e.g. 'voxtral', 'mlx_whisper').

    Returns:
        Nested dict: voice_name → fixture_id → sample dict.
    """
    index: dict[str, dict[str, dict[str, object]]] = {}
    results = report.get("results", [])
    if not isinstance(results, list):
        return index
    for backend_entry in results:
        if not isinstance(backend_entry, dict):
            continue
        if backend_entry.get("backend") != backend_name:
            continue
        by_voice = backend_entry.get("by_voice", [])
        if not isinstance(by_voice, list):
            continue
        for voice_entry in by_voice:
            if not isinstance(voice_entry, dict):
                continue
            voice = str(voice_entry.get("voice", ""))
            samples = voice_entry.get("samples", [])
            if not isinstance(samples, list):
                continue
            voice_map: dict[str, dict[str, object]] = {}
            for s in samples:
                if not isinstance(s, dict):
                    continue
                fid = str(s.get("fixture_id", ""))
                voice_map[fid] = s
            index[voice] = voice_map
    return index


# ---------------------------------------------------------------------------
# Ensemble run
# ---------------------------------------------------------------------------


def _run_ensemble(
    voxtral_report: dict[str, object],
    whisper_report: dict[str, object],
    vocabulary: set[str],
) -> dict[str, object]:
    """Run ensemble over all common fixture × voice pairs.

    Args:
        voxtral_report: Parsed Voxtral bake-off JSON.
        whisper_report: Parsed Whisper+MedASR bake-off JSON.
        vocabulary: Set of lowercase RT vocabulary terms.

    Returns:
        Bake-off report dict in the same schema as input reports, backend
        name is 'ensemble_voxtral_whisper'.
    """
    # Lazy import to avoid circular issues if script is imported in tests
    from transcriber_radrx.ensemble import ensemble_transcriptions  # noqa: PLC0415

    vox_by_voice = _extract_samples_by_fixture(voxtral_report, "voxtral")
    whi_by_voice = _extract_samples_by_fixture(whisper_report, "mlx_whisper")

    common_voices = sorted(set(vox_by_voice.keys()) & set(whi_by_voice.keys()))
    logger.info("Common voices: {}", common_voices)

    by_voice_output: list[dict[str, object]] = []
    total_review_words = 0

    for voice in common_voices:
        vox_fixtures = vox_by_voice[voice]
        whi_fixtures = whi_by_voice[voice]
        common_fixtures = sorted(set(vox_fixtures.keys()) & set(whi_fixtures.keys()))
        logger.info("voice={} common_fixtures={}", voice, len(common_fixtures))

        samples_out: list[dict[str, object]] = []
        wer_values: list[float] = []

        for fid in common_fixtures:
            vox_sample = vox_fixtures[fid]
            whi_sample = whi_fixtures[fid]
            text_voxtral = str(vox_sample.get("raw_transcription", ""))
            text_whisper = str(whi_sample.get("raw_transcription", ""))
            ground_truth = str(vox_sample.get("ground_truth", ""))

            result = ensemble_transcriptions(
                text_voxtral=text_voxtral,
                text_whisper=text_whisper,
                vocabulary=vocabulary,
                fixture_id=fid,
                voice=voice,
            )
            total_review_words += result.review_count

            # Compute WER for this sample
            try:
                raw_wer = jiwer.wer(ground_truth, result.text_ensemble) if ground_truth else 0.0
            except Exception:
                logger.exception("WER computation failed for fixture_id={} voice={}", fid, voice)
                raw_wer = 1.0
            wer_values.append(raw_wer)

            # Build sample dict — same schema as input report samples
            sample_out: dict[str, object] = {
                "fixture_id": fid,
                "category": vox_sample.get("category", ""),
                "dense_subtype": vox_sample.get("dense_subtype", ""),
                "ground_truth": ground_truth,
                "raw_transcription": result.text_ensemble,
                "corrected_transcription": result.text_ensemble,
                "raw_wer": raw_wer,
                "corrected_wer": raw_wer,
                "vocabulary_terms": vox_sample.get("vocabulary_terms", []),
                "terms_found": [],
                "terms_total": vox_sample.get("terms_total", 0),
                "terms_missing": [],
                "error": None,
                # Ensemble provenance extensions
                "ensemble_needs_review": result.needs_review,
                "ensemble_review_count": result.review_count,
                "ensemble_agreement_rate": round(result.agreement_rate, 4),
                "ensemble_voxtral_chosen": result.voxtral_chosen,
                "ensemble_whisper_chosen": result.whisper_chosen,
                "ensemble_context_rule_count": result.context_rule_count,
            }
            samples_out.append(sample_out)

        avg_raw_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
        by_voice_output.append(
            {
                "voice": voice,
                "summary": {
                    "sample_count": len(samples_out),
                    "avg_raw_wer": round(avg_raw_wer, 6),
                },
                "samples": samples_out,
            }
        )

    ensemble_report: dict[str, object] = {
        "vocabulary": voxtral_report.get("vocabulary", ""),
        "sample_count": sum(len(bv["samples"]) for bv in by_voice_output),  # type: ignore[arg-type]
        "dense_only": voxtral_report.get("dense_only", True),
        "seed": voxtral_report.get("seed", 0),
        "system_prompt": "ensemble_voxtral_whisper",
        "noise": voxtral_report.get("noise", False),
        "fixture_ids": voxtral_report.get("fixture_ids", []),
        "voices": common_voices,
        "backends": ["ensemble_voxtral_whisper"],
        "source_report": _ENSEMBLE_JSON.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [
            {
                "backend": "ensemble_voxtral_whisper",
                "by_voice": by_voice_output,
            }
        ],
    }

    return ensemble_report, total_review_words  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Console comparison table
# ---------------------------------------------------------------------------


def _compute_backend_stats(report: dict[str, object], backend_name: str) -> dict[str, object]:
    """Compute aggregate WER and failure counts for one backend from a bake-off JSON.

    Args:
        report: Parsed bake-off JSON.
        backend_name: Backend name to extract.

    Returns:
        Dict with avg_raw_wer, sample_count.
    """
    wer_values: list[float] = []
    results = report.get("results", [])
    if not isinstance(results, list):
        return {"avg_raw_wer": 0.0, "sample_count": 0}
    for backend_entry in results:
        if not isinstance(backend_entry, dict):
            continue
        if backend_entry.get("backend") != backend_name:
            continue
        by_voice = backend_entry.get("by_voice", [])
        if not isinstance(by_voice, list):
            continue
        for voice_entry in by_voice:
            if not isinstance(voice_entry, dict):
                continue
            summary = voice_entry.get("summary", {})
            if isinstance(summary, dict):
                avg = float(summary.get("avg_raw_wer", 0.0))
                n = int(summary.get("sample_count", 0))
                wer_values.extend([avg] * n)
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    return {"avg_raw_wer": avg_wer, "sample_count": len(wer_values)}


def _print_comparison_table(
    voxtral_report: dict[str, object],
    whisper_report: dict[str, object],
    ensemble_report: dict[str, object],
    voxtral_gate: object,
    whisper_gate: object,
    ensemble_gate: object,
    review_word_count: int,
) -> None:
    """Print a console comparison table across all three backends.

    Args:
        voxtral_report: Parsed Voxtral bake-off JSON.
        whisper_report: Parsed Whisper bake-off JSON.
        ensemble_report: Computed ensemble bake-off JSON.
        voxtral_gate: SafetyGateReport for Voxtral.
        whisper_gate: SafetyGateReport for Whisper.
        ensemble_gate: SafetyGateReport for ensemble.
        review_word_count: Total words flagged for human review across all pairs.
    """
    from tests.validation.metrics.safety_gate import SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM  # noqa: PLC0415

    def _gate_stats(gate_report: object, backend: str) -> dict[str, object]:
        """Extract stats from a SafetyGateReport for one backend."""
        br = gate_report.backend_results.get(backend)  # type: ignore[union-attr]
        if br is None:
            return {"crit": 0, "high": 0, "med": 0, "raw_gate": "?", "post_gate": "?", "wer": 0.0}
        c = br.overall_counts
        return {
            "crit": c.get(SEVERITY_CRITICAL, 0),
            "high": c.get(SEVERITY_HIGH, 0),
            "med": c.get(SEVERITY_MEDIUM, 0),
            "raw_gate": br.overall_raw_gate,
            "post_gate": br.overall_post_correction_gate,
            "wer": br.overall_safety_adjusted_wer,
        }

    vox_stats = _gate_stats(voxtral_gate, "voxtral")
    whi_stats = _gate_stats(whisper_gate, "mlx_whisper")
    ens_stats = _gate_stats(ensemble_gate, "ensemble_voxtral_whisper")

    # Compute raw WER from reports
    vox_wer_info = _compute_backend_stats(voxtral_report, "voxtral")
    whi_wer_info = _compute_backend_stats(whisper_report, "mlx_whisper")
    ens_wer_info = _compute_backend_stats(ensemble_report, "ensemble_voxtral_whisper")

    col_hdr = f"{'Backend':<28} {'Raw WER':>8}  {'CRIT':>5} {'HIGH':>5} {'MED':>4}"
    gate_hdr = f"  {'raw_gate':<10} {'post_gate':<10} {'needs_review':>12}"
    header = col_hdr + gate_hdr
    separator = "-" * len(header)
    print("\n=== Ensemble vs Individual Backends ===")
    print(header)
    print(separator)
    for name, wer_info, stats in [
        ("voxtral", vox_wer_info, vox_stats),
        ("mlx_whisper", whi_wer_info, whi_stats),
        ("ensemble_voxtral_whisper", ens_wer_info, ens_stats),
    ]:
        review_col = str(review_word_count) if name == "ensemble_voxtral_whisper" else "n/a"
        print(
            f"{name:<28} {wer_info['avg_raw_wer']:>8.4f}  "
            f"{stats['crit']:>5} {stats['high']:>5} {stats['med']:>4}  "
            f"{stats['raw_gate']:<10} {stats['post_gate']:<10} {review_col:>12}"
        )
    print()
    print(f"Words flagged for human review across all pairs: {review_word_count}")

    # Ensemble improvement assessment
    best_individual_wer = min(vox_wer_info["avg_raw_wer"], whi_wer_info["avg_raw_wer"])
    ensemble_wer = ens_wer_info["avg_raw_wer"]
    if ensemble_wer <= best_individual_wer:
        print(f"✓ Ensemble WER ({ensemble_wer:.4f}) <= best individual WER ({best_individual_wer:.4f})")
    else:
        print(f"✗ Ensemble WER ({ensemble_wer:.4f}) > best individual WER ({best_individual_wer:.4f})")

    vox_failures = (vox_stats["crit"] or 0) + (vox_stats["high"] or 0)
    whi_failures = (whi_stats["crit"] or 0) + (whi_stats["high"] or 0)
    ens_failures = (ens_stats["crit"] or 0) + (ens_stats["high"] or 0)
    best_individual_failures = min(vox_failures, whi_failures)
    if ens_failures <= best_individual_failures:
        print(f"✓ Ensemble CRIT+HIGH ({ens_failures}) <= best individual ({best_individual_failures})")
    else:
        print(f"✗ Ensemble CRIT+HIGH ({ens_failures}) > best individual ({best_individual_failures})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the ensemble aggregator and print comparison table.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Lazy import here so tests importing this module don't need all deps loaded
    from tests.validation.metrics.safety_gate import evaluate_report, report_to_dict  # noqa: PLC0415

    # Verify inputs exist
    for path in (_VOXTRAL_JSON, _WHISPER_JSON, _VOCAB_FILE):
        if not path.exists():
            logger.error("Required file not found: {}", path)
            return 1

    logger.info("Loading Voxtral report: {}", _VOXTRAL_JSON.name)
    with _VOXTRAL_JSON.open(encoding="utf-8") as fh:
        voxtral_report: dict[str, object] = json.load(fh)

    logger.info("Loading Whisper report: {}", _WHISPER_JSON.name)
    with _WHISPER_JSON.open(encoding="utf-8") as fh:
        whisper_report: dict[str, object] = json.load(fh)

    logger.info("Loading vocabulary: {}", _VOCAB_FILE.name)
    vocabulary = _load_vocabulary(_VOCAB_FILE)
    logger.info("Vocabulary terms: {}", len(vocabulary))

    logger.info("Running ensemble aggregation …")
    ensemble_result = _run_ensemble(voxtral_report, whisper_report, vocabulary)
    # _run_ensemble returns a tuple (report, total_review_words)
    if isinstance(ensemble_result, tuple):
        ensemble_report, total_review_words = ensemble_result
    else:
        ensemble_report = ensemble_result  # type: ignore[assignment]
        total_review_words = 0

    # Write ensemble JSON
    with _ENSEMBLE_JSON.open("w", encoding="utf-8") as fh:
        json.dump(ensemble_report, fh, indent=2)
    logger.info("Ensemble report written to: {}", _ENSEMBLE_JSON)

    # Safety-gate on all three reports
    voxtral_report["source_report"] = _VOXTRAL_JSON.name
    whisper_report["source_report"] = _WHISPER_JSON.name
    ensemble_report["source_report"] = _ENSEMBLE_JSON.name

    logger.info("Running safety-gate on Voxtral …")
    voxtral_gate = evaluate_report(voxtral_report)

    logger.info("Running safety-gate on Whisper …")
    whisper_gate = evaluate_report(whisper_report)

    logger.info("Running safety-gate on Ensemble …")
    ensemble_gate = evaluate_report(ensemble_report)

    # Write safety-gate JSON for ensemble
    sg_output_path = _ENSEMBLE_JSON.with_suffix(_ENSEMBLE_JSON.suffix + ".safety_gate.json")
    with sg_output_path.open("w", encoding="utf-8") as fh:
        json.dump(report_to_dict(ensemble_gate), fh, indent=2)
    logger.info("Ensemble safety-gate report written to: {}", sg_output_path)

    # Print comparison table
    _print_comparison_table(
        voxtral_report,
        whisper_report,
        ensemble_report,
        voxtral_gate,
        whisper_gate,
        ensemble_gate,
        total_review_words,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
