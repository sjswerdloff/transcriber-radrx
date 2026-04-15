"""Compute ensemble UWR from bake-off JSON reports.

Reads paired Whisper + Voxtral transcriptions from bake-off JSONs,
runs the ensemble, and reports UWR (Unresolved Word Rate) — the
fraction of words the ensemble cannot resolve automatically.

Supports three correction modes:
  raw     — no correction, raw ASR output only
  phrase  — phrase-level regex corrections (PhraseCorrectorPipeline)
  full    — phrase + single-word corrections (correct_full)

Usage::

    # Compare all three modes on all reports in a directory
    uv run python tests/validation/scripts/compute_ensemble_uwr.py \
        --reports tests/validation/reports/bakeoff_esl_*.json \
                  tests/validation/reports/bakeoff_commonwealth_*.json

    # Single report, specific mode
    uv run python tests/validation/scripts/compute_ensemble_uwr.py \
        --reports tests/validation/reports/bakeoff_esl_dense_rt.json \
        --mode full

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transcriber_radrx.corrector import CorrectionDictionary  # noqa: E402
from transcriber_radrx.ensemble import ensemble_transcriptions  # noqa: E402
from transcriber_radrx.phrase_corrector import PhraseCorrectorPipeline  # noqa: E402


def _load_vocab(path: Path) -> set[str]:
    terms: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                terms.add(s.lower())
    return terms


def _extract_backends(report: dict[str, object]) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Index a bake-off report by backend → voice → fixture_id → sample."""
    backends: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for be in report.get("results", []):
        if not isinstance(be, dict):
            continue
        bname = str(be.get("backend", ""))
        voice_map: dict[str, dict[str, dict[str, object]]] = {}
        for vr in be.get("by_voice", []):
            if not isinstance(vr, dict):
                continue
            voice = str(vr.get("voice", ""))
            fmap: dict[str, dict[str, object]] = {}
            for s in vr.get("samples", []):
                if isinstance(s, dict):
                    fmap[str(s.get("fixture_id", ""))] = s
            voice_map[voice] = fmap
        backends[bname] = voice_map
    return backends


def compute_uwr(
    report_path: Path,
    vocabulary: set[str],
    *,
    mode: str = "raw",
    corrector: CorrectionDictionary | None = None,
    phrase_pipeline: PhraseCorrectorPipeline | None = None,
) -> tuple[int, int, int, float]:
    """Compute UWR for a single bake-off report.

    Args:
        report_path: Path to bake-off JSON.
        vocabulary: Set of lowercase RT vocabulary terms.
        mode: "raw", "phrase", or "full".
        corrector: CorrectionDictionary instance (needed for "full" mode).
        phrase_pipeline: PhraseCorrectorPipeline instance (needed for "phrase"/"full").

    Returns:
        Tuple of (n_pairs, total_words, review_words, uwr_percent).
    """
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    backends = _extract_backends(report)
    vox = backends.get("voxtral", {})
    whi = backends.get("mlx_whisper", {})
    common_voices = sorted(set(vox) & set(whi))

    total_words = 0
    review_words = 0
    n_pairs = 0

    for voice in common_voices:
        common_fids = sorted(set(vox[voice]) & set(whi[voice]))
        for fid in common_fids:
            vox_raw = str(vox[voice][fid].get("raw_transcription", ""))
            whi_raw = str(whi[voice][fid].get("raw_transcription", ""))

            if mode == "phrase" and phrase_pipeline is not None:
                vox_text, _ = phrase_pipeline.correct(vox_raw)
                whi_text, _ = phrase_pipeline.correct(whi_raw)
            elif mode == "full" and corrector is not None:
                vox_text, _, _ = corrector.correct_full(vox_raw)
                whi_text, _, _ = corrector.correct_full(whi_raw)
            else:
                vox_text = vox_raw
                whi_text = whi_raw

            result = ensemble_transcriptions(
                text_voxtral=vox_text,
                text_whisper=whi_text,
                vocabulary=vocabulary,
                fixture_id=fid,
                voice=voice,
            )
            total_words += len(result.words)
            review_words += result.review_count
            n_pairs += 1

    uwr = review_words / total_words * 100 if total_words > 0 else 0.0
    return n_pairs, total_words, review_words, uwr


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reports",
        nargs="+",
        type=Path,
        required=True,
        help="Bake-off JSON report files to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "phrase", "full", "all"],
        default="all",
        help="Correction mode. 'all' runs all three and prints a comparison table.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=_REPO_ROOT / "data" / "rt_vocabulary.txt",
        help="RT vocabulary file.",
    )
    args = parser.parse_args(argv)

    vocab = _load_vocab(args.vocabulary)

    # Lazy-init correctors only when needed
    corrector: CorrectionDictionary | None = None
    phrase_pipeline: PhraseCorrectorPipeline | None = None

    modes = ["raw", "phrase", "full"] if args.mode == "all" else [args.mode]

    if "phrase" in modes or "full" in modes:
        phrase_pipeline = PhraseCorrectorPipeline()
    if "full" in modes:
        corrector = CorrectionDictionary(args.vocabulary, enable_phonetic=True)

    # Header
    if len(modes) > 1:
        mode_headers = "  ".join(f"{m:>8s}" for m in modes)
        print(f"{'Report':<40s}  {mode_headers}")
        print("-" * (42 + 10 * len(modes)))
    else:
        print(f"{'Report':<40s}  {'UWR':>8s}  {'pairs':>6s}  {'words':>7s}  {'review':>7s}")
        print("-" * 75)

    for report_path in sorted(args.reports):
        label = report_path.stem
        if len(modes) > 1:
            values = []
            for mode in modes:
                _, _, _, uwr = compute_uwr(
                    report_path,
                    vocab,
                    mode=mode,
                    corrector=corrector,
                    phrase_pipeline=phrase_pipeline,
                )
                values.append(f"{uwr:7.2f}%")
            print(f"{label:<40s}  {'  '.join(values)}")
        else:
            n_pairs, total_words, review_words, uwr = compute_uwr(
                report_path,
                vocab,
                mode=modes[0],
                corrector=corrector,
                phrase_pipeline=phrase_pipeline,
            )
            print(f"{label:<40s}  {uwr:7.2f}%  {n_pairs:>6d}  {total_words:>7d}  {review_words:>7d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
