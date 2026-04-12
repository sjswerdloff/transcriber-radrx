"""Mine ASR substitution patterns from bake-off reports.

Analyzes bake-off JSON reports to find systematic patterns in how ASR
backends misrecognize RT vocabulary terms. Groups results by L1 language
background and computes accent penalty (ESL miss rate vs Commonwealth
miss rate) to separate accent-driven failures from domain vocabulary
failures.

Usage::

    # Accent penalty analysis (ESL vs CW miss rates)
    uv run python tests/validation/scripts/mine_substitution_patterns.py \
        --esl-reports tests/validation/reports/bakeoff_esl_*.json \
        --cw-reports tests/validation/reports/bakeoff_commonwealth_*.json

    # Just show most-missed terms across all reports
    uv run python tests/validation/scripts/mine_substitution_patterns.py \
        --esl-reports tests/validation/reports/bakeoff_esl_*.json \
        --mode missed-terms

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# L2-Arctic speaker → L1 language mapping (Zhao et al., Interspeech 2018)
L2ARCTIC_L1: dict[str, str] = {
    "ABA": "Arabic",
    "SKA": "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",
    "BWC": "Mandarin",
    "LXC": "Mandarin",
    "NCC": "Mandarin",
    "TXHC": "Mandarin",
    "ASI": "Hindi",
    "RRBI": "Hindi",
    "TNI": "Hindi",
    "SVBI": "Hindi",
    "HJK": "Korean",
    "HKK": "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",
    "ERMS": "Spanish",
    "MBMPS": "Spanish",
    "NJS": "Spanish",
    "EBVS": "Spanish",
    "THV": "Vietnamese",
    "TLV": "Vietnamese",
    "PNV": "Vietnamese",
    "HQTV": "Vietnamese",
}


def _get_l1(voice_name: str) -> str:
    """Infer L1 background from voice name."""
    for code, lang in L2ARCTIC_L1.items():
        if code in voice_name:
            return lang
    if "reza" in voice_name.lower():
        return "Persian"
    if "kusal" in voice_name.lower():
        return "Sinhala"
    return "Native"


def _is_esl(voice_name: str) -> bool:
    return _get_l1(voice_name) != "Native"


def _load_reports(paths: list[Path]) -> list[dict[str, object]]:
    reports = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def _get_term_miss_rates(
    reports: list[dict[str, object]],
    voice_filter: object = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute per-term miss counts and total counts.

    Args:
        reports: List of parsed bake-off JSONs.
        voice_filter: Callable(voice_name) -> bool. None = all voices.

    Returns:
        (term_missed_counts, term_total_counts)
    """
    term_missed: dict[str, int] = defaultdict(int)
    term_total: dict[str, int] = defaultdict(int)

    for report in reports:
        for be in report.get("results", []):
            if not isinstance(be, dict):
                continue
            for vr in be.get("by_voice", []):
                if not isinstance(vr, dict):
                    continue
                voice = str(vr.get("voice", ""))
                if voice_filter is not None and not voice_filter(voice):
                    continue
                for s in vr.get("samples", []):
                    if not isinstance(s, dict):
                        continue
                    vocab = s.get("vocabulary_terms", [])
                    missing = {str(t) for t in s.get("terms_missing", [])}
                    for t in vocab:
                        t = str(t)
                        term_total[t] += 1
                        if t in missing:
                            term_missed[t] += 1
    return dict(term_missed), dict(term_total)


def _find_substitution(ground_truth: str, transcription: str, term: str) -> str | None:
    """Find what ASR produced in place of a missed term.

    Uses word-position anchoring: finds the term's position in ground truth,
    locates the preceding anchor word in the transcription, then extracts the
    words that follow the anchor.
    """
    gt_words = ground_truth.lower().split()
    tr_words = transcription.lower().split()
    term_words = term.lower().split()

    for i in range(len(gt_words) - len(term_words) + 1):
        if gt_words[i : i + len(term_words)] == term_words:
            anchor_before = gt_words[i - 1] if i > 0 else None
            if anchor_before:
                for ti in range(len(tr_words)):
                    if tr_words[ti] == anchor_before:
                        start = ti + 1
                        end = min(len(tr_words), start + len(term_words) + 1)
                        return " ".join(tr_words[start:end])

            # Positional fallback
            ratio = i / max(len(gt_words), 1)
            est = int(ratio * len(tr_words))
            end = min(len(tr_words), est + len(term_words) + 1)
            return " ".join(tr_words[est:end])
    return None


def show_accent_penalty(
    esl_reports: list[dict[str, object]],
    cw_reports: list[dict[str, object]],
    *,
    top_n: int = 25,
    min_esl_count: int = 10,
    min_cw_count: int = 5,
) -> None:
    """Print accent penalty table: terms with biggest ESL-CW gap."""
    all_reports = esl_reports + cw_reports

    esl_missed, esl_total = _get_term_miss_rates(all_reports, _is_esl)
    cw_missed, cw_total = _get_term_miss_rates(all_reports, lambda v: not _is_esl(v))

    gaps: list[tuple[str, float, float, float]] = []
    for term in set(esl_total) & set(cw_total):
        if esl_total[term] >= min_esl_count and cw_total[term] >= min_cw_count:
            esl_rate = esl_missed.get(term, 0) / esl_total[term]
            cw_rate = cw_missed.get(term, 0) / cw_total[term]
            gaps.append((term, esl_rate, cw_rate, esl_rate - cw_rate))

    gaps.sort(key=lambda x: -x[3])

    print("=== ACCENT PENALTY: terms with biggest ESL-CW gap ===")
    print(f"{'Term':30s}  {'ESL miss%':>10s}  {'CW miss%':>10s}  {'Gap':>8s}")
    print("-" * 65)
    for term, esl_r, cw_r, gap in gaps[:top_n]:
        print(f"{term:30s}  {esl_r * 100:9.1f}%  {cw_r * 100:9.1f}%  {gap * 100:+7.1f}%")

    print()
    equal = [(t, e, c, g) for t, e, c, g in gaps if abs(g) < 0.05 and e > 0.3]
    equal.sort(key=lambda x: -x[1])
    if equal:
        print("=== DOMAIN FAILURES (equal miss rate — not accent-driven) ===")
        for term, esl_r, cw_r, gap in equal[:15]:
            print(f"{term:30s}  {esl_r * 100:9.1f}%  {cw_r * 100:9.1f}%  {gap * 100:+7.1f}%")


def show_substitutions(
    reports: list[dict[str, object]],
    *,
    target_terms: list[str] | None = None,
    top_n: int = 5,
) -> None:
    """Print what ASR actually produces in place of missed terms."""
    if target_terms is None:
        target_terms = [
            "Gy",
            "GyE",
            "IMRT",
            "IGRT",
            "VMAT",
            "brachytherapy",
            "chemoradiation",
            "medulloblastoma",
            "orchidectomy",
            "dose painting",
            "fiducial marker",
            "neoadjuvant",
            "oropharyngeal",
            "lumpectomy",
            "seminoma",
            "vulvar",
        ]

    subs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    for report in reports:
        for be in report.get("results", []):
            if not isinstance(be, dict):
                continue
            backend = str(be.get("backend", ""))
            for vr in be.get("by_voice", []):
                if not isinstance(vr, dict):
                    continue
                voice = str(vr.get("voice", ""))
                l1 = _get_l1(voice)
                for s in vr.get("samples", []):
                    if not isinstance(s, dict):
                        continue
                    missing = [str(t) for t in s.get("terms_missing", [])]
                    gt = str(s.get("ground_truth", ""))
                    tr = str(s.get("raw_transcription", ""))
                    for term in missing:
                        if term in target_terms:
                            sub = _find_substitution(gt, tr, term)
                            if sub:
                                subs[(term, backend)].append((l1, sub))

    for term in target_terms:
        for backend in ["mlx_whisper", "voxtral"]:
            entries = subs.get((term, backend), [])
            if not entries:
                continue
            counts: dict[str, int] = defaultdict(int)
            for _, sub in entries:
                counts[sub] += 1
            top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
            top_str = "  |  ".join(f'"{p}"({c})' for p, c in top)
            print(f"{term:25s} [{backend:12s}]  {top_str}")
        print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esl-reports", nargs="+", type=Path, default=[], help="ESL bake-off JSON reports.")
    parser.add_argument("--cw-reports", nargs="+", type=Path, default=[], help="Commonwealth bake-off JSON reports.")
    parser.add_argument(
        "--mode",
        choices=["accent-penalty", "substitutions", "missed-terms", "all"],
        default="all",
        help="Analysis mode.",
    )
    parser.add_argument("--top", type=int, default=25, help="Number of top results to show.")
    parser.add_argument("--terms", nargs="+", default=None, help="Specific terms to mine substitutions for.")
    args = parser.parse_args(argv)

    all_reports = _load_reports(args.esl_reports) + _load_reports(args.cw_reports)
    esl_reports = _load_reports(args.esl_reports)
    cw_reports = _load_reports(args.cw_reports)

    if args.mode in ("accent-penalty", "all") and esl_reports and cw_reports:
        show_accent_penalty(esl_reports, cw_reports, top_n=args.top)
        print()

    if args.mode in ("substitutions", "all") and all_reports:
        show_substitutions(all_reports, target_terms=args.terms, top_n=5)

    if args.mode == "missed-terms":
        missed, total = _get_term_miss_rates(all_reports)
        combined: list[tuple[str, int, float]] = []
        for term in total:
            rate = missed.get(term, 0) / total[term]
            combined.append((term, missed.get(term, 0), rate))
        combined.sort(key=lambda x: -x[1])
        print(f"{'Term':30s}  {'Missed':>8s}  {'Rate':>8s}")
        print("-" * 50)
        for term, cnt, rate in combined[: args.top]:
            print(f"{term:30s}  {cnt:>8d}  {rate * 100:7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
