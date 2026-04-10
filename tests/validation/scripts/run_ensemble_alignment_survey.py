"""Ensemble alignment survey: Voxtral vs Whisper on particle therapy corpus.

Loads both bake-off JSONs from tests/validation/reports/, runs word-level
alignment on every fixture × voice pair present in both, and prints:
- A summary table (fixture_id, voice, agreement_rate, n_sub, n_ins)
- The inline diff for every pair with agreement_rate < 0.8

Used by Silas to examine the disagreement landscape before designing
Phase 2 token-class decision rules.

Usage::

    uv run python tests/validation/scripts/run_ensemble_alignment_survey.py

The script resolves the report paths relative to the project root, so it
can be run from any directory as long as the virtualenv is active.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from any directory by resolving the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from transcriber_radrx.ensemble.aligner import (  # noqa: E402
    align_transcriptions,
    format_alignment_diff,
    summarize_alignment,
)

_REPORTS_DIR = _PROJECT_ROOT / "tests" / "validation" / "reports"
_VOXTRAL_JSON = _REPORTS_DIR / "bakeoff_particle_voxtral_2026-04-09.json"
_WHISPER_JSON = _REPORTS_DIR / "bakeoff_particle_whisper_medasr_2026-04-09.json"

_LOW_AGREEMENT_THRESHOLD = 0.8


def _load_transcriptions(json_path: Path, backend_filter: str) -> dict[tuple[str, str], str]:
    """Load raw transcriptions keyed by (fixture_id, voice) for a given backend.

    Args:
        json_path: Path to bake-off JSON report.
        backend_filter: Substring to match the backend name (case-insensitive).

    Returns:
        Dict mapping (fixture_id, voice) → raw_transcription string.
    """
    with json_path.open() as fh:
        data = json.load(fh)

    transcriptions: dict[tuple[str, str], str] = {}
    for result in data.get("results", []):
        backend = result.get("backend", "")
        if backend_filter.lower() not in backend.lower():
            continue
        for voice_entry in result.get("by_voice", []):
            voice = voice_entry.get("voice", "")
            for sample in voice_entry.get("samples", []):
                fixture_id = sample.get("fixture_id", "")
                raw = sample.get("raw_transcription", "")
                if fixture_id and raw is not None:
                    transcriptions[(fixture_id, voice)] = raw.strip()
    return transcriptions


def _print_header() -> None:
    header = f"{'fixture_id':<18} {'voice':<22} {'agree':>6} {'subs':>5} {'ins_a':>6} {'ins_b':>6}"
    print(header)
    print("-" * len(header))


def main() -> None:
    """Run the alignment survey and print results to stdout."""
    if not _VOXTRAL_JSON.exists():
        print(f"ERROR: Voxtral JSON not found: {_VOXTRAL_JSON}", file=sys.stderr)
        sys.exit(1)
    if not _WHISPER_JSON.exists():
        print(f"ERROR: Whisper JSON not found: {_WHISPER_JSON}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Voxtral transcriptions from: {_VOXTRAL_JSON.name}")
    voxtral_map = _load_transcriptions(_VOXTRAL_JSON, "voxtral")
    print(f"  Found {len(voxtral_map)} (fixture, voice) pairs for voxtral backend")

    print(f"Loading Whisper transcriptions from: {_WHISPER_JSON.name}")
    whisper_map = _load_transcriptions(_WHISPER_JSON, "whisper")
    print(f"  Found {len(whisper_map)} (fixture, voice) pairs for whisper backend")

    common_keys = sorted(voxtral_map.keys() & whisper_map.keys())
    print(f"\nPairs present in both: {len(common_keys)}\n")

    _print_header()

    low_agreement_pairs: list[tuple[tuple[str, str], str]] = []

    for key in common_keys:
        fixture_id, voice = key
        text_a = voxtral_map[key]
        text_b = whisper_map[key]

        spans = align_transcriptions(text_a, text_b)
        summary = summarize_alignment(spans)

        print(
            f"{fixture_id:<18} {voice:<22} {summary.agreement_rate:>6.3f} "
            f"{summary.substitutions:>5} {summary.insertions_a:>6} {summary.insertions_b:>6}"
        )

        if summary.agreement_rate < _LOW_AGREEMENT_THRESHOLD:
            diff = format_alignment_diff(spans)
            low_agreement_pairs.append((key, diff))

    print()
    print(f"Pairs with agreement_rate < {_LOW_AGREEMENT_THRESHOLD}: {len(low_agreement_pairs)}")

    if low_agreement_pairs:
        print()
        print("=" * 80)
        print("INLINE DIFFS FOR LOW-AGREEMENT PAIRS  [Voxtral | Whisper]")
        print("=" * 80)
        for (fixture_id, voice), diff in low_agreement_pairs:
            print(f"\n{fixture_id} / {voice}")
            print(f"  {diff}")


if __name__ == "__main__":
    main()
