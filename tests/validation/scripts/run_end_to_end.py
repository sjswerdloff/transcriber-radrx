"""End-to-end validation runner: text → piper → Whisper → corrector → WER.

Bypasses the acoustic chain for the first empirical check. Takes a sample
of text fixtures, synthesizes clean-tier audio with piper, transcribes
with Whisper MLX (vocabulary biasing + correction dictionary), and
compares the output to ground truth using jiwer.

Usage:
    python tests/validation/scripts/run_end_to_end.py \\
        --fixtures tests/validation/fixtures/rt_dictation_samples.jsonl \\
        --voice-model /path/to/en_US-amy-medium.onnx \\
        --piper-bin /path/to/piper \\
        --vocabulary tests/validation/fixtures/rt_dictation_samples.jsonl \\
        --limit 10

For empirical safety verification, the runner produces two pipelines:
- phonetic_off: corrector with phonetic matching disabled (the default)
- phonetic_on: corrector with phonetic matching enabled

This shows empirically whether the safety default is the right default.

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tests.validation.audio_synthesis.piper_tts import (
    TextFixture,
    load_fixtures,
    synthesize_fixtures,
)
from transcriber_radrx.transcriber import transcribe

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _normalize_for_wer(text: str) -> str:
    """Normalize text for WER comparison.

    Whisper and piper may differ on capitalization, punctuation, and
    whitespace. WER should focus on word content, not surface details.
    """
    import re

    text = text.lower()
    # Strip punctuation but keep word-internal apostrophes and hyphens
    text = re.sub(r"[^\w\s'\-]", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


def _wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate using jiwer."""
    import jiwer

    ref_norm = _normalize_for_wer(reference)
    hyp_norm = _normalize_for_wer(hypothesis)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return float(jiwer.wer(ref_norm, hyp_norm))


def _sample_fixtures(
    fixtures: list[TextFixture],
    *,
    limit: int,
    seed: int,
    include_categories: set[str] | None = None,
) -> list[TextFixture]:
    """Pick a stratified sample of fixtures across categories.

    Ensures the sample includes homograph_trap cases if available.
    """
    if include_categories:
        fixtures = [f for f in fixtures if f.raw.get("category") in include_categories]

    random.seed(seed)

    # Stratify by category: at least 1 from each present category if possible
    by_category: dict[str, list[TextFixture]] = {}
    for f in fixtures:
        cat = str(f.raw.get("category", "unknown"))
        by_category.setdefault(cat, []).append(f)

    selected: list[TextFixture] = []
    # Always include up to 5 homograph traps if present
    if "homograph_trap" in by_category:
        traps = by_category["homograph_trap"]
        selected.extend(random.sample(traps, min(5, len(traps))))  # noqa: S311

    # Fill remaining slots round-robin across other categories
    other_cats = [c for c in by_category if c != "homograph_trap"]
    remaining = limit - len(selected)
    while remaining > 0 and other_cats:
        for cat in list(other_cats):
            pool = [f for f in by_category[cat] if f not in selected]
            if not pool:
                other_cats.remove(cat)
                continue
            selected.append(random.choice(pool))  # noqa: S311
            remaining -= 1
            if remaining <= 0:
                break

    return selected[:limit]


def _find_audio_for_fixture(
    fixture_id: str,
    manifest_entries: Sequence[dict[str, object]],
) -> Path | None:
    """Find the audio path for a given fixture id from manifest entries."""
    for entry in manifest_entries:
        if entry.get("text_id") == fixture_id:
            path_str = str(entry.get("audio_path", ""))
            path = Path(path_str)
            if path.is_absolute():
                return path
            return REPO_ROOT / path
    return None


def _run_transcription_pass(
    fixtures: list[TextFixture],
    manifest_entries: Sequence[dict[str, object]],
    vocabulary_path: Path,
    *,
    enable_phonetic: bool,
    whisper_model: str,
) -> list[dict[str, object]]:
    """Run one transcription pass over the fixture set with given settings."""
    results: list[dict[str, object]] = []
    for fixture in fixtures:
        audio_path = _find_audio_for_fixture(fixture.id, manifest_entries)
        if audio_path is None or not audio_path.exists():
            logger.warning("No audio for fixture %s, skipping", fixture.id)
            continue

        logger.info(
            "Transcribing %s (enable_phonetic=%s): %s",
            fixture.id,
            enable_phonetic,
            fixture.text[:60],
        )
        result = transcribe(
            audio_path,
            model=whisper_model,
            vocabulary_path=vocabulary_path,
            enable_phonetic_correction=enable_phonetic,
        )

        ground_truth = fixture.text
        raw_wer = _wer(ground_truth, result.text)
        corrected_wer = _wer(ground_truth, result.corrected_text)

        category = str(fixture.raw.get("category", "unknown"))
        must_not_become = fixture.raw.get("must_not_become", []) or []

        # Homograph-trap safety check: no corrected text should contain terms
        # that the corrector must not introduce
        trap_violations: list[str] = []
        if category == "homograph_trap" and isinstance(must_not_become, list):
            for term in must_not_become:
                term_str = str(term)
                if term_str in result.corrected_text:
                    trap_violations.append(term_str)

        results.append(
            {
                "fixture_id": fixture.id,
                "category": category,
                "ground_truth": ground_truth,
                "raw_transcription": result.text,
                "corrected_transcription": result.corrected_text,
                "raw_wer": raw_wer,
                "corrected_wer": corrected_wer,
                "num_corrections": len(result.corrections),
                "correction_methods": sorted({c.method for c in result.corrections}),
                "trap_violations": trap_violations,
                "enable_phonetic": enable_phonetic,
            },
        )
    return results


def _summarize(results: list[dict[str, object]]) -> dict[str, object]:
    """Produce summary metrics for a single transcription pass."""
    if not results:
        return {"sample_count": 0}

    def _sum(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in results)

    sample_count = len(results)
    avg_raw_wer = _sum("raw_wer") / sample_count
    avg_corrected_wer = _sum("corrected_wer") / sample_count
    total_trap_violations = sum(len(list(r.get("trap_violations", []))) for r in results)
    total_corrections = int(_sum("num_corrections"))

    by_category: dict[str, dict[str, float]] = {}
    for r in results:
        cat = str(r.get("category", "unknown"))
        bucket = by_category.setdefault(cat, {"n": 0.0, "raw_wer": 0.0, "corrected_wer": 0.0})
        bucket["n"] += 1
        bucket["raw_wer"] += float(r.get("raw_wer", 0.0))
        bucket["corrected_wer"] += float(r.get("corrected_wer", 0.0))

    category_summary: dict[str, dict[str, float]] = {}
    for cat, bucket in by_category.items():
        n = bucket["n"] or 1
        category_summary[cat] = {
            "n": int(bucket["n"]),
            "avg_raw_wer": round(bucket["raw_wer"] / n, 4),
            "avg_corrected_wer": round(bucket["corrected_wer"] / n, 4),
        }

    return {
        "sample_count": sample_count,
        "avg_raw_wer": round(avg_raw_wer, 4),
        "avg_corrected_wer": round(avg_corrected_wer, 4),
        "total_corrections_applied": total_corrections,
        "total_trap_violations": total_trap_violations,
        "by_category": category_summary,
    }


def _print_report(
    summary_off: dict[str, object],
    summary_on: dict[str, object],
    results_off: list[dict[str, object]],
    results_on: list[dict[str, object]],
) -> None:
    """Human-readable side-by-side comparison of phonetic off vs on."""
    print("\n" + "=" * 78)
    print("  END-TO-END VALIDATION RESULTS")
    print("=" * 78)

    print("\n  Pipeline: text → piper → Whisper MLX → corrector → WER")
    print(f"  Samples: {summary_off.get('sample_count', 0)}")
    print()
    print("  |                        | phonetic OFF | phonetic ON  |")
    print("  |------------------------|--------------|--------------|")
    print(
        f"  | avg raw WER            | {summary_off.get('avg_raw_wer', 0):.4f}       "
        f"| {summary_on.get('avg_raw_wer', 0):.4f}       |",
    )
    print(
        f"  | avg corrected WER      | {summary_off.get('avg_corrected_wer', 0):.4f}       "
        f"| {summary_on.get('avg_corrected_wer', 0):.4f}       |",
    )
    print(
        f"  | corrections applied    | {summary_off.get('total_corrections_applied', 0):>12} "
        f"| {summary_on.get('total_corrections_applied', 0):>12} |",
    )
    print(
        f"  | homograph violations   | {summary_off.get('total_trap_violations', 0):>12} "
        f"| {summary_on.get('total_trap_violations', 0):>12} |",
    )

    off_violations = [r for r in results_off if r.get("trap_violations")]
    on_violations = [r for r in results_on if r.get("trap_violations")]

    if off_violations or on_violations:
        print("\n  HOMOGRAPH TRAP VIOLATIONS (safety failures):")
        for pass_name, rset in [("phonetic_off", off_violations), ("phonetic_on", on_violations)]:
            for r in rset:
                print(f"    [{pass_name}] {r['fixture_id']}: {list(r.get('trap_violations', []))}")
                print(f"       ground truth: {r['ground_truth']}")
                print(f"       corrected   : {r['corrected_transcription']}")

    print("\n  PER-SAMPLE DIFF (phonetic_off):")
    for r in results_off:
        gt = str(r["ground_truth"])
        rt = str(r["raw_transcription"])
        ct = str(r["corrected_transcription"])
        raw_wer = float(r["raw_wer"])
        cor_wer = float(r["corrected_wer"])
        print(
            f"\n    {r['fixture_id']} [{r['category']}] raw WER={raw_wer:.3f} corrected WER={cor_wer:.3f}",
        )
        print(f"      GT : {gt[:100]}")
        print(f"      RAW: {rt[:100]}")
        if ct.strip() != rt.strip():
            print(f"      COR: {ct[:100]}")

    print("\n" + "=" * 78)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    """CLI entry point for the end-to-end runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl",
        help="Path to text fixtures JSONL",
    )
    parser.add_argument(
        "--voice-model",
        type=Path,
        required=True,
        help="Path to piper ONNX voice model",
    )
    parser.add_argument(
        "--piper-bin",
        type=Path,
        default=None,
        help="Path to piper binary (default: 'piper' in PATH)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl",
        help="Vocabulary file used for Whisper initial_prompt and corrector. "
        "Default: the fixtures file itself (uses vocabulary_terms field aggregation "
        "isn't implemented — pass data/rt_vocabulary.txt for the real vocab).",
    )
    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper MLX model identifier",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of fixtures to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file for detailed results",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Only sample from these categories (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.voice_model.exists():
        print(f"ERROR: voice model not found: {args.voice_model}", file=sys.stderr)
        return 1
    if not args.fixtures.exists():
        print(f"ERROR: fixtures not found: {args.fixtures}", file=sys.stderr)
        return 1
    if not args.vocabulary.exists():
        print(f"ERROR: vocabulary not found: {args.vocabulary}", file=sys.stderr)
        return 1

    all_fixtures = load_fixtures(args.fixtures)
    include_cats = set(args.categories) if args.categories else None
    sampled = _sample_fixtures(all_fixtures, limit=args.limit, seed=args.seed, include_categories=include_cats)
    if not sampled:
        print("No fixtures selected (empty sample)", file=sys.stderr)
        return 1

    print(f"Selected {len(sampled)} fixtures:")
    for f in sampled:
        print(f"  {f.id} [{f.raw.get('category', '?')}]: {f.text[:70]}")

    # Synthesize clean-tier audio in a temp directory
    with tempfile.TemporaryDirectory(prefix="radrx-e2e-") as tmpdir:
        audio_dir = Path(tmpdir) / "audio"
        audio_dir.mkdir(parents=True)

        piper_cmd = [str(args.piper_bin)] if args.piper_bin else None
        print("\nSynthesizing audio with piper...")
        manifest_entries = synthesize_fixtures(
            sampled,
            voice_model=args.voice_model,
            output_dir=audio_dir,
            piper_cmd=piper_cmd,
            repo_root=REPO_ROOT,
        )
        if len(manifest_entries) != len(sampled):
            print(
                f"WARNING: synthesized {len(manifest_entries)}/{len(sampled)} (some failed)",
                file=sys.stderr,
            )

        print("\nRunning transcription pass 1/2: phonetic correction OFF (default)")
        results_off = _run_transcription_pass(
            sampled,
            manifest_entries,
            args.vocabulary,
            enable_phonetic=False,
            whisper_model=args.whisper_model,
        )

        print("\nRunning transcription pass 2/2: phonetic correction ON")
        results_on = _run_transcription_pass(
            sampled,
            manifest_entries,
            args.vocabulary,
            enable_phonetic=True,
            whisper_model=args.whisper_model,
        )

    summary_off = _summarize(results_off)
    summary_on = _summarize(results_on)

    _print_report(summary_off, summary_on, results_off, results_on)

    if args.output:
        report = {
            "whisper_model": args.whisper_model,
            "voice_model": str(args.voice_model),
            "vocabulary": str(args.vocabulary),
            "sample_count": len(sampled),
            "seed": args.seed,
            "phonetic_off": {
                "summary": summary_off,
                "samples": results_off,
            },
            "phonetic_on": {
                "summary": summary_on,
                "samples": results_on,
            },
        }
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nDetailed report written to {args.output}")

    # Exit non-zero if homograph traps were violated under ANY configuration
    if summary_off.get("total_trap_violations", 0) or summary_on.get("total_trap_violations", 0):
        print("\nWARNING: homograph trap violations detected — clinical safety failure", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
