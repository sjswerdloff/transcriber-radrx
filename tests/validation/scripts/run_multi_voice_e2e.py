"""Multi-voice end-to-end validation wrapper.

Runs the end-to-end pipeline against several piper voices and compares
transcription quality (WER) across voices. Useful for measuring how
robust the ASR + correction pipeline is to speaker/accent variation.

Uses the existing run_end_to_end internals (fixture sampling,
synthesize_fixtures, _run_transcription_pass, _summarize) so the
single-voice runner remains the authoritative source of truth.

Usage:
    python tests/validation/scripts/run_multi_voice_e2e.py \\
        --piper-voices-root /path/to/piper-voices \\
        --piper-bin /path/to/piper \\
        --vocabulary data/rt_vocabulary.txt \\
        --limit 10 \\
        --output multi_voice_report.json

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tests.validation.audio_synthesis.piper_tts import (
    load_fixtures,
    synthesize_fixtures,
)
from tests.validation.scripts.run_end_to_end import (
    _run_transcription_pass,
    _sample_fixtures,
    _summarize,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VoiceSpec:
    """A single piper voice to evaluate.

    Attributes:
        name: Short name for the voice (e.g. "alan").
        model_path: Absolute path to the ONNX voice model.
        language: Language code from the voice directory (e.g. "en_GB").
        quality: Quality tier ("low", "medium", or "high").
        speaker_id: Optional speaker index for multi-speaker models (VCTK).
    """

    name: str
    model_path: Path
    language: str
    quality: str
    speaker_id: int | None = None

    @property
    def display_name(self) -> str:
        """Voice identifier suitable for manifests and reports."""
        parts = [self.language, self.name, self.quality]
        base = "-".join(parts)
        if self.speaker_id is not None:
            return f"{base}-speaker{self.speaker_id}"
        return base


# Default quality tier per voice. When "high" exists we prefer it; otherwise
# "medium"; otherwise "low". The single-low voice (southern_english_female)
# is intentionally included to test robustness against degraded voices.
#
# VCTK is intentionally excluded from the default list: it's a 109-speaker
# multi-speaker model, which would inflate memory and requires a piper
# speaker-info cross-reference to be meaningful. The single-speaker voices
# below cover UK accents (en_GB, 8 voices) and US gender balance (en_US,
# 4M/4F).
DEFAULT_VOICES: list[tuple[str, str, str]] = [
    # UK (en_GB) — accent and quality spread, 8 voices
    ("alan", "en_GB", "medium"),
    ("alba", "en_GB", "medium"),
    ("aru", "en_GB", "medium"),
    ("cori", "en_GB", "high"),
    ("jenny_dioco", "en_GB", "medium"),
    ("northern_english_male", "en_GB", "medium"),
    ("semaine", "en_GB", "medium"),
    ("southern_english_female", "en_GB", "low"),
    # US (en_US) — 4 female + 4 male
    ("amy", "en_US", "medium"),
    ("kristin", "en_US", "medium"),
    ("lessac", "en_US", "high"),
    ("ljspeech", "en_US", "high"),
    ("hfc_male", "en_US", "medium"),
    ("joe", "en_US", "medium"),
    ("norman", "en_US", "medium"),
    ("ryan", "en_US", "high"),
]


def _resolve_voice(
    piper_voices_root: Path,
    name: str,
    language: str,
    quality: str,
) -> Path | None:
    """Locate the ONNX file for a named voice."""
    lang_group = language.split("_")[0]
    candidate = piper_voices_root / lang_group / language / name / quality / f"{language}-{name}-{quality}.onnx"
    if candidate.exists():
        return candidate
    return None


def _load_voice_specs(
    piper_voices_root: Path,
    requested: list[tuple[str, str, str]],
) -> list[VoiceSpec]:
    """Resolve a list of (name, language, quality) tuples to VoiceSpec objects.

    Skips (with a warning) any voices whose model file is missing.
    """
    specs: list[VoiceSpec] = []
    for name, language, quality in requested:
        model_path = _resolve_voice(piper_voices_root, name, language, quality)
        if model_path is None:
            logger.warning("Voice %s (%s/%s) not found under %s", name, language, quality, piper_voices_root)
            continue
        specs.append(VoiceSpec(name=name, model_path=model_path, language=language, quality=quality))
    return specs


def _run_one_voice(
    voice: VoiceSpec,
    sampled_fixtures: list,  # type: ignore[type-arg]
    vocabulary_path: Path,
    whisper_model: str,
    piper_cmd: list[str] | None,
    *,
    workdir: Path,
) -> dict[str, object]:
    """Synthesize and transcribe the fixture set with a single voice.

    Returns a report dict with per-sample results and summary metrics for
    both phonetic pipelines (off and on).
    """
    audio_dir = workdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Synthesizing %d fixtures", voice.display_name, len(sampled_fixtures))
    manifest_entries = synthesize_fixtures(
        sampled_fixtures,
        voice_model=voice.model_path,
        output_dir=audio_dir,
        voice_name=voice.display_name,
        piper_cmd=piper_cmd,
        repo_root=REPO_ROOT,
    )

    synthesized = len(manifest_entries)
    if synthesized == 0:
        logger.error("[%s] Synthesis produced zero audio files; skipping transcription", voice.display_name)
        return {
            "voice": voice.display_name,
            "model_path": str(voice.model_path),
            "synthesized": 0,
            "phonetic_off": {"summary": {"sample_count": 0}, "samples": []},
            "phonetic_on": {"summary": {"sample_count": 0}, "samples": []},
        }

    logger.info("[%s] Transcribing pass 1/2: phonetic OFF", voice.display_name)
    results_off = _run_transcription_pass(
        sampled_fixtures,
        manifest_entries,
        vocabulary_path,
        enable_phonetic=False,
        whisper_model=whisper_model,
    )
    logger.info("[%s] Transcribing pass 2/2: phonetic ON", voice.display_name)
    results_on = _run_transcription_pass(
        sampled_fixtures,
        manifest_entries,
        vocabulary_path,
        enable_phonetic=True,
        whisper_model=whisper_model,
    )

    return {
        "voice": voice.display_name,
        "model_path": str(voice.model_path),
        "synthesized": synthesized,
        "phonetic_off": {
            "summary": _summarize(results_off),
            "samples": results_off,
        },
        "phonetic_on": {
            "summary": _summarize(results_on),
            "samples": results_on,
        },
    }


def _aggregate_cross_voice(voice_reports: list[dict[str, object]]) -> dict[str, object]:
    """Build a cross-voice comparison summary."""
    rows: list[dict[str, object]] = []
    for vr in voice_reports:
        off_summary = vr.get("phonetic_off", {}).get("summary", {})  # type: ignore[union-attr]
        on_summary = vr.get("phonetic_on", {}).get("summary", {})  # type: ignore[union-attr]
        rows.append(
            {
                "voice": vr.get("voice"),
                "synthesized": vr.get("synthesized", 0),
                "phonetic_off_raw_wer": off_summary.get("avg_raw_wer"),
                "phonetic_off_corrected_wer": off_summary.get("avg_corrected_wer"),
                "phonetic_off_trap_violations": off_summary.get("total_trap_violations", 0),
                "phonetic_on_raw_wer": on_summary.get("avg_raw_wer"),
                "phonetic_on_corrected_wer": on_summary.get("avg_corrected_wer"),
                "phonetic_on_trap_violations": on_summary.get("total_trap_violations", 0),
                "phonetic_on_corrections": on_summary.get("total_corrections_applied", 0),
            },
        )

    return {"by_voice": rows}


def _print_cross_voice_report(aggregate: dict[str, object]) -> None:
    """Human-readable cross-voice comparison table."""
    rows = aggregate.get("by_voice", [])
    if not isinstance(rows, list):
        return

    print("\n" + "=" * 100)
    print("  MULTI-VOICE E2E VALIDATION RESULTS")
    print("=" * 100)
    print()
    print(
        f"  {'voice':<40} {'n':>4}  {'raw WER (off)':>14}  {'cor WER (off)':>14}"
        f"  {'raw WER (on)':>13}  {'cor WER (on)':>13}  {'traps (on)':>10}",
    )
    print("  " + "-" * 98)
    for row in rows:
        voice = str(row.get("voice", "?"))
        n = row.get("synthesized", 0)
        off_raw = row.get("phonetic_off_raw_wer") or 0
        off_cor = row.get("phonetic_off_corrected_wer") or 0
        on_raw = row.get("phonetic_on_raw_wer") or 0
        on_cor = row.get("phonetic_on_corrected_wer") or 0
        traps_on = row.get("phonetic_on_trap_violations", 0)
        print(
            f"  {voice:<40} {n:>4}  {float(off_raw):>14.4f}  {float(off_cor):>14.4f}"
            f"  {float(on_raw):>13.4f}  {float(on_cor):>13.4f}  {int(traps_on):>10}",
        )
    print("=" * 100)
    print()


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    """CLI entry point for multi-voice E2E validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=REPO_ROOT / "data/rt_vocabulary.txt",
        help="Vocabulary file used for Whisper initial_prompt and corrector.",
    )
    parser.add_argument(
        "--piper-voices-root",
        type=Path,
        default=Path("/Users/stuartswerdloff/PythonProjects/PiperTTS/piper-voices"),
        help="Root directory of the piper-voices tree.",
    )
    parser.add_argument(
        "--piper-bin",
        type=Path,
        default=Path("/Users/stuartswerdloff/.pyenv/versions/piper311/bin/piper"),
        help="Path to piper binary.",
    )
    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper MLX model identifier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of fixtures to sample per voice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (same across voices for fair comparison).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file for detailed report.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Only sample from these fixture categories.",
    )
    parser.add_argument(
        "--voices",
        nargs="*",
        default=None,
        help=(
            "Subset of voice names to run. If omitted, runs the default panel: "
            "8 UK voices (alan, alba, aru, cori, jenny_dioco, northern_english_male, "
            "semaine, southern_english_female) + 8 US voices (amy, kristin, lessac, "
            "ljspeech, hfc_male, joe, norman, ryan). VCTK is excluded — its "
            "109-speaker model is heavy and needs speaker-info lookup to be "
            "meaningful for accent testing."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Filter DEFAULT_VOICES to the requested subset, if any
    requested_voices = DEFAULT_VOICES
    if args.voices:
        wanted = set(args.voices)
        requested_voices = [v for v in DEFAULT_VOICES if v[0] in wanted]
        missing = wanted - {v[0] for v in DEFAULT_VOICES}
        if missing:
            print(f"WARNING: unknown voice names: {sorted(missing)}", file=sys.stderr)

    voices = _load_voice_specs(args.piper_voices_root, requested_voices)
    if not voices:
        print("No valid voices to run", file=sys.stderr)
        return 1

    print(f"Resolved {len(voices)} voice(s):")
    for v in voices:
        print(f"  {v.display_name}  ({v.model_path})")

    # Sample fixtures once so every voice transcribes the same set
    all_fixtures = load_fixtures(args.fixtures)
    include_cats = set(args.categories) if args.categories else None
    sampled = _sample_fixtures(
        all_fixtures,
        limit=args.limit,
        seed=args.seed,
        include_categories=include_cats,
    )
    if not sampled:
        print("No fixtures selected (empty sample)", file=sys.stderr)
        return 1

    print(f"\nSampled {len(sampled)} fixtures (seed={args.seed}):")
    for f in sampled:
        print(f"  {f.id} [{f.raw.get('category', '?')}]: {f.text[:70]}")
    print()

    voice_reports: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="radrx-multi-voice-") as tmpdir:
        tmp_root = Path(tmpdir)
        for i, voice in enumerate(voices, start=1):
            print(f"\n[{i}/{len(voices)}] Voice: {voice.display_name}")
            voice_workdir = tmp_root / voice.display_name
            voice_workdir.mkdir(parents=True, exist_ok=True)
            try:
                report = _run_one_voice(
                    voice,
                    sampled,
                    vocabulary_path=args.vocabulary,
                    whisper_model=args.whisper_model,
                    piper_cmd=[str(args.piper_bin)] if args.piper_bin else None,
                    workdir=voice_workdir,
                )
            except Exception:
                logger.exception("Voice %s failed; continuing to next", voice.display_name)
                report = {
                    "voice": voice.display_name,
                    "model_path": str(voice.model_path),
                    "synthesized": 0,
                    "error": True,
                    "phonetic_off": {"summary": {"sample_count": 0}, "samples": []},
                    "phonetic_on": {"summary": {"sample_count": 0}, "samples": []},
                }
            voice_reports.append(report)

    aggregate = _aggregate_cross_voice(voice_reports)
    _print_cross_voice_report(aggregate)

    if args.output:
        full_report = {
            "whisper_model": args.whisper_model,
            "vocabulary": str(args.vocabulary),
            "sample_count": len(sampled),
            "seed": args.seed,
            "fixtures": [f.id for f in sampled],
            "voices": voice_reports,
            "aggregate": aggregate,
        }
        args.output.write_text(json.dumps(full_report, indent=2, ensure_ascii=False))
        print(f"Detailed report written to {args.output}")

    # Exit non-zero if any voice produced trap violations under either config
    any_violations = any(
        (
            vr.get("phonetic_off", {}).get("summary", {}).get("total_trap_violations", 0)  # type: ignore[union-attr]
            or vr.get("phonetic_on", {}).get("summary", {}).get("total_trap_violations", 0)
        )  # type: ignore[union-attr]
        for vr in voice_reports
    )
    return 2 if any_violations else 0


if __name__ == "__main__":
    sys.exit(main())
