"""Multi-backend end-to-end validation wrapper.

Runs the E2E pipeline across multiple ASR backends (Whisper, MedASR, ...)
and compares per-backend quality on the same audio. Synthesizes audio
once per voice via piper, then transcribes that audio with every backend
under test. Reports WER and per-term accuracy (how many of each
fixture's `vocabulary_terms` survived in the raw transcription).

Primary use: test Stuart's hypothesis — if MedASR's advantage is medical
vocabulary competence rather than acoustic robustness, MedASR should
beat Whisper on clean TTS of dense clinical content.

Phonetic correction is forced OFF to isolate backend signal from the
corrector. The corrector is the same across backends; we want the raw
ASR quality here.

Usage:
    python tests/validation/scripts/run_multi_backend_e2e.py \\
        --backends mlx_whisper medasr \\
        --voices alan lessac \\
        --dense-only \\
        --output bakeoff.json

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tests.validation.audio_synthesis.noise_injection import (
    DEFAULT_CROSSFADE_MS,
    NOISE_PRESETS,
    inject_manifest,
)
from tests.validation.audio_synthesis.piper_tts import (
    load_fixtures,
    resolve_piper_bin,
    resolve_piper_voices_root,
    synthesize_fixtures,
)
from tests.validation.scripts.run_end_to_end import (
    _find_audio_for_fixture,
    _sample_fixtures,
    _wer,
)
from tests.validation.scripts.run_multi_voice_e2e import (
    DEFAULT_VOICES,
    VoiceSpec,
    _load_voice_specs,
)
from transcriber_radrx.asr_backends import get_backend
from transcriber_radrx.transcriber import transcribe_with_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.validation.audio_synthesis.piper_tts import TextFixture
    from transcriber_radrx.asr_backends.base import ASRBackend

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BackendSpec:
    """A single backend to evaluate.

    Attributes:
        name: Backend registry name (e.g. "mlx_whisper", "medasr").
        model_id: Optional model override. If None, the backend's default
            model is used.
    """

    name: str
    model_id: str | None = None

    @property
    def label(self) -> str:
        """Short identifier suitable for reports and manifests."""
        if self.model_id is None:
            return self.name
        # e.g. mlx_whisper:whisper-large-v3-mlx
        short = self.model_id.split("/")[-1]
        return f"{self.name}:{short}"


def _parse_backend_arg(arg: str) -> BackendSpec:
    """Parse a CLI backend argument of the form name[:model_id]."""
    if ":" in arg:
        name, model_id = arg.split(":", 1)
        return BackendSpec(name=name, model_id=model_id)
    return BackendSpec(name=arg)


def _instantiate_backend(spec: BackendSpec) -> ASRBackend:
    """Instantiate and load an ASR backend."""
    kwargs: dict[str, object] = {}
    if spec.model_id is not None:
        kwargs["model_id"] = spec.model_id
    backend = get_backend(spec.name, **kwargs)
    logger.info("Loading backend %s (model=%s)", backend.name, backend.model_id)
    backend.load()
    return backend


def _term_found(term: str, hypothesis_norm: str) -> bool:
    """Check whether a vocabulary term survived in the normalized hypothesis.

    Uses a whitespace-bounded substring match on the lowercased hypothesis
    so that "Gy" (length 2) is not accidentally found inside "foggy", but
    "spinal cord" is found as a multi-word phrase.
    """
    term_norm = term.lower().strip()
    if not term_norm:
        return True
    pattern = r"(?:^|\s)" + re.escape(term_norm) + r"(?:\s|$|[.,;:!?])"
    return re.search(pattern, hypothesis_norm) is not None


def _term_accuracy(
    vocabulary_terms: list[str],
    hypothesis: str,
) -> tuple[int, int, list[str]]:
    """Score per-term accuracy for one fixture.

    Returns (found, total, missing_terms). If vocabulary_terms is empty,
    returns (0, 0, []) and the caller can skip the fixture for term metrics.
    """
    if not vocabulary_terms:
        return 0, 0, []
    # Normalize hypothesis: lowercase + collapse whitespace + strip most
    # punctuation but keep periods so phrases like "3 D" don't glue together.
    hyp_norm = re.sub(r"[^\w\s.]", " ", hypothesis.lower())
    hyp_norm = " ".join(hyp_norm.split())
    missing: list[str] = []
    found = 0
    for term in vocabulary_terms:
        if _term_found(str(term), hyp_norm):
            found += 1
        else:
            missing.append(str(term))
    return found, len(vocabulary_terms), missing


def _run_backend_pass(
    backend: ASRBackend,
    fixtures: list[TextFixture],
    manifest_entries: Sequence[dict[str, object]],
    vocabulary_path: Path,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, object]]:
    """Run one transcription pass over the fixture set with a loaded backend.

    Args:
        backend: Loaded ASR backend instance.
        fixtures: Fixture set to transcribe.
        manifest_entries: Audio manifest from piper synthesis.
        vocabulary_path: Vocabulary file for biasing + correction.
        system_prompt: Optional instruction-following directive for
            audio-LLM backends. Classical ASRs ignore it.
    """
    results: list[dict[str, object]] = []
    for fixture in fixtures:
        audio_path = _find_audio_for_fixture(fixture.id, manifest_entries)
        if audio_path is None or not audio_path.exists():
            logger.warning("No audio for fixture %s, skipping", fixture.id)
            continue

        try:
            tr = transcribe_with_backend(
                audio_path,
                backend,
                vocabulary_path=vocabulary_path,
                enable_phonetic_correction=False,
                system_prompt=system_prompt,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Backend %s failed on fixture %s",
                backend.name,
                fixture.id,
            )
            results.append(
                {
                    "fixture_id": fixture.id,
                    "category": str(fixture.raw.get("category", "unknown")),
                    "dense_subtype": fixture.raw.get("dense_subtype"),
                    "ground_truth": fixture.text,
                    "raw_transcription": "",
                    "corrected_transcription": "",
                    "raw_wer": 1.0,
                    "corrected_wer": 1.0,
                    "vocabulary_terms": fixture.raw.get("vocabulary_terms", []),
                    "terms_found": 0,
                    "terms_total": len(fixture.raw.get("vocabulary_terms", []) or []),
                    "terms_missing": list(fixture.raw.get("vocabulary_terms", []) or []),
                    "error": True,
                },
            )
            continue

        vocabulary_terms = list(fixture.raw.get("vocabulary_terms", []) or [])
        terms_found, terms_total, terms_missing = _term_accuracy(
            vocabulary_terms,
            tr.text,
        )

        results.append(
            {
                "fixture_id": fixture.id,
                "category": str(fixture.raw.get("category", "unknown")),
                "dense_subtype": fixture.raw.get("dense_subtype"),
                "ground_truth": fixture.text,
                "raw_transcription": tr.text,
                "corrected_transcription": tr.corrected_text,
                "raw_wer": _wer(fixture.text, tr.text),
                "corrected_wer": _wer(fixture.text, tr.corrected_text),
                "vocabulary_terms": vocabulary_terms,
                "terms_found": terms_found,
                "terms_total": terms_total,
                "terms_missing": terms_missing,
                "error": False,
            },
        )
    return results


def _summarize_backend_voice(
    results: list[dict[str, object]],
) -> dict[str, object]:
    """Summary metrics for one backend × voice cell."""
    if not results:
        return {"sample_count": 0}

    n = len(results)
    total_raw = sum(float(r.get("raw_wer", 0.0)) for r in results)
    total_cor = sum(float(r.get("corrected_wer", 0.0)) for r in results)
    total_found = sum(int(r.get("terms_found", 0)) for r in results)
    total_terms = sum(int(r.get("terms_total", 0)) for r in results)
    errors = sum(1 for r in results if r.get("error"))

    term_recall = (total_found / total_terms) if total_terms > 0 else None

    return {
        "sample_count": n,
        "error_count": errors,
        "avg_raw_wer": round(total_raw / n, 4),
        "avg_corrected_wer": round(total_cor / n, 4),
        "terms_found": total_found,
        "terms_total": total_terms,
        "term_recall": round(term_recall, 4) if term_recall is not None else None,
    }


def _aggregate_backend(
    backend_label: str,
    voice_reports: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate one backend's results across all voices."""
    all_samples: list[dict[str, object]] = []
    for vr in voice_reports:
        samples = vr.get("samples", [])
        if isinstance(samples, list):
            all_samples.extend(samples)

    overall = _summarize_backend_voice(all_samples)
    return {
        "backend": backend_label,
        "by_voice": [
            {
                "voice": vr.get("voice"),
                "summary": vr.get("summary"),
                "samples": vr.get("samples", []),
            }
            for vr in voice_reports
        ],
        "overall": overall,
    }


def _print_report(backend_aggs: list[dict[str, object]]) -> None:
    """Human-readable comparison across backends."""
    print("\n" + "=" * 100)
    print("  MULTI-BACKEND BAKE-OFF RESULTS")
    print("=" * 100)
    print()
    print(
        f"  {'backend':<45} {'n':>4}  {'raw WER':>10}  {'corr WER':>10}  {'term recall':>12}  {'terms':>10}",
    )
    print("  " + "-" * 96)
    for agg in backend_aggs:
        overall = agg.get("overall", {})
        if not isinstance(overall, dict):
            continue
        label = str(agg.get("backend", "?"))
        n = int(overall.get("sample_count", 0))
        raw = float(overall.get("avg_raw_wer") or 0.0)
        cor = float(overall.get("avg_corrected_wer") or 0.0)
        tr = overall.get("term_recall")
        tr_str = f"{float(tr):.4f}" if tr is not None else "   n/a"
        terms = f"{overall.get('terms_found', 0)}/{overall.get('terms_total', 0)}"
        print(f"  {label:<45} {n:>4}  {raw:>10.4f}  {cor:>10.4f}  {tr_str:>12}  {terms:>10}")
    print()
    print("  Per-backend × per-voice breakdown:")
    for agg in backend_aggs:
        print(f"\n    {agg.get('backend')}:")
        for vr in agg.get("by_voice", []):  # type: ignore[union-attr]
            if not isinstance(vr, dict):
                continue
            s = vr.get("summary", {})
            if not isinstance(s, dict):
                continue
            raw = float(s.get("avg_raw_wer") or 0.0)
            tr = s.get("term_recall")
            tr_str = f"{float(tr):.4f}" if tr is not None else "n/a"
            print(f"      {vr.get('voice'):<40}  raw WER={raw:.4f}  term recall={tr_str}")
    print("=" * 100)
    print()


def _filter_dense(fixtures: list[TextFixture]) -> list[TextFixture]:
    """Keep only fixtures with a dense_subtype field set."""
    return [f for f in fixtures if f.raw.get("dense_subtype")]


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    """CLI entry point for multi-backend bake-off."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mlx_whisper", "medasr"],
        help=(
            "Backends to compare. Each entry is 'name' (use backend default model) "
            "or 'name:model_id' to override. Example: mlx_whisper medasr"
        ),
    )
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
        default=resolve_piper_voices_root(),
        help=(
            "Root directory of the piper-voices tree. Defaults to "
            "$PIPER_VOICES_ROOT, then ./piper-voices, then ~/piper-voices. "
            "If none of those exist, must be passed explicitly."
        ),
    )
    parser.add_argument(
        "--piper-bin",
        type=Path,
        default=resolve_piper_bin(),
        help=(
            "Path to the piper binary. Defaults to $PIPER_BIN, then "
            "'piper' on $PATH (via shutil.which). If neither is set, "
            "must be passed explicitly."
        ),
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["alan", "lessac"],
        help=(
            "Voice names from the default piper voice panel. Default: alan, lessac "
            "(one UK, one US) — enough to separate backend signal from accent "
            "variance on a first run."
        ),
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        default=True,
        help=("Restrict to dense clinical fixtures (default). Use --no-dense-only to run against the full fixture set."),
    )
    parser.add_argument(
        "--no-dense-only",
        dest="dense_only",
        action="store_false",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=("Max fixtures per run. 0 (default) = no limit — run all fixtures (after dense-only filter if set)."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Instruction-following directive for audio-LLM backends "
            "(Granite-Speech, Voxtral, etc.). Classical ASRs (Whisper, "
            "MedASR, Cohere) ignore it with a log warning. Example: "
            "'Transcribe this radiation oncology dictation verbatim. "
            "Preserve Gy as Gy. Preserve PTV, CTV, IMRT, VMAT exactly. "
            "Use numeric digits for doses.'"
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help=(
            "Read the system prompt from a file instead of passing it "
            "on the command line. Useful for multi-line prompts or when "
            "the prompt contains shell-unfriendly characters."
        ),
    )
    parser.add_argument(
        "--noise-preset",
        choices=list(NOISE_PRESETS.keys()),
        default=None,
        help=(
            "If set, mix MUSAN background noise into every synthesized "
            "audio sample at the preset's target SNR before transcription. "
            "quiet=20 dB, moderate=10 dB, busy=5 dB. Default: no noise "
            "(clean tier)."
        ),
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=REPO_ROOT / "tests/validation/corpora/restricted/musan/noise",
        help="Path to MUSAN noise directory (contains free-sound/, sound-bible/).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Random seed for reproducible noise selection.",
    )
    parser.add_argument(
        "--noise-crossfade-ms",
        type=float,
        default=DEFAULT_CROSSFADE_MS,
        help="Linear crossfade length at noise splice seams, in milliseconds.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Resolve system prompt: file takes precedence if both are given.
    system_prompt: str | None = None
    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().strip()
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate external dependencies before doing any work.
    if args.piper_voices_root is None:
        print(
            "ERROR: could not locate a piper-voices tree. Set "
            "$PIPER_VOICES_ROOT or pass --piper-voices-root. See the "
            "'External dependencies' section of the repository README "
            "for installation instructions.",
            file=sys.stderr,
        )
        return 1
    if args.piper_bin is None:
        print(
            "ERROR: could not locate a piper binary. Install piper-tts "
            "('pip install piper-tts' or 'brew install piper-tts'), or "
            "set $PIPER_BIN, or pass --piper-bin. See the 'External "
            "dependencies' section of the repository README.",
            file=sys.stderr,
        )
        return 1

    # Resolve voices
    wanted = set(args.voices)
    requested_voices = [v for v in DEFAULT_VOICES if v[0] in wanted]
    missing_voices = wanted - {v[0] for v in DEFAULT_VOICES}
    if missing_voices:
        print(f"WARNING: unknown voice names: {sorted(missing_voices)}", file=sys.stderr)
    voices: list[VoiceSpec] = _load_voice_specs(args.piper_voices_root, requested_voices)
    if not voices:
        print("No valid voices to run", file=sys.stderr)
        return 1
    print(f"Resolved {len(voices)} voice(s): {[v.display_name for v in voices]}")

    # Load + filter + sample fixtures
    all_fixtures = load_fixtures(args.fixtures)
    if args.dense_only:
        all_fixtures = _filter_dense(all_fixtures)
        print(f"Dense-only filter: {len(all_fixtures)} fixtures with dense_subtype")
    if args.limit and args.limit > 0:
        sampled = _sample_fixtures(
            all_fixtures,
            limit=args.limit,
            seed=args.seed,
        )
    else:
        sampled = all_fixtures
    if not sampled:
        print("No fixtures selected", file=sys.stderr)
        return 1
    print(f"Selected {len(sampled)} fixtures")

    # Parse backend specs (but DO NOT instantiate yet — instantiate after
    # synthesis so any synth failure doesn't waste the model load).
    backend_specs = [_parse_backend_arg(b) for b in args.backends]
    print(f"Backends to run: {[s.label for s in backend_specs]}")

    backend_aggs: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="radrx-multi-backend-") as tmpdir:
        tmp_root = Path(tmpdir)

        # 1. Synthesize audio for every voice up-front. Reused across all backends.
        voice_manifests: dict[str, list[dict[str, object]]] = {}
        for voice in voices:
            audio_dir = tmp_root / voice.display_name / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[synth] {voice.display_name}")
            manifest_entries = synthesize_fixtures(
                sampled,
                voice_model=voice.model_path,
                output_dir=audio_dir,
                voice_name=voice.display_name,
                piper_cmd=[str(args.piper_bin)] if args.piper_bin else None,
                repo_root=REPO_ROOT,
            )
            voice_manifests[voice.display_name] = list(manifest_entries)
            print(f"  synthesized {len(manifest_entries)} / {len(sampled)}")

        # 1b. Optional noise injection stage. Replaces the clean voice
        # manifests with noisy-tier manifests pointing at mixed WAVs in a
        # sibling directory. Noise coverage uses prefer-long-first with
        # crossfaded splicing (see tests/validation/audio_synthesis/
        # noise_injection.py).
        if args.noise_preset is not None:
            preset = NOISE_PRESETS[args.noise_preset]
            print(f"\n[noise] injecting MUSAN noise at {preset.snr_db:.1f} dB SNR ({preset.name})")
            for voice in voices:
                clean_entries = voice_manifests.get(voice.display_name, [])
                if not clean_entries:
                    continue
                noisy_dir = tmp_root / voice.display_name / f"noisy-{preset.name}"
                noisy_entries = inject_manifest(
                    clean_entries,
                    clean_audio_dir=tmp_root / voice.display_name / "audio",
                    output_dir=noisy_dir,
                    preset=preset,
                    noise_dir=args.noise_dir,
                    seed=args.noise_seed,
                    crossfade_ms=args.noise_crossfade_ms,
                    repo_root=REPO_ROOT,
                )
                voice_manifests[voice.display_name] = noisy_entries
                print(f"  {voice.display_name}: {len(noisy_entries)} / {len(clean_entries)} mixed")

        # 2. For each backend, load once and transcribe all voices × fixtures.
        for spec in backend_specs:
            print(f"\n[backend] {spec.label}")
            try:
                backend = _instantiate_backend(spec)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load backend %s", spec.label)
                backend_aggs.append(
                    {
                        "backend": spec.label,
                        "by_voice": [],
                        "overall": {
                            "sample_count": 0,
                            "error_count": 0,
                            "load_failed": True,
                        },
                    },
                )
                continue

            voice_reports: list[dict[str, object]] = []
            for voice in voices:
                manifest_entries = voice_manifests.get(voice.display_name, [])
                if not manifest_entries:
                    continue
                print(f"  [{spec.label}] transcribing {voice.display_name}")
                results = _run_backend_pass(
                    backend,
                    sampled,
                    manifest_entries,
                    args.vocabulary,
                    system_prompt=system_prompt,
                )
                summary = _summarize_backend_voice(results)
                voice_reports.append(
                    {
                        "voice": voice.display_name,
                        "summary": summary,
                        "samples": results,
                    },
                )

            backend.unload()
            agg = _aggregate_backend(spec.label, voice_reports)
            backend_aggs.append(agg)

    _print_report(backend_aggs)

    if args.output:
        noise_metadata: dict[str, object] | None = None
        if args.noise_preset is not None:
            preset = NOISE_PRESETS[args.noise_preset]
            noise_metadata = {
                "preset": preset.name,
                "snr_db": preset.snr_db,
                "description": preset.description,
                "noise_dir": str(args.noise_dir),
                "seed": args.noise_seed,
                "crossfade_ms": args.noise_crossfade_ms,
            }
        full_report = {
            "vocabulary": str(args.vocabulary),
            "sample_count": len(sampled),
            "dense_only": args.dense_only,
            "seed": args.seed,
            "system_prompt": system_prompt,
            "noise": noise_metadata,
            "fixture_ids": [f.id for f in sampled],
            "voices": [v.display_name for v in voices],
            "backends": [s.label for s in backend_specs],
            "results": backend_aggs,
        }
        args.output.write_text(json.dumps(full_report, indent=2, ensure_ascii=False))
        print(f"Detailed report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
