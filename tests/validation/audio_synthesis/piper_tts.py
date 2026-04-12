"""Piper TTS pipeline for validation audio generation (clean tier).

Reads text fixtures from `rt_dictation_samples.jsonl`, invokes piper to
synthesize each sample as a WAV file, resamples to 16 kHz (Whisper/Parakeet
expected rate), and writes an `audio_manifest.jsonl` matching the schema
in tests/validation/SCHEMA.md.

Piper outputs raw int16 PCM at 22050 Hz. We convert to 16 kHz mono WAV
using scipy's polyphase resampler for accurate rate conversion.

Usage:
    # Generate clean-tier audio for all fixtures with default voice
    python -m tests.validation.audio_synthesis.piper_tts \\
        --fixtures tests/validation/fixtures/rt_dictation_samples.jsonl \\
        --output-dir tests/validation/audio/synthetic/clean \\
        --voice-model /path/to/en_US-amy-medium.onnx

Authors: vivian-1a61bc9a
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Piper --output_raw produces 22050 Hz, 16-bit signed, mono PCM
PIPER_SAMPLE_RATE = 22050
# ASR models (Whisper, Parakeet) expect 16 kHz input
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = 16


def _looks_like_piper_voices_tree(candidate: Path) -> bool:
    """Return True if ``candidate`` looks like a populated piper-voices tree.

    The rhasspy/piper-voices repository layout is
    ``{root}/{lang-group}/{language}/{name}/{quality}/...onnx`` — for
    example ``{root}/en/en_US/amy/medium/en_US-amy-medium.onnx``. This
    function checks that ``candidate`` is a directory with an ``en/``
    subdirectory that itself contains at least one language subdirectory
    (e.g. ``en_US`` or ``en_GB``). That is a cheap, layout-aware check
    that rejects a directory that happens to be named ``piper-voices``
    but does not actually contain the expected English voice tree.
    """
    if not candidate.is_dir():
        return False
    en_dir = candidate / "en"
    if not en_dir.is_dir():
        return False
    return any(child.is_dir() and child.name.startswith("en_") for child in en_dir.iterdir())


def resolve_piper_voices_root() -> Path | None:
    """Locate the piper-voices tree on the current machine.

    Resolution order (first candidate that looks like a populated
    piper-voices tree wins):

    1. ``$PIPER_VOICES_ROOT`` environment variable, if set and pointing
       at a directory containing the expected ``en/en_*/`` layout.
    2. ``./piper-voices`` in the current working directory (a common
       location when the rhasspy/piper-voices repo has been cloned
       alongside the project).
    3. ``~/piper-voices`` in the user's home directory.

    Layout assumption: candidates are expected to match the
    rhasspy/piper-voices HuggingFace / GitHub layout, i.e.
    ``{root}/en/en_US/amy/medium/en_US-amy-medium.onnx``. Candidates
    that are directories but do not contain the expected ``en/en_*/``
    layout are rejected with a debug log so that a stray directory
    named ``piper-voices`` does not mask a real voice tree further
    down the resolution order.

    This function contains no developer-specific absolute paths.
    Contributors whose voices are installed at a non-standard location
    must either set ``$PIPER_VOICES_ROOT``, create a symlink, or pass
    ``--piper-voices-root`` on the CLI.

    Returns:
        The resolved voices-root Path, or None if no candidate matches.
        A None return should be treated as a configuration error by
        the caller (argparse default + explicit error message).
    """
    candidates: list[Path] = []
    env_value = os.environ.get("PIPER_VOICES_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            Path.cwd() / "piper-voices",
            Path.home() / "piper-voices",
        ],
    )
    for candidate in candidates:
        if _looks_like_piper_voices_tree(candidate):
            return candidate
        if candidate.is_dir():
            logger.debug(
                "Candidate %s exists but does not contain the expected en/en_*/ layout; skipping.",
                candidate,
            )
    return None


def resolve_piper_bin() -> Path | None:
    """Locate the piper binary on the current machine.

    Resolution order (first existing path wins):

    1. ``$PIPER_BIN`` environment variable, if set and pointing at an
       existing file.
    2. ``piper`` on the user's ``$PATH`` (via ``shutil.which``). This
       finds the binary installed by ``pip install piper-tts``, by
       ``uv pip install piper-tts``, or by package managers like
       Homebrew.

    This function contains no developer-specific absolute paths.
    Contributors whose piper binary is installed at a non-standard
    location must either set ``$PIPER_BIN``, put the binary on
    ``$PATH``, or pass ``--piper-bin`` on the CLI.

    Returns:
        The resolved piper binary Path, or None if none exists.
    """
    env_value = os.environ.get("PIPER_BIN")
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_file():
            return env_path
    which_path = shutil.which("piper")
    if which_path:
        return Path(which_path)
    return None


@dataclass(frozen=True)
class TextFixture:
    """A single text fixture loaded from the JSONL file.

    Only includes fields needed for audio synthesis. Preserves the
    full record for manifest cross-referencing via the `raw` attribute.
    """

    id: str
    text: str
    language: str
    raw: dict[str, object]


def load_fixtures(path: Path) -> list[TextFixture]:
    """Load text fixtures from a JSONL file.

    Args:
        path: Path to `rt_dictation_samples.jsonl`.

    Returns:
        List of TextFixture objects in file order.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If any line is not valid JSON or missing required fields.
    """
    if not path.exists():
        msg = f"Fixtures file not found: {path}"
        raise FileNotFoundError(msg)

    fixtures: list[TextFixture] = []
    with path.open(encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON on line {line_num}: {e}"
                raise ValueError(msg) from e

            for required in ("id", "text"):
                if required not in record:
                    msg = f"Missing required field '{required}' on line {line_num}"
                    raise ValueError(msg)

            # Per SCHEMA.md, id/text/language must be strings. Validate
            # rather than silently coercing — schema violations should be
            # loud so they're caught early in fixture creation, not hidden
            # behind str() coercion of ints or other types.
            for string_field in ("id", "text"):
                if not isinstance(record[string_field], str):
                    msg = f"Field '{string_field}' on line {line_num} is {type(record[string_field]).__name__}, expected str"
                    raise TypeError(msg)
            language_raw = record.get("language", "en")
            if not isinstance(language_raw, str):
                msg = f"Field 'language' on line {line_num} is {type(language_raw).__name__}, expected str"
                raise TypeError(msg)

            fixtures.append(
                TextFixture(
                    id=record["id"],
                    text=record["text"],
                    language=language_raw,
                    raw=record,
                )
            )

    logger.info("Loaded %d fixtures from %s", len(fixtures), path)
    return fixtures


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 mono audio using scipy polyphase filtering.

    Uses scipy.signal.resample_poly for high-quality rate conversion.
    scipy is a hard dependency of the audio extra — there is no silent
    fallback. Silent quality degradation in medical code is a footgun.

    Args:
        audio: Int16 mono audio samples.
        src_rate: Source sample rate (Hz).
        dst_rate: Target sample rate (Hz).

    Returns:
        Int16 mono audio at dst_rate.

    Raises:
        ImportError: If scipy is not installed (audio extra not present).
    """
    if src_rate == dst_rate:
        return audio

    from math import gcd

    from scipy.signal import resample_poly

    common = gcd(src_rate, dst_rate)
    up = dst_rate // common
    down = src_rate // common
    # resample_poly expects float; convert then cast back
    resampled_float: np.ndarray = resample_poly(audio.astype(np.float64), up, down)
    clipped: np.ndarray = np.clip(resampled_float, -32768, 32767).astype(np.int16)
    return clipped


def synthesize_one(
    text: str,
    voice_model: Path,
    *,
    piper_cmd: list[str] | None = None,
    speaker_id: int | None = None,
) -> np.ndarray:
    """Synthesize a single text sample with piper and return int16 PCM at 22050 Hz.

    Args:
        text: Text to synthesize.
        voice_model: Path to piper ONNX voice model.
        piper_cmd: Piper command prefix (default: ["piper"]).
        speaker_id: Optional speaker index for multi-speaker models (e.g. L2-Arctic,
            VCTK). When set, ``--speaker N`` is appended to the piper command.

    Returns:
        Int16 mono PCM samples at 22050 Hz (piper's native rate).

    Raises:
        RuntimeError: If piper fails or produces no audio.
    """
    cmd_prefix = list(piper_cmd) if piper_cmd is not None else ["piper"]
    cmd = [*cmd_prefix, "--model", str(voice_model), "--output_raw"]
    if speaker_id is not None:
        cmd = [*cmd, "--speaker", str(speaker_id)]

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        msg = f"piper failed (exit {e.returncode}): {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = f"piper timed out after 60s on text: {text[:80]}"
        raise RuntimeError(msg) from e

    if not result.stdout:
        msg = f"piper produced no audio for text: {text[:80]}"
        raise RuntimeError(msg)

    return np.frombuffer(result.stdout, dtype=np.int16)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write int16 mono audio to a WAV file.

    Args:
        path: Output path.
        audio: Int16 mono samples.
        sample_rate: Sample rate in Hz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_BIT_DEPTH // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def build_manifest_entry(
    fixture: TextFixture,
    audio_path: Path,
    voice_name: str,
    audio_samples: np.ndarray,
    repo_root: Path,
    *,
    speaker_id: int | None = None,
) -> dict[str, object]:
    """Build an audio_manifest.jsonl entry for a synthesized file.

    Conforms to tests/validation/SCHEMA.md audio manifest schema.

    Args:
        fixture: The text fixture synthesized.
        audio_path: Absolute path to written WAV file.
        voice_name: Piper voice identifier (e.g. "en_US-amy-medium"). When
            ``speaker_id`` is also supplied, it is appended to form a unique
            identifier such as ``"en_US-l2arctic-medium-speaker3"``.
        audio_samples: The int16 audio written (at TARGET_SAMPLE_RATE).
        repo_root: Repository root for computing relative paths.
        speaker_id: Optional speaker index for multi-speaker models. When set,
            the ``tts_voice`` field is extended with a ``-speakerN`` suffix so
            that per-speaker results can be distinguished in the manifest.

    Returns:
        Dict matching the audio manifest schema.
    """
    duration = len(audio_samples) / TARGET_SAMPLE_RATE
    try:
        relative_path = audio_path.relative_to(repo_root)
    except ValueError:
        relative_path = audio_path

    tts_voice = f"{voice_name}-speaker{speaker_id}" if speaker_id is not None else voice_name

    return {
        "audio_id": f"{fixture.id}-piper-{tts_voice}-clean",
        "text_id": fixture.id,
        "audio_path": str(relative_path),
        "tier": "clean",
        "tts_engine": "piper",
        "tts_voice": tts_voice,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "duration_seconds": round(duration, 3),
        "channels": TARGET_CHANNELS,
        "bit_depth": TARGET_BIT_DEPTH,
        "acoustic_simulation": None,
        "noise_profile": None,
        "snr_db": None,
    }


def synthesize_fixtures(
    fixtures: Iterable[TextFixture],
    voice_model: Path,
    output_dir: Path,
    *,
    voice_name: str | None = None,
    piper_cmd: list[str] | None = None,
    repo_root: Path | None = None,
    speaker_id: int | None = None,
) -> list[dict[str, object]]:
    """Synthesize audio for multiple fixtures and return manifest entries.

    Args:
        fixtures: Text fixtures to synthesize.
        voice_model: Path to piper ONNX voice model.
        output_dir: Directory to write WAV files into.
        voice_name: Voice identifier for manifest (default: model filename stem).
        piper_cmd: Piper command prefix (default: ["piper"]).
        repo_root: Repo root for relative paths in manifest.
        speaker_id: Optional speaker index for multi-speaker models (e.g.
            L2-Arctic, VCTK). Forwarded to ``synthesize_one`` (adds
            ``--speaker N`` to piper command) and ``build_manifest_entry``
            (appends ``-speakerN`` suffix to ``tts_voice``).

    Returns:
        List of manifest entries (one per successful synthesis).
    """
    if voice_name is None:
        voice_name = voice_model.stem
    if repo_root is None:
        repo_root = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for fixture in fixtures:
        try:
            raw_audio = synthesize_one(fixture.text, voice_model, piper_cmd=piper_cmd, speaker_id=speaker_id)
        except RuntimeError:
            logger.exception("Failed to synthesize %s", fixture.id)
            continue

        resampled = _resample_linear(raw_audio, PIPER_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        # Include speaker suffix in filename when dealing with multi-speaker models
        # so that different speakers' WAVs don't collide in the same output_dir.
        file_voice_tag = f"{voice_name}-speaker{speaker_id}" if speaker_id is not None else voice_name
        audio_path = output_dir / f"{fixture.id}-piper-{file_voice_tag}.wav"
        write_wav(audio_path, resampled, TARGET_SAMPLE_RATE)

        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=audio_path,
            voice_name=voice_name,
            audio_samples=resampled,
            repo_root=repo_root,
            speaker_id=speaker_id,
        )
        entries.append(entry)
        logger.info("Synthesized %s (%.2fs)", fixture.id, entry["duration_seconds"])

    return entries


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    """Write audio manifest as JSONL.

    Args:
        path: Output path for `audio_manifest.jsonl`.
        entries: Manifest entries to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    logger.info("Wrote manifest with %d entries to %s", len(entries), path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for piper TTS batch synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesize validation audio with piper (clean tier)",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Path to rt_dictation_samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write WAV files into",
    )
    parser.add_argument(
        "--voice-model",
        type=Path,
        required=True,
        help="Path to piper ONNX voice model",
    )
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Voice identifier for manifest (default: model filename stem)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fixtures to synthesize (for testing)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path (default: <output-dir>/audio_manifest.jsonl)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for relative paths (default: cwd)",
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

    fixtures = load_fixtures(args.fixtures)
    if args.limit is not None:
        fixtures = fixtures[: args.limit]

    entries = synthesize_fixtures(
        fixtures,
        voice_model=args.voice_model,
        output_dir=args.output_dir,
        voice_name=args.voice_name,
        repo_root=args.repo_root,
    )

    manifest_path = args.manifest or (args.output_dir / "audio_manifest.jsonl")
    write_manifest(manifest_path, entries)

    print(f"Synthesized {len(entries)}/{len(fixtures)} fixtures", file=sys.stderr)
    return 0 if len(entries) == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
