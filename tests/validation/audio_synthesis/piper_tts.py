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

            fixtures.append(
                TextFixture(
                    id=str(record["id"]),
                    text=str(record["text"]),
                    language=str(record.get("language", "en")),
                    raw=record,
                )
            )

    logger.info("Loaded %d fixtures from %s", len(fixtures), path)
    return fixtures


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 mono audio using polyphase filtering.

    Uses scipy.signal.resample_poly for high-quality rate conversion.
    Falls back to linear interpolation if scipy is unavailable.

    Args:
        audio: Int16 mono audio samples.
        src_rate: Source sample rate (Hz).
        dst_rate: Target sample rate (Hz).

    Returns:
        Int16 mono audio at dst_rate.
    """
    if src_rate == dst_rate:
        return audio

    try:
        from math import gcd

        from scipy.signal import resample_poly

        common = gcd(src_rate, dst_rate)
        up = dst_rate // common
        down = src_rate // common
        # resample_poly expects float; convert then cast back
        resampled_float = resample_poly(audio.astype(np.float64), up, down)
        return np.clip(resampled_float, -32768, 32767).astype(np.int16)
    except ImportError:
        logger.warning(
            "scipy not available, using linear interpolation for resampling (quality reduced)",
        )
        src_len = len(audio)
        dst_len = round(src_len * dst_rate / src_rate)
        if dst_len == 0:
            return np.zeros(0, dtype=np.int16)
        x_src = np.arange(src_len, dtype=np.float64)
        x_dst = np.linspace(0, src_len - 1, dst_len, dtype=np.float64)
        interp = np.interp(x_dst, x_src, audio.astype(np.float64))
        return np.clip(interp, -32768, 32767).astype(np.int16)


def synthesize_one(
    text: str,
    voice_model: Path,
    *,
    piper_cmd: list[str] | None = None,
) -> np.ndarray:
    """Synthesize a single text sample with piper and return int16 PCM at 22050 Hz.

    Args:
        text: Text to synthesize.
        voice_model: Path to piper ONNX voice model.
        piper_cmd: Piper command prefix (default: ["piper"]).

    Returns:
        Int16 mono PCM samples at 22050 Hz (piper's native rate).

    Raises:
        RuntimeError: If piper fails or produces no audio.
    """
    cmd = [*(piper_cmd or ["piper"]), "--model", str(voice_model), "--output_raw"]

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
) -> dict[str, object]:
    """Build an audio_manifest.jsonl entry for a synthesized file.

    Conforms to tests/validation/SCHEMA.md audio manifest schema.

    Args:
        fixture: The text fixture synthesized.
        audio_path: Absolute path to written WAV file.
        voice_name: Piper voice identifier (e.g. "en_US-amy-medium").
        audio_samples: The int16 audio written (at TARGET_SAMPLE_RATE).
        repo_root: Repository root for computing relative paths.

    Returns:
        Dict matching the audio manifest schema.
    """
    duration = len(audio_samples) / TARGET_SAMPLE_RATE
    try:
        relative_path = audio_path.relative_to(repo_root)
    except ValueError:
        relative_path = audio_path

    return {
        "audio_id": f"{fixture.id}-piper-{voice_name}-clean",
        "text_id": fixture.id,
        "audio_path": str(relative_path),
        "tier": "clean",
        "tts_engine": "piper",
        "tts_voice": voice_name,
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
) -> list[dict[str, object]]:
    """Synthesize audio for multiple fixtures and return manifest entries.

    Args:
        fixtures: Text fixtures to synthesize.
        voice_model: Path to piper ONNX voice model.
        output_dir: Directory to write WAV files into.
        voice_name: Voice identifier for manifest (default: model filename stem).
        piper_cmd: Piper command prefix (default: ["piper"]).
        repo_root: Repo root for relative paths in manifest.

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
            raw_audio = synthesize_one(fixture.text, voice_model, piper_cmd=piper_cmd)
        except RuntimeError:
            logger.exception("Failed to synthesize %s", fixture.id)
            continue

        resampled = _resample_linear(raw_audio, PIPER_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        audio_path = output_dir / f"{fixture.id}-piper-{voice_name}.wav"
        write_wav(audio_path, resampled, TARGET_SAMPLE_RATE)

        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=audio_path,
            voice_name=voice_name,
            audio_samples=resampled,
            repo_root=repo_root,
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
