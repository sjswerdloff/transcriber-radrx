"""macOS TTS pipeline for validation audio generation (clean tier).

Uses macOS's built-in ``say`` command and ``afconvert`` to synthesize
audio for bake-off voices that piper does not cover: Australian, Irish,
Indian, South African, and other Commonwealth English accents.

Pipeline: ``say -v <voice> -o temp.aiff -- <text>``
       → ``afconvert -f WAVE -d LEI16@16000 -c 1 temp.aiff output.wav``
       → read WAV as int16 numpy array.

Both ``say`` and ``afconvert`` are macOS builtins. No additional PATH
configuration is required. This module raises ``RuntimeError`` on
non-Darwin platforms so callers discover the dependency immediately.

The output contract is identical to ``piper_tts.py``: each entry in the
returned list matches the audio manifest schema in
``tests/validation/SCHEMA.md``, and every WAV file is 16 kHz mono
int16.

Usage:
    python -m tests.validation.audio_synthesis.macos_tts \\
        --fixtures tests/validation/fixtures/rt_dictation_samples.jsonl \\
        --output-dir tests/validation/audio/synthetic/macos-clean \\
        --voice Karen

Authors: silas-397300f6
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.validation.audio_synthesis.piper_tts import TextFixture  # noqa: TCH001

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# ASR models (Whisper, Parakeet, etc.) expect 16 kHz input.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = 16

# macOS ``say`` novelty voices to exclude — these are sound effects,
# not realistic speech voices. A voice is considered novelty if its
# name is in this set (case-insensitive).
_NOVELTY_VOICE_NAMES: frozenset[str] = frozenset(
    {
        "bells",
        "boing",
        "bubbles",
        "cellos",
        "jester",
        "organ",
        "superstar",
        "trinoids",
        "whisper",
        "wobble",
        "zarvox",
        "bad news",
        "bahh",
        "deranged",
        "good news",
        "hysterical",
        "pipe organ",
        "princess",
        "ralph",
    }
)


def _check_platform() -> None:
    """Raise RuntimeError if not running on macOS (Darwin).

    Raises:
        RuntimeError: If sys.platform is not 'darwin'.
    """
    if sys.platform != "darwin":
        msg = (
            "macos_tts requires macOS (sys.platform='darwin'). "
            f"Current platform: {sys.platform!r}. "
            "Use piper_tts for cross-platform synthesis."
        )
        raise RuntimeError(msg)


# Module-level platform guard — fails immediately on import on non-Darwin.
_check_platform()


def list_available_voices() -> list[tuple[str, str]]:
    """Parse ``say -v '?'`` output and return English voice entries.

    Queries the macOS ``say`` command for the list of installed voices,
    filters to English (``en_*``) locales, and excludes novelty voices
    (sound effects rather than realistic speech).

    Returns:
        List of ``(voice_name, locale)`` tuples for realistic English
        voices, e.g. ``[("Karen", "en_AU"), ("Daniel", "en_GB"), ...]``.

    Raises:
        RuntimeError: If ``say -v '?'`` fails.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["say", "-v", "?"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = f"say -v '?' failed (exit {e.returncode}): {e.stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = "say -v '?' timed out after 10s"
        raise RuntimeError(msg) from e

    voices: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        # Each line has the form:
        #   VoiceName          locale    # Sample sentence
        # e.g.:
        #   Karen              en_AU     # G'day! My name is Karen...
        # Voice names with spaces (e.g. "Matilda (Premium)") fill the
        # first column followed by the locale. We split on the locale
        # pattern (two letters, underscore, two letters).
        line = line.strip()
        if not line:
            continue

        # Find locale token — the first token matching xx_XX pattern.
        parts = line.split()
        locale: str | None = None
        locale_idx: int = -1
        for i, token in enumerate(parts):
            # Locale tokens look like en_AU, fr_FR, zh_TW etc.
            if len(token) == 5 and token[2] == "_" and token[:2].isalpha() and token[3:].isalpha():
                locale = token
                locale_idx = i
                break

        if locale is None or not locale.startswith("en_"):
            continue

        # Voice name is everything before the locale token.
        voice_name = " ".join(parts[:locale_idx]).strip()
        if not voice_name:
            continue

        # Exclude novelty/effect voices by name (case-insensitive).
        if voice_name.lower() in _NOVELTY_VOICE_NAMES:
            logger.debug("Excluding novelty voice: %r", voice_name)
            continue

        voices.append((voice_name, locale))

    return voices


def synthesize_one(
    text: str,
    voice_name: str,
    *,
    say_cmd: list[str] | None = None,
) -> np.ndarray:
    """Synthesize a single text sample with macOS ``say`` and return int16 PCM at 16kHz.

    Pipeline:
        1. ``say -v <voice> -o temp.aiff -- <text>`` → AIFF at the
           voice's native sample rate.
        2. ``afconvert -f WAVE -d LEI16@16000 -c 1 temp.aiff out.wav``
           → 16 kHz mono signed-16-bit WAV.
        3. Read the WAV back as a numpy int16 array.

    The ``--`` separator before ``<text>`` prevents text that begins with
    a hyphen from being parsed as a flag by ``say``.

    Args:
        text: Text to synthesize.
        voice_name: macOS voice name (e.g. "Karen", "Daniel", "Rishi").
        say_cmd: Override for the ``say`` command prefix. Useful in tests
            to inject a mock or an alternative path.

    Returns:
        Int16 mono PCM samples at ``TARGET_SAMPLE_RATE`` (16000 Hz).

    Raises:
        RuntimeError: If ``say`` or ``afconvert`` fails, or if the
            resulting WAV is empty.
    """
    cmd_prefix = list(say_cmd) if say_cmd is not None else ["say"]

    with (
        tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as aiff_tmp,
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp,
    ):
        aiff_path = Path(aiff_tmp.name)
        wav_path = Path(wav_tmp.name)

    try:
        _say_to_aiff(cmd_prefix, voice_name, text, aiff_path)
        _afconvert_to_wav(aiff_path, wav_path)
        return _read_wav_as_int16(wav_path, text)
    finally:
        aiff_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


def _say_to_aiff(
    cmd_prefix: list[str],
    voice_name: str,
    text: str,
    aiff_path: Path,
) -> None:
    """Run ``say -v <voice> -o <aiff_path> -- <text>``."""
    cmd = [*cmd_prefix, "-v", voice_name, "-o", str(aiff_path), "--", text]
    try:
        subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        msg = f"say failed (exit {e.returncode}): {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = f"say timed out after 60s on text: {text[:80]!r}"
        raise RuntimeError(msg) from e


def _afconvert_to_wav(aiff_path: Path, wav_path: Path) -> None:
    """Convert an AIFF to 16kHz mono int16 WAV via ``afconvert``."""
    cmd = [
        "afconvert",
        "-f",
        "WAVE",
        "-d",
        f"LEI16@{TARGET_SAMPLE_RATE}",
        "-c",
        "1",
        str(aiff_path),
        str(wav_path),
    ]
    try:
        subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        msg = f"afconvert failed (exit {e.returncode}): {stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = f"afconvert timed out after 60s for {aiff_path}"
        raise RuntimeError(msg) from e


def _read_wav_as_int16(wav_path: Path, original_text: str) -> np.ndarray:
    """Read a WAV file and return samples as int16 numpy array."""
    with wave.open(str(wav_path), "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())

    if not raw_bytes:
        msg = f"afconvert produced an empty WAV for text: {original_text[:80]!r}"
        raise RuntimeError(msg)

    return np.frombuffer(raw_bytes, dtype=np.int16)


def build_manifest_entry(
    fixture: TextFixture,
    audio_path: Path,
    voice_name: str,
    audio_samples: np.ndarray,
    repo_root: Path,
) -> dict[str, object]:
    """Build an audio_manifest.jsonl entry for a macOS TTS synthesized file.

    Conforms to tests/validation/SCHEMA.md audio manifest schema.
    The ``tts_engine`` field is set to ``"macos_say"`` to distinguish
    these entries from piper-generated entries.

    Args:
        fixture: The text fixture that was synthesized.
        audio_path: Absolute path to the written WAV file.
        voice_name: macOS voice name (e.g. "Karen", "Daniel").
        audio_samples: The int16 audio written (at ``TARGET_SAMPLE_RATE``).
        repo_root: Repository root for computing relative paths.

    Returns:
        Dict matching the audio manifest schema with ``tts_engine="macos_say"``.
    """
    duration = len(audio_samples) / TARGET_SAMPLE_RATE
    try:
        relative_path = audio_path.relative_to(repo_root)
    except ValueError:
        relative_path = audio_path

    return {
        "audio_id": f"{fixture.id}-macos_say-{voice_name}-clean",
        "text_id": fixture.id,
        "audio_path": str(relative_path),
        "tier": "clean",
        "tts_engine": "macos_say",
        "tts_voice": voice_name,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "duration_seconds": round(duration, 3),
        "channels": TARGET_CHANNELS,
        "bit_depth": TARGET_BIT_DEPTH,
        "acoustic_simulation": None,
        "noise_profile": None,
        "snr_db": None,
    }


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


def synthesize_fixtures(
    fixtures: Iterable[TextFixture],
    voice_name: str,
    output_dir: Path,
    *,
    display_name: str | None = None,
    say_cmd: list[str] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Synthesize audio for multiple fixtures and return manifest entries.

    Args:
        fixtures: Text fixtures to synthesize.
        voice_name: macOS voice name for synthesis (e.g. "Karen").
        output_dir: Directory to write WAV files into.
        display_name: Voice identifier for manifest entries. Defaults
            to ``voice_name`` if not provided.
        say_cmd: Override for the ``say`` command prefix (used in tests).
        repo_root: Repo root for relative paths in manifest. Defaults
            to the current working directory.

    Returns:
        List of manifest entries (one per successfully synthesized fixture).
    """
    effective_display = display_name if display_name is not None else voice_name
    effective_repo_root = repo_root if repo_root is not None else Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for fixture in fixtures:
        try:
            audio = synthesize_one(fixture.text, voice_name, say_cmd=say_cmd)
        except RuntimeError:
            logger.exception("Failed to synthesize %s with voice %r", fixture.id, voice_name)
            continue

        audio_path = output_dir / f"{fixture.id}-macos_say-{effective_display}.wav"
        write_wav(audio_path, audio, TARGET_SAMPLE_RATE)

        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=audio_path,
            voice_name=effective_display,
            audio_samples=audio,
            repo_root=effective_repo_root,
        )
        entries.append(entry)
        logger.info("Synthesized %s (%.2fs)", fixture.id, entry["duration_seconds"])

    return entries


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    """Write audio manifest as JSONL.

    Args:
        path: Output path for ``audio_manifest.jsonl``.
        entries: Manifest entries to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    logger.info("Wrote manifest with %d entries to %s", len(entries), path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for macOS TTS batch synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesize validation audio with macOS say (clean tier)",
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
        "--voice",
        required=True,
        help="macOS voice name (e.g. 'Karen', 'Daniel', 'Rishi')",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Voice identifier for manifest (default: voice name)",
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
        "--list-voices",
        action="store_true",
        help="List available English voices and exit",
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

    if args.list_voices:
        voices = list_available_voices()
        for name, locale in sorted(voices, key=lambda t: (t[1], t[0])):
            print(f"  {name:<30} {locale}")
        return 0

    from tests.validation.audio_synthesis.piper_tts import load_fixtures

    fixtures = load_fixtures(args.fixtures)
    if args.limit is not None:
        fixtures = fixtures[: args.limit]

    entries = synthesize_fixtures(
        fixtures,
        voice_name=args.voice,
        output_dir=args.output_dir,
        display_name=args.display_name,
        repo_root=args.repo_root,
    )

    manifest_path = args.manifest or (args.output_dir / "audio_manifest.jsonl")
    write_manifest(manifest_path, entries)

    print(f"Synthesized {len(entries)}/{len(fixtures)} fixtures", file=sys.stderr)
    return 0 if len(entries) == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
