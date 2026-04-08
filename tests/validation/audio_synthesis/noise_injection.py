"""Additive background noise injection for validation audio (noisy tier).

Takes clean-tier WAV files and mixes in background noise sampled from the
MUSAN corpus at a target signal-to-noise ratio (SNR). Produces the 'noisy'
tier of the validation audio manifest: clean speech with realistic ambient
noise overlay, approximating what an ASR system encounters in real clinical
environments (fans, HVAC, distant conversation, equipment hum).

Noise presets model different acoustic environments by SNR level:

- quiet: 20 dB SNR, quiet office / dictation booth
- moderate: 10 dB SNR, typical clinic room with HVAC and nearby activity
- busy: 5 dB SNR, busy hallway / LINAC vault with equipment running

Coverage policy (what plays during the signal):

1. Prefer single-file coverage. If any noise clip in the pool is at least
   as long as the signal, pick one at random and crop. No seams.
2. Otherwise splice multiple *different* clips with a short linear
   crossfade (default 30 ms) at each seam until the full signal duration
   is covered. Sampling is without replacement until the pool is
   exhausted, then refilled. Repeating the same short clip across a
   whole dictation would be unrealistic and introduces audible
   click-per-loop artefacts, so we avoid it.

Composable with acoustic_sim.py: clean → acoustic → noisy is supported
via CLI by running each stage in sequence.

Authors: silas-397300f6 (for review by vivian-1a61bc9a, owner of
audio_synthesis)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.validation.audio_synthesis.acoustic_sim import read_wav, write_wav

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Int16 full-scale value for float<->int conversion.
INT16_FULL_SCALE = 32768.0
INT16_MAX = 32767
INT16_MIN = -32768
# Headroom kept below int16 peak to avoid clipping artefacts after mixing.
MIX_PEAK_TARGET = 0.95
# RMS floor for signals that are effectively silent; prevents division by
# zero when scaling noise relative to signal level.
RMS_FLOOR = 1e-9
# Default crossfade duration at splice seams (milliseconds). 30 ms is long
# enough to hide click artefacts at 16 kHz without audibly blurring the
# noise character at the seam.
DEFAULT_CROSSFADE_MS = 30.0


@dataclass(frozen=True)
class NoisePreset:
    """Named noise injection configuration.

    Attributes:
        name: Identifier used in manifest entries.
        snr_db: Target signal-to-noise ratio in decibels. Higher = cleaner.
        description: Human-readable clinical context for this preset.
    """

    name: str
    snr_db: float
    description: str


NOISE_PRESETS: dict[str, NoisePreset] = {
    "quiet": NoisePreset(
        name="quiet",
        snr_db=20.0,
        description="quiet office or dictation booth",
    ),
    "moderate": NoisePreset(
        name="moderate",
        snr_db=10.0,
        description="typical clinic room with HVAC and nearby activity",
    ),
    "busy": NoisePreset(
        name="busy",
        snr_db=5.0,
        description="busy hallway or LINAC vault with equipment running",
    ),
}


@dataclass(frozen=True)
class NoiseClipInfo:
    """Probed metadata for a noise WAV file.

    Attributes:
        path: Absolute path to the WAV file.
        samples: Number of audio samples (frames) in the file.
        sample_rate: Sample rate in Hz.
    """

    path: Path
    samples: int
    sample_rate: int


def rms(audio: np.ndarray) -> float:
    """Compute the root-mean-square amplitude of an audio signal.

    Args:
        audio: Float or int audio samples.

    Returns:
        RMS amplitude as float. Returns 0.0 for an empty array.
    """
    if audio.size == 0:
        return 0.0
    audio_float = audio.astype(np.float64)
    return float(np.sqrt(np.mean(audio_float * audio_float)))


def list_noise_files(noise_dir: Path, categories: tuple[str, ...] | None = None) -> list[Path]:
    """Enumerate noise WAV files under a MUSAN noise directory.

    Args:
        noise_dir: Path to musan/noise (containing free-sound/, sound-bible/).
        categories: Subdirectory names to include. None = include all.

    Returns:
        Sorted list of WAV file paths.

    Raises:
        FileNotFoundError: If noise_dir does not exist.
        ValueError: If no WAV files are found under the requested categories.
    """
    if not noise_dir.is_dir():
        msg = f"Noise directory not found: {noise_dir}"
        raise FileNotFoundError(msg)

    if categories is None:
        files = sorted(noise_dir.rglob("*.wav"))
    else:
        files = []
        for category in categories:
            cat_dir = noise_dir / category
            if not cat_dir.is_dir():
                logger.warning("Noise category directory missing: %s", cat_dir)
                continue
            files.extend(sorted(cat_dir.rglob("*.wav")))
        files.sort()

    if not files:
        msg = f"No noise WAV files found under {noise_dir} (categories={categories})"
        raise ValueError(msg)

    return files


def probe_noise_clips(files: list[Path], expected_rate: int) -> list[NoiseClipInfo]:
    """Read WAV headers to collect clip length metadata.

    Only opens the header, not the audio body, so this is cheap even for
    the full MUSAN corpus (~930 files). Clips whose sample rate does not
    match ``expected_rate`` are skipped with a warning rather than
    crashing the whole run.

    Args:
        files: Candidate WAV files.
        expected_rate: Sample rate that all returned clips must match.

    Returns:
        List of NoiseClipInfo for compatible files, in input order.

    Raises:
        ValueError: If no files remain after filtering by sample rate.
    """
    clips: list[NoiseClipInfo] = []
    for path in files:
        try:
            with wave.open(str(path), "rb") as wf:
                rate = wf.getframerate()
                samples = wf.getnframes()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
        except wave.Error:
            logger.warning("Failed to probe noise clip header: %s", path)
            continue
        if rate != expected_rate:
            logger.warning(
                "Skipping noise clip at %d Hz (expected %d Hz): %s",
                rate,
                expected_rate,
                path,
            )
            continue
        if channels != 1:
            logger.warning("Skipping non-mono noise clip (%d channels): %s", channels, path)
            continue
        if width != 2:
            logger.warning("Skipping non-16-bit noise clip (%d bytes/sample): %s", width, path)
            continue
        clips.append(NoiseClipInfo(path=path, samples=samples, sample_rate=rate))

    if not clips:
        msg = f"No noise clips found at rate {expected_rate} Hz across {len(files)} candidates"
        raise ValueError(msg)

    return clips


def crossfade_concat(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Concatenate two int16 signals with a linear crossfade at the seam.

    The last ``fade_samples`` of ``a`` are faded out while the first
    ``fade_samples`` of ``b`` are faded in, and the overlapping region
    is summed. This eliminates audible click artefacts that arise from
    a hard concatenation of two unrelated noise samples.

    Output length is ``len(a) + len(b) - fade_samples`` in the crossfaded
    case. If either input is shorter than ``fade_samples`` the inputs
    are simply concatenated (no fade possible).

    Args:
        a: First int16 mono signal.
        b: Second int16 mono signal.
        fade_samples: Length of the linear crossfade in samples.

    Returns:
        Int16 signal containing ``a`` followed by ``b`` with a smooth
        seam.
    """
    if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b])

    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    tail = a[-fade_samples:].astype(np.float64) * fade_out
    head = b[:fade_samples].astype(np.float64) * fade_in
    seam = tail + head

    joined = np.concatenate(
        [
            a[:-fade_samples].astype(np.float64),
            seam,
            b[fade_samples:].astype(np.float64),
        ],
    )
    joined_int16: np.ndarray = np.clip(joined, INT16_MIN, INT16_MAX).astype(np.int16)
    return joined_int16


def build_noise_coverage(
    clips: list[NoiseClipInfo],
    target_samples: int,
    rng: np.random.Generator,
    crossfade_samples: int = 0,
) -> tuple[np.ndarray, list[Path]]:
    """Construct a noise sequence that exactly covers the target length.

    Selection strategy:

    1. **Prefer single-file coverage**: if any clip in ``clips`` is at
       least ``target_samples`` long, pick one uniformly at random from
       that eligible pool, crop from a random start offset, and return
       it. This is the cleanest and most realistic case — one continuous
       noise stream with no splice seams.
    2. **Splice different clips**: if no single clip is long enough,
       draw clips uniformly at random *without replacement* from the
       pool and crossfade-concatenate them until the accumulated length
       reaches ``target_samples``. If the pool is exhausted before
       coverage is reached (rare with MUSAN's 930 clips but possible
       for extremely long signals), the pool is refilled and sampling
       continues. The result is cropped to exactly ``target_samples``.

    Single-clip looping is never used here; repeating the same short
    clip across an entire dictation sounds unrealistic and introduces
    a click at every loop boundary.

    Args:
        clips: Pool of probed noise clips (all same sample rate).
        target_samples: Desired output length in samples.
        rng: Random generator for clip selection and crop offsets.
        crossfade_samples: Linear crossfade length in samples, applied
            at every splice seam. Set to 0 to disable.

    Returns:
        Tuple of (int16 noise array of length ``target_samples``,
        list of file paths used in order).

    Raises:
        ValueError: If ``clips`` is empty or ``target_samples`` is not
            positive.
    """
    if not clips:
        msg = "Cannot build noise coverage from an empty clip pool"
        raise ValueError(msg)
    if target_samples <= 0:
        msg = f"target_samples must be positive, got {target_samples}"
        raise ValueError(msg)

    # 1. Single-file coverage from the eligible pool.
    eligible = [c for c in clips if c.samples >= target_samples]
    if eligible:
        chosen = eligible[int(rng.integers(0, len(eligible)))]
        audio, _ = read_wav(chosen.path)
        max_offset = chosen.samples - target_samples
        start = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
        cropped: np.ndarray = audio[start : start + target_samples].copy()
        return cropped, [chosen.path]

    # 2. Splice path: draw without replacement, crossfade, repeat until
    # coverage.
    segments: list[np.ndarray] = []
    used: list[Path] = []
    accumulated = 0
    available: list[NoiseClipInfo] = list(clips)

    while accumulated < target_samples:
        if not available:
            # Pool exhausted before coverage achieved; refill.
            available = list(clips)
        idx = int(rng.integers(0, len(available)))
        chosen = available.pop(idx)
        audio, _ = read_wav(chosen.path)
        segments.append(audio)
        used.append(chosen.path)
        # Accumulated length after crossfade-concatenating n segments is
        # sum(len) - (n-1)*crossfade (each seam overlaps by crossfade).
        accumulated = sum(int(s.shape[0]) for s in segments) - max(0, len(segments) - 1) * crossfade_samples

    joined: np.ndarray = segments[0]
    for seg in segments[1:]:
        joined = crossfade_concat(joined, seg, crossfade_samples)

    if len(joined) < target_samples:
        msg = (
            "Splice did not produce enough samples "
            f"({len(joined)} < {target_samples}) — this is a bug in the "
            "coverage estimator."
        )
        raise RuntimeError(msg)

    out: np.ndarray = joined[:target_samples].copy()
    return out, used


def mix_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, float]:
    """Mix additive noise with a clean signal at a target SNR.

    The noise is scaled so that 20 * log10(signal_rms / scaled_noise_rms)
    equals target_snr_db. If the mixed signal would clip the int16 range,
    the mixed signal (signal + scaled noise together) is attenuated
    uniformly to keep the peak within MIX_PEAK_TARGET of full scale; this
    preserves the SNR ratio.

    Args:
        signal: Int16 mono clean speech.
        noise: Int16 mono noise, same length as signal.
        target_snr_db: Target SNR in dB. Higher = cleaner speech.

    Returns:
        Tuple of (mixed int16 audio, achieved SNR in dB). The achieved
        SNR is the value actually realised after any post-mix gain
        scaling. Uniform gain preserves the ratio exactly, so
        achieved_snr_db == target_snr_db in normal operation.

    Raises:
        ValueError: If signal and noise lengths differ.
    """
    if signal.shape != noise.shape:
        msg = f"signal and noise shapes differ: {signal.shape} vs {noise.shape}"
        raise ValueError(msg)

    signal_f = signal.astype(np.float64)
    noise_f = noise.astype(np.float64)

    signal_rms = rms(signal_f)
    noise_rms = rms(noise_f)

    if signal_rms < RMS_FLOOR:
        logger.warning("Signal RMS near zero; cannot meaningfully mix noise.")
        return signal.copy(), float("inf")
    if noise_rms < RMS_FLOOR:
        logger.warning("Noise RMS near zero; returning clean signal unchanged.")
        return signal.copy(), float("inf")

    # Solve: target_snr_db = 20 * log10(signal_rms / (noise_rms * gain))
    #   => gain = signal_rms / (noise_rms * 10**(target_snr_db / 20))
    gain = signal_rms / (noise_rms * (10.0 ** (target_snr_db / 20.0)))
    scaled_noise = noise_f * gain

    mixed = signal_f + scaled_noise

    # Peak-limit to avoid int16 clipping. Uniform gain preserves SNR.
    peak = float(np.abs(mixed).max())
    target_peak = MIX_PEAK_TARGET * INT16_FULL_SCALE
    if peak > target_peak:
        attenuation = target_peak / peak
        mixed = mixed * attenuation

    mixed_int16 = np.clip(mixed, INT16_MIN, INT16_MAX).astype(np.int16)
    return mixed_int16, float(target_snr_db)


def build_manifest_entry(
    clean_entry: dict[str, object],
    noisy_path: Path,
    preset: NoisePreset,
    noise_files_used: list[Path],
    audio_samples: np.ndarray,
    sample_rate: int,
    repo_root: Path,
) -> dict[str, object]:
    """Build a noisy-tier manifest entry from a clean-tier entry.

    Follows the schema defined in tests/validation/SCHEMA.md:
        noise_profile = {
            background_source, background_categories,
            background_files, linac_hum_added, snr_db
        }

    Args:
        clean_entry: The clean-tier manifest entry this was derived from.
        noisy_path: Absolute path to the mixed WAV file.
        preset: Noise preset used.
        noise_files_used: Source noise WAV paths spliced into this sample,
            in order. Single-file coverage produces a list of length 1;
            splice coverage lists every file used at each seam.
        audio_samples: The mixed audio (for duration calculation).
        sample_rate: Sample rate of output in Hz.
        repo_root: Repo root for relative path calculation.

    Returns:
        Noisy-tier manifest entry.
    """
    try:
        relative_path = noisy_path.relative_to(repo_root)
    except ValueError:
        relative_path = noisy_path

    duration = len(audio_samples) / sample_rate

    relative_noise_files: list[str] = []
    for noise_file in noise_files_used:
        try:
            relative_noise_files.append(str(noise_file.relative_to(repo_root)))
        except ValueError:
            relative_noise_files.append(str(noise_file))

    # Categories are inferred from the parent directory of each noise file
    # (e.g. free-sound, sound-bible). Deduplicated, sorted for stability.
    categories = sorted({noise_file.parent.name for noise_file in noise_files_used})

    return {
        "audio_id": f"{clean_entry['text_id']}-piper-{clean_entry['tts_voice']}-noisy-{preset.name}",
        "text_id": clean_entry["text_id"],
        "audio_path": str(relative_path),
        "tier": "noisy",
        "tts_engine": clean_entry["tts_engine"],
        "tts_voice": clean_entry["tts_voice"],
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(duration, 3),
        "channels": 1,
        "bit_depth": 16,
        "acoustic_simulation": None,
        "noise_profile": {
            "background_source": "musan",
            "background_categories": categories,
            "background_files": relative_noise_files,
            "linac_hum_added": False,
            "snr_db": preset.snr_db,
        },
        "snr_db": preset.snr_db,
    }


def inject_manifest(
    clean_manifest: Iterable[dict[str, object]],
    clean_audio_dir: Path,
    output_dir: Path,
    preset: NoisePreset,
    noise_dir: Path,
    *,
    categories: tuple[str, ...] | None = None,
    seed: int = 0,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Apply noise injection to all clean-tier audio files in a manifest.

    Args:
        clean_manifest: Iterable of clean-tier manifest entries.
        clean_audio_dir: Fallback directory for resolving clean audio paths.
        output_dir: Directory to write mixed WAV files into.
        preset: Noise preset to apply.
        noise_dir: Path to MUSAN noise directory (contains free-sound/,
            sound-bible/).
        categories: Optional subset of noise sub-categories to sample from.
        seed: Random seed for reproducible noise selection.
        crossfade_ms: Crossfade duration at splice seams in milliseconds.
        repo_root: Repo root for path resolution and manifest relative paths.

    Returns:
        List of noisy-tier manifest entries (one per successful mix).
    """
    if repo_root is None:
        repo_root = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    noise_files = list_noise_files(noise_dir, categories)
    rng = np.random.default_rng(seed)

    clip_pool_cache: dict[int, list[NoiseClipInfo]] = {}

    entries: list[dict[str, object]] = []

    for clean_entry in clean_manifest:
        clean_path_str = str(clean_entry["audio_path"])
        clean_path = repo_root / clean_path_str
        if not clean_path.exists():
            clean_path = clean_audio_dir / Path(clean_path_str).name

        if not clean_path.exists():
            logger.warning("Clean audio missing for %s: %s", clean_entry["text_id"], clean_path)
            continue

        signal, rate = read_wav(clean_path)

        if rate not in clip_pool_cache:
            clip_pool_cache[rate] = probe_noise_clips(noise_files, expected_rate=rate)
        clips = clip_pool_cache[rate]

        crossfade_samples = int(round(crossfade_ms * rate / 1000.0))

        try:
            noise_coverage, used_paths = build_noise_coverage(
                clips,
                target_samples=len(signal),
                rng=rng,
                crossfade_samples=crossfade_samples,
            )
            mixed, _achieved = mix_noise(signal, noise_coverage, preset.snr_db)
        except (ValueError, RuntimeError):
            logger.exception("Noise injection failed for %s", clean_entry["text_id"])
            continue

        out_name = f"{clean_entry['text_id']}-piper-{clean_entry['tts_voice']}-noisy-{preset.name}.wav"
        out_path = output_dir / out_name
        write_wav(out_path, mixed, rate)

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            noisy_path=out_path,
            preset=preset,
            noise_files_used=used_paths,
            audio_samples=mixed,
            sample_rate=rate,
            repo_root=repo_root,
        )
        entries.append(entry)
        logger.info(
            "Mixed %s with %d noise file(s) at %.1f dB SNR (%s preset)",
            clean_entry["text_id"],
            len(used_paths),
            preset.snr_db,
            preset.name,
        )

    return entries


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for noise injection batch processing."""
    parser = argparse.ArgumentParser(
        description="Mix MUSAN background noise into clean-tier audio at a target SNR",
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        required=True,
        help="Path to clean-tier audio_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write mixed WAV files and manifest into",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        required=True,
        help="Path to MUSAN noise directory (contains free-sound/, sound-bible/)",
    )
    parser.add_argument(
        "--preset",
        choices=list(NOISE_PRESETS.keys()),
        default="moderate",
        help="Noise preset controlling target SNR",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Noise sub-categories to sample from (e.g. free-sound sound-bible). Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible noise selection",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=DEFAULT_CROSSFADE_MS,
        help="Linear crossfade length at splice seams, in milliseconds",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for path resolution",
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

    preset = NOISE_PRESETS[args.preset]
    categories = tuple(args.categories) if args.categories else None

    with args.clean_manifest.open(encoding="utf-8") as f:
        clean_entries = [json.loads(line) for line in f if line.strip()]

    entries = inject_manifest(
        clean_entries,
        clean_audio_dir=args.clean_manifest.parent,
        output_dir=args.output_dir,
        preset=preset,
        noise_dir=args.noise_dir,
        categories=categories,
        seed=args.seed,
        crossfade_ms=args.crossfade_ms,
        repo_root=args.repo_root,
    )

    manifest_path = args.output_dir / "audio_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Mixed {len(entries)}/{len(clean_entries)} fixtures at {preset.snr_db} dB SNR", file=sys.stderr)
    return 0 if len(entries) == len(clean_entries) else 1


if __name__ == "__main__":
    sys.exit(main())
