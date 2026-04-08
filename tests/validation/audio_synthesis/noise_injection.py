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


def fit_noise_length(
    noise: np.ndarray,
    target_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Crop or tile a noise clip to match a target length exactly.

    If the noise clip is longer than the target, a random start offset is
    chosen and the clip is cropped. If shorter, the clip is tiled (repeated)
    until it reaches the target length, then cropped to the exact size. The
    tiling start is also randomised so repeated calls do not always start
    at sample zero.

    Args:
        noise: Int or float noise samples.
        target_samples: Desired output length in samples.
        rng: Random generator for start-offset selection.

    Returns:
        A noise array of length exactly target_samples.

    Raises:
        ValueError: If noise is empty or target_samples is non-positive.
    """
    if noise.size == 0:
        msg = "Cannot fit empty noise clip to target length"
        raise ValueError(msg)
    if target_samples <= 0:
        msg = f"target_samples must be positive, got {target_samples}"
        raise ValueError(msg)

    if noise.size >= target_samples:
        max_offset = noise.size - target_samples
        start = int(rng.integers(0, max_offset + 1))
        cropped: np.ndarray = noise[start : start + target_samples].copy()
        return cropped

    # Tile: pick a random start inside the clip, then wrap-concatenate
    # until we have enough samples, then crop.
    start = int(rng.integers(0, noise.size))
    tiled = np.concatenate([noise[start:], noise])
    while tiled.size < target_samples:
        tiled = np.concatenate([tiled, noise])
    out: np.ndarray = tiled[:target_samples].copy()
    return out


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
        noise: Int16 mono noise, any length.
        target_snr_db: Target SNR in dB. Higher = cleaner speech.

    Returns:
        Tuple of (mixed int16 audio, achieved SNR in dB). The achieved SNR
        is the value actually realised after any post-mix gain scaling.
        Gain scaling uniform on both components preserves the ratio exactly,
        so achieved_snr == target_snr_db in normal operation.

    Raises:
        ValueError: If signal and noise lengths differ. Callers must use
            fit_noise_length before calling this function.
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


def inject_noise(
    signal: np.ndarray,
    sample_rate: int,
    noise_path: Path,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mix a clean speech signal with a MUSAN noise clip at a target SNR.

    Args:
        signal: Int16 mono clean speech.
        sample_rate: Sample rate of the signal in Hz.
        noise_path: Path to a MUSAN noise WAV. Must match signal sample rate.
        target_snr_db: Target SNR in dB.
        rng: Random generator for noise crop/tile offset.

    Returns:
        Tuple of (mixed int16 audio, metadata dict describing the mix).

    Raises:
        ValueError: If the noise file's sample rate differs from the signal's.
    """
    noise, noise_rate = read_wav(noise_path)
    if noise_rate != sample_rate:
        msg = f"Noise sample rate {noise_rate} Hz does not match signal {sample_rate} Hz: {noise_path}"
        raise ValueError(msg)

    fitted = fit_noise_length(noise, len(signal), rng)
    mixed, achieved_snr = mix_noise(signal, fitted, target_snr_db)

    metadata: dict[str, object] = {
        "noise_file": str(noise_path),
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(achieved_snr),
    }
    return mixed, metadata


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
        noise_files_used: Source noise WAV path(s) mixed into this sample.
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
        repo_root: Repo root for path resolution and manifest relative paths.

    Returns:
        List of noisy-tier manifest entries (one per successful mix).
    """
    if repo_root is None:
        repo_root = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    noise_files = list_noise_files(noise_dir, categories)
    rng = np.random.default_rng(seed)

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

        noise_idx = int(rng.integers(0, len(noise_files)))
        noise_path = noise_files[noise_idx]

        try:
            mixed, _meta = inject_noise(signal, rate, noise_path, preset.snr_db, rng)
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
            noise_files_used=[noise_path],
            audio_samples=mixed,
            sample_rate=rate,
            repo_root=repo_root,
        )
        entries.append(entry)
        logger.info(
            "Mixed %s with %s at %.1f dB SNR (%s preset)",
            clean_entry["text_id"],
            noise_path.name,
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
