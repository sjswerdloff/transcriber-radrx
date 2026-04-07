"""Acoustic room simulation for validation audio (acoustic tier).

Takes clean-tier WAV files and simulates realistic room acoustics using
pyroomacoustics. Produces the 'acoustic' tier of the validation audio
manifest: clean voice audio degraded by room reflections, reverberation,
and microphone distance — the conditions a real clinical dictation
microphone experiences in a treatment vault or clinic room.

Room presets model realistic clinical spaces:
- linac_vault: Large concrete-walled treatment room with long RT60
- exam_room: Small carpeted clinical room, low reverberation
- open_office: Medium office with moderate reverberation

Authors: vivian-1a61bc9a
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Speaker standing head height for source placement (meters)
SPEAKER_HEAD_HEIGHT_M = 1.7
# Desk mic typical height (meters)
DESK_MIC_HEIGHT_M = 1.2
# Ceiling mic clearance below ceiling (meters)
CEILING_MIC_CLEARANCE_M = 0.3
# Sabine's formula: RT60 = 0.161 * V / (S * alpha)
SABINE_COEFFICIENT = 0.161
# Physical absorption coefficient bounds
MIN_ABSORPTION = 0.05
MAX_ABSORPTION = 0.99
# Image-source reflection order (higher = more accurate, slower)
IMAGE_SOURCE_MAX_ORDER = 10


@dataclass(frozen=True)
class RoomPreset:
    """Named room acoustic configuration.

    Attributes:
        name: Identifier used in manifest.
        dimensions_m: [width, depth, height] in meters.
        rt60_seconds: Target reverberation time (time for -60 dB decay).
        mic_distance_m: Microphone distance from speaker in meters.
        mic_position: Descriptive label ("ceiling", "desk", "handheld").
        materials_absorption: Scalar 0-1, higher = more absorptive walls.
    """

    name: str
    dimensions_m: tuple[float, float, float]
    rt60_seconds: float
    mic_distance_m: float
    mic_position: str
    materials_absorption: float


# Clinical room presets. Dimensions and RT60 values based on published
# measurements of radiotherapy treatment vaults and clinic rooms.
ROOM_PRESETS: dict[str, RoomPreset] = {
    "linac_vault": RoomPreset(
        name="linac_vault",
        dimensions_m=(6.0, 8.0, 3.5),
        rt60_seconds=0.6,  # Large concrete room, long reverb tail
        mic_distance_m=1.5,
        mic_position="ceiling",
        materials_absorption=0.15,  # Concrete, minimal absorption
    ),
    "exam_room": RoomPreset(
        name="exam_room",
        dimensions_m=(3.5, 4.0, 2.7),
        rt60_seconds=0.3,  # Small carpeted room, short reverb
        mic_distance_m=0.8,
        mic_position="desk",
        materials_absorption=0.45,  # Soft furnishings
    ),
    "open_office": RoomPreset(
        name="open_office",
        dimensions_m=(5.0, 6.0, 3.0),
        rt60_seconds=0.4,  # Typical office acoustics
        mic_distance_m=1.0,
        mic_position="desk",
        materials_absorption=0.30,
    ),
}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV file into an int16 numpy array.

    Args:
        path: Path to input WAV file.

    Returns:
        Tuple of (int16 audio array, sample rate in Hz).

    Raises:
        ValueError: If file is not mono or not 16-bit.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            msg = f"Expected mono WAV, got {wf.getnchannels()} channels: {path}"
            raise ValueError(msg)
        if wf.getsampwidth() != 2:
            msg = f"Expected 16-bit WAV, got {wf.getsampwidth() * 8}-bit: {path}"
            raise ValueError(msg)

        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16)

    return audio, rate


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write int16 mono audio to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def simulate_room(
    audio: np.ndarray,
    sample_rate: int,
    preset: RoomPreset,
) -> np.ndarray:
    """Apply room acoustic simulation to clean audio.

    Uses pyroomacoustics' image-source method to compute a realistic
    room impulse response (RIR) and convolves the input audio with it.

    Args:
        audio: Int16 mono input audio.
        sample_rate: Sample rate in Hz.
        preset: Room configuration.

    Returns:
        Int16 mono audio with room acoustics applied. Same length as input
        (tail truncated rather than padded to preserve manifest duration).

    Raises:
        ImportError: If pyroomacoustics is not installed.
    """
    import pyroomacoustics as pra

    # Convert to float for processing
    audio_float = audio.astype(np.float64) / 32768.0

    # Compute absorption from RT60 using Sabine's formula:
    # RT60 = 0.161 * V / (S * alpha), solving for alpha
    width, depth, height = preset.dimensions_m
    volume = width * depth * height
    surface_area = 2 * (width * depth + width * height + depth * height)
    absorption = SABINE_COEFFICIENT * volume / (surface_area * preset.rt60_seconds)
    absorption = float(np.clip(absorption, MIN_ABSORPTION, MAX_ABSORPTION))

    # Build shoebox room with computed absorption
    room = pra.ShoeBox(
        list(preset.dimensions_m),
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=IMAGE_SOURCE_MAX_ORDER,
    )

    # Place source near one wall at standing head height
    source_pos = [width * 0.2, depth * 0.5, SPEAKER_HEAD_HEIGHT_M]

    # Place microphone so that the true 3D Euclidean distance from the
    # source equals preset.mic_distance_m. For a ceiling mic this means
    # solving for horizontal offset given the vertical separation.
    mic_z = height - CEILING_MIC_CLEARANCE_M if preset.mic_position == "ceiling" else DESK_MIC_HEIGHT_M
    vertical_delta = mic_z - SPEAKER_HEAD_HEIGHT_M
    vertical_sq = vertical_delta * vertical_delta
    target_sq = preset.mic_distance_m * preset.mic_distance_m

    if vertical_sq >= target_sq:
        # Vertical separation alone exceeds desired distance — mic directly
        # above/below source (horizontal offset zero). Actual distance will
        # be |vertical_delta|, which is recorded in the manifest.
        horizontal_offset = 0.0
        logger.warning(
            "[acoustic_sim] Preset %s: vertical separation %.2fm exceeds desired "
            "mic_distance_m=%.2f. Placing mic directly above/below source; "
            "actual distance will be %.2fm.",
            preset.name,
            abs(vertical_delta),
            preset.mic_distance_m,
            abs(vertical_delta),
        )
    else:
        horizontal_offset = math.sqrt(target_sq - vertical_sq)

    mic_pos = [
        source_pos[0] + horizontal_offset,
        depth * 0.5,
        mic_z,
    ]
    room.add_source(source_pos, signal=audio_float)
    room.add_microphone(mic_pos)

    room.simulate()

    # Extract the mic signal and normalize
    simulated = room.mic_array.signals[0]

    # Preserve original length (truncate reverb tail)
    simulated = simulated[: len(audio_float)]

    # Normalize to match input peak to avoid DC growth / clipping
    input_peak = np.abs(audio_float).max()
    output_peak = np.abs(simulated).max()
    if output_peak > 0 and input_peak > 0:
        simulated = simulated * (input_peak / output_peak)

    # Clip and convert back to int16
    simulated_int16: np.ndarray = np.clip(simulated * 32768.0, -32768, 32767).astype(np.int16)
    return simulated_int16


def build_manifest_entry(
    clean_entry: dict[str, object],
    simulated_path: Path,
    preset: RoomPreset,
    audio_samples: np.ndarray,
    sample_rate: int,
    repo_root: Path,
) -> dict[str, object]:
    """Build an acoustic-tier manifest entry from a clean-tier entry.

    Args:
        clean_entry: The clean-tier manifest entry this was derived from.
        simulated_path: Absolute path to the simulated WAV file.
        preset: Room configuration used.
        audio_samples: The simulated audio.
        sample_rate: Sample rate of output.
        repo_root: Repo root for relative paths.

    Returns:
        Acoustic-tier manifest entry matching SCHEMA.md.
    """
    try:
        relative_path = simulated_path.relative_to(repo_root)
    except ValueError:
        relative_path = simulated_path

    duration = len(audio_samples) / sample_rate

    return {
        "audio_id": f"{clean_entry['text_id']}-piper-{clean_entry['tts_voice']}-acoustic-{preset.name}",
        "text_id": clean_entry["text_id"],
        "audio_path": str(relative_path),
        "tier": "acoustic",
        "tts_engine": clean_entry["tts_engine"],
        "tts_voice": clean_entry["tts_voice"],
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(duration, 3),
        "channels": 1,
        "bit_depth": 16,
        "acoustic_simulation": {
            "room_type": preset.name,
            "room_dimensions_m": list(preset.dimensions_m),
            "rt60_seconds": preset.rt60_seconds,
            "mic_distance_m": preset.mic_distance_m,
            "mic_position": preset.mic_position,
        },
        "noise_profile": None,
        "snr_db": None,
    }


def simulate_manifest(
    clean_manifest: Iterable[dict[str, object]],
    clean_audio_dir: Path,
    output_dir: Path,
    preset: RoomPreset,
    *,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Apply acoustic simulation to all clean-tier audio files.

    Args:
        clean_manifest: Iterable of clean-tier manifest entries.
        clean_audio_dir: Directory containing clean WAV files (paths in
            manifest are typically relative to repo root, but we resolve
            via clean_audio_dir as fallback).
        output_dir: Directory to write simulated WAV files into.
        preset: Room configuration to apply.
        repo_root: Repo root for path resolution and manifest relative paths.

    Returns:
        List of acoustic-tier manifest entries.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for clean_entry in clean_manifest:
        clean_path_str = str(clean_entry["audio_path"])
        clean_path = repo_root / clean_path_str
        if not clean_path.exists():
            clean_path = clean_audio_dir / Path(clean_path_str).name

        if not clean_path.exists():
            logger.warning("Clean audio missing for %s: %s", clean_entry["text_id"], clean_path)
            continue

        audio, rate = read_wav(clean_path)
        try:
            simulated = simulate_room(audio, rate, preset)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            logger.exception("Simulation failed for %s", clean_entry["text_id"])
            continue

        out_name = f"{clean_entry['text_id']}-piper-{clean_entry['tts_voice']}-{preset.name}.wav"
        out_path = output_dir / out_name
        write_wav(out_path, simulated, rate)

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            simulated_path=out_path,
            preset=preset,
            audio_samples=simulated,
            sample_rate=rate,
            repo_root=repo_root,
        )
        entries.append(entry)
        logger.info("Simulated %s in %s room", clean_entry["text_id"], preset.name)

    return entries


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for acoustic simulation batch processing."""
    parser = argparse.ArgumentParser(
        description="Apply room acoustic simulation to clean-tier audio",
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
        help="Directory to write simulated WAV files and manifest into",
    )
    parser.add_argument(
        "--room",
        choices=list(ROOM_PRESETS.keys()),
        default="linac_vault",
        help="Room preset to apply",
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

    preset = ROOM_PRESETS[args.room]

    with args.clean_manifest.open(encoding="utf-8") as f:
        clean_entries = [json.loads(line) for line in f if line.strip()]

    entries = simulate_manifest(
        clean_entries,
        clean_audio_dir=args.clean_manifest.parent,
        output_dir=args.output_dir,
        preset=preset,
        repo_root=args.repo_root,
    )

    manifest_path = args.output_dir / "audio_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Simulated {len(entries)}/{len(clean_entries)} fixtures", file=sys.stderr)
    return 0 if len(entries) == len(clean_entries) else 1


if __name__ == "__main__":
    sys.exit(main())
