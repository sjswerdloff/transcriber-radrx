"""Tests for acoustic room simulation module.

Logic tests mock pyroomacoustics so they run without the native binary.
A real-simulation test is gated by requires_audio for CI systems that
have pyroomacoustics installed.

Authors: vivian-1a61bc9a
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.validation.audio_synthesis.acoustic_sim import (
    ROOM_PRESETS,
    build_manifest_entry,
    read_wav,
    simulate_manifest,
    simulate_room,
    write_wav,
)


def _write_test_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    """Write a sine-wave test WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * rate)
    t = np.arange(samples, dtype=np.float64) / rate
    audio = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


class TestRoomPresets:
    """Room preset definition contract tests."""

    def test_linac_vault_exists(self) -> None:
        """Contract: linac_vault preset is defined for the primary clinical use case."""
        assert "linac_vault" in ROOM_PRESETS

    def test_all_presets_have_required_fields(self) -> None:
        """Contract: every preset has complete configuration."""
        for preset in ROOM_PRESETS.values():
            assert isinstance(preset.name, str)
            assert len(preset.dimensions_m) == 3
            assert preset.rt60_seconds > 0
            assert preset.mic_distance_m > 0
            assert 0 < preset.materials_absorption < 1

    def test_linac_vault_is_larger_than_exam_room(self) -> None:
        """Contract: room size ordering matches physical reality."""
        linac = ROOM_PRESETS["linac_vault"]
        exam = ROOM_PRESETS["exam_room"]
        linac_vol = linac.dimensions_m[0] * linac.dimensions_m[1] * linac.dimensions_m[2]
        exam_vol = exam.dimensions_m[0] * exam.dimensions_m[1] * exam.dimensions_m[2]
        assert linac_vol > exam_vol

    def test_linac_vault_has_longer_rt60(self) -> None:
        """Contract: linac vault (concrete) has longer RT60 than exam room (soft)."""
        assert ROOM_PRESETS["linac_vault"].rt60_seconds > ROOM_PRESETS["exam_room"].rt60_seconds


class TestReadWriteWav:
    """Round-trip WAV I/O contract tests."""

    def test_round_trip_preserves_audio(self, tmp_path: Path) -> None:
        """Contract: read(write(x)) == x for int16 mono at given rate."""
        original = np.array([100, -200, 300, -400, 500], dtype=np.int16)
        path = tmp_path / "test.wav"

        write_wav(path, original, 16000)
        read_audio, read_rate = read_wav(path)

        np.testing.assert_array_equal(read_audio, original)
        assert read_rate == 16000

    def test_rejects_stereo_input(self, tmp_path: Path) -> None:
        """Contract: reader rejects non-mono WAV files."""
        path = tmp_path / "stereo.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(100, dtype=np.int16).tobytes())

        with pytest.raises(ValueError, match="mono"):
            read_wav(path)

    def test_rejects_non_16bit_input(self, tmp_path: Path) -> None:
        """Contract: reader rejects non-16-bit WAV files."""
        path = tmp_path / "8bit.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)  # 8-bit
            wf.setframerate(16000)
            wf.writeframes(np.zeros(100, dtype=np.int8).tobytes())

        with pytest.raises(ValueError, match="16-bit"):
            read_wav(path)


class TestSimulateRoom:
    """Room simulation logic tests (mocked pyroomacoustics)."""

    def test_preserves_input_length(self) -> None:
        """Contract: output length equals input length (reverb tail truncated)."""
        audio = np.ones(16000, dtype=np.int16) * 5000  # 1 second
        preset = ROOM_PRESETS["linac_vault"]

        with patch("pyroomacoustics.ShoeBox") as mock_room_class, patch("pyroomacoustics.Material"):
            mock_room = MagicMock()
            # Simulated output is longer than input (includes reverb tail)
            mock_room.mic_array.signals = [np.zeros(20000, dtype=np.float64)]
            mock_room_class.return_value = mock_room

            result = simulate_room(audio, 16000, preset)

        assert len(result) == len(audio)

    def test_returns_int16(self) -> None:
        """Contract: output dtype is int16 regardless of internal float processing."""
        audio = np.ones(1000, dtype=np.int16) * 100
        preset = ROOM_PRESETS["exam_room"]

        with patch("pyroomacoustics.ShoeBox") as mock_room_class, patch("pyroomacoustics.Material"):
            mock_room = MagicMock()
            mock_room.mic_array.signals = [np.zeros(1000, dtype=np.float64)]
            mock_room_class.return_value = mock_room

            result = simulate_room(audio, 16000, preset)

        assert result.dtype == np.int16

    def test_invokes_pyroomacoustics_shoebox(self) -> None:
        """Contract: simulation uses pyroomacoustics.ShoeBox with preset dimensions."""
        audio = np.ones(100, dtype=np.int16) * 100
        preset = ROOM_PRESETS["linac_vault"]

        with patch("pyroomacoustics.ShoeBox") as mock_room_class, patch("pyroomacoustics.Material"):
            mock_room = MagicMock()
            mock_room.mic_array.signals = [np.zeros(100, dtype=np.float64)]
            mock_room_class.return_value = mock_room

            simulate_room(audio, 16000, preset)

            assert mock_room_class.called
            dimensions_arg = mock_room_class.call_args.args[0]
            assert list(preset.dimensions_m) == list(dimensions_arg)


class TestBuildManifestEntry:
    """Acoustic-tier manifest entry contract tests."""

    def test_tier_is_acoustic(self, tmp_path: Path) -> None:
        """Contract: entries have tier='acoustic'."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = ROOM_PRESETS["linac_vault"]
        audio = np.zeros(16000, dtype=np.int16)

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            simulated_path=tmp_path / "t-001-acoustic.wav",
            preset=preset,
            audio_samples=audio,
            sample_rate=16000,
            repo_root=tmp_path,
        )

        assert entry["tier"] == "acoustic"

    def test_acoustic_simulation_object_present(self, tmp_path: Path) -> None:
        """Contract: acoustic_simulation object includes room metadata."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = ROOM_PRESETS["linac_vault"]
        audio = np.zeros(16000, dtype=np.int16)

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            simulated_path=tmp_path / "t-001.wav",
            preset=preset,
            audio_samples=audio,
            sample_rate=16000,
            repo_root=tmp_path,
        )

        sim_obj = entry["acoustic_simulation"]
        assert sim_obj is not None
        assert isinstance(sim_obj, dict)
        assert sim_obj["room_type"] == "linac_vault"
        assert sim_obj["rt60_seconds"] == preset.rt60_seconds
        assert sim_obj["mic_distance_m"] == preset.mic_distance_m

    def test_noise_fields_remain_null(self, tmp_path: Path) -> None:
        """Contract: acoustic tier has null noise_profile (noise is a separate tier)."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = ROOM_PRESETS["linac_vault"]

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            simulated_path=tmp_path / "t-001.wav",
            preset=preset,
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )

        assert entry["noise_profile"] is None
        assert entry["snr_db"] is None

    def test_audio_id_includes_room_name(self, tmp_path: Path) -> None:
        """Contract: audio_id has -acoustic-{room} suffix for disambiguation."""
        clean_entry: dict[str, object] = {
            "text_id": "rond-0001",
            "tts_engine": "piper",
            "tts_voice": "en_US-amy-medium",
        }
        preset = ROOM_PRESETS["linac_vault"]

        entry = build_manifest_entry(
            clean_entry=clean_entry,
            simulated_path=tmp_path / "rond-0001.wav",
            preset=preset,
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )

        assert "acoustic" in str(entry["audio_id"])
        assert "linac_vault" in str(entry["audio_id"])


class TestSimulateManifest:
    """End-to-end manifest simulation contract tests."""

    def test_skips_missing_clean_audio(self, tmp_path: Path) -> None:
        """Contract: missing clean audio files are logged and skipped."""
        clean_entry = {
            "text_id": "missing-001",
            "audio_path": "tests/validation/audio/synthetic/clean/missing.wav",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }

        entries = simulate_manifest(
            [clean_entry],
            clean_audio_dir=tmp_path,
            output_dir=tmp_path / "acoustic",
            preset=ROOM_PRESETS["linac_vault"],
            repo_root=tmp_path,
        )

        assert len(entries) == 0

    def test_processes_existing_clean_audio(self, tmp_path: Path) -> None:
        """Contract: existing clean audio is simulated and written."""
        clean_audio = tmp_path / "clean" / "t-001-piper-amy.wav"
        _write_test_wav(clean_audio, seconds=0.5, rate=16000)

        clean_entry = {
            "text_id": "t-001",
            "audio_path": "clean/t-001-piper-amy.wav",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }

        with patch(
            "tests.validation.audio_synthesis.acoustic_sim.simulate_room",
        ) as mock_sim:
            mock_sim.return_value = np.zeros(8000, dtype=np.int16)

            entries = simulate_manifest(
                [clean_entry],
                clean_audio_dir=tmp_path / "clean",
                output_dir=tmp_path / "acoustic",
                preset=ROOM_PRESETS["linac_vault"],
                repo_root=tmp_path,
            )

        assert len(entries) == 1
        assert (tmp_path / "acoustic" / "t-001-piper-amy-linac_vault.wav").exists()


@pytest.mark.requires_audio
def test_real_pyroomacoustics_simulation() -> None:
    """Real pyroomacoustics simulation on a sine tone.

    Gated by requires_audio marker — skipped unless pyroomacoustics is
    installed and the marker is selected. Verifies the integration works
    end to end with real audio and real simulation.
    """
    pytest.importorskip("pyroomacoustics")

    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    preset = ROOM_PRESETS["linac_vault"]
    result = simulate_room(tone, sample_rate, preset)

    assert len(result) == len(tone)
    assert result.dtype == np.int16
    # Simulated audio should not be identical to input (room added reverb)
    assert not np.array_equal(result, tone)


class TestRoomPresetDataclass:
    """RoomPreset dataclass contract."""

    def test_is_immutable(self) -> None:
        """Contract: RoomPreset is frozen for safety across parallel runs."""
        preset = ROOM_PRESETS["linac_vault"]
        with pytest.raises((AttributeError, Exception)):
            preset.rt60_seconds = 999.0  # type: ignore[misc]
