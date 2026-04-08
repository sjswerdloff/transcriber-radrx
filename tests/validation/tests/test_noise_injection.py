"""Tests for MUSAN noise injection module.

Contract tests for SNR math, tiling/cropping of noise clips, manifest
construction, and CLI entry. The bulk of tests use synthetic in-memory
signals so they are fast and deterministic. A single real-corpus
integration test is gated by requires_audio and the presence of the
extracted MUSAN noise directory.

Authors: silas-397300f6
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.validation.audio_synthesis.noise_injection import (
    INT16_FULL_SCALE,
    MIX_PEAK_TARGET,
    NOISE_PRESETS,
    NoisePreset,
    build_manifest_entry,
    fit_noise_length,
    inject_manifest,
    inject_noise,
    list_noise_files,
    main,
    mix_noise,
    rms,
)

# Default MUSAN noise directory used by the integration test. Parameterised
# via environment assumption: if the corpus is not extracted, the test
# is skipped.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MUSAN_NOISE_DIR = _REPO_ROOT / "tests/validation/corpora/restricted/musan/noise"


def _write_sine_wav(path: Path, seconds: float = 1.0, rate: int = 16000, amplitude: int = 10000) -> None:
    """Write a sine-wave test WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * rate)
    t = np.arange(samples, dtype=np.float64) / rate
    audio = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def _write_white_noise_wav(
    path: Path,
    seconds: float = 1.0,
    rate: int = 16000,
    amplitude: int = 5000,
    seed: int = 0,
) -> None:
    """Write a white-noise test WAV file with deterministic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = int(seconds * rate)
    audio = (rng.standard_normal(samples) * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


class TestRms:
    """Contract tests for the rms helper."""

    def test_silence_returns_zero(self) -> None:
        """Contract: all-zero signal has zero RMS."""
        audio = np.zeros(1000, dtype=np.int16)
        assert rms(audio) == 0.0

    def test_empty_returns_zero(self) -> None:
        """Contract: empty array returns zero (no division-by-zero)."""
        audio = np.array([], dtype=np.int16)
        assert rms(audio) == 0.0

    def test_constant_signal_equals_abs_value(self) -> None:
        """Contract: DC signal of value v has RMS |v|."""
        audio = np.full(1000, 100, dtype=np.int16)
        assert rms(audio) == pytest.approx(100.0)

    def test_sine_rms_equals_amplitude_over_sqrt_two(self) -> None:
        """Contract: a pure sine of amplitude A has RMS A/sqrt(2)."""
        rate = 16000
        t = np.arange(rate, dtype=np.float64) / rate
        amplitude = 10000.0
        sine = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.int16)
        expected = amplitude / np.sqrt(2.0)
        # Int16 quantisation + sample count / period mismatch admits a
        # small tolerance.
        assert rms(sine) == pytest.approx(expected, rel=0.01)


class TestFitNoiseLength:
    """Contract tests for noise length fitting (crop / tile)."""

    def test_exact_length_returns_input_unchanged(self) -> None:
        """Contract: noise already at target length is returned (same values)."""
        noise = np.arange(1000, dtype=np.int16)
        rng = np.random.default_rng(0)
        out = fit_noise_length(noise, 1000, rng)
        assert len(out) == 1000
        # Exact-length path picks a start offset in [0, 0] so returns full clip.
        np.testing.assert_array_equal(out, noise)

    def test_longer_noise_is_cropped_to_target(self) -> None:
        """Contract: noise longer than target is cropped to exact target length."""
        noise = np.arange(5000, dtype=np.int16)
        rng = np.random.default_rng(42)
        out = fit_noise_length(noise, 1000, rng)
        assert len(out) == 1000

    def test_shorter_noise_is_tiled_to_target(self) -> None:
        """Contract: noise shorter than target is tiled (repeated) to target length."""
        noise = np.arange(100, dtype=np.int16)
        rng = np.random.default_rng(0)
        out = fit_noise_length(noise, 1000, rng)
        assert len(out) == 1000

    def test_tiled_output_contains_input_values(self) -> None:
        """Contract: tiled noise consists only of values from the source clip."""
        noise = np.arange(100, dtype=np.int16)
        rng = np.random.default_rng(0)
        out = fit_noise_length(noise, 500, rng)
        assert set(out.tolist()).issubset(set(noise.tolist()))

    def test_empty_noise_raises(self) -> None:
        """Contract: empty noise clip raises ValueError."""
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="empty"):
            fit_noise_length(np.array([], dtype=np.int16), 1000, rng)

    def test_nonpositive_target_raises(self) -> None:
        """Contract: zero or negative target samples raises ValueError."""
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="positive"):
            fit_noise_length(np.ones(10, dtype=np.int16), 0, rng)

    def test_deterministic_with_fixed_seed(self) -> None:
        """Contract: same seed produces the same crop offset."""
        noise = np.arange(5000, dtype=np.int16)
        out1 = fit_noise_length(noise, 1000, np.random.default_rng(7))
        out2 = fit_noise_length(noise, 1000, np.random.default_rng(7))
        np.testing.assert_array_equal(out1, out2)


class TestMixNoise:
    """Contract tests for the SNR mixer."""

    def test_shape_mismatch_raises(self) -> None:
        """Contract: mismatched lengths raise ValueError (caller must pre-fit)."""
        signal = np.ones(100, dtype=np.int16)
        noise = np.ones(200, dtype=np.int16)
        with pytest.raises(ValueError, match="shapes differ"):
            mix_noise(signal, noise, 10.0)

    def test_achieves_target_snr_on_known_signals(self) -> None:
        """Contract: achieved SNR on mixed output matches target within 0.5 dB.

        Verified by measuring the RMS of the clean signal and the RMS of
        (mixed - signal), i.e. the noise component actually present.
        """
        rate = 16000
        t = np.arange(rate, dtype=np.float64) / rate
        signal = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(rate) * 3000).astype(np.int16)

        target_snr = 10.0
        mixed, _achieved = mix_noise(signal, noise, target_snr)

        residual_noise = mixed.astype(np.float64) - signal.astype(np.float64)
        measured_snr_db = 20.0 * np.log10(rms(signal) / rms(residual_noise))
        assert abs(measured_snr_db - target_snr) < 0.5

    def test_higher_snr_means_less_noise_energy(self) -> None:
        """Contract: higher target SNR results in quieter noise in mix."""
        rate = 16000
        t = np.arange(rate, dtype=np.float64) / rate
        signal = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(rate) * 3000).astype(np.int16)

        mixed_quiet, _ = mix_noise(signal, noise, 20.0)
        mixed_loud, _ = mix_noise(signal, noise, 5.0)

        noise_quiet_rms = rms(mixed_quiet.astype(np.float64) - signal.astype(np.float64))
        noise_loud_rms = rms(mixed_loud.astype(np.float64) - signal.astype(np.float64))
        assert noise_quiet_rms < noise_loud_rms

    def test_silent_signal_returns_signal_unchanged(self) -> None:
        """Contract: silent signal (RMS < floor) returns unchanged signal and inf SNR."""
        signal = np.zeros(1000, dtype=np.int16)
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(1000) * 1000).astype(np.int16)
        mixed, achieved = mix_noise(signal, noise, 10.0)
        np.testing.assert_array_equal(mixed, signal)
        assert achieved == float("inf")

    def test_silent_noise_returns_signal_unchanged(self) -> None:
        """Contract: silent noise returns clean signal unchanged."""
        signal = (np.ones(1000) * 100).astype(np.int16)
        noise = np.zeros(1000, dtype=np.int16)
        mixed, achieved = mix_noise(signal, noise, 10.0)
        np.testing.assert_array_equal(mixed, signal)
        assert achieved == float("inf")

    def test_output_is_int16(self) -> None:
        """Contract: mixer always returns int16."""
        signal = np.ones(1000, dtype=np.int16) * 1000
        noise = np.ones(1000, dtype=np.int16) * 100
        mixed, _ = mix_noise(signal, noise, 10.0)
        assert mixed.dtype == np.int16

    def test_peak_limiting_prevents_clipping(self) -> None:
        """Contract: mixed peak stays within MIX_PEAK_TARGET of int16 full scale."""
        # Construct a near-peak signal and loud noise so the naive sum
        # would clip.
        signal = np.full(1000, 30000, dtype=np.int16)
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(1000) * 15000).astype(np.int16)
        mixed, _ = mix_noise(signal, noise, 0.0)
        peak = float(np.abs(mixed).max())
        assert peak <= MIX_PEAK_TARGET * INT16_FULL_SCALE + 1  # +1 for int rounding


class TestListNoiseFiles:
    """Contract tests for noise file enumeration."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        """Contract: nonexistent noise directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list_noise_files(tmp_path / "does-not-exist")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """Contract: directory with no WAV files raises ValueError."""
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="No noise WAV files"):
            list_noise_files(tmp_path / "empty")

    def test_enumerates_wav_files(self, tmp_path: Path) -> None:
        """Contract: finds all WAV files under the root."""
        (tmp_path / "free-sound").mkdir()
        (tmp_path / "sound-bible").mkdir()
        _write_white_noise_wav(tmp_path / "free-sound" / "a.wav", seconds=0.5)
        _write_white_noise_wav(tmp_path / "sound-bible" / "b.wav", seconds=0.5)
        files = list_noise_files(tmp_path)
        assert len(files) == 2

    def test_category_filtering(self, tmp_path: Path) -> None:
        """Contract: category argument limits enumeration to named subdirs."""
        (tmp_path / "free-sound").mkdir()
        (tmp_path / "sound-bible").mkdir()
        _write_white_noise_wav(tmp_path / "free-sound" / "a.wav", seconds=0.5)
        _write_white_noise_wav(tmp_path / "sound-bible" / "b.wav", seconds=0.5)
        files = list_noise_files(tmp_path, categories=("free-sound",))
        assert len(files) == 1
        assert files[0].name == "a.wav"


class TestNoisePresets:
    """Contract tests for the preset table."""

    def test_all_presets_have_required_fields(self) -> None:
        """Contract: every preset has a populated name, snr_db, description."""
        for preset in NOISE_PRESETS.values():
            assert isinstance(preset, NoisePreset)
            assert preset.name
            assert isinstance(preset.snr_db, float)
            assert preset.description

    def test_presets_ordered_quiet_to_busy(self) -> None:
        """Contract: quiet > moderate > busy in SNR (higher SNR = cleaner)."""
        assert NOISE_PRESETS["quiet"].snr_db > NOISE_PRESETS["moderate"].snr_db
        assert NOISE_PRESETS["moderate"].snr_db > NOISE_PRESETS["busy"].snr_db

    def test_preset_is_frozen(self) -> None:
        """Contract: NoisePreset is frozen for safe parallel use."""
        preset = NOISE_PRESETS["quiet"]
        with pytest.raises((AttributeError, Exception)):
            preset.snr_db = 999.0  # type: ignore[misc]


class TestInjectNoise:
    """Contract tests for the single-sample noise injection."""

    def test_rejects_sample_rate_mismatch(self, tmp_path: Path) -> None:
        """Contract: noise at wrong sample rate raises ValueError."""
        noise_path = tmp_path / "noise.wav"
        _write_white_noise_wav(noise_path, seconds=0.5, rate=8000)
        signal = np.zeros(16000, dtype=np.int16)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="sample rate"):
            inject_noise(signal, 16000, noise_path, 10.0, rng)

    def test_returns_metadata_with_file_and_snr(self, tmp_path: Path) -> None:
        """Contract: metadata dict includes noise file path and target SNR."""
        noise_path = tmp_path / "noise.wav"
        _write_white_noise_wav(noise_path, seconds=0.5)
        signal = (np.ones(16000) * 1000).astype(np.int16)
        rng = np.random.default_rng(0)
        _mixed, metadata = inject_noise(signal, 16000, noise_path, 15.0, rng)
        assert metadata["noise_file"] == str(noise_path)
        assert metadata["target_snr_db"] == 15.0


class TestBuildManifestEntry:
    """Noisy-tier manifest entry contract tests."""

    def test_tier_is_noisy(self, tmp_path: Path) -> None:
        """Contract: entries have tier='noisy'."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = NOISE_PRESETS["moderate"]
        entry = build_manifest_entry(
            clean_entry=clean_entry,
            noisy_path=tmp_path / "t-001-noisy.wav",
            preset=preset,
            noise_files_used=[tmp_path / "free-sound" / "noise-0001.wav"],
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )
        assert entry["tier"] == "noisy"

    def test_noise_profile_matches_schema(self, tmp_path: Path) -> None:
        """Contract: noise_profile object has the shape from SCHEMA.md."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = NOISE_PRESETS["moderate"]
        entry = build_manifest_entry(
            clean_entry=clean_entry,
            noisy_path=tmp_path / "t-001.wav",
            preset=preset,
            noise_files_used=[tmp_path / "free-sound" / "noise-0001.wav"],
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )
        noise_profile = entry["noise_profile"]
        assert isinstance(noise_profile, dict)
        assert noise_profile["background_source"] == "musan"
        assert noise_profile["background_categories"] == ["free-sound"]
        assert len(noise_profile["background_files"]) == 1
        assert noise_profile["linac_hum_added"] is False
        assert noise_profile["snr_db"] == preset.snr_db
        # Top-level snr_db also set for quick filtering.
        assert entry["snr_db"] == preset.snr_db

    def test_multiple_noise_files_deduplicated_categories(self, tmp_path: Path) -> None:
        """Contract: multiple noise files with same category yield a single category entry."""
        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        preset = NOISE_PRESETS["quiet"]
        entry = build_manifest_entry(
            clean_entry=clean_entry,
            noisy_path=tmp_path / "t-001.wav",
            preset=preset,
            noise_files_used=[
                tmp_path / "free-sound" / "a.wav",
                tmp_path / "free-sound" / "b.wav",
                tmp_path / "sound-bible" / "c.wav",
            ],
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )
        noise_profile = entry["noise_profile"]
        assert isinstance(noise_profile, dict)
        assert noise_profile["background_categories"] == ["free-sound", "sound-bible"]

    def test_audio_id_includes_preset_name(self, tmp_path: Path) -> None:
        """Contract: audio_id suffix distinguishes noisy-tier runs by preset."""
        clean_entry: dict[str, object] = {
            "text_id": "rond-0001",
            "tts_engine": "piper",
            "tts_voice": "en_US-amy-medium",
        }
        preset = NOISE_PRESETS["busy"]
        entry = build_manifest_entry(
            clean_entry=clean_entry,
            noisy_path=tmp_path / "rond-0001.wav",
            preset=preset,
            noise_files_used=[tmp_path / "free-sound" / "x.wav"],
            audio_samples=np.zeros(16000, dtype=np.int16),
            sample_rate=16000,
            repo_root=tmp_path,
        )
        assert "noisy" in str(entry["audio_id"])
        assert "busy" in str(entry["audio_id"])


class TestInjectManifest:
    """End-to-end manifest contract tests with synthetic inputs."""

    def test_skips_missing_clean_audio(self, tmp_path: Path) -> None:
        """Contract: missing clean files are logged and skipped."""
        noise_dir = tmp_path / "musan_noise" / "free-sound"
        noise_dir.mkdir(parents=True)
        _write_white_noise_wav(noise_dir / "n.wav", seconds=0.5)

        clean_entry: dict[str, object] = {
            "text_id": "missing-001",
            "audio_path": "clean/missing.wav",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        entries = inject_manifest(
            [clean_entry],
            clean_audio_dir=tmp_path / "clean",
            output_dir=tmp_path / "noisy",
            preset=NOISE_PRESETS["moderate"],
            noise_dir=tmp_path / "musan_noise",
            repo_root=tmp_path,
        )
        assert entries == []

    def test_processes_existing_clean_audio(self, tmp_path: Path) -> None:
        """Contract: existing clean audio is mixed and a manifest entry emitted."""
        clean_path = tmp_path / "clean" / "t-001-piper-amy.wav"
        _write_sine_wav(clean_path, seconds=1.0)

        noise_dir = tmp_path / "musan_noise" / "free-sound"
        noise_dir.mkdir(parents=True)
        _write_white_noise_wav(noise_dir / "n.wav", seconds=2.0)

        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "audio_path": "clean/t-001-piper-amy.wav",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }
        entries = inject_manifest(
            [clean_entry],
            clean_audio_dir=tmp_path / "clean",
            output_dir=tmp_path / "noisy",
            preset=NOISE_PRESETS["moderate"],
            noise_dir=tmp_path / "musan_noise",
            repo_root=tmp_path,
        )
        assert len(entries) == 1
        assert (tmp_path / "noisy" / "t-001-piper-amy-noisy-moderate.wav").exists()

    def test_deterministic_across_runs_with_same_seed(self, tmp_path: Path) -> None:
        """Contract: same seed selects the same noise file and crop."""
        clean_path = tmp_path / "clean" / "t-001-piper-amy.wav"
        _write_sine_wav(clean_path, seconds=1.0)

        noise_dir = tmp_path / "musan_noise" / "free-sound"
        noise_dir.mkdir(parents=True)
        for i in range(5):
            _write_white_noise_wav(noise_dir / f"n{i}.wav", seconds=2.0, seed=i)

        clean_entry: dict[str, object] = {
            "text_id": "t-001",
            "audio_path": "clean/t-001-piper-amy.wav",
            "tts_engine": "piper",
            "tts_voice": "amy",
        }

        out1 = tmp_path / "noisy1"
        out2 = tmp_path / "noisy2"
        entries1 = inject_manifest(
            [clean_entry],
            clean_audio_dir=tmp_path / "clean",
            output_dir=out1,
            preset=NOISE_PRESETS["moderate"],
            noise_dir=tmp_path / "musan_noise",
            seed=123,
            repo_root=tmp_path,
        )
        entries2 = inject_manifest(
            [clean_entry],
            clean_audio_dir=tmp_path / "clean",
            output_dir=out2,
            preset=NOISE_PRESETS["moderate"],
            noise_dir=tmp_path / "musan_noise",
            seed=123,
            repo_root=tmp_path,
        )

        # Same noise file chosen
        assert entries1[0]["noise_profile"] == entries2[0]["noise_profile"]

        # Same mixed samples written
        with wave.open(str(out1 / "t-001-piper-amy-noisy-moderate.wav"), "rb") as w1:
            data1 = w1.readframes(w1.getnframes())
        with wave.open(str(out2 / "t-001-piper-amy-noisy-moderate.wav"), "rb") as w2:
            data2 = w2.readframes(w2.getnframes())
        assert data1 == data2


class TestMainCli:
    """CLI entry-point contract tests."""

    def test_writes_manifest_and_returns_zero_on_success(self, tmp_path: Path) -> None:
        """Contract: happy-path CLI run writes audio_manifest.jsonl and returns 0."""
        clean_path = tmp_path / "clean" / "t-001-piper-amy.wav"
        _write_sine_wav(clean_path, seconds=1.0)

        clean_manifest_path = tmp_path / "clean" / "audio_manifest.jsonl"
        clean_manifest_path.write_text(
            json.dumps(
                {
                    "text_id": "t-001",
                    "audio_path": "clean/t-001-piper-amy.wav",
                    "tts_engine": "piper",
                    "tts_voice": "amy",
                },
            )
            + "\n",
            encoding="utf-8",
        )

        noise_dir = tmp_path / "musan_noise" / "free-sound"
        noise_dir.mkdir(parents=True)
        _write_white_noise_wav(noise_dir / "n.wav", seconds=2.0)

        out_dir = tmp_path / "noisy"
        rc = main(
            [
                "--clean-manifest",
                str(clean_manifest_path),
                "--output-dir",
                str(out_dir),
                "--noise-dir",
                str(tmp_path / "musan_noise"),
                "--preset",
                "moderate",
                "--repo-root",
                str(tmp_path),
            ],
        )
        assert rc == 0
        manifest = out_dir / "audio_manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tier"] == "noisy"


@pytest.mark.requires_audio
def test_real_musan_corpus_round_trip(tmp_path: Path) -> None:
    """Real MUSAN corpus round-trip: mix a sine fixture with real noise.

    Gated by requires_audio marker and the presence of the extracted MUSAN
    noise directory. Skipped in CI unless both are present. Confirms that
    the module works end-to-end against the actual corpus file formats.
    """
    if not _DEFAULT_MUSAN_NOISE_DIR.is_dir():
        pytest.skip(f"MUSAN noise corpus not present at {_DEFAULT_MUSAN_NOISE_DIR}")

    clean_path = tmp_path / "clean" / "sine.wav"
    _write_sine_wav(clean_path, seconds=2.0)

    clean_entry: dict[str, object] = {
        "text_id": "sine-0001",
        "audio_path": "clean/sine.wav",
        "tts_engine": "piper",
        "tts_voice": "test",
    }

    entries = inject_manifest(
        [clean_entry],
        clean_audio_dir=tmp_path / "clean",
        output_dir=tmp_path / "noisy",
        preset=NOISE_PRESETS["moderate"],
        noise_dir=_DEFAULT_MUSAN_NOISE_DIR,
        seed=0,
        repo_root=tmp_path,
    )
    assert len(entries) == 1
    out_wav = tmp_path / "noisy" / "sine-0001-piper-test-noisy-moderate.wav"
    assert out_wav.exists()
