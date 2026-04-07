"""Tests for piper TTS synthesis pipeline.

These tests mock the piper subprocess so they run without requiring
the piper binary or a real voice model. Real synthesis is exercised
via the validation tier (requires_audio marker).

Authors: vivian-1a61bc9a
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.validation.audio_synthesis.piper_tts import (
    PIPER_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    TextFixture,
    _resample_linear,
    build_manifest_entry,
    load_fixtures,
    synthesize_fixtures,
    synthesize_one,
    write_manifest,
    write_wav,
)


def _make_fake_audio(seconds: float, rate: int = PIPER_SAMPLE_RATE) -> bytes:
    """Generate fake int16 PCM bytes for mocking piper output."""
    samples = int(seconds * rate)
    # Simple sine wave so it's non-trivial (not all zeros)
    t = np.arange(samples, dtype=np.float64) / rate
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    return tone.tobytes()


class TestLoadFixtures:
    """Fixture loading contract tests."""

    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        """Contract: all valid JSONL lines become TextFixture objects."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text(
            '{"id": "test-001", "text": "Hello world.", "language": "en"}\n'
            '{"id": "test-002", "text": "Second sample.", "language": "en"}\n',
        )

        fixtures = load_fixtures(fixtures_file)

        assert len(fixtures) == 2
        assert fixtures[0].id == "test-001"
        assert fixtures[0].text == "Hello world."
        assert fixtures[1].id == "test-002"

    def test_defaults_language_to_en(self, tmp_path: Path) -> None:
        """Contract: missing language field defaults to 'en'."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text('{"id": "test-001", "text": "Hello."}\n')

        fixtures = load_fixtures(fixtures_file)

        assert fixtures[0].language == "en"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Contract: blank lines are ignored."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text(
            '{"id": "test-001", "text": "A."}\n\n{"id": "test-002", "text": "B."}\n',
        )

        fixtures = load_fixtures(fixtures_file)

        assert len(fixtures) == 2

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Contract: missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_fixtures(tmp_path / "nonexistent.jsonl")

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Contract: malformed JSON raises ValueError with line number."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text("not json\n")

        with pytest.raises(ValueError, match="line 1"):
            load_fixtures(fixtures_file)

    def test_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        """Contract: missing 'id' or 'text' raises ValueError."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text('{"id": "test-001"}\n')

        with pytest.raises(ValueError, match="text"):
            load_fixtures(fixtures_file)

    def test_raises_typeerror_on_non_string_id(self, tmp_path: Path) -> None:
        """Contract: non-string id raises TypeError rather than silently coercing."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text('{"id": 12345, "text": "Hello."}\n')

        with pytest.raises(TypeError, match="id"):
            load_fixtures(fixtures_file)

    def test_raises_typeerror_on_non_string_text(self, tmp_path: Path) -> None:
        """Contract: non-string text raises TypeError."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text('{"id": "t-001", "text": 42}\n')

        with pytest.raises(TypeError, match="text"):
            load_fixtures(fixtures_file)

    def test_raises_typeerror_on_non_string_language(self, tmp_path: Path) -> None:
        """Contract: non-string language raises TypeError."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text('{"id": "t-001", "text": "Hello.", "language": 1}\n')

        with pytest.raises(TypeError, match="language"):
            load_fixtures(fixtures_file)

    def test_preserves_raw_record(self, tmp_path: Path) -> None:
        """Contract: optional fields are preserved in fixture.raw for manifest use."""
        fixtures_file = tmp_path / "fixtures.jsonl"
        fixtures_file.write_text(
            '{"id": "t-001", "text": "A.", "vocabulary_terms": ["Gy"], "category": "dose_prescription"}\n',
        )

        fixtures = load_fixtures(fixtures_file)

        assert fixtures[0].raw["vocabulary_terms"] == ["Gy"]
        assert fixtures[0].raw["category"] == "dose_prescription"


class TestResampling:
    """Resampling function contract tests."""

    def test_no_op_when_rates_equal(self) -> None:
        """Contract: identical rates return input unchanged."""
        audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        result = _resample_linear(audio, 22050, 22050)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_halves_length(self) -> None:
        """Contract: 2:1 downsampling produces ~half the samples."""
        # 1 second at 22050 Hz → ~0.726 seconds at 16 kHz (22050→16000)
        audio = np.ones(22050, dtype=np.int16) * 1000
        result = _resample_linear(audio, 22050, 16000)
        # Expected length: 22050 * 16000 / 22050 = 16000
        assert abs(len(result) - 16000) <= 2  # tolerance for rounding

    def test_preserves_int16_dtype(self) -> None:
        """Contract: output dtype is always int16."""
        audio = np.ones(1000, dtype=np.int16) * 5000
        result = _resample_linear(audio, 22050, 16000)
        assert result.dtype == np.int16

    def test_handles_empty_input(self) -> None:
        """Contract: empty input returns empty output."""
        audio = np.zeros(0, dtype=np.int16)
        result = _resample_linear(audio, 22050, 16000)
        assert len(result) == 0


class TestSynthesizeOne:
    """Single-sample synthesis contract tests."""

    def test_passes_text_to_piper_stdin(self) -> None:
        """Contract: text is sent to piper via stdin as UTF-8."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

            synthesize_one("Hello world.", Path("/fake/model.onnx"))

            call = mock_run.call_args
            assert call.kwargs["input"] == b"Hello world."

    def test_invokes_piper_with_model_and_output_raw(self) -> None:
        """Contract: piper is invoked with --model and --output_raw flags."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

            synthesize_one("Hello.", Path("/fake/model.onnx"))

            cmd = mock_run.call_args.args[0]
            assert "piper" in cmd
            assert "--model" in cmd
            assert "/fake/model.onnx" in cmd
            assert "--output_raw" in cmd

    def test_returns_int16_array(self) -> None:
        """Contract: returns numpy int16 array from piper stdout."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

            result = synthesize_one("Hello.", Path("/fake/model.onnx"))

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.int16

    def test_raises_runtime_error_on_piper_failure(self) -> None:
        """Contract: piper non-zero exit raises RuntimeError with stderr."""
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["piper"],
                stderr=b"model not found",
            )

            with pytest.raises(RuntimeError, match="model not found"):
                synthesize_one("Hello.", Path("/fake/model.onnx"))

    def test_raises_runtime_error_on_empty_output(self) -> None:
        """Contract: empty piper output raises RuntimeError."""
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"", stderr=b"", returncode=0)

            with pytest.raises(RuntimeError, match="no audio"):
                synthesize_one("Hello.", Path("/fake/model.onnx"))


class TestWriteWav:
    """WAV file writing contract tests."""

    def test_writes_readable_wav(self, tmp_path: Path) -> None:
        """Contract: written WAV can be read back with correct metadata."""
        audio = np.ones(16000, dtype=np.int16) * 1000
        path = tmp_path / "test.wav"

        write_wav(path, audio, TARGET_SAMPLE_RATE)

        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # int16 = 2 bytes
            assert wf.getframerate() == TARGET_SAMPLE_RATE
            assert wf.getnframes() == 16000

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Contract: nested parent directories are created automatically."""
        audio = np.zeros(100, dtype=np.int16)
        path = tmp_path / "nested" / "dirs" / "test.wav"

        write_wav(path, audio, TARGET_SAMPLE_RATE)

        assert path.exists()


class TestManifestEntry:
    """Manifest entry construction contract tests."""

    def test_entry_matches_schema(self, tmp_path: Path) -> None:
        """Contract: manifest entry has all required fields from SCHEMA.md."""
        fixture = TextFixture(id="t-001", text="Hello.", language="en", raw={"id": "t-001"})
        audio = np.zeros(16000, dtype=np.int16)  # 1 second
        audio_path = tmp_path / "t-001-piper-en_US-amy-medium.wav"

        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=audio_path,
            voice_name="en_US-amy-medium",
            audio_samples=audio,
            repo_root=tmp_path,
        )

        # Required fields from SCHEMA.md
        required_fields = {
            "audio_id",
            "text_id",
            "audio_path",
            "tier",
            "tts_engine",
            "tts_voice",
            "sample_rate_hz",
            "duration_seconds",
            "channels",
        }
        assert required_fields.issubset(entry.keys())

    def test_audio_id_follows_convention(self, tmp_path: Path) -> None:
        """Contract: audio_id is `{text_id}-piper-{voice_name}-clean`."""
        fixture = TextFixture(id="rond-0001", text="Hi.", language="en", raw={})
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "rond-0001.wav",
            voice_name="en_US-amy-medium",
            audio_samples=np.zeros(16000, dtype=np.int16),
            repo_root=tmp_path,
        )

        assert entry["audio_id"] == "rond-0001-piper-en_US-amy-medium-clean"

    def test_duration_calculated_from_samples(self, tmp_path: Path) -> None:
        """Contract: duration_seconds = len(samples) / sample_rate."""
        fixture = TextFixture(id="t-001", text="Hi.", language="en", raw={})
        audio = np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.int16)  # 2 seconds

        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "t-001.wav",
            voice_name="en_US-amy-medium",
            audio_samples=audio,
            repo_root=tmp_path,
        )

        assert entry["duration_seconds"] == 2.0

    def test_tier_is_clean(self, tmp_path: Path) -> None:
        """Contract: clean-tier manifest always has tier='clean'."""
        fixture = TextFixture(id="t-001", text="Hi.", language="en", raw={})
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "t-001.wav",
            voice_name="amy",
            audio_samples=np.zeros(100, dtype=np.int16),
            repo_root=tmp_path,
        )

        assert entry["tier"] == "clean"

    def test_null_fields_for_acoustic_and_noise(self, tmp_path: Path) -> None:
        """Contract: clean-tier entries have null acoustic and noise fields."""
        fixture = TextFixture(id="t-001", text="Hi.", language="en", raw={})
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "t-001.wav",
            voice_name="amy",
            audio_samples=np.zeros(100, dtype=np.int16),
            repo_root=tmp_path,
        )

        assert entry["acoustic_simulation"] is None
        assert entry["noise_profile"] is None
        assert entry["snr_db"] is None


class TestSynthesizeFixtures:
    """End-to-end synthesis pipeline contract tests."""

    def test_writes_wav_per_fixture(self, tmp_path: Path) -> None:
        """Contract: one WAV file is written per fixture."""
        fixtures = [
            TextFixture(id="t-001", text="First.", language="en", raw={}),
            TextFixture(id="t-002", text="Second.", language="en", raw={}),
        ]
        fake_audio = _make_fake_audio(0.5)

        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

            entries = synthesize_fixtures(
                fixtures,
                voice_model=Path("/fake/amy.onnx"),
                output_dir=tmp_path,
                voice_name="amy",
                repo_root=tmp_path,
            )

        assert len(entries) == 2
        assert (tmp_path / "t-001-piper-amy.wav").exists()
        assert (tmp_path / "t-002-piper-amy.wav").exists()

    def test_skips_failed_syntheses(self, tmp_path: Path) -> None:
        """Contract: a failed synthesis does not block subsequent fixtures."""
        fixtures = [
            TextFixture(id="t-001", text="First.", language="en", raw={}),
            TextFixture(id="t-002", text="Second.", language="en", raw={}),
        ]
        fake_audio = _make_fake_audio(0.5)

        call_count = {"n": 0}

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise subprocess.CalledProcessError(returncode=1, cmd=["piper"], stderr=b"fail")
            return MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

        with patch(
            "tests.validation.audio_synthesis.piper_tts.subprocess.run",
            side_effect=side_effect,
        ):
            entries = synthesize_fixtures(
                fixtures,
                voice_model=Path("/fake/amy.onnx"),
                output_dir=tmp_path,
                voice_name="amy",
                repo_root=tmp_path,
            )

        # Only the second one succeeded
        assert len(entries) == 1
        assert entries[0]["text_id"] == "t-002"

    def test_output_wav_is_at_target_sample_rate(self, tmp_path: Path) -> None:
        """Contract: output WAV files are at 16 kHz, not piper's 22050 Hz."""
        fixtures = [TextFixture(id="t-001", text="Hi.", language="en", raw={})]
        fake_audio = _make_fake_audio(1.0)  # 1 second at 22050 Hz

        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)

            synthesize_fixtures(
                fixtures,
                voice_model=Path("/fake/amy.onnx"),
                output_dir=tmp_path,
                voice_name="amy",
                repo_root=tmp_path,
            )

        wav_path = tmp_path / "t-001-piper-amy.wav"
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == TARGET_SAMPLE_RATE


class TestWriteManifest:
    """Manifest file writing contract tests."""

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        """Contract: manifest is valid JSONL parseable line by line."""
        entries: list[dict[str, object]] = [
            {"audio_id": "a-001", "text_id": "t-001"},
            {"audio_id": "a-002", "text_id": "t-002"},
        ]
        path = tmp_path / "manifest.jsonl"

        write_manifest(path, entries)

        with path.open() as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed == entries

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Contract: nested parent directories are created."""
        path = tmp_path / "nested" / "manifest.jsonl"
        write_manifest(path, [{"audio_id": "a"}])
        assert path.exists()
