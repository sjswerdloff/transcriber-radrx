"""Tests for Commonwealth English and ESL voice panels.

Verifies panel constants, speaker-ID expansion, and the synthesize_one /
build_manifest_entry speaker_id plumbing — all without requiring a real
piper binary or voice model on disk.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.validation.audio_synthesis.piper_tts import (
    PIPER_SAMPLE_RATE,
    TextFixture,
    build_manifest_entry,
    synthesize_one,
)
from tests.validation.scripts.run_multi_voice_e2e import (
    COMMONWEALTH_VOICES,
    ESL_VOICES,
    L2ARCTIC_SPEAKERS,
    VOICE_PANELS,
    VoiceSpec,
    expand_esl_voice_specs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_audio(seconds: float, rate: int = PIPER_SAMPLE_RATE) -> bytes:
    """Generate fake int16 PCM bytes for mocking piper output."""
    samples = int(seconds * rate)
    t = np.arange(samples, dtype=np.float64) / rate
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    return tone.tobytes()


# ---------------------------------------------------------------------------
# COMMONWEALTH_VOICES panel
# ---------------------------------------------------------------------------


class TestCommonwealthVoices:
    """Contract tests for the COMMONWEALTH_VOICES panel constant."""

    def test_has_eight_entries(self) -> None:
        """Contract: Commonwealth panel contains exactly 8 voice entries."""
        assert len(COMMONWEALTH_VOICES) == 8

    def test_all_entries_are_en_gb(self) -> None:
        """Contract: every voice in the Commonwealth panel uses the en_GB language code."""
        for name, language, _quality in COMMONWEALTH_VOICES:
            assert language == "en_GB", f"Voice {name!r} has unexpected language {language!r}"

    def test_all_entries_have_non_empty_name_and_quality(self) -> None:
        """Contract: every entry has a non-empty name and quality string."""
        for name, _language, quality in COMMONWEALTH_VOICES:
            assert name, f"Entry ({name!r}, {_language!r}, {quality!r}) has empty name"
            assert quality, f"Entry ({name!r}, {_language!r}, {quality!r}) has empty quality"

    def test_quality_values_are_valid(self) -> None:
        """Contract: quality values are restricted to low/medium/high."""
        valid_qualities = {"low", "medium", "high"}
        for name, _language, quality in COMMONWEALTH_VOICES:
            assert quality in valid_qualities, (
                f"Voice {name!r} has invalid quality {quality!r}; expected one of {valid_qualities}"
            )


# ---------------------------------------------------------------------------
# ESL_VOICES panel
# ---------------------------------------------------------------------------


class TestEslVoices:
    """Contract tests for the ESL_VOICES panel constant."""

    def test_has_three_entries(self) -> None:
        """Contract: ESL panel has exactly 3 top-level entries (l2arctic, reza_ibrahim, kusal)."""
        assert len(ESL_VOICES) == 3

    def test_l2arctic_is_first_entry(self) -> None:
        """Contract: first ESL entry is l2arctic (the multi-speaker model)."""
        name, language, quality = ESL_VOICES[0]
        assert name == "l2arctic"
        assert language == "en_US"

    def test_single_speaker_names_present(self) -> None:
        """Contract: reza_ibrahim and kusal are in the panel."""
        names = {entry[0] for entry in ESL_VOICES}
        assert "reza_ibrahim" in names
        assert "kusal" in names


# ---------------------------------------------------------------------------
# L2ARCTIC_SPEAKERS mapping
# ---------------------------------------------------------------------------


class TestL2arcticSpeakers:
    """Contract tests for the L2ARCTIC_SPEAKERS mapping."""

    def test_has_twenty_four_entries(self) -> None:
        """Contract: exactly 24 speakers in L2-Arctic corpus mapping."""
        assert len(L2ARCTIC_SPEAKERS) == 24

    def test_speaker_ids_are_unique(self) -> None:
        """Contract: no two speakers share the same speaker_id."""
        speaker_ids = [sid for _, sid in L2ARCTIC_SPEAKERS.values()]
        assert len(speaker_ids) == len(set(speaker_ids)), "Duplicate speaker IDs found in L2ARCTIC_SPEAKERS"

    def test_speaker_ids_are_non_negative_integers(self) -> None:
        """Contract: all speaker IDs are non-negative integers."""
        for label, (_l1_lang, speaker_id) in L2ARCTIC_SPEAKERS.items():
            assert isinstance(speaker_id, int), f"Speaker {label!r} has non-int ID: {speaker_id!r}"
            assert speaker_id >= 0, f"Speaker {label!r} has negative ID: {speaker_id}"

    def test_l1_languages_are_strings(self) -> None:
        """Contract: all L1 language values are non-empty strings."""
        for label, (l1_lang, _speaker_id) in L2ARCTIC_SPEAKERS.items():
            assert isinstance(l1_lang, str) and l1_lang, f"Speaker {label!r} has invalid L1 language"

    def test_known_speakers_present(self) -> None:
        """Contract: spot-check a selection of expected speaker labels."""
        expected = {"ABA", "BWC", "HJK", "ERMS", "THV", "ASI", "ZHAA"}
        missing = expected - L2ARCTIC_SPEAKERS.keys()
        assert not missing, f"Expected speaker labels missing: {missing}"


# ---------------------------------------------------------------------------
# VOICE_PANELS registry
# ---------------------------------------------------------------------------


class TestVoicePanels:
    """Contract tests for the VOICE_PANELS registry."""

    def test_contains_default_panel(self) -> None:
        """Contract: VOICE_PANELS has a 'default' key."""
        assert "default" in VOICE_PANELS

    def test_contains_commonwealth_panel(self) -> None:
        """Contract: VOICE_PANELS has a 'commonwealth' key."""
        assert "commonwealth" in VOICE_PANELS

    def test_contains_esl_panel(self) -> None:
        """Contract: VOICE_PANELS has an 'esl' key."""
        assert "esl" in VOICE_PANELS

    def test_all_panels_are_lists(self) -> None:
        """Contract: every panel value is a list."""
        for panel_name, panel in VOICE_PANELS.items():
            assert isinstance(panel, list), f"Panel {panel_name!r} is not a list"

    def test_commonwealth_panel_matches_constant(self) -> None:
        """Contract: VOICE_PANELS['commonwealth'] is the same object as COMMONWEALTH_VOICES."""
        assert VOICE_PANELS["commonwealth"] is COMMONWEALTH_VOICES

    def test_esl_panel_matches_constant(self) -> None:
        """Contract: VOICE_PANELS['esl'] is the same object as ESL_VOICES."""
        assert VOICE_PANELS["esl"] is ESL_VOICES


# ---------------------------------------------------------------------------
# expand_esl_voice_specs
# ---------------------------------------------------------------------------


class TestExpandEslVoiceSpecs:
    """Contract tests for expand_esl_voice_specs."""

    def _make_fake_root(self, tmp_path: Path) -> Path:
        """Create a minimal piper-voices tree with l2arctic, reza_ibrahim, and kusal models."""
        for name, lang in [("l2arctic", "en_US"), ("reza_ibrahim", "en_US"), ("kusal", "en_US")]:
            lang_group = lang.split("_")[0]
            model_dir = tmp_path / lang_group / lang / name / "medium"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / f"{lang}-{name}-medium.onnx").write_text("fake")
        return tmp_path

    def test_returns_twenty_six_specs_when_all_voices_found(self, tmp_path: Path) -> None:
        """Contract: 24 l2arctic speakers + 2 single-speaker voices = 26 VoiceSpec entries."""
        root = self._make_fake_root(tmp_path)
        specs = expand_esl_voice_specs(root)
        assert len(specs) == 26

    def test_l2arctic_entries_have_speaker_ids(self, tmp_path: Path) -> None:
        """Contract: l2arctic-prefixed specs each carry a non-None speaker_id."""
        root = self._make_fake_root(tmp_path)
        specs = expand_esl_voice_specs(root)
        l2arctic_specs = [s for s in specs if s.name.startswith("l2arctic-")]
        assert len(l2arctic_specs) == 24
        for spec in l2arctic_specs:
            assert spec.speaker_id is not None, f"l2arctic spec {spec.name!r} has no speaker_id"

    def test_l2arctic_speaker_ids_are_unique(self, tmp_path: Path) -> None:
        """Contract: no two l2arctic speaker VoiceSpecs share the same speaker_id."""
        root = self._make_fake_root(tmp_path)
        specs = expand_esl_voice_specs(root)
        l2arctic_specs = [s for s in specs if s.name.startswith("l2arctic-")]
        ids = [s.speaker_id for s in l2arctic_specs]
        assert len(ids) == len(set(ids))

    def test_single_speaker_voices_have_no_speaker_id(self, tmp_path: Path) -> None:
        """Contract: reza_ibrahim and kusal pass through with speaker_id=None."""
        root = self._make_fake_root(tmp_path)
        specs = expand_esl_voice_specs(root)
        single_speaker = [s for s in specs if s.name in ("reza_ibrahim", "kusal")]
        assert len(single_speaker) == 2
        for spec in single_speaker:
            assert spec.speaker_id is None, f"Single-speaker voice {spec.name!r} has unexpected speaker_id"

    def test_missing_voice_is_skipped_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Contract: a voice not found on disk is skipped; a warning is emitted."""
        # Only l2arctic on disk — reza_ibrahim and kusal are absent
        lang_group = "en"
        lang = "en_US"
        name = "l2arctic"
        model_dir = tmp_path / lang_group / lang / name / "medium"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{lang}-{name}-medium.onnx").write_text("fake")

        with caplog.at_level(logging.WARNING):
            specs = expand_esl_voice_specs(tmp_path)

        # Only l2arctic speakers — reza_ibrahim and kusal missing
        assert len(specs) == 24
        assert any("reza_ibrahim" in record.message for record in caplog.records)

    def test_accepts_custom_esl_voices_list(self, tmp_path: Path) -> None:
        """Contract: custom esl_voices list overrides the default ESL_VOICES constant."""
        # Only put kusal on disk
        lang = "en_US"
        lang_group = lang.split("_")[0]
        model_dir = tmp_path / lang_group / lang / "kusal" / "medium"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{lang}-kusal-medium.onnx").write_text("fake")

        custom = [("kusal", "en_US", "medium")]
        specs = expand_esl_voice_specs(tmp_path, esl_voices=custom)
        assert len(specs) == 1
        assert specs[0].name == "kusal"
        assert specs[0].speaker_id is None

    def test_display_name_includes_l2arctic_label(self, tmp_path: Path) -> None:
        """Contract: l2arctic VoiceSpec display_name encodes speaker label and speaker_id."""
        root = self._make_fake_root(tmp_path)
        specs = expand_esl_voice_specs(root)
        l2arctic_specs = {s.name: s for s in specs if s.name.startswith("l2arctic-")}
        # Spot-check ABA: display_name = "{language}-l2arctic-ABA-{quality}-speaker{id}"
        aba_spec = l2arctic_specs.get("l2arctic-ABA")
        assert aba_spec is not None
        dn = aba_spec.display_name
        assert "l2arctic-ABA" in dn
        assert f"speaker{aba_spec.speaker_id}" in dn


# ---------------------------------------------------------------------------
# synthesize_one — speaker_id plumbing
# ---------------------------------------------------------------------------


class TestSynthesizeOneSpeakerId:
    """Contract tests for speaker_id in synthesize_one."""

    def test_no_speaker_flag_when_speaker_id_is_none(self) -> None:
        """Contract: --speaker is NOT added to the piper command when speaker_id is None."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)
            synthesize_one("Hello.", Path("/fake/model.onnx"), speaker_id=None)
            cmd = mock_run.call_args.args[0]
        assert "--speaker" not in cmd

    def test_speaker_flag_appended_when_speaker_id_set(self) -> None:
        """Contract: --speaker N is appended when speaker_id is provided."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)
            synthesize_one("Hello.", Path("/fake/model.onnx"), speaker_id=7)
            cmd = mock_run.call_args.args[0]
        assert "--speaker" in cmd
        speaker_idx = cmd.index("--speaker")
        assert cmd[speaker_idx + 1] == "7"

    def test_speaker_id_zero_is_valid(self) -> None:
        """Contract: speaker_id=0 is a valid value and produces --speaker 0."""
        fake_audio = _make_fake_audio(0.5)
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_audio, stderr=b"", returncode=0)
            synthesize_one("Hello.", Path("/fake/model.onnx"), speaker_id=0)
            cmd = mock_run.call_args.args[0]
        assert "--speaker" in cmd
        speaker_idx = cmd.index("--speaker")
        assert cmd[speaker_idx + 1] == "0"

    def test_raises_runtime_error_on_piper_failure_with_speaker_id(self) -> None:
        """Contract: speaker_id does not mask piper errors."""
        with patch("tests.validation.audio_synthesis.piper_tts.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["piper"],
                stderr=b"speaker out of range",
            )
            with pytest.raises(RuntimeError, match="speaker out of range"):
                synthesize_one("Hello.", Path("/fake/model.onnx"), speaker_id=99)


# ---------------------------------------------------------------------------
# build_manifest_entry — speaker_id in tts_voice
# ---------------------------------------------------------------------------


class TestBuildManifestEntrySpeakerId:
    """Contract tests for speaker_id integration in build_manifest_entry."""

    def _make_fixture(self) -> TextFixture:
        return TextFixture(id="t-001", text="Hello.", language="en", raw={})

    def test_tts_voice_excludes_speaker_suffix_when_none(self, tmp_path: Path) -> None:
        """Contract: tts_voice is plain voice_name when speaker_id is None."""
        fixture = self._make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="en_US-amy-medium",
            audio_samples=audio,
            repo_root=tmp_path,
            speaker_id=None,
        )
        assert entry["tts_voice"] == "en_US-amy-medium"

    def test_tts_voice_includes_speaker_suffix_when_set(self, tmp_path: Path) -> None:
        """Contract: tts_voice is extended with -speakerN when speaker_id is provided."""
        fixture = self._make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="en_US-l2arctic-medium",
            audio_samples=audio,
            repo_root=tmp_path,
            speaker_id=3,
        )
        assert entry["tts_voice"] == "en_US-l2arctic-medium-speaker3"

    def test_audio_id_uses_speaker_extended_voice_name(self, tmp_path: Path) -> None:
        """Contract: audio_id uses the full tts_voice (with speaker suffix) in its key."""
        fixture = self._make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="en_US-l2arctic-medium",
            audio_samples=audio,
            repo_root=tmp_path,
            speaker_id=21,
        )
        assert entry["audio_id"] == "t-001-piper-en_US-l2arctic-medium-speaker21-clean"

    def test_speaker_id_zero_produces_speaker0_suffix(self, tmp_path: Path) -> None:
        """Contract: speaker_id=0 produces a -speaker0 suffix (not omitted)."""
        fixture = self._make_fixture()
        audio = np.zeros(100, dtype=np.int16)
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="en_US-l2arctic-medium",
            audio_samples=audio,
            repo_root=tmp_path,
            speaker_id=0,
        )
        assert entry["tts_voice"] == "en_US-l2arctic-medium-speaker0"

    def test_required_schema_fields_present_with_speaker_id(self, tmp_path: Path) -> None:
        """Contract: all required manifest schema fields are present even with speaker_id."""
        fixture = self._make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="en_US-l2arctic-medium",
            audio_samples=audio,
            repo_root=tmp_path,
            speaker_id=7,
        )
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


# ---------------------------------------------------------------------------
# VoiceSpec.display_name with speaker_id
# ---------------------------------------------------------------------------


class TestVoiceSpecDisplayName:
    """Contract tests for VoiceSpec.display_name speaker_id suffix."""

    def test_display_name_no_speaker_id(self) -> None:
        """Contract: display_name without speaker_id is language-name-quality."""
        spec = VoiceSpec(
            name="alan",
            model_path=Path("/fake/alan.onnx"),
            language="en_GB",
            quality="medium",
        )
        assert spec.display_name == "en_GB-alan-medium"

    def test_display_name_with_speaker_id(self) -> None:
        """Contract: display_name with speaker_id appends -speakerN suffix."""
        spec = VoiceSpec(
            name="l2arctic-ABA",
            model_path=Path("/fake/l2arctic.onnx"),
            language="en_US",
            quality="medium",
            speaker_id=21,
        )
        assert spec.display_name == "en_US-l2arctic-ABA-medium-speaker21"

    def test_display_name_speaker_id_zero(self) -> None:
        """Contract: speaker_id=0 produces -speaker0 in display_name."""
        spec = VoiceSpec(
            name="l2arctic-TXHC",
            model_path=Path("/fake/l2arctic.onnx"),
            language="en_US",
            quality="medium",
            speaker_id=0,
        )
        assert "speaker0" in spec.display_name
