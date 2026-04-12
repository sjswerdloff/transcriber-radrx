"""Tests for macOS TTS synthesis backend.

All tests mock the subprocess calls so they run without the macOS ``say``
command or real audio synthesis. The platform guard in ``macos_tts`` is
also mocked out so tests run on non-Darwin CI.

Authors: silas-397300f6
"""

from __future__ import annotations

import io
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.validation.audio_synthesis import piper_tts as _piper_tts_mod

# ---------------------------------------------------------------------------
# Platform guard: macos_tts raises RuntimeError on non-Darwin at import
# time. We patch sys.platform before importing the module in all tests so
# the test suite runs on Linux CI as well as macOS.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force sys.platform to 'darwin' for all tests in this module."""
    monkeypatch.setattr(sys, "platform", "darwin")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """Return raw bytes of a minimal 16-bit mono WAV file."""
    samples = int(seconds * rate)
    t = np.arange(samples, dtype=np.float64) / rate
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(tone.tobytes())
    return buf.getvalue()


def _make_fixture(fixture_id: str = "dense-0001") -> _piper_tts_mod.TextFixture:
    """Return a minimal TextFixture for testing."""
    return _piper_tts_mod.TextFixture(id=fixture_id, text="Hello.", language="en", raw={})


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------


class TestPlatformGuard:
    """Contract: module raises RuntimeError on non-Darwin platforms."""

    def test_raises_on_non_darwin(self) -> None:
        """Contract: _check_platform raises RuntimeError on Linux."""
        import importlib

        import tests.validation.audio_synthesis.macos_tts as mod

        importlib.reload(mod)
        with patch.object(sys, "platform", "linux"), pytest.raises(RuntimeError, match="macOS"):
            mod._check_platform()

    def test_no_error_on_darwin(self) -> None:
        """Contract: _check_platform does not raise when platform is darwin."""
        import importlib

        import tests.validation.audio_synthesis.macos_tts as mod

        importlib.reload(mod)
        with patch.object(sys, "platform", "darwin"):
            mod._check_platform()


# ---------------------------------------------------------------------------
# list_available_voices
# ---------------------------------------------------------------------------

_SAMPLE_SAY_OUTPUT = """\
Alex               en_US    # Most people recognize me by my voice.
Daniel             en_GB    # Hello, my name is Daniel. I am a British-English voice.
Karen              en_AU    # Hello, my name is Karen. I am an Australian-English voice.
Moira              en_IE    # Hello, my name is Moira. I am an Irish-English voice.
Rishi              en_IN    # Hello, my name is Rishi. I am an Indian-English voice.
Tessa              en_ZA    # Hello, my name is Tessa. I am a South-African-English voice.
Matilda (Premium)  en_AU    # Hello, my name is Matilda.
Thomas             fr_FR    # Bonjour, je m'appelle Thomas.
Bells              en_US    # Ding dong. (novelty)
Boing              en_US    # Doing! (novelty)
Zarvox             en_US    # That looks like a fish to me. (novelty)
Whisper            en_US    # Shh! I'm trying to talk quietly.
"""


class TestListAvailableVoices:
    """Contract tests for list_available_voices."""

    def _run_with_output(self, output: str) -> list[tuple[str, str]]:
        """Run list_available_voices with a mocked say -v '?' output."""
        import tests.validation.audio_synthesis.macos_tts as mod

        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", return_value=mock_result):
            return mod.list_available_voices()

    def test_returns_list_of_tuples(self) -> None:
        """Contract: return type is list of (str, str) tuples."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        assert isinstance(voices, list)
        for item in voices:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, locale = item
            assert isinstance(name, str)
            assert isinstance(locale, str)

    def test_filters_to_english_locales(self) -> None:
        """Contract: only en_* voices are returned."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        for _name, locale in voices:
            assert locale.startswith("en_"), f"Non-English locale returned: {locale!r}"

    def test_excludes_novelty_voices(self) -> None:
        """Contract: novelty voices (Bells, Boing, Zarvox, Whisper) are excluded."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        voice_names = {name for name, _ in voices}
        novelty = {"Bells", "Boing", "Zarvox", "Whisper"}
        overlap = novelty & voice_names
        assert not overlap, f"Novelty voices were not filtered: {overlap}"

    def test_includes_expected_commonwealth_voices(self) -> None:
        """Contract: Daniel, Karen, Moira, Rishi, Tessa are returned from sample output."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        voice_names = {name for name, _ in voices}
        expected = {"Daniel", "Karen", "Moira", "Rishi", "Tessa"}
        missing = expected - voice_names
        assert not missing, f"Expected voices not returned: {missing}"

    def test_multi_word_voice_name_preserved(self) -> None:
        """Contract: voice names containing spaces are returned intact."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        voice_names = {name for name, _ in voices}
        assert "Matilda (Premium)" in voice_names

    def test_correct_locale_for_karen(self) -> None:
        """Contract: Karen is associated with en_AU locale."""
        voices = self._run_with_output(_SAMPLE_SAY_OUTPUT)
        by_name = dict(voices)
        assert by_name.get("Karen") == "en_AU"

    def test_raises_runtime_error_on_subprocess_failure(self) -> None:
        """Contract: RuntimeError raised when say -v '?' exits non-zero."""
        import tests.validation.audio_synthesis.macos_tts as mod

        with (
            patch(
                "tests.validation.audio_synthesis.macos_tts.subprocess.run",
                side_effect=subprocess.CalledProcessError(returncode=1, cmd=["say", "-v", "?"], stderr=b"error"),
            ),
            pytest.raises(RuntimeError, match="say -v"),
        ):
            mod.list_available_voices()

    def test_raises_runtime_error_on_timeout(self) -> None:
        """Contract: RuntimeError raised when say -v '?' times out."""
        import tests.validation.audio_synthesis.macos_tts as mod

        with (
            patch(
                "tests.validation.audio_synthesis.macos_tts.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["say", "-v", "?"], timeout=10),
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            mod.list_available_voices()

    def test_empty_output_returns_empty_list(self) -> None:
        """Contract: empty say output returns an empty list without error."""
        voices = self._run_with_output("")
        assert voices == []


# ---------------------------------------------------------------------------
# synthesize_one
# ---------------------------------------------------------------------------


class TestSynthesizeOne:
    """Contract tests for synthesize_one."""

    def _make_say_afconvert_side_effect(self, wav_bytes: bytes) -> object:
        """Return a side_effect callable simulating say + afconvert.

        The first subprocess.run call (say) does nothing.
        The second (afconvert) writes wav_bytes to the output path.
        """
        call_count = 0

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if call_count == 2:
                # afconvert — write the WAV to the output path argument
                Path(cmd[-1]).write_bytes(wav_bytes)
            return mock_result

        return side_effect

    def test_returns_int16_numpy_array(self) -> None:
        """Contract: synthesize_one returns an int16 numpy array."""
        import tests.validation.audio_synthesis.macos_tts as mod

        wav_bytes = _make_wav_bytes(0.1)
        with patch(
            "tests.validation.audio_synthesis.macos_tts.subprocess.run",
            side_effect=self._make_say_afconvert_side_effect(wav_bytes),
        ):
            audio = mod.synthesize_one("Hello.", "Karen")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16

    def test_say_command_uses_correct_voice(self) -> None:
        """Contract: say is invoked with -v <voice_name>."""
        import tests.validation.audio_synthesis.macos_tts as mod

        wav_bytes = _make_wav_bytes(0.1)
        captured_cmds: list[list[str]] = []

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            captured_cmds.append(list(cmd))
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if len(captured_cmds) == 2:
                Path(cmd[-1]).write_bytes(wav_bytes)
            return mock_result

        with patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect):
            mod.synthesize_one("Hello.", "Karen")

        say_cmd = captured_cmds[0]
        assert "-v" in say_cmd
        v_idx = say_cmd.index("-v")
        assert say_cmd[v_idx + 1] == "Karen"

    def test_say_command_uses_double_dash_separator(self) -> None:
        """Contract: say command includes -- before the text to prevent flag parsing."""
        import tests.validation.audio_synthesis.macos_tts as mod

        wav_bytes = _make_wav_bytes(0.1)
        captured_cmds: list[list[str]] = []

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            captured_cmds.append(list(cmd))
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if len(captured_cmds) == 2:
                Path(cmd[-1]).write_bytes(wav_bytes)
            return mock_result

        with patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect):
            mod.synthesize_one("-odd-text starting with dash", "Karen")

        say_cmd = captured_cmds[0]
        assert "--" in say_cmd

    def test_afconvert_targets_16khz_mono(self) -> None:
        """Contract: afconvert is called with LEI16@16000 and -c 1 for 16kHz mono."""
        import tests.validation.audio_synthesis.macos_tts as mod

        wav_bytes = _make_wav_bytes(0.1)
        captured_cmds: list[list[str]] = []

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            captured_cmds.append(list(cmd))
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if len(captured_cmds) == 2:
                Path(cmd[-1]).write_bytes(wav_bytes)
            return mock_result

        with patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect):
            mod.synthesize_one("Hello.", "Daniel")

        afconvert_cmd = captured_cmds[1]
        assert "LEI16@16000" in afconvert_cmd
        assert "-c" in afconvert_cmd
        c_idx = afconvert_cmd.index("-c")
        assert afconvert_cmd[c_idx + 1] == "1"

    def test_custom_say_cmd_prefix_used(self) -> None:
        """Contract: say_cmd override replaces the default 'say' command."""
        import tests.validation.audio_synthesis.macos_tts as mod

        wav_bytes = _make_wav_bytes(0.1)
        captured_cmds: list[list[str]] = []

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            captured_cmds.append(list(cmd))
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if len(captured_cmds) == 2:
                Path(cmd[-1]).write_bytes(wav_bytes)
            return mock_result

        custom_cmd = ["/custom/path/say", "--extra-flag"]
        with patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect):
            mod.synthesize_one("Hello.", "Karen", say_cmd=custom_cmd)

        say_cmd = captured_cmds[0]
        assert say_cmd[0] == "/custom/path/say"
        assert say_cmd[1] == "--extra-flag"

    def test_raises_runtime_error_when_say_fails(self) -> None:
        """Contract: RuntimeError raised when say exits non-zero."""
        import tests.validation.audio_synthesis.macos_tts as mod

        with (
            patch(
                "tests.validation.audio_synthesis.macos_tts.subprocess.run",
                side_effect=subprocess.CalledProcessError(returncode=1, cmd=["say"], stderr=b"unknown voice"),
            ),
            pytest.raises(RuntimeError, match="say failed"),
        ):
            mod.synthesize_one("Hello.", "FakeVoice")

    def test_raises_runtime_error_when_say_times_out(self) -> None:
        """Contract: RuntimeError raised when say times out."""
        import tests.validation.audio_synthesis.macos_tts as mod

        with (
            patch(
                "tests.validation.audio_synthesis.macos_tts.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["say"], timeout=60),
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            mod.synthesize_one("Hello.", "Karen")

    def test_raises_runtime_error_when_afconvert_fails(self) -> None:
        """Contract: RuntimeError raised when afconvert exits non-zero."""
        import tests.validation.audio_synthesis.macos_tts as mod

        call_count = 0

        def side_effect(_cmd: list[str], **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            raise subprocess.CalledProcessError(returncode=1, cmd=["afconvert"], stderr=b"conversion failed")

        with (
            patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect),
            pytest.raises(RuntimeError, match="afconvert failed"),
        ):
            mod.synthesize_one("Hello.", "Karen")

    def test_raises_runtime_error_on_empty_wav_output(self) -> None:
        """Contract: RuntimeError raised when afconvert produces an empty WAV."""
        import tests.validation.audio_synthesis.macos_tts as mod

        call_count = 0

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
            if call_count == 2:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"")
                Path(cmd[-1]).write_bytes(buf.getvalue())
            return mock_result

        with (
            patch("tests.validation.audio_synthesis.macos_tts.subprocess.run", side_effect=side_effect),
            pytest.raises(RuntimeError, match="empty"),
        ):
            mod.synthesize_one("Hello.", "Karen")


# ---------------------------------------------------------------------------
# build_manifest_entry
# ---------------------------------------------------------------------------


class TestBuildManifestEntry:
    """Contract tests for build_manifest_entry."""

    def test_tts_engine_is_macos_say(self, tmp_path: Path) -> None:
        """Contract: tts_engine field is always 'macos_say'."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Karen",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["tts_engine"] == "macos_say"

    def test_tts_voice_matches_voice_name(self, tmp_path: Path) -> None:
        """Contract: tts_voice is set to the supplied voice_name."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Matilda (Premium)",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["tts_voice"] == "Matilda (Premium)"

    def test_audio_id_contains_macos_say_and_voice_name(self, tmp_path: Path) -> None:
        """Contract: audio_id encodes fixture id, tts engine, voice name."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Karen",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        audio_id = str(entry["audio_id"])
        assert "dense-0001" in audio_id
        assert "macos_say" in audio_id
        assert "Karen" in audio_id

    def test_sample_rate_is_16000(self, tmp_path: Path) -> None:
        """Contract: sample_rate_hz is always TARGET_SAMPLE_RATE (16000)."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Karen",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["sample_rate_hz"] == 16000

    def test_duration_computed_from_samples(self, tmp_path: Path) -> None:
        """Contract: duration_seconds = len(audio_samples) / 16000."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        # 8000 samples at 16kHz = 0.5 s
        audio = np.zeros(8000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Karen",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["duration_seconds"] == pytest.approx(0.5, abs=0.001)

    def test_tier_is_clean(self, tmp_path: Path) -> None:
        """Contract: tier field is 'clean' (no noise injection)."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Karen",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["tier"] == "clean"

    def test_required_schema_fields_present(self, tmp_path: Path) -> None:
        """Contract: all required audio manifest schema fields are present."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "test.wav",
            voice_name="Moira",
            audio_samples=audio,
            repo_root=tmp_path,
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
            "bit_depth",
            "acoustic_simulation",
            "noise_profile",
            "snr_db",
        }
        assert required_fields.issubset(entry.keys())

    def test_audio_path_relative_to_repo_root(self, tmp_path: Path) -> None:
        """Contract: audio_path is relative to repo_root when possible."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        audio_file = tmp_path / "subdir" / "audio.wav"
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=audio_file,
            voice_name="Daniel",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        audio_path_str = str(entry["audio_path"])
        # Should be relative, not absolute
        assert not audio_path_str.startswith(str(tmp_path))
        assert "subdir" in audio_path_str

    def test_text_id_matches_fixture_id(self, tmp_path: Path) -> None:
        """Contract: text_id is set to the fixture's id."""
        import tests.validation.audio_synthesis.macos_tts as mod

        fixture = _make_fixture()
        audio = np.zeros(16000, dtype=np.int16)
        entry = mod.build_manifest_entry(
            fixture=fixture,
            audio_path=tmp_path / "audio.wav",
            voice_name="Tessa",
            audio_samples=audio,
            repo_root=tmp_path,
        )
        assert entry["text_id"] == "dense-0001"


# ---------------------------------------------------------------------------
# MACOS_COMMONWEALTH_VOICES panel
# ---------------------------------------------------------------------------


class TestMacosCommonwealthVoices:
    """Contract tests for the MACOS_COMMONWEALTH_VOICES panel constant."""

    def test_is_list_of_two_tuples(self) -> None:
        """Contract: MACOS_COMMONWEALTH_VOICES is a list of (str, str) tuples."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert isinstance(MACOS_COMMONWEALTH_VOICES, list)
        for entry in MACOS_COMMONWEALTH_VOICES:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, locale = entry
            assert isinstance(name, str)
            assert isinstance(locale, str)

    def test_contains_karen_en_au(self) -> None:
        """Contract: Karen (en_AU) is in the panel."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert ("Karen", "en_AU") in MACOS_COMMONWEALTH_VOICES

    def test_contains_daniel_en_gb(self) -> None:
        """Contract: Daniel (en_GB) is in the panel."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert ("Daniel", "en_GB") in MACOS_COMMONWEALTH_VOICES

    def test_contains_moira_en_ie(self) -> None:
        """Contract: Moira (en_IE) is in the panel."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert ("Moira", "en_IE") in MACOS_COMMONWEALTH_VOICES

    def test_contains_rishi_en_in(self) -> None:
        """Contract: Rishi (en_IN) is in the panel."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert ("Rishi", "en_IN") in MACOS_COMMONWEALTH_VOICES

    def test_contains_tessa_en_za(self) -> None:
        """Contract: Tessa (en_ZA) is in the panel."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        assert ("Tessa", "en_ZA") in MACOS_COMMONWEALTH_VOICES

    def test_all_locales_start_with_en(self) -> None:
        """Contract: all locales in the panel are English (en_*)."""
        from tests.validation.scripts.run_multi_voice_e2e import MACOS_COMMONWEALTH_VOICES

        for name, locale in MACOS_COMMONWEALTH_VOICES:
            assert locale.startswith("en_"), f"Voice {name!r} has non-English locale {locale!r}"


# ---------------------------------------------------------------------------
# VoiceSpec with tts_engine="macos_say"
# ---------------------------------------------------------------------------


class TestVoiceSpecMacosEngine:
    """Contract tests for VoiceSpec with tts_engine='macos_say'."""

    def test_default_tts_engine_is_piper(self) -> None:
        """Contract: tts_engine defaults to 'piper' for backward compatibility."""
        from tests.validation.scripts.run_multi_voice_e2e import VoiceSpec

        spec = VoiceSpec(
            name="alan",
            model_path=Path("/fake/alan.onnx"),
            language="en_GB",
            quality="medium",
        )
        assert spec.tts_engine == "piper"

    def test_tts_engine_macos_say_accepted(self) -> None:
        """Contract: tts_engine='macos_say' is accepted without error."""
        from tests.validation.scripts.run_multi_voice_e2e import VoiceSpec

        spec = VoiceSpec(
            name="Karen",
            model_path=Path("/usr/bin/say"),
            language="en_AU",
            quality="system",
            tts_engine="macos_say",
        )
        assert spec.tts_engine == "macos_say"

    def test_display_name_for_macos_voice(self) -> None:
        """Contract: display_name for macOS voice is language-name-quality."""
        from tests.validation.scripts.run_multi_voice_e2e import VoiceSpec

        spec = VoiceSpec(
            name="Karen",
            model_path=Path("/usr/bin/say"),
            language="en_AU",
            quality="system",
            tts_engine="macos_say",
        )
        assert spec.display_name == "en_AU-Karen-system"

    def test_display_name_macos_no_speaker_id(self) -> None:
        """Contract: macOS VoiceSpec display_name has no speaker suffix when no speaker_id."""
        from tests.validation.scripts.run_multi_voice_e2e import VoiceSpec

        spec = VoiceSpec(
            name="Daniel",
            model_path=Path("/usr/bin/say"),
            language="en_GB",
            quality="system",
            tts_engine="macos_say",
        )
        assert "speaker" not in spec.display_name


# ---------------------------------------------------------------------------
# _load_macos_voice_specs
# ---------------------------------------------------------------------------


class TestLoadMacosVoiceSpecs:
    """Contract tests for _load_macos_voice_specs."""

    def test_returns_voice_specs(self) -> None:
        """Contract: returns a list of VoiceSpec objects."""
        from tests.validation.scripts.run_multi_voice_e2e import (
            MACOS_COMMONWEALTH_VOICES,
            VoiceSpec,
            _load_macos_voice_specs,
        )

        specs = _load_macos_voice_specs(MACOS_COMMONWEALTH_VOICES)
        assert all(isinstance(s, VoiceSpec) for s in specs)

    def test_length_matches_input(self) -> None:
        """Contract: one VoiceSpec returned per input tuple."""
        from tests.validation.scripts.run_multi_voice_e2e import (
            MACOS_COMMONWEALTH_VOICES,
            _load_macos_voice_specs,
        )

        specs = _load_macos_voice_specs(MACOS_COMMONWEALTH_VOICES)
        assert len(specs) == len(MACOS_COMMONWEALTH_VOICES)

    def test_all_specs_have_macos_say_engine(self) -> None:
        """Contract: every returned spec has tts_engine='macos_say'."""
        from tests.validation.scripts.run_multi_voice_e2e import (
            MACOS_COMMONWEALTH_VOICES,
            _load_macos_voice_specs,
        )

        specs = _load_macos_voice_specs(MACOS_COMMONWEALTH_VOICES)
        for spec in specs:
            assert spec.tts_engine == "macos_say", f"Spec {spec.name!r} has wrong engine: {spec.tts_engine!r}"

    def test_all_specs_have_system_quality(self) -> None:
        """Contract: every returned spec has quality='system'."""
        from tests.validation.scripts.run_multi_voice_e2e import (
            MACOS_COMMONWEALTH_VOICES,
            _load_macos_voice_specs,
        )

        specs = _load_macos_voice_specs(MACOS_COMMONWEALTH_VOICES)
        for spec in specs:
            assert spec.quality == "system"

    def test_model_path_is_say_sentinel(self) -> None:
        """Contract: model_path is /usr/bin/say sentinel for all macOS specs."""
        from tests.validation.scripts.run_multi_voice_e2e import (
            MACOS_COMMONWEALTH_VOICES,
            _load_macos_voice_specs,
        )

        specs = _load_macos_voice_specs(MACOS_COMMONWEALTH_VOICES)
        for spec in specs:
            assert spec.model_path == Path("/usr/bin/say")

    def test_voice_names_preserved(self) -> None:
        """Contract: voice names from input tuples are preserved in VoiceSpec."""
        from tests.validation.scripts.run_multi_voice_e2e import _load_macos_voice_specs

        voices = [("Karen", "en_AU"), ("Daniel", "en_GB")]
        specs = _load_macos_voice_specs(voices)
        names = [s.name for s in specs]
        assert names == ["Karen", "Daniel"]

    def test_empty_input_returns_empty_list(self) -> None:
        """Contract: empty input produces empty output."""
        from tests.validation.scripts.run_multi_voice_e2e import _load_macos_voice_specs

        specs = _load_macos_voice_specs([])
        assert specs == []
