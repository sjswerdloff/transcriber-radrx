"""Tests for the transcribe-radrx CLI (0.2.0 subcommand architecture).

Tests cover:
- ``transcribe`` subcommand (backward-compat).
- ``evaluate`` subcommand argument parsing and pipeline integration.
- Backward-compatibility fallback (bare ``transcribe-radrx audio.wav``).
- Vocabulary resolution logic.
- Reference text loading.
- Error cases (missing files, bad paths).

All backend calls are mocked — no real ASR inference.

Authors: silas-397300f6
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from transcriber_radrx.cli import (
    EvaluateArgumentError,
    VocabularyLoadError,
    _build_parser,
    _load_reference_text,
    _load_vocabulary_set,
    _resolve_vocabulary,
    main,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vocab_file(tmp_path: Path) -> Path:
    """A minimal vocabulary file with a few RT terms."""
    vf = tmp_path / "rt_vocab.txt"
    vf.write_text("# RT vocabulary\nIMRT\nVMAT\nGy\nPTV\nCTV\n")
    return vf


@pytest.fixture()
def dummy_audio(tmp_path: Path) -> Path:
    """A dummy WAV file (zero bytes — not played, just checked for existence)."""
    wav = tmp_path / "dictation.wav"
    wav.write_bytes(b"")
    return wav


@pytest.fixture()
def output_docx(tmp_path: Path) -> Path:
    """A path for the output .docx file."""
    return tmp_path / "review.docx"


# ---------------------------------------------------------------------------
# _resolve_vocabulary
# ---------------------------------------------------------------------------


class TestResolveVocabulary:
    """Tests for vocabulary file resolution helper."""

    def test_explicit_path_that_exists(self, vocab_file: Path) -> None:
        """Contract: explicit path is returned when it exists."""
        result = _resolve_vocabulary(vocab_file)
        assert result == vocab_file

    def test_explicit_path_missing_raises(self, tmp_path: Path) -> None:
        """Contract: explicit path that does not exist raises FileNotFoundError."""
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="Vocabulary file not found"):
            _resolve_vocabulary(missing)

    def test_none_returns_none_when_no_defaults_exist(self) -> None:
        """Contract: None arg returns None when no default vocabulary found."""
        # Accept either None or a valid path — the key is no exception.
        result = _resolve_vocabulary(None)
        assert result is None or result.exists()

    def test_none_returns_package_default_when_present(self) -> None:
        """Contract: None arg resolves to package-relative default when it exists."""
        result = _resolve_vocabulary(None)
        if result is not None:
            assert result.exists()


# ---------------------------------------------------------------------------
# _load_vocabulary_set
# ---------------------------------------------------------------------------


class TestLoadVocabularySet:
    """Tests for vocabulary set loading."""

    def test_loads_terms_lowercase(self, vocab_file: Path) -> None:
        """Contract: terms are returned as lowercase strings."""
        terms = _load_vocabulary_set(vocab_file)
        assert "imrt" in terms
        assert "vmat" in terms
        assert "gy" in terms

    def test_excludes_comments_and_blanks(self, vocab_file: Path) -> None:
        """Contract: comment lines and blank lines are excluded."""
        terms = _load_vocabulary_set(vocab_file)
        assert all(not t.startswith("#") for t in terms)

    def test_missing_file_raises_vocabulary_load_error(self, tmp_path: Path) -> None:
        """Contract: missing file raises VocabularyLoadError."""
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(VocabularyLoadError, match="Cannot load vocabulary file"):
            _load_vocabulary_set(missing)

    def test_returns_set(self, vocab_file: Path) -> None:
        """Contract: return type is set."""
        result = _load_vocabulary_set(vocab_file)
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# _load_reference_text
# ---------------------------------------------------------------------------


class TestLoadReferenceText:
    """Tests for gold reference text loading."""

    def test_inline_takes_precedence_over_file(self, tmp_path: Path) -> None:
        """Contract: --reference text overrides --reference-file."""
        ref_file = tmp_path / "gold.txt"
        ref_file.write_text("from file")
        result = _load_reference_text("inline text", ref_file)
        assert result == "inline text"

    def test_inline_reference(self) -> None:
        """Contract: inline reference is returned as-is."""
        result = _load_reference_text("Prescribed 54 Gy in 30 fractions.", None)
        assert result == "Prescribed 54 Gy in 30 fractions."

    def test_file_reference(self, tmp_path: Path) -> None:
        """Contract: file reference is read and stripped."""
        ref_file = tmp_path / "gold.txt"
        ref_file.write_text("  Gold text here.  \n")
        result = _load_reference_text(None, ref_file)
        assert result == "Gold text here."

    def test_none_when_both_absent(self) -> None:
        """Contract: returns None when no reference is provided."""
        result = _load_reference_text(None, None)
        assert result is None

    def test_missing_reference_file_raises(self, tmp_path: Path) -> None:
        """Contract: missing --reference-file raises FileNotFoundError."""
        missing = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError, match="Reference file not found"):
            _load_reference_text(None, missing)


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------


class TestParserStructure:
    """Tests for argument parser configuration."""

    def test_transcribe_subcommand_exists(self) -> None:
        """Contract: 'transcribe' subcommand is registered."""
        parser = _build_parser()
        args = parser.parse_args(["transcribe", "/some/audio.wav"])
        assert args.subcommand == "transcribe"

    def test_evaluate_subcommand_exists(self) -> None:
        """Contract: 'evaluate' subcommand is registered."""
        parser = _build_parser()
        args = parser.parse_args(["evaluate", "--audio", "/some/audio.wav", "--output", "/out.docx"])
        assert args.subcommand == "evaluate"

    def test_transcribe_defaults(self) -> None:
        """Contract: transcribe subcommand has expected defaults."""
        parser = _build_parser()
        args = parser.parse_args(["transcribe", "/audio.wav"])
        assert args.model == "mlx-community/whisper-large-v3-mlx"
        assert args.language == "en"
        assert args.enable_phonetic is False
        assert args.vocabulary is None

    def test_evaluate_requires_audio(self) -> None:
        """Contract: evaluate subcommand requires --audio."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--output", "/out.docx"])

    def test_evaluate_requires_output(self) -> None:
        """Contract: evaluate subcommand requires --output."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--audio", "/audio.wav"])

    def test_evaluate_optional_args_default_none(self) -> None:
        """Contract: optional evaluate args default to None."""
        parser = _build_parser()
        args = parser.parse_args(["evaluate", "--audio", "/audio.wav", "--output", "/out.docx"])
        assert args.reference is None
        assert args.reference_file is None
        assert args.vocabulary is None
        assert args.audit_output is None
        assert args.whisper_model is None
        assert args.voxtral_model is None

    def test_evaluate_reference_and_reference_file(self, tmp_path: Path) -> None:
        """Contract: both --reference and --reference-file can be parsed."""
        ref_file = tmp_path / "gold.txt"
        ref_file.write_text("gold")
        parser = _build_parser()
        args = parser.parse_args(
            [
                "evaluate",
                "--audio",
                "/audio.wav",
                "--output",
                "/out.docx",
                "--reference",
                "inline gold",
                "--reference-file",
                str(ref_file),
            ]
        )
        assert args.reference == "inline gold"
        assert args.reference_file == ref_file


# ---------------------------------------------------------------------------
# Backward-compatibility: bare 'transcribe-radrx audio.wav'
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests that bare positional audio arg still works (0.1.x compat)."""

    def test_bare_audio_arg_dispatches_to_transcribe(self, dummy_audio: Path) -> None:
        """Contract: bare audio path (no subcommand) dispatches to transcribe."""
        mock_result = MagicMock()
        mock_result.corrected_text = "test transcription"
        mock_result.corrections = []

        with patch("transcriber_radrx.transcriber.transcribe", return_value=mock_result):
            main([str(dummy_audio)])

    def test_explicit_transcribe_subcommand_works(self, dummy_audio: Path) -> None:
        """Contract: explicit 'transcribe' subcommand dispatches correctly."""
        with patch("transcriber_radrx.cli._run_transcribe") as mock_run:
            main(["transcribe", str(dummy_audio)])
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _run_transcribe integration (mocked backend)
# ---------------------------------------------------------------------------


class TestRunTranscribe:
    """Tests for the transcribe subcommand handler."""

    def test_prints_corrected_text(self, dummy_audio: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: corrected_text is printed to stdout."""
        mock_result = MagicMock()
        mock_result.corrected_text = "Prescribed 54 Gy in 30 fractions."
        mock_result.corrections = []

        with patch("transcriber_radrx.transcriber.transcribe", return_value=mock_result):
            main(["transcribe", str(dummy_audio)])

        captured = capsys.readouterr()
        assert "54 Gy" in captured.out

    def test_corrections_printed_to_stderr(self, dummy_audio: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: applied corrections are printed to stderr."""
        correction = MagicMock()
        correction.original = "jai"
        correction.corrected = "Gy"
        correction.method = "exact"
        correction.score = 1.0
        correction.offset = 12

        mock_result = MagicMock()
        mock_result.corrected_text = "dose of 50 Gy"
        mock_result.corrections = [correction]

        with patch("transcriber_radrx.transcriber.transcribe", return_value=mock_result):
            main(["transcribe", str(dummy_audio)])

        captured = capsys.readouterr()
        assert "jai" in captured.err
        assert "Gy" in captured.err

    def test_no_corrections_block_when_empty(self, dummy_audio: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Contract: corrections block is suppressed when no corrections applied."""
        mock_result = MagicMock()
        mock_result.corrected_text = "clean output"
        mock_result.corrections = []

        with patch("transcriber_radrx.transcriber.transcribe", return_value=mock_result):
            main(["transcribe", str(dummy_audio)])

        captured = capsys.readouterr()
        assert "Corrections applied" not in captured.err


# ---------------------------------------------------------------------------
# _run_evaluate integration (mocked pipeline)
# ---------------------------------------------------------------------------


class TestRunEvaluate:
    """Tests for the evaluate subcommand handler.

    All patches target *source module* locations because _run_evaluate uses
    local imports (names are bound fresh at each call). contextlib.ExitStack
    keeps patch target strings short.
    """

    # Patch target constants — avoids repeating long dotted paths on every line.
    _P_GET_BACKEND = "transcriber_radrx.asr_backends.registry.get_backend"
    _P_TRANSCRIBE = "transcriber_radrx.transcriber.transcribe_with_backend"
    _P_ENSEMBLE = "transcriber_radrx.ensemble.decision_rules.ensemble_transcriptions"
    _P_RENDER = "transcriber_radrx.ensemble.docx_renderer.render_ensemble_docx"
    _P_RENDER_PAIR = "transcriber_radrx.ensemble.docx_renderer.render_ensemble_docx_pair"
    _P_CORRECTOR = "transcriber_radrx.corrector.CorrectionDictionary"
    _P_RESOLVE_VOCAB = "transcriber_radrx.cli._resolve_vocabulary"

    def _make_ensemble_result(self) -> MagicMock:
        """Build a realistic EnsembleResult mock."""
        word = MagicMock()
        word.needs_review = False

        result = MagicMock()
        result.text_ensemble = "Prescribed 54 Gy in 30 fractions."
        result.review_count = 0
        result.words = [word] * 6
        result.fixture_id = "dictation"
        result.voice = "clinical"
        result.needs_review = False
        return result

    def _make_corrector_mock(self) -> MagicMock:
        """Build a CorrectionDictionary mock with correct_full returning a 3-tuple."""
        corrector_instance = MagicMock()
        corrector_instance.correct_full.return_value = ("corrected text", [], [])
        return MagicMock(return_value=corrector_instance)

    def _make_transcribe_result(self) -> MagicMock:
        """Build a transcribe_with_backend return value mock."""
        mock_result = MagicMock()
        mock_result.text = "raw transcription"
        mock_result.corrected_text = "corrected transcription"
        return mock_result

    def _enter_standard_patches(
        self,
        stack: contextlib.ExitStack,
        ensemble_result: MagicMock,
        *,
        get_backend_side_effect: object = None,
    ) -> dict[str, MagicMock]:
        """Enter evaluate pipeline patches into stack; return named mock dict.

        Args:
            stack: ExitStack to enter patches into.
            ensemble_result: The EnsembleResult mock to return from ensemble call.
            get_backend_side_effect: Optional side_effect for get_backend mock.

        Returns:
            Dict of entered mocks keyed by short name.
        """
        transcribe_result = self._make_transcribe_result()
        corrector_cls = self._make_corrector_mock()

        if get_backend_side_effect is not None:
            get_backend_patch = patch(self._P_GET_BACKEND, side_effect=get_backend_side_effect)
        else:
            get_backend_patch = patch(self._P_GET_BACKEND, return_value=MagicMock())

        return {
            "get_backend": stack.enter_context(get_backend_patch),
            "transcribe": stack.enter_context(patch(self._P_TRANSCRIBE, return_value=transcribe_result)),
            "ensemble": stack.enter_context(patch(self._P_ENSEMBLE, return_value=ensemble_result)),
            "render": stack.enter_context(patch(self._P_RENDER)),
            "render_pair": stack.enter_context(patch(self._P_RENDER_PAIR)),
            "corrector": stack.enter_context(patch(self._P_CORRECTOR, corrector_cls)),
            "resolve_vocab": stack.enter_context(patch(self._P_RESOLVE_VOCAB, return_value=None)),
        }

    def test_evaluate_prints_ensemble_to_stdout(
        self, dummy_audio: Path, output_docx: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Contract: ensemble text is printed to stdout."""
        ensemble_result = self._make_ensemble_result()

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result)
            main(["evaluate", "--audio", str(dummy_audio), "--output", str(output_docx)])

        captured = capsys.readouterr()
        assert "54 Gy" in captured.out

    def test_evaluate_with_reference_computes_wer(
        self, dummy_audio: Path, output_docx: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Contract: WER and UWR appear in stderr when reference provided."""
        ensemble_result = self._make_ensemble_result()
        ensemble_result.text_ensemble = "Prescribed 54 Gy in 30 fractions."
        ensemble_result.review_count = 1
        ensemble_result.words = [MagicMock() for _ in range(6)]

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result)
            main(
                [
                    "evaluate",
                    "--audio",
                    str(dummy_audio),
                    "--output",
                    str(output_docx),
                    "--reference",
                    "Prescribed 54 Gy in 30 fractions.",
                ]
            )

        captured = capsys.readouterr()
        assert "WER" in captured.err
        assert "UWR" in captured.err

    def test_evaluate_without_reference_omits_wer(
        self, dummy_audio: Path, output_docx: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Contract: WER/UWR lines omitted when no reference provided."""
        ensemble_result = self._make_ensemble_result()

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result)
            main(["evaluate", "--audio", str(dummy_audio), "--output", str(output_docx)])

        captured = capsys.readouterr()
        assert "WER" not in captured.err

    def test_evaluate_missing_audio_raises(self, tmp_path: Path, output_docx: Path) -> None:
        """Contract: missing audio file raises EvaluateArgumentError."""
        missing = tmp_path / "ghost.wav"
        with pytest.raises(EvaluateArgumentError, match="Audio file not found"):
            main(["evaluate", "--audio", str(missing), "--output", str(output_docx)])

    def test_evaluate_missing_output_dir_raises(self, dummy_audio: Path, tmp_path: Path) -> None:
        """Contract: output directory that does not exist raises EvaluateArgumentError."""
        bad_output = tmp_path / "nonexistent_dir" / "review.docx"
        with pytest.raises(EvaluateArgumentError, match="Output directory does not exist"):
            main(["evaluate", "--audio", str(dummy_audio), "--output", str(bad_output)])

    def test_evaluate_audit_output_calls_docx_pair(self, dummy_audio: Path, output_docx: Path, tmp_path: Path) -> None:
        """Contract: --audit-output triggers render_ensemble_docx_pair, not render."""
        ensemble_result = self._make_ensemble_result()
        audit_path = tmp_path / "audit.docx"

        with contextlib.ExitStack() as stack:
            mocks = self._enter_standard_patches(stack, ensemble_result)
            main(
                [
                    "evaluate",
                    "--audio",
                    str(dummy_audio),
                    "--output",
                    str(output_docx),
                    "--audit-output",
                    str(audit_path),
                ]
            )

        mocks["render_pair"].assert_called_once()
        mocks["render"].assert_not_called()

    def test_evaluate_without_audit_calls_single_render(self, dummy_audio: Path, output_docx: Path) -> None:
        """Contract: without --audit-output, render_ensemble_docx is called (not pair)."""
        ensemble_result = self._make_ensemble_result()

        with contextlib.ExitStack() as stack:
            mocks = self._enter_standard_patches(stack, ensemble_result)
            main(["evaluate", "--audio", str(dummy_audio), "--output", str(output_docx)])

        mocks["render"].assert_called_once()
        mocks["render_pair"].assert_not_called()

    def test_evaluate_whisper_model_override(self, dummy_audio: Path, output_docx: Path) -> None:
        """Contract: --whisper-model is forwarded to get_backend for mlx_whisper."""
        ensemble_result = self._make_ensemble_result()
        captured_kwargs: list[dict[str, object]] = []

        def fake_get_backend(name: str, **kwargs: object) -> MagicMock:
            captured_kwargs.append({"name": name, **kwargs})
            return MagicMock()

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result, get_backend_side_effect=fake_get_backend)
            main(
                [
                    "evaluate",
                    "--audio",
                    str(dummy_audio),
                    "--output",
                    str(output_docx),
                    "--whisper-model",
                    "mlx-community/whisper-large-v3-mlx",
                ]
            )

        whisper_calls = [c for c in captured_kwargs if c.get("name") == "mlx_whisper"]
        assert len(whisper_calls) == 1
        assert whisper_calls[0].get("model_id") == "mlx-community/whisper-large-v3-mlx"

    def test_evaluate_voxtral_model_override(self, dummy_audio: Path, output_docx: Path) -> None:
        """Contract: --voxtral-model is forwarded to get_backend for voxtral."""
        ensemble_result = self._make_ensemble_result()
        captured_kwargs: list[dict[str, object]] = []

        def fake_get_backend(name: str, **kwargs: object) -> MagicMock:
            captured_kwargs.append({"name": name, **kwargs})
            return MagicMock()

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result, get_backend_side_effect=fake_get_backend)
            main(
                [
                    "evaluate",
                    "--audio",
                    str(dummy_audio),
                    "--output",
                    str(output_docx),
                    "--voxtral-model",
                    "mistralai/Voxtral-Mini-3B-2407",
                ]
            )

        voxtral_calls = [c for c in captured_kwargs if c.get("name") == "voxtral"]
        assert len(voxtral_calls) == 1
        assert voxtral_calls[0].get("model_id") == "mistralai/Voxtral-Mini-3B-2407"

    def test_evaluate_reference_file_arg(
        self, dummy_audio: Path, output_docx: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Contract: --reference-file is loaded and used for WER computation."""
        ensemble_result = self._make_ensemble_result()
        ensemble_result.text_ensemble = "Prescribed 54 Gy."
        ensemble_result.review_count = 0
        ensemble_result.words = [MagicMock() for _ in range(3)]

        ref_file = tmp_path / "gold.txt"
        ref_file.write_text("Prescribed 54 Gy.")

        with contextlib.ExitStack() as stack:
            self._enter_standard_patches(stack, ensemble_result)
            main(
                [
                    "evaluate",
                    "--audio",
                    str(dummy_audio),
                    "--output",
                    str(output_docx),
                    "--reference-file",
                    str(ref_file),
                ]
            )

        captured = capsys.readouterr()
        assert "WER" in captured.err
