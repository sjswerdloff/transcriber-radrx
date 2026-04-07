"""Tests for the ASR backend registry and base protocol.

These tests cover registry behavior and contract compliance. Backend-
specific behavior (like actually running Whisper or MedASR) is exercised
by the validation suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber_radrx.asr_backends import (
    BACKENDS,
    ASRBackend,
    ASRBackendError,
    UnsupportedFeatureError,
    get_backend,
    register_backend,
)


class TestRegistry:
    """Contract tests for the backend registry."""

    def test_default_backends_registered(self) -> None:
        """Contract: mlx_whisper and medasr are always available by name."""
        assert "mlx_whisper" in BACKENDS
        assert "medasr" in BACKENDS

    def test_get_unknown_backend_raises_keyerror(self) -> None:
        """Contract: requesting an unregistered backend raises KeyError with
        a helpful message listing available backends."""
        with pytest.raises(KeyError, match="Unknown ASR backend"):
            get_backend("nonexistent_backend")

    def test_register_custom_backend(self) -> None:
        """Contract: register_backend adds a new factory to the registry."""

        class _FakeBackend:
            name = "fake"
            model_id = "fake/model"

            def load(self) -> None: ...

            def transcribe_wav(
                self,
                audio_path: Path,  # noqa: ARG002
                *,
                language: str = "en",  # noqa: ARG002
                initial_prompt: str | None = None,  # noqa: ARG002
            ) -> str:
                return "fake transcription"

            def unload(self) -> None: ...

        def factory(**kwargs: object) -> ASRBackend:  # noqa: ARG001
            return _FakeBackend()  # type: ignore[return-value]

        try:
            register_backend("fake_test_backend", factory)
            backend = get_backend("fake_test_backend")
            assert backend.name == "fake"
            assert backend.transcribe_wav(Path("/nonexistent")) == "fake transcription"
        finally:
            BACKENDS.pop("fake_test_backend", None)


class TestExceptions:
    """Base exception hierarchy."""

    def test_asr_backend_error_is_runtime_error(self) -> None:
        assert issubclass(ASRBackendError, RuntimeError)

    def test_unsupported_feature_error_is_asr_backend_error(self) -> None:
        assert issubclass(UnsupportedFeatureError, ASRBackendError)


class TestMlxWhisperBackend:
    """Contract tests for the MLX Whisper backend.

    Uses the unittest.mock patch pattern so tests run without needing
    the mlx_whisper runtime. Real transcription is exercised by the
    validation suite.
    """

    def test_default_model_is_large_v3_non_turbo(self) -> None:
        """Contract: default model is whisper-large-v3 (NOT turbo).

        Turbo is the fast variant with a known accuracy trade-off.
        The default should be the full-quality model; turbo is opt-in
        via an explicit model_id argument.
        """
        from transcriber_radrx.asr_backends.mlx_whisper import MlxWhisperBackend

        backend = MlxWhisperBackend()
        assert backend.model_id == "mlx-community/whisper-large-v3-mlx"
        assert "turbo" not in backend.model_id

    def test_custom_model_id(self) -> None:
        from transcriber_radrx.asr_backends.mlx_whisper import MlxWhisperBackend

        backend = MlxWhisperBackend(model_id="mlx-community/whisper-large-v3-turbo")
        assert backend.model_id == "mlx-community/whisper-large-v3-turbo"

    def test_name_is_stable(self) -> None:
        from transcriber_radrx.asr_backends.mlx_whisper import MlxWhisperBackend

        backend = MlxWhisperBackend()
        assert backend.name == "mlx_whisper"

    def test_transcribe_missing_file_raises(self) -> None:
        from transcriber_radrx.asr_backends.mlx_whisper import MlxWhisperBackend

        backend = MlxWhisperBackend()
        with pytest.raises(FileNotFoundError):
            backend.transcribe_wav(Path("/nonexistent/audio.wav"))


class TestMedASRBackend:
    """Contract tests for the MedASR backend.

    Real model loading is skipped here because it requires gated-repo
    authentication and significant download time. These tests cover
    the construction contract only. A requires_audio marked test
    in the validation suite exercises the real path.
    """

    def test_default_model_is_ainergiz_fp16(self) -> None:
        """Contract: default is ainergiz/medasr-mlx-fp16.

        This is the port compatible with the vendored loader
        (weights.npz format) and the model card reports 0.0% WER delta
        vs the PyTorch source, so it is effectively lossless.
        """
        from transcriber_radrx.asr_backends.medasr import MedASRBackend

        backend = MedASRBackend()
        assert backend.model_id == "ainergiz/medasr-mlx-fp16"

    def test_default_decode_mode_is_greedy(self) -> None:
        """Contract: default decode mode is greedy (no kenlm dependency)."""
        from transcriber_radrx.asr_backends.medasr import MedASRBackend

        backend = MedASRBackend()
        assert backend.decode_mode == "greedy"

    def test_name_is_stable(self) -> None:
        from transcriber_radrx.asr_backends.medasr import MedASRBackend

        backend = MedASRBackend()
        assert backend.name == "medasr"

    def test_transcribe_missing_file_raises(self) -> None:
        from transcriber_radrx.asr_backends.medasr import MedASRBackend

        backend = MedASRBackend()
        with pytest.raises(FileNotFoundError):
            backend.transcribe_wav(Path("/nonexistent/audio.wav"))

    def test_beam_mode_accepts_kenlm_params(self) -> None:
        """Contract: beam-mode constructor accepts alpha, beta, width."""
        from transcriber_radrx.asr_backends.medasr import MedASRBackend

        backend = MedASRBackend(
            decode_mode="beam",
            beam_width=64,
            kenlm_alpha=0.3,
            kenlm_beta=0.5,
        )
        assert backend.decode_mode == "beam"
        assert backend.beam_width == 64
        assert backend.kenlm_alpha == 0.3
        assert backend.kenlm_beta == 0.5
