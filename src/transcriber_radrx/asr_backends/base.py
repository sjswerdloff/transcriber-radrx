"""ASR backend protocol and shared exceptions.

Every concrete backend (MLX Whisper, MedASR, Parakeet, etc.) implements
the ASRBackend protocol so the rest of the pipeline can treat them
interchangeably.

Authors: silas-397300f6
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class ASRBackendError(RuntimeError):
    """Base exception for ASR backend failures."""


class UnsupportedFeatureError(ASRBackendError):
    """Raised when a backend is asked for a feature it does not support.

    For example, Parakeet TDT does not support hotword/vocabulary biasing
    via a prompt, so it raises this if transcribe_wav is called with an
    initial_prompt argument.
    """


@runtime_checkable
class ASRBackend(Protocol):
    """Protocol for a pluggable ASR backend.

    Contract:
        - `name` is a stable short identifier (e.g., "mlx_whisper", "medasr")
          used in reports and manifests.
        - `model_id` identifies the weights/config (e.g., a HuggingFace repo).
        - `load()` eagerly loads the model. Safe to call multiple times;
          subsequent calls are no-ops.
        - `transcribe_wav()` takes a path to a 16 kHz mono WAV file and
          returns the raw transcribed text. No post-processing correction
          — that is applied downstream by `transcriber_radrx.corrector`.
        - `initial_prompt` is a soft-bias string (Whisper-style) if the
          backend supports it. Backends that don't may ignore it, or
          raise UnsupportedFeatureError if the caller insists.
        - `system_prompt` is an instruction-following directive for
          audio-LLM backends (Granite-Speech, Voxtral, Phi-4-multimodal,
          etc.). It is DIFFERENT from `initial_prompt`: `initial_prompt`
          is a soft decoder bias (Whisper-style), `system_prompt` is a
          chat-template system instruction. Classical ASRs that cannot
          be instructed (Whisper, MedASR, Cohere Transcribe) log a
          warning on first use and ignore it. Audio-LLM backends fold
          it into their chat template.
        - `unload()` frees model memory. Called between model comparisons
          in the bake-off runner to keep peak memory bounded.

    Backends should NOT apply any text normalization beyond what their
    native inference produces. Normalization for WER comparison happens
    downstream so all backends are normalized identically.
    """

    name: str
    model_id: str

    def load(self) -> None:
        """Eagerly load the model. Idempotent."""
        ...

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Transcribe one 16 kHz mono WAV file.

        Args:
            audio_path: Path to a mono 16 kHz WAV file.
            language: Language code (most backends default to English).
            initial_prompt: Optional vocabulary biasing prompt (Whisper-
                style soft bias). Ignored by backends that do not support
                it unless strict=True.
            system_prompt: Optional instruction-following directive for
                audio-LLM backends. Classical ASRs ignore with a warning.
                Audio-LLMs fold it into their chat template.

        Returns:
            Raw transcription text, unmodified.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ASRBackendError: If inference fails.
        """
        ...

    def unload(self) -> None:
        """Release model resources. Idempotent."""
        ...
