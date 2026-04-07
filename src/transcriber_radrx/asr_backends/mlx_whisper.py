"""MLX Whisper backend.

Wraps `mlx_whisper.transcribe()` behind the ASRBackend protocol. This
backend is the Whisper baseline for the model bake-off.

Dependency: `transcriber-radrx[asr-whisper-mlx]` extra, which pulls
`mlx-whisper>=0.4.0`. Apple Silicon only.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
from pathlib import Path

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError

logger = logging.getLogger(__name__)


class MlxWhisperBackend(ASRBackend):
    """MLX-native Whisper backend for Apple Silicon.

    Attributes:
        name: Always "mlx_whisper".
        model_id: HuggingFace model identifier for mlx-whisper (e.g.
            "mlx-community/whisper-large-v3-mlx", "mlx-community/whisper-large-v3-turbo").
    """

    name = "mlx_whisper"

    def __init__(
        self,
        model_id: str = "mlx-community/whisper-large-v3-mlx",
    ) -> None:
        """Initialize the MLX Whisper backend.

        Args:
            model_id: mlx-whisper model identifier. Defaults to the full
                large-v3 model, NOT the turbo variant. Turbo is faster but
                has a known accuracy trade-off from the reduced decoder.
        """
        self.model_id = model_id
        self._loaded = False
        self._system_prompt_warned = False

    def load(self) -> None:
        """Lazily import mlx_whisper and mark the backend ready.

        mlx_whisper downloads/loads the model on first transcribe() call,
        so this method primarily exists to surface import errors early
        (so a missing dependency fails fast, not mid-batch).
        """
        if self._loaded:
            return
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            msg = "mlx_whisper is not installed. Install with: uv sync --extra asr-whisper-mlx"
            raise ASRBackendError(msg) from e
        self._loaded = True

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Transcribe one WAV file with mlx_whisper.

        Whisper is not an instruction-following model, so `system_prompt`
        is ignored with a one-time log warning. If you want to bias
        Whisper toward specific vocabulary, use `initial_prompt` instead.
        """
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        if system_prompt is not None and not self._system_prompt_warned:
            logger.info(
                "[%s] system_prompt ignored (Whisper is not instruction-following; "
                "use initial_prompt for soft vocabulary biasing)",
                self.name,
            )
            self._system_prompt_warned = True

        self.load()
        import mlx_whisper

        decode_options: dict[str, str] = {"language": language}
        if initial_prompt is not None:
            decode_options["initial_prompt"] = initial_prompt

        logger.info("[%s] transcribing %s with model %s", self.name, audio_path, self.model_id)
        try:
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_id,
                **decode_options,
            )
        except Exception as e:
            msg = f"mlx_whisper transcription failed: {e}"
            raise ASRBackendError(msg) from e

        raw_text: str = result.get("text", "") if isinstance(result, dict) else ""
        return raw_text

    def unload(self) -> None:
        """Release MLX model resources.

        mlx_whisper caches the model globally at the module level, so
        unload is effectively a hint. Marking the backend as unloaded
        lets a subsequent load() re-verify the import.
        """
        self._loaded = False
