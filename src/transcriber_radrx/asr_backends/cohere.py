"""Cohere Transcribe backend.

Wraps Cohere Labs' `cohere-transcribe-03-2026` model via native Hugging
Face transformers. Cohere Transcribe is a 2B-parameter Conformer
encoder + Transformer decoder model that currently leads the Hugging
Face Open ASR Leaderboard at ~5.42% average English WER (as of
March 2026).

Runs on Apple Silicon via `device_map="auto"` (which routes to MPS when
available) with fp16 weights. Memory footprint: ~4.5 GB loaded.

Dependency: `transcriber-radrx[asr-cohere]` extra, which pulls
transformers>=5.4.0 (Cohere's native `CohereAsrForConditionalGeneration`
was added in 5.4), torch>=2.2 (for MPS fp16 support), soundfile, scipy
(for resampling).

Cohere Transcribe is an encoder-decoder model. Like Whisper, it supports
a `language` hint passed through the processor. Unlike Whisper, the
README does NOT document an `initial_prompt` mechanism, so we ignore the
`initial_prompt` argument (with a log message on first use) rather than
raising.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError

logger = logging.getLogger(__name__)


class CohereBackend(ASRBackend):
    """Cohere Transcribe backend via native Hugging Face transformers.

    Attributes:
        name: Always "cohere".
        model_id: HuggingFace repo ID of the Cohere Transcribe weights.
        device: Torch device string ("auto", "mps", "cuda", "cpu").
        torch_dtype: Weight dtype as a string ("float16", "bfloat16",
            "float32"). Float16 is the recommended default on Apple
            Silicon — bfloat16 has limited MPS support as of early 2026.
    """

    name = "cohere"

    def __init__(
        self,
        model_id: str = "CohereLabs/cohere-transcribe-03-2026",
        *,
        device: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 256,
    ) -> None:
        """Initialize the Cohere Transcribe backend.

        Args:
            model_id: HuggingFace repo ID. Defaults to the March-2026
                Cohere Transcribe 2B model, which leads the Open ASR
                Leaderboard.
            device: Torch device placement. "auto" delegates to
                `device_map="auto"` (which picks MPS on Apple Silicon,
                CUDA on NVIDIA, CPU otherwise). Pass "mps", "cuda", or
                "cpu" to force a specific device.
            torch_dtype: Weight dtype. "float16" fits in ~4.5 GB and is
                the right default for 16–32 GB Apple Silicon machines.
                Use "float32" if you hit MPS numerical issues with fp16
                (unlikely on Cohere's Conformer).
            max_new_tokens: Max tokens generated per audio clip. 256 is
                sufficient for ~60 seconds of dense clinical dictation.
                Increase for longer audio.
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._prompt_warning_logged = False

    def load(self) -> None:
        """Download weights (if not cached) and load the transformers model.

        Raises:
            ASRBackendError: If dependencies are missing, download fails,
                or model instantiation fails.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        except ImportError as e:
            msg = "Cohere backend dependencies not installed. Install with: uv sync --extra asr-cohere"
            raise ASRBackendError(msg) from e

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.torch_dtype not in dtype_map:
            msg = f"Unsupported torch_dtype: {self.torch_dtype!r}. Use one of {sorted(dtype_map)}"
            raise ASRBackendError(msg)
        dtype = dtype_map[self.torch_dtype]

        logger.info("[%s] loading processor for %s", self.name, self.model_id)
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = f"Failed to load AutoProcessor for {self.model_id}: {e}"
            raise ASRBackendError(msg) from e
        self._processor = processor

        logger.info(
            "[%s] loading model %s (dtype=%s, device=%s)",
            self.name,
            self.model_id,
            self.torch_dtype,
            self.device,
        )
        device_map: str | None = self.device if self.device != "cpu" else None
        try:
            model = CohereAsrForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=device_map,
                torch_dtype=dtype,
            )
        except Exception as e:
            msg = f"Failed to load Cohere Transcribe weights from {self.model_id}: {e}"
            raise ASRBackendError(msg) from e
        model.eval()
        self._model = model

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe one 16 kHz mono WAV file with Cohere Transcribe.

        Args:
            audio_path: Path to a WAV file. Will be loaded and resampled
                to 16 kHz mono if needed.
            language: Language code passed to the processor (e.g. "en").
            initial_prompt: Ignored with a log message on first use.
                Cohere Transcribe does not document a prompt mechanism.

        Returns:
            Raw transcription text.
        """
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        if initial_prompt is not None and not self._prompt_warning_logged:
            logger.info(
                "[%s] initial_prompt ignored (Cohere Transcribe has no documented prompt channel)",
                self.name,
            )
            self._prompt_warning_logged = True

        self.load()

        import numpy as np
        import soundfile as sf
        import torch

        if self._model is None or self._processor is None:
            msg = "Cohere backend was not loaded; call load() first"
            raise ASRBackendError(msg)

        # Load audio and ensure 16 kHz mono float32.
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False)
        except Exception as e:
            msg = f"Failed to read audio {audio_path}: {e}"
            raise ASRBackendError(msg) from e
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        target_sr = 16000
        if sr != target_sr:
            try:
                from scipy.signal import resample_poly
            except ImportError as e:
                msg = "scipy is required for resampling. Install with: uv sync --extra asr-cohere"
                raise ASRBackendError(msg) from e
            audio = resample_poly(audio, target_sr, sr).astype(np.float32, copy=False)
            sr = target_sr

        try:
            inputs = self._processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                language=language,
            )
            # Match the model's device AND dtype. The processor always
            # returns float32 mel features, but the model weights may be
            # fp16/bf16 — without an explicit cast, the first conv layer
            # will fail with "Input type (float) and bias type (c10::Half)
            # should be the same". Only cast floating-point tensors;
            # integer tensors (attention_mask, input_ids) must stay as-is.
            model_param = next(self._model.parameters())
            model_device = model_param.device
            model_dtype = model_param.dtype
            cast_inputs: dict[str, Any] = {}
            for k, v in inputs.items():
                if not hasattr(v, "to"):
                    cast_inputs[k] = v
                    continue
                if torch.is_floating_point(v):
                    cast_inputs[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    cast_inputs[k] = v.to(device=model_device)
            inputs = cast_inputs
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        except Exception as e:
            msg = f"Cohere Transcribe inference failed on {audio_path}: {e}"
            raise ASRBackendError(msg) from e

        if not decoded:
            return ""
        return str(decoded[0]).strip()

    def unload(self) -> None:
        """Release model + processor references. Idempotent."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                torch.mps.empty_cache()  # type: ignore[attr-defined]
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
