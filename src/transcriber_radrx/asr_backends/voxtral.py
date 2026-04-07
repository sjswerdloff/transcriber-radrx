"""Mistral Voxtral backend.

Wraps `mistralai/Voxtral-Mini-3B-2507` (or the Small 24B variant on
larger hardware) via native Hugging Face transformers. Voxtral is a
**multimodal audio-LLM** — a Ministral text backbone with a causal
audio encoder bolted in front — used via AutoProcessor +
VoxtralForConditionalGeneration with a chat template that can mix
audio and text turns.

Like Granite-Speech, Voxtral is instruction-following: the caller
can pass a chat-template system prompt and the model will follow it
at transcription time. Unlike Granite-Speech, Voxtral has a dedicated
`apply_transcription_request()` path on the processor for the
transcription-only case. This backend uses the chat-template path
(not the dedicated transcription request) so the same code supports
both neutral and domain-prompted transcription via the
`system_prompt` parameter.

NOTE: This backend is **audio-LLM-shaped** and belongs to the RT
dictation validation track, not the kitchen ambient AI track. For
streaming kitchen ASR, the Voxtral Mini 4B Realtime variant is the
right candidate — see Vivian and Violet's analyses. That variant is
ASR-only (no chat template) and requires vLLM or an Apple Silicon
community MLX port, neither of which lives in this backend.

Apple Silicon: runs via `device_map="auto"`, bfloat16 by default.
3B fits in ~7 GB unified RAM; 24B Small requires M3 Ultra.

Dependency: `transcriber-radrx[asr-voxtral]` extra, which pulls
transformers>=5.2 (Voxtral class was registered then), torch>=2.2,
torchaudio>=2.2, mistral-common (required for Voxtral tokenizer in
some code paths), accelerate, soundfile, scipy, numpy, huggingface-hub.

References:
- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- Cora's July 2025 analysis: 2025-07-15_2145_Mistral_releases_Voxtral_its_first_open_source_AI_analysis.md

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError

logger = logging.getLogger(__name__)

# Neutral default system prompt for transcription-only use. When the
# caller passes an explicit system_prompt, it replaces this.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a transcription assistant. When given audio, output ONLY the "
    "verbatim transcription of the speech. Do not summarize, interpret, "
    "or add commentary. Do not answer questions about the audio — just "
    "transcribe what is said."
)

_DEFAULT_USER_TEXT = "Transcribe the following audio verbatim."


class VoxtralBackend(ASRBackend):
    """Mistral Voxtral backend via native Hugging Face transformers.

    Attributes:
        name: Always "voxtral".
        model_id: HuggingFace repo ID (e.g. "mistralai/Voxtral-Mini-3B-2507",
            "mistralai/Voxtral-Small-24B-2507"). Note: the Voxtral Mini
            4B Realtime variant is ASR-only and uses a different code
            path; do NOT use this backend for that model.
        device: Torch device placement.
        torch_dtype: Weight dtype as a string ("bfloat16" default,
            "float16" if bf16 has MPS issues).
        max_new_tokens: Max generated tokens per audio clip.
    """

    name = "voxtral"

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        *,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> None:
        """Initialize the Voxtral backend.

        Args:
            model_id: HuggingFace repo ID. Defaults to the Mini 3B
                variant for local Apple Silicon use. Use the Small 24B
                variant on M3 Ultra for higher quality.
            device: Torch device placement. "auto" picks MPS/CUDA/CPU.
            torch_dtype: Weight dtype. bfloat16 is Mistral's default;
                float16 fallback if MPS bf16 misbehaves.
            max_new_tokens: Max generated tokens per audio clip.
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._initial_prompt_warned = False

    def load(self) -> None:
        """Download and load model + processor. Idempotent."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, VoxtralForConditionalGeneration
        except ImportError as e:
            msg = "Voxtral backend dependencies not installed. Install with: uv sync --extra asr-voxtral"
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
            model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=device_map,
                torch_dtype=dtype,
            )
        except Exception as e:
            msg = f"Failed to load Voxtral weights from {self.model_id}: {e}"
            raise ASRBackendError(msg) from e
        model.eval()  # type: ignore[no-untyped-call]
        self._model = model

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",  # noqa: ARG002  # protocol requires; Voxtral auto-detects
        initial_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Transcribe one 16 kHz mono WAV file with Voxtral.

        Args:
            audio_path: Path to a WAV file.
            language: Ignored — Voxtral does automatic language detection.
            initial_prompt: Ignored with a one-time warning. For domain
                biasing, use `system_prompt` to give the model an
                instruction-following directive.
            system_prompt: Chat-template system instruction. If None,
                a neutral "transcribe verbatim" default is used. Pass
                a domain prompt here to test instructable transcription
                (e.g. "Preserve Gy as Gy. Preserve PTV/CTV/IMRT.").

        Returns:
            Raw transcription text, decoded from the model's generation
            after stripping input prompt tokens.
        """
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        if initial_prompt is not None and not self._initial_prompt_warned:
            logger.info(
                "[%s] initial_prompt ignored (Voxtral is instruction-following; use system_prompt instead for domain biasing)",
                self.name,
            )
            self._initial_prompt_warned = True

        self.load()

        import torch

        if self._model is None or self._processor is None:
            msg = "Voxtral backend was not loaded; call load() first"
            raise ASRBackendError(msg)

        # Build chat template with audio + text turns. Voxtral accepts
        # a file path directly in the audio content field.
        effective_system = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT
        conversation = [
            {"role": "system", "content": effective_system},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(audio_path)},
                    {"type": "text", "text": _DEFAULT_USER_TEXT},
                ],
            },
        ]

        try:
            inputs = self._processor.apply_chat_template(conversation, return_tensors="pt")

            # Move to device and cast floats to model dtype.
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
                    cast_inputs[k] = v.to(model_device)
            inputs = cast_inputs

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            # Strip the input prompt tokens.
            input_len = inputs["input_ids"].shape[-1]
            new_tokens = outputs[:, input_len:]
            decoded = self._processor.batch_decode(  # type: ignore[no-untyped-call]
                new_tokens,
                skip_special_tokens=True,
            )
        except Exception as e:
            msg = f"Voxtral inference failed on {audio_path}: {e}"
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
