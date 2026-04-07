"""IBM Granite-Speech backend.

Wraps `ibm-granite/granite-speech-3.3-2b` (or -8b) via native Hugging
Face transformers. Granite-Speech is a **speech-LLM** — a Granite
text LLM with an audio encoder adapter, used via AutoProcessor +
AutoModelForSpeechSeq2Seq. Unlike classical ASRs (Whisper, MedASR,
Cohere), Granite-Speech is **instruction-following**: the caller
passes a chat-template system prompt and an audio input, and the
model generates text conditioned on both.

This is the backend we use to test Stuart's "instructable
transcription" hypothesis: an audio-LLM given a domain prompt like
"Transcribe this radiation oncology dictation verbatim. Preserve Gy
as Gy. Preserve PTV/CTV/IMRT exactly. Use numeric digits for doses."
should be able to cooperate in a way Whisper/MedASR/Cohere
structurally cannot (because they have no instruction channel).

Apple Silicon: runs via `device_map="auto"` (MPS when available),
bfloat16 by default per the model card. 2B fits comfortably in 16 GB
unified RAM; 8B fits in 32 GB (or bigger M3 Ultra configurations).

Dependency: `transcriber-radrx[asr-granite]` extra, which pulls
transformers>=4.52, torch>=2.2, torchaudio>=2.2 (required by the
processor for audio loading), peft>=0.13 (the audio encoder ships as
a PEFT adapter on top of base Granite), accelerate (required for
device_map="auto"), soundfile, scipy, numpy, huggingface-hub.

Reference: https://huggingface.co/ibm-granite/granite-speech-3.3-2b

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError

logger = logging.getLogger(__name__)

# Granite's chat template expects identity/role in the system prompt
# and task-specific instructions in the user prompt. Putting the
# transcription instructions in the system role causes the model to
# echo them back instead of transcribing — it generates output that
# looks like a continuation of its own system message.
#
# So we keep the system prompt as the generic Granite identity from
# the HF model card example, and inject any caller-supplied
# system_prompt (which is really task guidance) into the user content
# alongside the <|audio|> token.
_GRANITE_IDENTITY_SYSTEM_PROMPT = "You are Granite, developed by IBM. You are a helpful AI assistant."

# Default user task when the caller does not pass a system_prompt.
# Matches the HF model card example for pure transcription.
_DEFAULT_TRANSCRIPTION_TASK = (
    "Please transcribe the audio verbatim. Output only the transcription "
    "with no commentary, no summary, and no interpretation."
)


class GraniteSpeechBackend(ASRBackend):
    """IBM Granite-Speech backend via native Hugging Face transformers.

    Attributes:
        name: Always "granite_speech".
        model_id: HuggingFace repo ID of the Granite-Speech weights
            (e.g. "ibm-granite/granite-speech-3.3-2b" or -8b).
        device: Torch device placement ("auto", "mps", "cuda", "cpu").
        torch_dtype: Weight dtype as a string ("bfloat16" default per
            model card, "float16" if you hit MPS bf16 issues,
            "float32" as a last resort).
        max_new_tokens: Max tokens per transcription. 512 is enough for
            ~2 minutes of RT dictation; bump for longer audio.
    """

    name = "granite_speech"

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-speech-3.3-2b",
        *,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> None:
        """Initialize the Granite-Speech backend.

        Args:
            model_id: HuggingFace repo ID. Defaults to the 2B variant
                for local Apple Silicon use. Use the 8B variant on
                M3 Ultra for higher quality.
            device: Torch device placement. "auto" delegates to
                `device_map="auto"`.
            torch_dtype: Weight dtype. bfloat16 is the model card
                default; float16 if bf16 has MPS issues; float32 is
                the fallback.
            max_new_tokens: Max generated tokens per audio clip.
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._initial_prompt_warned = False

    def load(self) -> None:
        """Download and load model + processor. Idempotent."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as e:
            msg = "Granite-Speech backend dependencies not installed. Install with: uv sync --extra asr-granite"
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
        self._tokenizer = processor.tokenizer

        logger.info(
            "[%s] loading model %s (dtype=%s, device=%s)",
            self.name,
            self.model_id,
            self.torch_dtype,
            self.device,
        )
        device_map: str | None = self.device if self.device != "cpu" else None
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                device_map=device_map,
                torch_dtype=dtype,
            )
        except Exception as e:
            msg = f"Failed to load Granite-Speech weights from {self.model_id}: {e}"
            raise ASRBackendError(msg) from e
        model.eval()  # type: ignore[no-untyped-call]
        self._model = model

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",  # noqa: ARG002  # protocol requires, Granite is English-primarily
        initial_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Transcribe one 16 kHz mono WAV file with Granite-Speech.

        Args:
            audio_path: Path to a WAV file.
            language: Language code (Granite is primarily English).
            initial_prompt: Ignored with a one-time warning. Granite-
                Speech is not a Whisper-style decoder with soft
                prompting; for vocabulary biasing, use system_prompt
                to give the model an instruction-following directive.
            system_prompt: Chat-template system instruction. If None,
                a neutral "transcribe verbatim" default is used. Pass
                a domain prompt here to test instructable
                transcription.

        Returns:
            Raw transcription text, extracted from the model's
            generation after stripping the input prompt tokens and
            special tokens.
        """
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        if initial_prompt is not None and not self._initial_prompt_warned:
            logger.info(
                "[%s] initial_prompt ignored (Granite-Speech is instruction-"
                "following; use system_prompt instead for domain biasing)",
                self.name,
            )
            self._initial_prompt_warned = True

        self.load()

        import numpy as np
        import soundfile as sf
        import torch

        if self._model is None or self._processor is None or self._tokenizer is None:
            msg = "Granite-Speech backend was not loaded; call load() first"
            raise ASRBackendError(msg)

        # Load audio via soundfile (not torchaudio — newer torchaudio
        # requires torchcodec which is an extra dep we'd rather avoid).
        # Ensure mono float32 at 16 kHz, then wrap in a torch tensor of
        # shape [1, num_samples] as the Granite processor expects.
        try:
            audio_np, sr = sf.read(str(audio_path), always_2d=False)
        except Exception as e:
            msg = f"Failed to read audio {audio_path}: {e}"
            raise ASRBackendError(msg) from e
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32, copy=False)
        target_sr = 16000
        if sr != target_sr:
            from scipy.signal import resample_poly

            audio_np = resample_poly(audio_np, target_sr, sr).astype(np.float32, copy=False)
        wav = torch.from_numpy(audio_np).unsqueeze(0)  # [1, num_samples]

        # Granite chat template: system = identity, user = task
        # (with <|audio|> as the audio placeholder). The caller's
        # system_prompt — which is really task-specific guidance — goes
        # into the USER content, not the system content. See the
        # _GRANITE_IDENTITY_SYSTEM_PROMPT comment for why.
        task_text = system_prompt if system_prompt is not None else _DEFAULT_TRANSCRIPTION_TASK
        user_content = f"<|audio|>{task_text}"
        chat = [
            {"role": "system", "content": _GRANITE_IDENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Run processor + model.generate.
        try:
            model_param = next(self._model.parameters())
            model_device = model_param.device
            model_dtype = model_param.dtype
            inputs = self._processor(
                prompt_text,
                wav,
                device=str(model_device),
                return_tensors="pt",
            )
            # Move to device and cast floating-point tensors to model dtype.
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
                    num_beams=1,
                )

            # Strip the input prompt tokens; only decode the newly
            # generated portion.
            input_len = inputs["input_ids"].shape[-1]
            new_tokens = outputs[:, input_len:]
            decoded = self._tokenizer.batch_decode(  # type: ignore[no-untyped-call]
                new_tokens,
                skip_special_tokens=True,
                add_special_tokens=False,
            )
        except Exception as e:
            msg = f"Granite-Speech inference failed on {audio_path}: {e}"
            raise ASRBackendError(msg) from e

        if not decoded:
            return ""
        return str(decoded[0]).strip()

    def unload(self) -> None:
        """Release model, processor, tokenizer references. Idempotent."""
        self._model = None
        self._processor = None
        self._tokenizer = None
        try:
            import torch

            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                torch.mps.empty_cache()  # type: ignore[attr-defined]
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
