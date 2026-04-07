"""Google MedASR backend (MLX port).

Wraps the vendored ainergiz/medasr-mlx inference library behind the
ASRBackend protocol. Uses the `ainergiz/medasr-mlx-fp16` weights by
default — these ship in the `weights.npz` format the loader expects
and the model card reports **0.0% WER delta vs the PyTorch source**,
so fp16 here is effectively lossless against the original model.

The drankush-ai/medasr-mlx-fp32 port exists but ships in
`model.safetensors` format and is NOT compatible with this loader.
If you want fp32, use a different loader.

MedASR is a Conformer-CTC model. It does NOT support initial_prompt
the way Whisper does — CTC has no prompting channel. If an
initial_prompt is supplied, it is logged and ignored (no error), so
the bake-off runner can pass the same arguments to every backend.

The only vocabulary biasing mechanism MedASR offers is KenLM beam
decoding. A 6-gram LM (`lm_6.kenlm`) is shipped with the model repo.
The backend exposes this via the `decode_mode` and `kenlm_path`
parameters.

Dependency: `transcriber-radrx[asr-medasr]` extra, which pulls
mlx, soundfile, scipy, transformers (for AutoProcessor), huggingface_hub,
pyctcdecode, and kenlm.

Apple Silicon only.

Authors: silas-397300f6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError

if TYPE_CHECKING:
    pass  # Vendored types are not imported at type-check time

logger = logging.getLogger(__name__)


class MedASRBackend(ASRBackend):
    """MLX-native Google MedASR backend.

    Attributes:
        name: Always "medasr".
        model_id: HuggingFace model identifier for the MLX port (e.g.
            "drankush-ai/medasr-mlx-fp32", "ainergiz/medasr-mlx-fp16").
        decode_mode: "greedy" (default) or "beam" for KenLM beam search.
        kenlm_path: Optional path to a KenLM .kenlm file. If None and
            decode_mode is "beam", the shipped lm_6.kenlm is downloaded
            from the hf_processor_id repo.
        hf_processor_id: HF repo used for the AutoProcessor and the
            default KenLM. Almost always "google/medasr" since the MLX
            ports strip the processor config.
    """

    name = "medasr"

    def __init__(
        self,
        model_id: str = "ainergiz/medasr-mlx-fp16",
        *,
        decode_mode: str = "greedy",
        kenlm_path: str | None = None,
        beam_width: int = 128,
        kenlm_alpha: float = 0.5,
        kenlm_beta: float = 1.0,
        hf_processor_id: str = "google/medasr",
    ) -> None:
        """Initialize the MedASR MLX backend.

        Args:
            model_id: HuggingFace repo ID of the MLX-converted weights.
                Defaults to `ainergiz/medasr-mlx-fp16` — compatible with
                the vendored loader (weights.npz format) and reports
                0.0% WER delta vs the PyTorch source.
            decode_mode: "greedy" or "beam". Beam requires KenLM support.
            kenlm_path: Explicit path to a .kenlm file. If None and
                decode_mode is "beam", the default lm_6.kenlm is fetched
                from the HF processor repo.
            beam_width: Beam width for beam-search decoding.
            kenlm_alpha: KenLM language model weight.
            kenlm_beta: KenLM word insertion bonus.
            hf_processor_id: HF repo for the AutoProcessor. Use this if
                the MLX port does not ship its own processor.
        """
        self.model_id = model_id
        self.decode_mode = decode_mode
        self.kenlm_path = kenlm_path
        self.beam_width = beam_width
        self.kenlm_alpha = kenlm_alpha
        self.kenlm_beta = kenlm_beta
        self.hf_processor_id = hf_processor_id
        # The vendored library types aren't visible to mypy (intentionally
        # excluded), so we store these as Any. Real typing happens at
        # runtime via the vendored _medasr_mlx_lib package.
        self._model: Any = None
        self._processor: Any = None
        self._decoder: Any = None
        self._model_dir: Path | None = None

    def load(self) -> None:
        """Download the weights (if not cached) and load the MLX model.

        Raises:
            ASRBackendError: If any dependency is missing, if the model
                download fails, or if the load fails.
        """
        if self._model is not None:
            return

        try:
            import huggingface_hub
            import mlx.core  # noqa: F401
            from transformers import AutoProcessor

            from transcriber_radrx.asr_backends._medasr_mlx_lib import (
                CTCTextDecoder,
                DecoderConfig,
                load_mlx_model,
            )
        except ImportError as e:
            msg = "MedASR backend dependencies not installed. Install with: uv sync --extra asr-medasr"
            raise ASRBackendError(msg) from e

        logger.info("[%s] downloading weights for %s", self.name, self.model_id)
        try:
            model_dir = Path(huggingface_hub.snapshot_download(self.model_id))
        except Exception as e:
            msg = f"Failed to download MedASR weights from {self.model_id}: {e}"
            raise ASRBackendError(msg) from e
        self._model_dir = model_dir

        logger.info("[%s] loading processor from %s", self.name, self.hf_processor_id)
        processor_path = model_dir / "processor"
        try:
            if processor_path.exists():
                processor = AutoProcessor.from_pretrained(str(processor_path))  # type: ignore[no-untyped-call]
            else:
                processor = AutoProcessor.from_pretrained(self.hf_processor_id)  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = (
                f"Failed to load AutoProcessor. If the repo is gated, ensure "
                f"you have accepted the terms at https://huggingface.co/{self.hf_processor_id} "
                f"and logged in with `huggingface-cli login`. Error: {e}"
            )
            raise ASRBackendError(msg) from e
        self._processor = processor

        logger.info("[%s] loading MLX model from %s", self.name, model_dir)
        try:
            self._model = load_mlx_model(model_dir)
        except Exception as e:
            msg = f"Failed to load MLX MedASR weights: {e}"
            raise ASRBackendError(msg) from e

        logger.info("[%s] building %s decoder", self.name, self.decode_mode)
        self._decoder = CTCTextDecoder(
            processor=processor,
            config=DecoderConfig(
                mode=self.decode_mode,
                hf_model_id=self.hf_processor_id,
                kenlm_path=self.kenlm_path,
                alpha=self.kenlm_alpha,
                beta=self.kenlm_beta,
                beam_width=self.beam_width,
            ),
        )

    def transcribe_wav(
        self,
        audio_path: Path,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe one 16 kHz mono WAV file.

        Args:
            audio_path: Path to a WAV file. Will be loaded and resampled
                to 16 kHz mono if needed.
            language: Ignored — MedASR is English-only.
            initial_prompt: Ignored with a warning on first use. CTC
                models do not support Whisper-style prompting.

        Returns:
            Raw transcription text from the CTC decoder.
        """
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        if language != "en":
            logger.warning(
                "[%s] MedASR is English-only; language=%s ignored",
                self.name,
                language,
            )
        if initial_prompt is not None:
            logger.info(
                "[%s] initial_prompt ignored (CTC has no prompting channel)",
                self.name,
            )

        self.load()

        import mlx.core as mx
        import numpy as np

        from transcriber_radrx.asr_backends._medasr_mlx_lib import load_audio_mono

        speech, sr = load_audio_mono(audio_path, target_sr=16000)
        speech = speech.astype(np.float32, copy=False)

        if self._processor is None or self._model is None or self._decoder is None:
            msg = "MedASR backend was not loaded; call load() first"
            raise ASRBackendError(msg)

        try:
            features = self._processor(
                speech,
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors="np",
            )
            input_features = mx.array(features["input_features"])
            attention_mask = mx.array(features["attention_mask"].astype(np.bool_))
            logits = self._model(input_features=input_features, attention_mask=attention_mask)
            logits_np = np.asarray(logits)
            pred_ids = np.asarray(mx.argmax(logits, axis=-1))
            text: str = str(self._decoder.decode(logits_np, pred_ids=pred_ids))
        except Exception as e:
            msg = f"MedASR inference failed on {audio_path}: {e}"
            raise ASRBackendError(msg) from e

        return text

    def unload(self) -> None:
        """Release the MLX model reference. Idempotent."""
        self._model = None
        self._processor = None
        self._decoder = None
