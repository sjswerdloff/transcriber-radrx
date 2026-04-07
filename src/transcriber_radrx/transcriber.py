"""Offline batch transcription pipeline with pluggable ASR backends.

The pipeline composes three stages:

1. ASR backend (pluggable) — transcribes audio to raw text. Whisper MLX
   is the default. Other backends (MedASR, Parakeet, etc) slot in via the
   `transcriber_radrx.asr_backends` registry.
2. Vocabulary biasing (backend-specific) — for backends that support it
   (Whisper family), the vocabulary file is passed as `initial_prompt`.
   Backends that do not support prompting (CTC models like MedASR) ignore
   the prompt and rely on post-processing correction.
3. Correction dictionary (post-processing) — tiered exact/case-insensitive
   matching with safety guards. See `corrector.py`.

The `transcribe()` entry point preserves backward compatibility with the
original single-backend API. New code should use `transcribe_with_backend()`
directly for explicit backend selection.

Authors: vivian-1a61bc9a, silas-397300f6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from transcriber_radrx.corrector import Correction, CorrectionDictionary

if TYPE_CHECKING:
    from transcriber_radrx.asr_backends import ASRBackend

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of transcribing a single audio file.

    Attributes:
        text: Raw transcription text from the ASR backend.
        corrected_text: Text after post-processing correction dictionary.
        audio_path: Path to the source audio file.
        model: ASR model identifier used (backend-specific).
        language: Detected or specified language.
        corrections: Structured list of corrections applied.
        backend_name: Short identifier of the ASR backend used.
    """

    text: str
    corrected_text: str
    audio_path: Path
    model: str
    language: str = "en"
    corrections: list[Correction] = field(default_factory=list)
    backend_name: str = "mlx_whisper"


def build_initial_prompt(vocabulary_path: Path) -> str:
    """Build Whisper initial_prompt from vocabulary file.

    Reads domain terms (one per line, # comments ignored) and constructs
    a prompt string that biases the Whisper decoder toward these terms.

    Args:
        vocabulary_path: Path to vocabulary file.

    Returns:
        Prompt string containing domain terms.
    """
    terms: list[str] = []
    with vocabulary_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)

    # Whisper's initial_prompt works best with terms in natural sentence context.
    # For domain vocabulary, a comma-separated list in a framing sentence works.
    prompt = "This is a radiotherapy clinical dictation. Key terms include: " + ", ".join(terms[:200]) + "."
    logger.info("Built initial_prompt with %d terms (truncated to 200)", len(terms))
    return prompt


def apply_corrections(
    text: str,
    vocabulary_path: Path | None,
    *,
    enable_phonetic: bool = False,
) -> tuple[str, list[Correction]]:
    """Apply post-processing correction to transcribed text.

    Args:
        text: Raw transcription text.
        vocabulary_path: Path to vocabulary file. If None, no corrections applied.
        enable_phonetic: Enable phonetic matching tier (DEFAULT: False for safety).

    Returns:
        Tuple of (corrected_text, list of corrections).
    """
    if vocabulary_path is None:
        return text, []

    corrector = CorrectionDictionary(
        str(vocabulary_path),
        enable_phonetic=enable_phonetic,
    )
    return corrector.correct(text)


def transcribe_with_backend(
    audio_path: Path,
    backend: ASRBackend,
    *,
    vocabulary_path: Path | None = None,
    language: str = "en",
    enable_phonetic_correction: bool = False,
    system_prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe one audio file using a specified ASR backend.

    This is the preferred entry point for multi-backend work. For
    backward compatibility with older callers, see `transcribe()`.

    Args:
        audio_path: Path to audio file.
        backend: An ASRBackend instance (load() may be called lazily).
        vocabulary_path: Optional vocabulary file. Used for initial_prompt
            biasing (if the backend supports it) and post-processing
            correction. Backends that do not support prompting will log
            and ignore the prompt.
        language: Language code.
        enable_phonetic_correction: Enable phonetic corrector tier.
        system_prompt: Optional instruction-following directive for
            audio-LLM backends (Granite-Speech, Voxtral, Phi-4-multimodal).
            This is DIFFERENT from `initial_prompt`: the latter is a
            Whisper-style soft decoder bias built from `vocabulary_path`,
            the former is a chat-template system instruction that lets
            instructable models apply domain rules (e.g. "preserve Gy
            as Gy, preserve PTV/CTV/IMRT, use numeric digits for doses").
            Classical ASRs (Whisper, MedASR, Cohere) ignore it with a
            log warning.

    Returns:
        TranscriptionResult with raw and corrected text.
    """
    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    initial_prompt = None
    if vocabulary_path is not None:
        initial_prompt = build_initial_prompt(vocabulary_path)

    logger.info(
        "Transcribing %s with backend=%s model=%s",
        audio_path,
        backend.name,
        backend.model_id,
    )
    raw_text = backend.transcribe_wav(
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
        system_prompt=system_prompt,
    )

    corrected_text, corrections = apply_corrections(
        raw_text,
        vocabulary_path,
        enable_phonetic=enable_phonetic_correction,
    )

    if corrections:
        logger.info("Applied %d corrections to transcription", len(corrections))

    return TranscriptionResult(
        text=raw_text,
        corrected_text=corrected_text,
        audio_path=audio_path,
        model=backend.model_id,
        language=language,
        corrections=corrections,
        backend_name=backend.name,
    )


def transcribe(
    audio_path: Path,
    *,
    model: str = "mlx-community/whisper-large-v3-turbo",
    vocabulary_path: Path | None = None,
    language: str = "en",
    enable_phonetic_correction: bool = False,
    backend_name: str = "mlx_whisper",
) -> TranscriptionResult:
    """Backward-compatible single-call transcription entry point.

    Maintains the original API for callers that predate the backend
    registry. Instantiates the requested backend internally.

    Pipeline:
    1. ASR backend (default: mlx_whisper) with optional vocabulary biasing
    2. Post-processing correction dictionary (if vocab provided)

    Args:
        audio_path: Path to audio file.
        model: Model identifier (backend-specific).
        vocabulary_path: Optional vocabulary file for biasing and correction.
        language: Language code.
        enable_phonetic_correction: Enable phonetic corrector tier.
        backend_name: Which ASRBackend to use. Defaults to "mlx_whisper".

    Returns:
        TranscriptionResult with raw and corrected text.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        KeyError: If backend_name is not registered.
        ASRBackendError: If the backend dependencies are missing or
            inference fails.
    """
    from transcriber_radrx.asr_backends import get_backend

    backend = get_backend(backend_name, model_id=model)
    return transcribe_with_backend(
        audio_path,
        backend,
        vocabulary_path=vocabulary_path,
        language=language,
        enable_phonetic_correction=enable_phonetic_correction,
    )
