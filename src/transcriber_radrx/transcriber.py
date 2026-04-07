"""Whisper MLX transcription engine with vocabulary biasing.

Wraps mlx-whisper for offline batch transcription of clinical dictations.
Vocabulary biasing via initial_prompt — feeds domain terms to the decoder
so it biases toward expected radiotherapy vocabulary.

Authors: vivian-1a61bc9a, silas-397300f6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from transcriber_radrx.corrector import Correction, CorrectionDictionary

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of transcribing a single audio file.

    Attributes:
        text: Raw transcription text from Whisper.
        corrected_text: Text after post-processing correction dictionary.
        audio_path: Path to the source audio file.
        model: Whisper model identifier used.
        language: Detected or specified language.
        corrections: Structured list of corrections applied (with score, method, offset).
    """

    text: str
    corrected_text: str
    audio_path: Path
    model: str
    language: str = "en"
    corrections: list[Correction] = field(default_factory=list)


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


def transcribe(
    audio_path: Path,
    *,
    model: str = "mlx-community/whisper-large-v3-turbo",
    vocabulary_path: Path | None = None,
    language: str = "en",
    enable_phonetic_correction: bool = False,
) -> TranscriptionResult:
    """Transcribe an audio file using Whisper MLX with optional vocabulary biasing.

    Pipeline:
    1. Whisper ASR with initial_prompt vocabulary biasing (if vocab provided)
    2. Post-processing correction dictionary (if vocab provided)

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, etc.).
        model: Whisper model identifier for mlx-whisper.
        vocabulary_path: Optional path to vocabulary file. Used for both
            Whisper initial_prompt biasing AND post-processing correction.
        language: Language code for transcription.
        enable_phonetic_correction: Enable phonetic correction tier (DEFAULT: False).
            Phonetic matching can produce false positives — only enable when
            verified safe for the target vocabulary.

    Returns:
        TranscriptionResult with raw and corrected text plus correction list.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ImportError: If mlx_whisper is not installed.
    """
    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    import mlx_whisper

    decode_options: dict[str, str] = {"language": language}
    if vocabulary_path is not None:
        decode_options["initial_prompt"] = build_initial_prompt(vocabulary_path)

    logger.info("Transcribing %s with model %s", audio_path, model)
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        **decode_options,
    )

    raw_text = result.get("text", "")

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
        model=model,
        language=language,
        corrections=corrections,
    )
