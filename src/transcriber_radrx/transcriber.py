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
        corrections: List of (original, corrected) pairs applied.
    """

    text: str
    corrected_text: str
    audio_path: Path
    model: str
    language: str = "en"
    corrections: list[tuple[str, str]] = field(default_factory=list)


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
    prompt = (
        "This is a radiotherapy clinical dictation. "
        "Key terms include: " + ", ".join(terms[:200]) + "."
    )
    logger.info("Built initial_prompt with %d terms (truncated to 200)", len(terms))
    return prompt


def transcribe(
    audio_path: Path,
    *,
    model: str = "mlx-community/whisper-large-v3-turbo",
    vocabulary_path: Path | None = None,
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe an audio file using Whisper MLX with optional vocabulary biasing.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, etc.).
        model: Whisper model identifier for mlx-whisper.
        vocabulary_path: Optional path to vocabulary file for initial_prompt biasing.
        language: Language code for transcription.

    Returns:
        TranscriptionResult with raw and corrected text.

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

    return TranscriptionResult(
        text=raw_text,
        corrected_text=raw_text,  # Correction applied separately by post-processor
        audio_path=audio_path,
        model=model,
        language=language,
    )
