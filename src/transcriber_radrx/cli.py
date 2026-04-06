"""Command-line interface for transcriber-radrx.

Authors: vivian-1a61bc9a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Entry point for transcribe-radrx CLI."""
    parser = argparse.ArgumentParser(
        description="Local radiotherapy clinical transcription with vocabulary biasing",
    )
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model identifier (default: whisper-large-v3-turbo)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Path to RT vocabulary file for initial_prompt biasing",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Path to correction dictionary for post-processing",
    )

    args = parser.parse_args(argv)

    from transcriber_radrx.transcriber import transcribe

    result = transcribe(
        args.audio,
        model=args.model,
        vocabulary_path=args.vocabulary,
        language=args.language,
    )

    if args.corrections:
        from transcriber_radrx.corrector import CorrectionDictionary

        corrector = CorrectionDictionary(str(args.corrections))
        corrected_text, corrections = corrector.correct(result.text)
        result.corrected_text = corrected_text
        result.corrections = [(c.original, c.corrected) for c in corrections]

    print(result.corrected_text)

    if result.corrections:
        print("\n--- Corrections applied ---", file=sys.stderr)
        for original, corrected in result.corrections:
            print(f"  {original} → {corrected}", file=sys.stderr)
