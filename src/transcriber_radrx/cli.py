"""Command-line interface for transcriber-radrx.

Authors: vivian-1a61bc9a, silas-397300f6
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
        help="Path to RT vocabulary file. Used for both Whisper initial_prompt "
        "biasing AND post-processing correction dictionary.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--enable-phonetic",
        action="store_true",
        help="Enable phonetic correction tier (DEFAULT: off). "
        "Phonetic matching can produce false positives — only enable when "
        "verified safe for the target vocabulary.",
    )

    args = parser.parse_args(argv)

    from transcriber_radrx.transcriber import transcribe

    result = transcribe(
        args.audio,
        model=args.model,
        vocabulary_path=args.vocabulary,
        language=args.language,
        enable_phonetic_correction=args.enable_phonetic,
    )

    print(result.corrected_text)

    if result.corrections:
        print("\n--- Corrections applied ---", file=sys.stderr)
        for c in result.corrections:
            print(
                f"  {c.original} → {c.corrected} (method={c.method}, score={c.score:.2f}, offset={c.offset})",
                file=sys.stderr,
            )
