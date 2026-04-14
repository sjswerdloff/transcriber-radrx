"""Command-line interface for transcriber-radrx.

Provides two subcommands:

``transcribe``
    Single-backend transcription (Whisper MLX).  Preserves the original 0.1
    behaviour so callers that relied on bare ``transcribe-radrx audio.wav``
    continue to work without changes.

``evaluate``
    Dual-backend ensemble evaluation (Whisper + Voxtral), with optional gold
    standard comparison and Word document output.  The primary command for
    clinician-facing use.

Authors: vivian-1a61bc9a, silas-397300f6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EvaluateArgumentError(ValueError):
    """Raised when evaluate subcommand arguments are inconsistent or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class VocabularyLoadError(OSError):
    """Raised when the RT vocabulary file cannot be loaded."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"Cannot load vocabulary file {path}: {cause}")
        self.path = path
        self.cause = cause


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Package-relative default vocabulary path (relative to installed package root).
_DEFAULT_VOCAB_RELPATH = "../../data/rt_vocabulary.txt"


def _resolve_vocabulary(vocabulary_arg: Path | None) -> Path | None:
    """Locate the vocabulary file from CLI arg or well-known defaults.

    Search order:
    1. Explicit ``--vocabulary`` argument (must exist, raises if it does not).
    2. ``data/rt_vocabulary.txt`` relative to the package directory.
    3. ``data/rt_vocabulary.txt`` in the current working directory.

    Args:
        vocabulary_arg: Value of ``--vocabulary`` flag, or None.

    Returns:
        Resolved Path, or None if no vocabulary file can be found.

    Raises:
        FileNotFoundError: If an explicit ``--vocabulary`` path was given but
            does not exist.
    """
    if vocabulary_arg is not None:
        if not vocabulary_arg.exists():
            msg = f"Vocabulary file not found: {vocabulary_arg}"
            raise FileNotFoundError(msg)
        return vocabulary_arg

    # Try package-relative location
    pkg_default = Path(__file__).parent / _DEFAULT_VOCAB_RELPATH
    pkg_default = pkg_default.resolve()
    if pkg_default.exists():
        return pkg_default

    # Try cwd-relative location
    cwd_default = Path.cwd() / "data" / "rt_vocabulary.txt"
    if cwd_default.exists():
        return cwd_default

    return None


def _load_vocabulary_set(vocabulary_path: Path) -> set[str]:
    """Load vocabulary terms from file into a lowercase set.

    Args:
        vocabulary_path: Path to vocabulary file (one term per line, # comments).

    Returns:
        Set of lowercase vocabulary terms.

    Raises:
        VocabularyLoadError: If the file cannot be read.
    """
    try:
        terms: set[str] = set()
        with vocabulary_path.open() as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    terms.add(line.lower())
    except OSError as exc:
        raise VocabularyLoadError(vocabulary_path, exc) from exc
    else:
        return terms


def _load_reference_text(reference: str | None, reference_file: Path | None) -> str | None:
    """Load gold standard text from inline arg or file.

    ``--reference`` takes precedence over ``--reference-file``.

    Args:
        reference: Inline reference text, or None.
        reference_file: Path to reference text file, or None.

    Returns:
        Reference string, or None if neither was provided.

    Raises:
        FileNotFoundError: If ``--reference-file`` was given but does not exist.
    """
    if reference is not None:
        return reference
    if reference_file is not None:
        if not reference_file.exists():
            msg = f"Reference file not found: {reference_file}"
            raise FileNotFoundError(msg)
        return reference_file.read_text().strip()
    return None


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _run_transcribe(args: argparse.Namespace) -> None:
    """Run the single-backend Whisper transcription subcommand.

    Args:
        args: Parsed namespace from the ``transcribe`` subparser.
    """
    vocabulary_path = _resolve_vocabulary(args.vocabulary)

    from transcriber_radrx.transcriber import transcribe

    result = transcribe(
        args.audio,
        model=args.model,
        vocabulary_path=vocabulary_path,
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


def _run_evaluate(args: argparse.Namespace) -> None:
    """Run the dual-backend ensemble evaluation subcommand.

    Pipeline:
    1. Load vocabulary (for ensemble tiebreaking and correction).
    2. Load Whisper and Voxtral backends.
    3. Transcribe audio with both backends.
    4. Apply phrase-level then word-level corrections to both outputs.
    5. Run ensemble decision rules.
    6. If gold reference provided: compute WER and UWR.
    7. Render review .docx (and optional audit .docx).
    8. Print ensemble text to stdout.
    9. Print WER / UWR / review count to stderr.

    Args:
        args: Parsed namespace from the ``evaluate`` subparser.

    Raises:
        EvaluateArgumentError: If audio file does not exist.
    """
    import logging

    import jiwer

    from transcriber_radrx.asr_backends.registry import get_backend
    from transcriber_radrx.corrector import CorrectionDictionary
    from transcriber_radrx.ensemble.decision_rules import ensemble_transcriptions
    from transcriber_radrx.ensemble.docx_renderer import render_ensemble_docx, render_ensemble_docx_pair
    from transcriber_radrx.transcriber import transcribe_with_backend

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("transcribe-radrx.evaluate")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    audio_path: Path = args.audio
    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise EvaluateArgumentError(msg)

    output_path: Path = args.output
    if not output_path.parent.exists():
        msg = f"Output directory does not exist: {output_path.parent}"
        raise EvaluateArgumentError(msg)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    vocabulary_path = _resolve_vocabulary(args.vocabulary)
    vocabulary_set: set[str] = set()
    if vocabulary_path is not None:
        vocabulary_set = _load_vocabulary_set(vocabulary_path)
        log.info("Loaded %d vocabulary terms from %s", len(vocabulary_set), vocabulary_path)
    else:
        log.warning("No vocabulary file found — ensemble tiebreaking and corrections disabled")

    # ------------------------------------------------------------------
    # Gold reference (optional)
    # ------------------------------------------------------------------
    reference_text = _load_reference_text(
        getattr(args, "reference", None),
        getattr(args, "reference_file", None),
    )

    # ------------------------------------------------------------------
    # Load backends
    # ------------------------------------------------------------------
    whisper_kwargs: dict[str, object] = {}
    if args.whisper_model:
        whisper_kwargs["model_id"] = args.whisper_model

    voxtral_kwargs: dict[str, object] = {}
    if args.voxtral_model:
        voxtral_kwargs["model_id"] = args.voxtral_model

    print("Loading Whisper large-v3 ...", file=sys.stderr, flush=True)
    whisper_backend = get_backend("mlx_whisper", **whisper_kwargs)

    print("Loading Voxtral Mini 3B ...", file=sys.stderr, flush=True)
    voxtral_backend = get_backend("voxtral", **voxtral_kwargs)

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    print(f"Transcribing {audio_path.name} with Whisper ...", file=sys.stderr, flush=True)
    whisper_result = transcribe_with_backend(
        audio_path,
        whisper_backend,
        vocabulary_path=vocabulary_path,
    )

    print(f"Transcribing {audio_path.name} with Voxtral ...", file=sys.stderr, flush=True)
    voxtral_result = transcribe_with_backend(
        audio_path,
        voxtral_backend,
        vocabulary_path=vocabulary_path,
    )

    # ------------------------------------------------------------------
    # Apply full corrections (phrase + word) to both outputs
    # ------------------------------------------------------------------
    whisper_text = whisper_result.corrected_text
    voxtral_text = voxtral_result.corrected_text

    if vocabulary_path is not None:
        corrector = CorrectionDictionary(str(vocabulary_path))
        whisper_text, _, _ = corrector.correct_full(whisper_result.text)
        voxtral_text, _, _ = corrector.correct_full(voxtral_result.text)
        log.info("Applied full corrections to both outputs")

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    print("Running ensemble decision rules ...", file=sys.stderr, flush=True)
    ensemble_result = ensemble_transcriptions(
        text_voxtral=voxtral_text,
        text_whisper=whisper_text,
        vocabulary=vocabulary_set,
        fixture_id=audio_path.stem,
        voice="clinical",
    )

    # ------------------------------------------------------------------
    # WER / UWR (if reference provided)
    # ------------------------------------------------------------------
    wer_value: float | None = None
    uwr_value: float | None = None

    if reference_text is not None:
        wer_value = jiwer.wer(reference_text, ensemble_result.text_ensemble)
        total_words = len(ensemble_result.words)
        uwr_value = ensemble_result.review_count / total_words if total_words > 0 else 0.0

    # ------------------------------------------------------------------
    # Render docx
    # ------------------------------------------------------------------
    print(f"Rendering review document to {output_path} ...", file=sys.stderr, flush=True)
    audit_output: Path | None = getattr(args, "audit_output", None)

    gold_texts: dict[tuple[str, str], str] | None = None
    show_gold = reference_text is not None
    if reference_text is not None:
        gold_texts = {(audio_path.stem, "clinical"): reference_text}

    if audit_output is not None:
        render_ensemble_docx_pair(
            [ensemble_result],
            audit_path=audit_output,
            review_path=output_path,
            show_gold=show_gold,
            gold_texts=gold_texts,
        )
        log.info("Audit document written to %s", audit_output)
    else:
        render_ensemble_docx(
            [ensemble_result],
            output_path,
            mode="review",
            show_gold=show_gold,
            gold_texts=gold_texts,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print(ensemble_result.text_ensemble)

    print("\n--- Ensemble summary ---", file=sys.stderr)
    print(f"  Review count : {ensemble_result.review_count}", file=sys.stderr)
    print(f"  Total words  : {len(ensemble_result.words)}", file=sys.stderr)

    if wer_value is not None and uwr_value is not None:
        print(f"  WER          : {wer_value:.4f} ({wer_value * 100:.1f}%)", file=sys.stderr)
        print(f"  UWR          : {uwr_value:.4f} ({uwr_value * 100:.1f}%)", file=sys.stderr)

    print(f"  Review docx  : {output_path}", file=sys.stderr)
    if audit_output is not None:
        print(f"  Audit docx   : {audit_output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="transcribe-radrx",
        description="Local radiotherapy clinical transcription with vocabulary biasing",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # ------------------------------------------------------------------
    # transcribe subcommand (preserves 0.1.x behaviour)
    # ------------------------------------------------------------------
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio using Whisper (single-backend)",
        description=(
            "Transcribe one audio file using the Whisper MLX backend with optional "
            "vocabulary biasing and post-processing correction."
        ),
    )
    transcribe_parser.add_argument("audio", type=Path, help="Path to audio file")
    transcribe_parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-mlx",
        help="Whisper model identifier (default: whisper-large-v3-mlx)",
    )
    transcribe_parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Path to RT vocabulary file. Used for both Whisper initial_prompt "
        "biasing AND post-processing correction dictionary.",
    )
    transcribe_parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    transcribe_parser.add_argument(
        "--enable-phonetic",
        action="store_true",
        help="Enable phonetic correction tier (DEFAULT: off). "
        "Phonetic matching can produce false positives — only enable when "
        "verified safe for the target vocabulary.",
    )

    # ------------------------------------------------------------------
    # evaluate subcommand (new in 0.2.0)
    # ------------------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Dual-backend ensemble evaluation with clinician review document",
        description=(
            "Transcribe audio with both Whisper and Voxtral, run the 2-backend "
            "ensemble, and produce a Word document for clinician review. "
            "Optionally compare against a gold-standard reference (WER / UWR)."
        ),
    )
    evaluate_parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to audio WAV file",
    )
    evaluate_parser.add_argument(
        "--reference",
        default=None,
        help="Gold-standard reference text (inline). Takes precedence over --reference-file.",
    )
    evaluate_parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        dest="reference_file",
        help="Path to a file containing the gold-standard reference text.",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the clinician review .docx file",
    )
    evaluate_parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="RT vocabulary file (default: data/rt_vocabulary.txt relative to package or cwd)",
    )
    evaluate_parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        dest="audit_output",
        help="Optional path for the full audit .docx (shows all ensemble decisions as Track Changes)",
    )
    evaluate_parser.add_argument(
        "--whisper-model",
        default=None,
        dest="whisper_model",
        help="Whisper model override (default: backend default)",
    )
    evaluate_parser.add_argument(
        "--voxtral-model",
        default=None,
        dest="voxtral_model",
        help="Voxtral model override (default: backend default)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_KNOWN_SUBCOMMANDS: frozenset[str] = frozenset({"transcribe", "evaluate"})


def main(argv: list[str] | None = None) -> None:
    """Entry point for transcribe-radrx CLI.

    Dispatches to the appropriate subcommand handler. When no subcommand is
    given, falls back to ``transcribe`` for backward compatibility (0.1.x
    callers that used bare ``transcribe-radrx audio.wav`` continue to work).

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).
    """
    effective_argv: list[str] = list(argv if argv is not None else sys.argv[1:])

    # Backward-compat: if the first non-flag arg is not a known subcommand,
    # prepend "transcribe" so 0.1.x callers still work.
    first_positional = next(
        (a for a in effective_argv if not a.startswith("-")),
        None,
    )
    if first_positional not in _KNOWN_SUBCOMMANDS:
        effective_argv = ["transcribe"] + effective_argv

    parser = _build_parser()
    args = parser.parse_args(effective_argv)

    if args.subcommand == "transcribe":
        _run_transcribe(args)
    elif args.subcommand == "evaluate":
        _run_evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)
