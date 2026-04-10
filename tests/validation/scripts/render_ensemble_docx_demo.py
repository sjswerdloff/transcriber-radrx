"""Demo script: render ensemble transcription results as Word documents with Track Changes.

Produces two demo documents in docs/demo/:

1. ``ensemble_review_particle.docx`` — particle therapy corpus (Voxtral vs Whisper)
2. ``ensemble_review_anatomy.docx`` — anatomy corpus (Voxtral vs Whisper)

Each document shows per-word Track Changes so a clinician can open it in Word
and use Accept/Reject Changes to resolve disagreements.

Usage::

    uv run python tests/validation/scripts/render_ensemble_docx_demo.py

Output:
    docs/demo/ensemble_review_particle.docx
    docs/demo/ensemble_review_anatomy.docx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Path bootstrap: ensure repo root is on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]  # tests/validation/scripts → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPORTS_DIR = _REPO_ROOT / "tests" / "validation" / "reports"
_VOCAB_FILE = _REPO_ROOT / "data" / "rt_vocabulary.txt"
_DOCS_DEMO_DIR = _REPO_ROOT / "docs" / "demo"

# Particle corpus
_PARTICLE_VOXTRAL = _REPORTS_DIR / "bakeoff_particle_voxtral_2026-04-09.json"
_PARTICLE_WHISPER = _REPORTS_DIR / "bakeoff_particle_whisper_medasr_2026-04-09.json"

# Anatomy corpus
_ANATOMY_VOXTRAL = _REPORTS_DIR / "bakeoff_anatomy_voxtral_2026-04-10.json"
_ANATOMY_WHISPER = _REPORTS_DIR / "bakeoff_anatomy_whisper_2026-04-10.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_vocabulary(path: Path) -> set[str]:
    """Load RT vocabulary terms from a text file.

    Lines starting with '#' and blank lines are ignored.
    Returns a set of lowercase terms.

    Args:
        path: Path to the vocabulary text file.

    Returns:
        Set of lowercase vocabulary terms.
    """
    terms: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                terms.add(stripped.lower())
    return terms


def _extract_samples_by_voice(
    report: dict[str, object],
    backend_name: str,
) -> dict[str, dict[str, dict[str, object]]]:
    """Extract samples indexed by voice → fixture_id.

    Args:
        report: Parsed bake-off JSON.
        backend_name: Backend name to filter (e.g. 'voxtral', 'mlx_whisper').

    Returns:
        Nested dict: voice_name → fixture_id → sample dict.
    """
    index: dict[str, dict[str, dict[str, object]]] = {}
    results = report.get("results", [])
    if not isinstance(results, list):
        return index
    for be in results:
        if not isinstance(be, dict) or be.get("backend") != backend_name:
            continue
        for ve in be.get("by_voice", []):
            if not isinstance(ve, dict):
                continue
            voice = str(ve.get("voice", ""))
            vm: dict[str, dict[str, object]] = {}
            for s in ve.get("samples", []):
                if isinstance(s, dict):
                    vm[str(s.get("fixture_id", ""))] = s
            index[voice] = vm
    return index


def _build_ensemble_results(
    vox_path: Path,
    whi_path: Path,
    vox_backend: str,
    whi_backend: str,
    vocabulary: set[str],
) -> tuple[list[object], dict[tuple[str, str], str]]:
    """Load two bake-off JSONs, run ensemble_transcriptions, return results + gold texts.

    Args:
        vox_path: Path to the Voxtral bake-off JSON.
        whi_path: Path to the Whisper bake-off JSON.
        vox_backend: Backend name for Voxtral (e.g. 'voxtral').
        whi_backend: Backend name for Whisper (e.g. 'mlx_whisper').
        vocabulary: Set of RT vocabulary terms.

    Returns:
        Tuple of (list[EnsembleResult], gold_texts dict).
    """
    from transcriber_radrx.ensemble import ensemble_transcriptions  # noqa: PLC0415

    with vox_path.open(encoding="utf-8") as f:
        vox_report: dict[str, object] = json.load(f)
    with whi_path.open(encoding="utf-8") as f:
        whi_report: dict[str, object] = json.load(f)

    vox_by_voice = _extract_samples_by_voice(vox_report, vox_backend)
    whi_by_voice = _extract_samples_by_voice(whi_report, whi_backend)

    common_voices = sorted(set(vox_by_voice.keys()) & set(whi_by_voice.keys()))
    logger.info("Common voices: {}", common_voices)

    results = []
    gold_texts: dict[tuple[str, str], str] = {}

    for voice in common_voices:
        vox_fixtures = vox_by_voice[voice]
        whi_fixtures = whi_by_voice[voice]
        common_fixtures = sorted(set(vox_fixtures.keys()) & set(whi_fixtures.keys()))
        logger.info("voice={} fixtures={}", voice, len(common_fixtures))

        for fid in common_fixtures:
            vox_s = vox_fixtures[fid]
            whi_s = whi_fixtures[fid]
            text_voxtral = str(vox_s.get("raw_transcription", ""))
            text_whisper = str(whi_s.get("raw_transcription", ""))
            ground_truth = str(vox_s.get("ground_truth", ""))

            er = ensemble_transcriptions(
                text_voxtral=text_voxtral,
                text_whisper=text_whisper,
                vocabulary=vocabulary,
                fixture_id=fid,
                voice=voice,
            )
            results.append(er)
            if ground_truth:
                gold_texts[(fid, voice)] = ground_truth

    return results, gold_texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Generate demo Word documents for the particle and anatomy corpora.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    from transcriber_radrx.ensemble.docx_renderer import render_ensemble_docx  # noqa: PLC0415

    # Ensure output directory exists
    _DOCS_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if not _VOCAB_FILE.exists():
        logger.error("Vocabulary file not found: {}", _VOCAB_FILE)
        return 1

    logger.info("Loading RT vocabulary …")
    vocab = _load_vocabulary(_VOCAB_FILE)
    logger.info("Vocabulary terms: {}", len(vocab))

    # ------------------------------------------------------------------
    # Particle corpus
    # ------------------------------------------------------------------
    particle_out = _DOCS_DEMO_DIR / "ensemble_review_particle.docx"
    if _PARTICLE_VOXTRAL.exists() and _PARTICLE_WHISPER.exists():
        logger.info("Building particle ensemble …")
        particle_results, particle_gold = _build_ensemble_results(
            _PARTICLE_VOXTRAL, _PARTICLE_WHISPER, "voxtral", "mlx_whisper", vocab
        )
        logger.info("Particle results: {}", len(particle_results))
        render_ensemble_docx(
            particle_results,  # type: ignore[arg-type]
            particle_out,
            title="Particle Therapy Ensemble Review",
            show_gold=True,
            gold_texts=particle_gold,  # type: ignore[arg-type]
            mode="audit",
        )
        particle_review_out = _DOCS_DEMO_DIR / "ensemble_review_particle_review.docx"
        render_ensemble_docx(
            particle_results,  # type: ignore[arg-type]
            particle_review_out,
            title="Particle Therapy Ensemble Review",
            show_gold=True,
            gold_texts=particle_gold,  # type: ignore[arg-type]
            mode="review",
        )
        print(f"Demo document written to {particle_out} (audit) — open in Word to see all tracked changes")
        print(f"Demo document written to {particle_review_out} (review) — only UWR items as tracked changes")
    else:
        logger.warning("Particle corpus files not found — skipping particle demo")

    # ------------------------------------------------------------------
    # Anatomy corpus
    # ------------------------------------------------------------------
    anatomy_out = _DOCS_DEMO_DIR / "ensemble_review_anatomy.docx"
    if _ANATOMY_VOXTRAL.exists() and _ANATOMY_WHISPER.exists():
        logger.info("Building anatomy ensemble …")
        anatomy_results, anatomy_gold = _build_ensemble_results(
            _ANATOMY_VOXTRAL, _ANATOMY_WHISPER, "voxtral", "mlx_whisper", vocab
        )
        logger.info("Anatomy results: {}", len(anatomy_results))
        render_ensemble_docx(
            anatomy_results,  # type: ignore[arg-type]
            anatomy_out,
            title="Anatomy Corpus Ensemble Review",
            show_gold=True,
            gold_texts=anatomy_gold,  # type: ignore[arg-type]
            mode="audit",
        )
        anatomy_review_out = _DOCS_DEMO_DIR / "ensemble_review_anatomy_review.docx"
        render_ensemble_docx(
            anatomy_results,  # type: ignore[arg-type]
            anatomy_review_out,
            title="Anatomy Corpus Ensemble Review",
            show_gold=True,
            gold_texts=anatomy_gold,  # type: ignore[arg-type]
            mode="review",
        )
        print(f"Demo document written to {anatomy_out} (audit) — open in Word to see all tracked changes")
        print(f"Demo document written to {anatomy_review_out} (review) — only UWR items as tracked changes")
    else:
        logger.warning("Anatomy corpus files not found — skipping anatomy demo")

    return 0


if __name__ == "__main__":
    sys.exit(main())
