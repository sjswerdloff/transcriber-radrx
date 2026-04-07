"""Extract RT dictation samples from ROND into the validation fixtures format.

ROND structure (Apache 2.0, Mayo Clinic):
- 1-Logic Reasoning/Logic_Reasoning.csv — Q&A pairs about RT physics/biology
- 2-Text Classification/Text_Classification.csv — short clinical phrases with treatment labels
- 3-NER (text format, harder to parse cleanly)
- 4-Text summarization — medical physics paper abstracts (too academic for dictation)
- 5-QA — open Q&A
- 6-Conversational/Oncology-Question.csv — patient-clinician dialogue

For dictation training, we want CLINICAL NARRATIVE text, not Q&A.
This script extracts text from the most dictation-relevant sources:
- Conversational answers (clinician responses, often dose/treatment-focused)
- Classification phrases (short clinical statements)
- Logic reasoning answers (factual statements)

Output: JSONL conforming to SCHEMA.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROND_RAW = REPO_ROOT / "tests/validation/corpora/redistributable/rond/raw"
DEFAULT_OUTPUT = REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl"

# RT vocabulary terms we look for to tag samples
RT_TERMS = {
    "IMRT",
    "VMAT",
    "SBRT",
    "SRS",
    "SRT",
    "3DCRT",
    "IGRT",
    "Gy",
    "cGy",
    "Gray",
    "PTV",
    "GTV",
    "CTV",
    "OAR",
    "ITV",
    "DIBH",
    "FFF",
    "TBI",
    "CSI",
    "TSET",
    "brachytherapy",
    "stereotactic",
    "hypofractionation",
    "boost",
    "proton",
    "photon",
    "electron",
    "linac",
    "accelerator",
    "fraction",
    "fractions",
    "dose",
    "fractionation",
    "isodose",
    "tumor",
    "tumour",
    "lesion",
    "metastasis",
    "metastases",
    "chemotherapy",
    "chemoradiation",
    "concurrent",
}

# Lowercase set for case-insensitive lookup
RT_TERMS_LOWER = {t.lower() for t in RT_TERMS}


def find_rt_terms(text: str) -> list[str]:
    """Return RT vocabulary terms found in text (case-insensitive lookup,
    canonical case in result)."""
    canonical_by_lower = {t.lower(): t for t in RT_TERMS}
    found: list[str] = []
    for word in re.findall(r"\b[\w-]+\b", text):
        lower = word.lower()
        if lower in canonical_by_lower and canonical_by_lower[lower] not in found:
            found.append(canonical_by_lower[lower])
    return found


def categorize(text: str, source_section: str) -> str:
    """Heuristically categorize a sample for the SCHEMA.md taxonomy."""
    lower = text.lower()
    if source_section == "conversational":
        return "consent_discussion"
    if any(kw in lower for kw in ["dose", "gy", "fraction", "cgy"]):
        return "dose_prescription"
    if any(kw in lower for kw in ["constraint", "oar", "v20", "dmax", "mean dose"]):
        return "oar_constraints"
    if any(kw in lower for kw in ["setup", "positioning", "immobiliz"]):
        return "setup_instructions"
    return "treatment_summary"


def difficulty(text: str, vocab_terms: list[str]) -> str:
    """Crude difficulty estimate based on vocabulary density."""
    word_count = len(text.split())
    if word_count == 0:
        return "low"
    density = len(vocab_terms) / word_count
    if density > 0.15:
        return "high"
    if density > 0.05:
        return "medium"
    return "low"


def load_csv(path: Path) -> list[list[str]]:
    """Load a CSV with cp1252 fallback (ROND CSVs use Windows encoding)."""
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.reader(f))
    except UnicodeDecodeError:
        with path.open(encoding="cp1252") as f:
            return list(csv.reader(f))


def extract_classification(rows: list[list[str]], start_id: int) -> list[dict]:
    """2-Text Classification: short clinical phrases."""
    samples: list[dict] = []
    next_id = start_id
    for row in rows:
        if len(row) < 1 or not row[0].strip():
            continue
        text = row[0].strip()
        if text == "Question" or text.lower() == "text":
            continue  # header
        if len(text.split()) < 4:
            continue  # too short for dictation
        vocab = find_rt_terms(text)
        sample = {
            "id": f"rond-cls-{next_id:04d}",
            "text": text,
            "category": categorize(text, "classification"),
            "source": "ROND",
            "source_section": "Text Classification",
            "license": "Apache-2.0",
            "language": "en",
            "vocabulary_terms": vocab,
            "expected_difficulty": difficulty(text, vocab),
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        samples.append(sample)
        next_id += 1
    return samples


def extract_logic_reasoning(rows: list[list[str]], start_id: int) -> list[dict]:
    """1-Logic Reasoning: physics/biology factual statements (questions used as text)."""
    samples: list[dict] = []
    next_id = start_id
    for row in rows:
        if len(row) < 1 or not row[0].strip():
            continue
        text = row[0].strip()
        if text == "Question":
            continue
        # Strip leading numbering like "1. " or "12. "
        text = re.sub(r"^\d+\.\s*", "", text)
        if len(text.split()) < 6:
            continue
        vocab = find_rt_terms(text)
        sample = {
            "id": f"rond-lr-{next_id:04d}",
            "text": text,
            "category": "treatment_summary",  # closest match for factual RT statements
            "source": "ROND",
            "source_section": "Logic Reasoning",
            "license": "Apache-2.0",
            "language": "en",
            "vocabulary_terms": vocab,
            "expected_difficulty": difficulty(text, vocab),
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        samples.append(sample)
        next_id += 1
    return samples


def extract_conversational(rows: list[list[str]], start_id: int) -> list[dict]:
    """6-Conversational: patient-clinician Q&A. We use the clinician answer
    as the dictation text."""
    samples: list[dict] = []
    next_id = start_id
    for row in rows:
        if len(row) < 2:
            continue
        answer = row[1].strip()
        if not answer or answer == "Answer":
            continue
        if len(answer.split()) < 4:
            continue
        vocab = find_rt_terms(answer)
        sample = {
            "id": f"rond-conv-{next_id:04d}",
            "text": answer,
            "category": "consent_discussion",
            "source": "ROND",
            "source_section": "Conversational",
            "license": "Apache-2.0",
            "language": "en",
            "vocabulary_terms": vocab,
            "expected_difficulty": difficulty(answer, vocab),
            "word_count": len(answer.split()),
            "char_count": len(answer),
        }
        samples.append(sample)
        next_id += 1
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rond-dir",
        type=Path,
        default=ROND_RAW,
        help="Path to ROND raw clone (default: tests/validation/corpora/redistributable/rond/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit total samples (for testing)")
    args = parser.parse_args()

    if not args.rond_dir.exists():
        print(f"ERROR: ROND directory not found: {args.rond_dir}", file=sys.stderr)
        print("Run: bash tests/validation/scripts/acquire_rond.sh", file=sys.stderr)
        return 1

    classification_csv = args.rond_dir / "2-Text Classification" / "Text_Classification.csv"
    logic_csv = args.rond_dir / "1-Logic Reasoning" / "Logic_Reasoning.csv"
    conversational_csv = args.rond_dir / "6-Conversational" / "Oncology-Question.csv"

    samples: list[dict] = []
    samples.extend(extract_classification(load_csv(classification_csv), start_id=1))
    samples.extend(extract_logic_reasoning(load_csv(logic_csv), start_id=1))
    samples.extend(extract_conversational(load_csv(conversational_csv), start_id=1))

    if args.limit is not None:
        samples = samples[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    by_section: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for s in samples:
        by_section[s["source_section"]] = by_section.get(s["source_section"], 0) + 1
        by_category[s["category"]] = by_category.get(s["category"], 0) + 1
        by_difficulty[s["expected_difficulty"]] = by_difficulty.get(s["expected_difficulty"], 0) + 1

    print(f"Wrote {len(samples)} samples to {args.output.relative_to(REPO_ROOT)}")
    print(f"  By section: {by_section}")
    print(f"  By category: {by_category}")
    print(f"  By difficulty: {by_difficulty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
