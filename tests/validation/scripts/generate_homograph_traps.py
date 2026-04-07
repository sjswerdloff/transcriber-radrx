"""Generate the homograph_trap fixture category.

These are hand-curated sentences that contain common English words which
collide with RT vocabulary acronyms or short terms. They are NEGATIVE
test cases: the validation runner expects the corrector to leave them
unchanged. Any false-positive correction is a clinical safety failure.

Source: empirically derived from Cora's PR #1 review (2026-04-07).
License: this fixture is original work, MIT compatible with the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl"

# Each entry: (text, words_that_must_not_become_acronyms)
# These are clinically plausible sentences a radiation oncologist might dictate
# or a clinician might transcribe, containing English homographs of RT terms.
HOMOGRAPH_TRAPS: list[tuple[str, list[str]]] = [
    # 'our' / 'or' / 'are' → OAR
    ("Our patient is supportive of treatment and wishes to proceed.", ["OAR", "SBRT"]),
    ("The lungs are at risk of significant pneumonitis.", ["OAR"]),
    ("Treatment may include chemotherapy or radiation alone.", ["OAR"]),
    ("Our team has reviewed the case in tumor board.", ["OAR"]),
    # 'guy' → Gy
    ("The guy in room three is ready for setup.", ["Gy"]),
    ("Several guys on staff have been trained on the new protocol.", ["Gy"]),
    # 'support' → SBRT
    ("Patient requires nutritional support during treatment.", ["SBRT"]),
    ("The family is providing emotional support throughout therapy.", ["SBRT"]),
    ("Dietary support has been arranged for the patient.", ["SBRT"]),
    # 'great' → 3DCRT
    ("The patient had a great response to the initial cycle.", ["3DCRT"]),
    ("There is great concern about the toxicity profile.", ["3DCRT"]),
    # 'gray' (color) → Gray (dose unit)
    ("The patient has gray hair and appears older than stated age.", ["Gray", "Gy"]),
    ("MRI shows gray matter changes in the temporal lobe.", ["Gray", "Gy"]),
    ("The grey area on the DRR represents the bony anatomy.", ["Gray", "Gy"]),
    # 'format' → VMAT
    ("The radiology report format has been updated.", ["VMAT"]),
    ("Please review the formatted treatment summary.", ["VMAT"]),
    # 'emerald' → IMRT (from review failure case)
    ("The patient wore an emerald necklace during her CT simulation.", ["IMRT"]),
    # mixed traps
    (
        "Our patient is supportive of treatment, the response was great, and we expect a good outcome.",
        ["OAR", "SBRT", "3DCRT"],
    ),
    ("The guy is here for a follow-up after his treatment, his lungs are healing well.", ["Gy", "OAR"]),
    # ones that DO contain real RT terms but also stop words (should still correct the real ones)
    ("Our patient received 50 Gy in 25 fractions to the PTV using IMRT.", ["OAR"]),
    ("The lungs are at risk so we limited the V20 to less than 30 percent.", ["OAR"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=["append", "write"],
        default="append",
        help="append (default) or write a fresh file",
    )
    args = parser.parse_args()

    samples = []
    for i, (text, must_not_become) in enumerate(HOMOGRAPH_TRAPS, start=1):
        sample = {
            "id": f"trap-{i:04d}",
            "text": text,
            "category": "homograph_trap",
            "source": "hand-curated",
            "license": "MIT",
            "language": "en",
            "vocabulary_terms": [],  # the trap is that vocab terms should NOT be added
            "must_not_become": must_not_become,
            "expected_difficulty": "high",
            "word_count": len(text.split()),
            "char_count": len(text),
            "notes": "Negative test: corrector must NOT introduce 'must_not_become' terms",
        }
        samples.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.mode == "append" else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    verb = "Appended" if args.mode == "append" else "Wrote"
    rel = args.output.relative_to(REPO_ROOT)
    print(f"{verb} {len(samples)} homograph traps to {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
