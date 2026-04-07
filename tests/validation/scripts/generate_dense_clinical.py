"""Generate the rt_dense_clinical fixture category.

Hand-curated sentences with high density of RT-specific vocabulary.
The purpose is to separate two hypotheses about MedASR's advantage:

  1. MedASR is better than Whisper because it handles noisy dictation
     acoustically. In that case, clean TTS of dense medical content
     would have both models near 0% WER.

  2. MedASR is better because it knows medical vocabulary distributions
     Whisper doesn't. In that case, clean TTS of dense medical content
     would have MedASR near 0% WER and Whisper miscoding terms.

These fixtures are the test content for that hypothesis. Each sentence
is a realistic RT clinical pattern with 5-15 domain terms.

License: original work, MIT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "tests/validation/fixtures/rt_dictation_samples.jsonl"


# Each entry: (text, vocabulary_terms_present, subtype)
# Subtypes help categorize what kind of clinical content we're stressing:
#   - dose: dose/fractionation language
#   - oar: organ-at-risk constraints
#   - technique: treatment modality and planning
#   - brachy: brachytherapy-specific
#   - srs_sbrt: stereotactic terminology
#   - imaging: image guidance and simulation
#   - chemo_rt: combined modality
DENSE_FIXTURES: list[tuple[str, list[str], str]] = [
    # Dose prescription
    (
        "Prescribed dose of 54 Gy in 30 fractions to the PTV with a simultaneous integrated boost to 60 Gy for the high-risk CTV.",
        ["Gy", "PTV", "CTV", "simultaneous integrated boost"],
        "dose",
    ),
    (
        "The patient will receive 50 Gy in 25 fractions delivered via IMRT with daily image guidance.",
        ["Gy", "IMRT", "IGRT"],
        "dose",
    ),
    (
        "Hypofractionated regimen of 40.05 Gy in 15 fractions with a sequential boost of 10 Gy in 4 fractions to the tumor bed.",
        ["Gy", "hypofractionation", "boost"],
        "dose",
    ),
    (
        "Total dose of 70 Gy in 35 fractions over 7 weeks with concurrent chemotherapy.",
        ["Gy", "chemoradiation"],
        "dose",
    ),
    # OAR constraints
    (
        "Organs at risk included the parotids, pharyngeal constrictors, brainstem, and cervical spinal cord.",
        ["OAR", "brainstem", "spinal cord"],
        "oar",
    ),
    (
        "Dose volume constraints: V20 to both lungs less than 30 percent, mean heart dose less than 25 Gy, and maximum spinal cord dose less than 45 Gy.",
        ["V20", "Gy", "spinal cord"],
        "oar",
    ),
    (
        "Target parotid mean dose was limited to 26 Gy with a V30 of less than 50 percent.",
        ["Gy", "V30"],
        "oar",
    ),
    (
        "The planning target volume abuts the bowel, so we constrained the V45 to less than 195 cubic centimeters.",
        ["PTV", "V45"],
        "oar",
    ),
    # Treatment technique
    (
        "Treatment was delivered using volumetric modulated arc therapy with daily kilovoltage image guidance.",
        ["VMAT", "IGRT"],
        "technique",
    ),
    (
        "The plan was optimized with intensity modulated radiation therapy using seven coplanar beams.",
        ["IMRT"],
        "technique",
    ),
    (
        "Three dimensional conformal radiation therapy was used for the whole pelvis field with a boost using IMRT.",
        ["3DCRT", "IMRT", "boost"],
        "technique",
    ),
    (
        "Deep inspiration breath hold technique was employed to reduce cardiac dose during left breast irradiation.",
        ["DIBH"],
        "technique",
    ),
    # Brachytherapy
    (
        "Brachytherapy boost was administered via high dose rate iridium-192 afterloading with three fractions of 7 Gy.",
        ["brachytherapy", "HDR", "Gy", "boost"],
        "brachy",
    ),
    (
        "Intracavitary brachytherapy was delivered using a Fletcher Suit tandem and ovoids applicator.",
        ["brachytherapy"],
        "brachy",
    ),
    # SRS/SBRT
    (
        "Stereotactic radiosurgery was planned with a single 18 Gy fraction to the 80 percent isodose line.",
        ["SRS", "Gy", "isodose", "stereotactic"],
        "srs_sbrt",
    ),
    (
        "Stereotactic body radiation therapy was delivered to the lung metastasis using 54 Gy in 3 fractions.",
        ["SBRT", "Gy", "stereotactic"],
        "srs_sbrt",
    ),
    (
        "The patient underwent fractionated stereotactic radiotherapy with 30 Gy in 5 fractions to the cavernous sinus.",
        ["SRT", "Gy", "stereotactic"],
        "srs_sbrt",
    ),
    # Imaging and simulation
    (
        "CT simulation was performed with a five millimeter slice thickness in the treatment position with thermoplastic mask immobilization.",
        [],
        "imaging",
    ),
    (
        "Daily cone beam CT was used for image guidance with a threshold of 3 millimeters for translational correction.",
        ["CBCT", "IGRT"],
        "imaging",
    ),
    (
        "PET CT was fused to the planning CT for gross tumor volume delineation.",
        ["GTV"],
        "imaging",
    ),
    # Combined modality
    (
        "Concurrent cisplatin was administered weekly at 40 milligrams per square meter during radiotherapy.",
        ["chemoradiation", "concurrent"],
        "chemo_rt",
    ),
    (
        "Neoadjuvant chemoradiation consisted of 50.4 Gy in 28 fractions with concurrent capecitabine.",
        ["Gy", "chemoradiation", "concurrent"],
        "chemo_rt",
    ),
    # Acronym dense (the kind of phrase a clinician dictates fast)
    (
        "IMRT plan delivered 60 Gy in 30 fractions to the PTV sparing the OARs including parotids and cord.",
        ["IMRT", "Gy", "PTV", "OAR", "spinal cord"],
        "acronym_dense",
    ),
    (
        "VMAT plan with SIB delivered 70 Gy to the GTV and 56 Gy to the CTV in 35 fractions with daily IGRT.",
        ["VMAT", "SIB", "Gy", "GTV", "CTV", "IGRT"],
        "acronym_dense",
    ),
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
    for i, (text, vocab, subtype) in enumerate(DENSE_FIXTURES, start=1):
        category = "acronym_dense" if subtype == "acronym_dense" else "dose_prescription"
        # Dense clinical content spans multiple SCHEMA.md categories; we use the
        # closest existing category for the schema check, plus a subtype for our
        # own analytics. (Adding a new category would require expanding the
        # schema validator.)
        if subtype == "oar":
            category = "oar_constraints"
        elif subtype in ("technique", "brachy", "srs_sbrt"):
            category = "treatment_summary"
        elif subtype == "imaging":
            category = "setup_instructions"
        elif subtype == "chemo_rt":
            category = "treatment_summary"

        samples.append(
            {
                "id": f"dense-{i:04d}",
                "text": text,
                "category": category,
                "source": "hand-curated",
                "license": "MIT",
                "language": "en",
                "vocabulary_terms": vocab,
                "expected_difficulty": "high",
                "word_count": len(text.split()),
                "char_count": len(text),
                "notes": f"Dense clinical content, subtype={subtype}. Tests vocabulary knowledge vs acoustic robustness.",
                "dense_subtype": subtype,
            },
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.mode == "append" else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    verb = "Appended" if args.mode == "append" else "Wrote"
    rel = args.output.relative_to(REPO_ROOT)
    print(f"{verb} {len(samples)} dense clinical fixtures to {rel}")

    # Summary by subtype
    by_subtype: dict[str, int] = {}
    for _, _, subtype in DENSE_FIXTURES:
        by_subtype[subtype] = by_subtype.get(subtype, 0) + 1
    print(f"  By subtype: {by_subtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
