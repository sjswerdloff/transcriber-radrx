# Validation Suite

End-to-end validation of the transcriber-radrx pipeline against realistic
RT clinical content. Measures word error rate (WER) per category, false
positive rates from the corrector, and model-vs-model comparisons.

## Why This Matters

The whole point of transcriber-radrx is to produce clinically accurate
transcriptions. Unit tests verify individual functions. The validation
suite verifies the **product**: does the pipeline actually work on real
RT dictation content?

This may be the most valuable part of the repo. Without it, we can ship
"54 tests passing" while still failing on the only thing that matters.

## Architecture

```
tests/validation/
├── corpora/                       # Source data (mostly gitignored)
│   ├── redistributable/           # Apache 2.0 / public domain — committable
│   │   ├── rond/                  # Mayo Clinic Radiation Oncology NLP DB
│   │   └── tg263/                 # AAPM TG-263 vocabulary
│   └── restricted/                # License unclear / not redistributable
│       ├── mtsamples/             # Web-sourced clinical reports
│       ├── rcr_dose_fractionation/  # PDF
│       └── corsair/               # Dose constraints
├── fixtures/                       # Curated, derived from corpora — committable
│   ├── rt_dictation_samples.jsonl # text + category + source + license
│   ├── vocabulary_enriched.txt    # base + TG-263 + RCR-derived terms
│   └── correction_test_cases.jsonl # ASR-style errors → expected corrections
├── audio/                          # Generated audio (gitignored, large)
│   └── synthetic/
│       ├── clean/                 # piper TTS output, 16kHz mono
│       ├── acoustic/              # +room reverb, distance mic simulation
│       └── noisy/                 # +equipment hum, background voices
├── scripts/
│   ├── acquire_rond.sh            # git clone Mayo repo
│   ├── acquire_tg263.py           # download AAPM spreadsheet
│   ├── acquire_mtsamples.py       # fetch from Kaggle mirror
│   ├── extract_rond_text.py       # ROND JSON → fixtures
│   ├── extract_mtsamples_rt.py    # CSV filter → fixtures
│   ├── enrich_vocabulary.py       # base + TG-263 + RCR → enriched
│   ├── generate_synthetic_audio.py  # text → piper → wav
│   ├── apply_acoustic_simulation.py # clean wav → reverb/noise
│   └── run_validation.py          # ASR → corrector → WER report
└── tests/
    ├── test_fixtures_schema.py    # validates committed fixtures
    └── test_validation_runner.py  # smoke test for the runner
```

## Running

The validation suite is gated by pytest markers. Default `pytest tests/`
does NOT run validation (avoids requiring 50GB of corpora and audio).

```bash
# Acquire corpora (one-time, takes time + bandwidth)
make acquire-corpora       # or run scripts manually

# Generate synthetic audio (one-time per voice/text combination)
make generate-audio

# Run validation against the current pipeline
pytest tests/validation/ -m validation -v

# Benchmark a specific model
python tests/validation/scripts/run_validation.py \
    --model mlx-community/whisper-large-v3-turbo \
    --audio tests/validation/audio/synthetic/clean \
    --vocabulary tests/validation/fixtures/vocabulary_enriched.txt \
    --output validation_report.json
```

## Fixtures Schema

`rt_dictation_samples.jsonl` — one JSON object per line:

```json
{
  "id": "rond-0001",
  "text": "The patient received 50 Gy in 25 fractions to the PTV using IMRT...",
  "category": "treatment_summary",
  "source": "ROND",
  "license": "Apache-2.0",
  "vocabulary_terms": ["Gy", "PTV", "IMRT"],
  "expected_difficulty": "low"
}
```

Categories:
- `treatment_summary` — full RT plan summary
- `dose_prescription` — focused on dose/fractionation language
- `oar_constraints` — organ-at-risk dose limits
- `setup_instructions` — patient positioning, immobilization
- `progress_note` — interim clinical notes during treatment
- `consent_discussion` — patient-clinician dialogue
- `acronym_dense` — high concentration of RT acronyms (stress test)
- `homograph_trap` — sentences with English words that collide with RT acronyms (negative test)

## Validation Metrics

The runner produces a JSON report with:

- **Overall WER** — standard word error rate
- **Per-category WER** — broken down by fixture category
- **Vocabulary recall** — fraction of RT vocabulary terms correctly transcribed
- **Vocabulary precision** — fraction of asserted vocabulary terms that were actually correct
- **Corrector improvement** — WER before vs. after post-processing
- **False positive count** — common English words wrongly corrected to RT acronyms
- **Per-term confusion matrix** — which terms get confused with which others

## Licensing Notes

| Source | License | Redistribute? |
|--------|---------|---------------|
| ROND | Apache 2.0 | Yes (curated subset committed) |
| AAPM TG-263 | Free for clinical use | Vocabulary list yes, full PDF no |
| MTSamples | Web-sourced, unclear | No (gitignored) |
| RCR Dose Fractionation | Free for clinical use, copyright RCR | No (gitignored) |
| CORSAIR | PMC open access | Yes for excerpts; we cite, not redistribute |
| Synthea | Apache 2.0 | Yes |
| MUSAN (noise only) | Attribution; mixed sub-component licenses | No — kept under restricted/ |

When uncertain → restricted/.
