# Restricted Corpora

Source data with licenses that do NOT permit redistribution, or where
the license is unclear. **Nothing in this directory should be committed.**

The whole directory is gitignored except for this README and `.gitkeep`.

## Sources

| Directory | Source | License Status | Notes |
|-----------|--------|----------------|-------|
| `mtsamples/` | MTSamples.com clinical reports | Web-sourced, unclear | Use Kaggle CSV mirror |
| `rcr_dose_fractionation/` | RCR Dose Fractionation 4th ed (PDF) | Free for clinical use, copyright RCR | Don't redistribute the PDF |
| `corsair/` | CORSAIR dose constraints | PMC open access (cite, don't redistribute) | |
| `musan/` | MUSAN noise corpus | OpenSLR free for research | ~11GB, used by audio synthesis |

## Acquisition

```bash
# From repo root
python tests/validation/scripts/acquire_mtsamples.py
bash tests/validation/scripts/acquire_musan.sh
```

## Working with restricted data

Data here is for local development only. Derived **fixtures** that we have
the right to redistribute (e.g. extracted vocabulary terms, statistical
summaries) belong in `tests/validation/fixtures/` and should reference the
source by citation, not by copying text.

For MTSamples specifically: extract category labels, fixture IDs, and
small representative excerpts only. Don't commit full reports.
