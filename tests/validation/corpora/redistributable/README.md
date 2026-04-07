# Redistributable Corpora

Source data with licenses that permit redistribution. Curated extracts
may be committed to the repo. Full raw downloads (often large) are
gitignored — use the acquisition scripts to fetch them.

## Sources

| Directory | Source | License | Notes |
|-----------|--------|---------|-------|
| `rond/` | Mayo Clinic Radiation Oncology NLP Database | Apache 2.0 | Full corpus may be committed in curated form |
| `tg263/` | AAPM TG-263 standardized nomenclature | Free for clinical use | Vocabulary list, not full PDF |

## Acquisition

```bash
# From repo root
bash tests/validation/scripts/acquire_rond.sh
python tests/validation/scripts/acquire_tg263.py
```

## What gets committed

Curated extracts only. The acquire scripts download full datasets to
`raw/` subdirectories which are gitignored. The extraction scripts
produce committed fixtures in `tests/validation/fixtures/`.
