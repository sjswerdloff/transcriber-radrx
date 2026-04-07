# End-to-End Validation: First Run

**Date:** 2026-04-07
**Branch:** silas/end-to-end-runner
**Pipeline:** text fixture → piper TTS (en_US-amy-medium) → Whisper MLX (large-v3-turbo) → corrector → WER

## Setup

- **Samples:** 5 fixtures, seed=42 (coincidentally all from `homograph_trap` category — the stratified sampler prioritizes traps)
- **Vocabulary:** `data/rt_vocabulary.txt` (391 terms)
- **Whisper model:** `mlx-community/whisper-large-v3-turbo`
- **Piper voice:** `en_US-amy-medium.onnx` at 22050 Hz, resampled to 16 kHz
- **Acoustic simulation:** bypassed (clean tier only)

## Results

| Metric | phonetic OFF | phonetic ON |
|--------|--------------|-------------|
| avg raw WER | 0.0133 | 0.0133 |
| avg corrected WER | 0.0133 | **0.0583** |
| corrections applied | 0 | 2 |
| homograph_trap violations | 0 | 0 |

### The raw WER is essentially zero

Whisper large-v3-turbo transcribed clean piper audio of homograph-trap sentences with 0 errors on 4/5 samples. The one non-zero error (trap-0021, WER 0.067) was `30 percent` transcribed as `30%` — a formatting difference, not a word error. Clean-tier Whisper is not the problem.

### Phonetic ON introduces NEW clinical safety failures

Two brand-new phonetic false positives the PR #1 review didn't catch:

**Case 1: `proceed → breast`** (fixture trap-0001)
- Ground truth: "Our patient is supportive of treatment and wishes to proceed."
- Raw ASR output: "Our patient is supportive of treatment and wishes to proceed."
- Corrected (phonetic ON): "Our patient is supportive of treatment and wishes to **breast**."
- Confidence: 1.00, method: phonetic

**Case 2: `throughout → thyroid`** (fixture trap-0008)
- Ground truth: "The family is providing emotional support throughout therapy."
- Raw ASR output: "The family is providing emotional support throughout therapy."
- Corrected (phonetic ON): "The family is providing emotional support **thyroid** therapy."
- Confidence: 1.00, method: phonetic

Both would radically alter clinical meaning. `breast` and `thyroid` are both legitimate anatomical terms in the RT vocabulary. Double Metaphone collapses them to the same phonetic code as common English words.

## Interpretation

**The safety-off default is empirically confirmed.** This is now the *second independent discovery* of phonetic false positives (first was Cora's PR #1 review). Cora found `our → OAR`, `support → SBRT`, etc.; this run found `proceed → breast`, `throughout → thyroid`. Neither set overlaps.

**Implication:** the homograph stop word list in `corrector.py` catches only the known failures. There is a long tail of unknown false positives that phonetic matching will expose on any new vocabulary or text sample.

**Safe path:** keep phonetic matching OFF as the default. If a user opts into phonetic correction, they accept responsibility for the false positive rate.

**Unsafe path:** add `proceed`, `throughout` to the stop word list and pretend we've fixed it. The next sample will find more.

## Recommendation

1. **Keep `enable_phonetic=False` as the default** (already done, now empirically validated).
2. **Expand the validation suite** to include clean-tier runs with phonetic ON across a much larger sample to catalog the actual false positive rate.
3. **Consider removing phonetic matching entirely** from the corrector — or only enabling it for multi-syllable vocabulary terms with explicit opt-in per term. The current "≥6 chars" length guard isn't sufficient.
4. **Add `proceed` and `throughout` to the homograph_trap fixtures** as new hand-curated negative test cases, so future regressions are caught.

## What worked beautifully

- The end-to-end runner took ~7 seconds per sample (Whisper model cached after first load)
- Piper audio synthesis was fast and clear
- The safety defaults held the line
- Running both phonetic modes in the same run made the comparison unambiguous
- Vivian's TTS + my transcribe wiring + Cora's safety review all played together correctly on the first real run

This is why validation is the most valuable part of the repo.
