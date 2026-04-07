# Multi-backend bake-off: MedASR vs Whisper on dense clinical fixtures

**Date:** 2026-04-07
**Author:** silas-397300f6
**Runner:** `tests/validation/scripts/run_multi_backend_e2e.py`
**Output JSON:** `tests/validation/reports/bakeoff_dense_first_run.json`
**PR:** (pending) silas/multi-backend-runner

## Setup

- **Backends:**
  - `mlx_whisper` at `mlx-community/whisper-large-v3-mlx` (non-turbo, the full large-v3)
  - `medasr` at `ainergiz/medasr-mlx-fp16` (Google MedASR, Conformer-CTC via vendored loader)
- **Voices:** `en_GB-alan-medium`, `en_US-lessac-high` (one UK, one US)
- **Fixtures:** 24 hand-curated dense clinical sentences (`dense_subtype` filter)
- **Audio pipeline:** piper TTS @ 22.05 kHz → 16 kHz resample → each backend transcribes the same audio
- **Corrector:** phonetic OFF (to isolate backend signal)
- **Total transcriptions:** 96 (2 backends × 2 voices × 24 fixtures)

## Headline results

| Backend | Avg raw WER | Term recall (terms_found / terms_total) |
|---|---:|---|
| **mlx_whisper (large-v3 non-turbo)** | **0.1185** | 50/126 = 39.7% |
| medasr (fp16) | 0.1959 | 49/126 = 38.9% |

**Whisper large-v3 beats MedASR by ~8 WER points on aggregate.** Term recall
is essentially tied.

Per-voice breakdown:

| Backend | alan (UK) | lessac (US) |
|---|---:|---:|
| Whisper | 0.1207 | 0.1163 |
| MedASR  | 0.1939 | 0.1978 |

Both backends are voice-insensitive on this corpus (< 0.01 WER delta between
UK and US). Backend effect dominates voice effect.

## Per-fixture nuance (why the headline is misleading)

The aggregate WER hides a real split. Stuart's original hypothesis — "if
MedASR's advantage is medical vocabulary competence, it should beat Whisper
on clean audio" — is **partially validated and partially refuted** by the
per-sample data.

### Where MedASR wins (the N=1 `Gy` signal generalizes)

**dense-0001** — clearest MedASR win:

    GT : Prescribed dose of 54 Gy in 30 fractions to the PTV with a
         simultaneous integrated boost to 60 Gy for the high-risk CTV.
    WH : Rescribed dose of 54 GI and 30 fractions to the PTV with a
         simultaneous integrated boost to 60 GI for the high risk, CTV.
    MED: prescribed dose of 54 Gy in 30 fractions to the PTV with a
         simultaneous integrated boost to 60 gy for the high-risk CTV.

Whisper says `GI` twice, garbles "in" → "and", drops the leading "P". MedASR
gets `Gy` correct both times, preserves `in`, and matches all acronyms.
WER 0.174 vs 0.043.

**dense-0016** — MedASR +3 WER points:

    WH : Stereotactic body radiation therapy was delivered to the lung
         metastasis using 54 GI in three fractions.
    MED: stereotactic body radiation therapy was delivered to the lung
         metastases using 54 Gy in three fractions.

Again, MedASR knows `Gy`, Whisper hears `GI`. **The `Gy → GI` substitution
is a systematic Whisper failure mode on this corpus.**

### Where MedASR loses (and what it tells us about MedASR's weaknesses)

**dense-0013 (brachytherapy, uncommon drug name):**

    GT : Brachytherapy boost was administered via high dose rate
         iridium-192 afterloading with three fractions of 7 Gy.
    WH : BRCA therapy boost was administered via high dose rate
         Iridium-192 after loading with 3 fractions of 7-Gi.
    MED: brachtherapy booost was administered via high-dse rate of
         riddium 192 after loading with three fractions of seven jy.

MedASR produces "brachtherapy", "booost", "high-dse", "riddium" — phonetic
misspellings on less common medical words. Whisper misreads "brachytherapy"
as "BRCA therapy" (the gene!) but otherwise is cleaner. **MedASR's medical
training doesn't extend to full spelling correctness for uncommon drug names;
it acts more like a naive phonetic transcriber there.**

**dense-0014 (Fletcher Suit applicator — CTC word-boundary failure):**

    GT : Intracavitary brachytherapy was delivered using a Fletcher Suit
         tandem and ovoids applicator.
    WH : Intercavitary brachytherapy was delivered using a Fletcher-Sue
         tandem and ovoids applicator.
    MED: intracavitary brachotherapy was delivered using a
         fleter suitetandeommentovoids applicator.

MedASR glues "suit tandem and ovoids" into `suitetandeommentovoids`. This is
a CTC model artifact — no word-boundary model, so the decoder has no prior
that word boundaries are correlated with acoustic boundaries. Whisper's
attention decoder handles this trivially.

**dense-0021 (spelling vs abbreviation — possible fixture bias):**

    GT : Concurrent cisplatin was administered weekly at 40 milligrams per
         square meter during radiotherapy.
    WH : Concurrent cisplatin was administered weekly at 40 milligrams per
         square meter during radiotherapy.        [PERFECT]
    MED: concurrent cisplatin was administered weekly at 40 mg/m2 during
         radiotherapy.

MedASR produces `mg/m2` where the ground truth has "milligrams per square
meter". Both are **clinically correct** — `mg/m2` is standard chemo dose
notation — but WER penalizes the abbreviation. Similar pattern in
`dense-0008` where MedASR emits `cm` instead of "cubic centimeters".

This is partly a fixture design issue. We could normalize unit expressions
before computing WER, but then we'd need a domain-specific normalizer. For
now, flag it as known bias: **MedASR appears to have been trained with
medical abbreviations normalized, so it emits them even when the speaker
said the spelled form.**

## What this means for the hypothesis

Stuart articulated it as:

> If MedASR's advantage is noise robustness, we should see both models tied
> on clean audio; if it's medical vocabulary competence, MedASR should win
> even on clean audio.

The data says: **both, but the win and loss modes are different categories.**

1. **MedASR does win on specific high-frequency medical tokens**, most
   prominently the `Gy` dose unit. Whisper systematically hears `Gy` as `GI`
   because its general-corpus training doesn't prime `Gy` as a plausible
   word. That's a real medical vocabulary advantage and it generalizes from
   the N=1 smoke test.

2. **MedASR loses on less-common medical words and on word-boundary
   detection.** Uncommon drug names, applicator names, and multi-word
   phrases fall apart in MedASR's output. This is partly an architectural
   weakness of CTC (no attention decoder, no language-model fluency) and
   partly a coverage weakness of the medical training corpus.

3. **Aggregate WER favors Whisper** because (a) the fluency wins on common
   words are larger than (b) the specific-token losses on `Gy`. Whisper's
   large-v3 is still the better general-purpose model, even on dense
   clinical content, even though it has a known systematic failure on one
   of the most important units in the corpus.

## Practical implications

1. **Whisper large-v3 is the right default for RT dictation** as-is —
   confirmed. MedASR is not a drop-in replacement that wins across the board.

2. **`Gy → GI` is a known Whisper failure mode that we can address in the
   corrector.** This is safe because `GI` is not a common clinical noun in
   RT dictation (it usually means "gastrointestinal" in a different
   context), and `Gy` with a number next to it is unambiguous. A context-
   aware corrector rule like `number + space + GI` → `number + space + Gy`
   would close the gap without needing MedASR.

3. **MedASR may still earn its keep** as a complementary backend under one
   of these patterns:
   - **Ensemble / voting** — combine with Whisper on specific token classes
     (dose units, common acronyms). Would need a token-level voting layer.
   - **KenLM-biased MedASR** — the shipped `lm_6.kenlm` may recover fluency
     for uncommon words. Not tested here; next step.
   - **Noisy-audio tier** — Google's benchmarks report MedASR at 4.6% WER
     on RAD-DICT (Mayo RT dictation with real acoustic conditions). We
     haven't tested either backend on noisy audio yet. MedASR's advantage
     may show up there and not on clean TTS.

4. **The per-term recall metric is coarse.** It registered ~40% for both
   backends (50/126 and 49/126), driven by both backends uniformly failing
   on TTS-spoken acronyms like "PTV" and "IMRT" (piper says "P T V" and the
   ASRs transcribe what they hear). For the next iteration, we should track
   the `Gy` preservation rate separately, since that's the single biggest
   clinical-safety item in this corpus.

## Next steps

1. **Add `Gy`-specific corrector rule** (`\d\s*GI\b` → `\d Gy`) to recover
   the Whisper failure case without needing a second backend.
2. **KenLM beam decoding for MedASR** — rerun MedASR with the shipped
   `lm_6.kenlm` to see if it closes the gap on uncommon words.
3. **Add real acoustic conditions** (MUSAN-mixed noise when available) and
   re-run. The hypothesis "MedASR is noise-robust" is still untested.
4. **Add more backends** — Parakeet TDT v3, Qwen3-ASR, IBM Granite Speech
   2B — to the bake-off, now that the runner supports it.
5. **Expand the fixture set.** 24 sentences is directional; 50–100 dense
   clinical sentences would give tighter confidence intervals, especially
   for per-fixture delta analysis.

## Honesty note

The N=1 smoke test result from `medasr_first_sample_2026-04-07.md` was a
real signal, but I over-generalized from it. On clean TTS of 24 dense
clinical sentences across 2 voices, Whisper **beats** MedASR overall by
nearly 2× on WER. MedASR's `Gy` win is real but local. The bake-off
infrastructure was exactly the tool needed to find out.

This is the same lesson as cycle 106: **build the finder, then let it find**.
The validation suite didn't agree with the first impression, and that's why
we run the validation suite.
