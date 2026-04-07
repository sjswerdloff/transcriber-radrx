# MedASR vs Whisper: First Head-to-Head Sample

**Date:** 2026-04-07
**Branch:** silas/asr-backends-and-medasr
**Purpose:** Smoke test of the new ASR backend infrastructure + first empirical MedASR vs Whisper result on dense clinical content.

This is **one sample**, not a published result. It exists to verify the infrastructure and to surface the directional signal for the bake-off still to come.

## Setup

- **Backends:** `mlx_whisper` (mlx-community/whisper-large-v3-mlx, full non-turbo) and `medasr` (ainergiz/medasr-mlx-fp16)
- **Voice:** en_US-amy-medium, piper 22050 Hz synthesis
- **Vocabulary biasing:** None for this smoke test — raw ASR capability only
- **Correction dictionary:** Not applied — raw backend output

## The Sample

**Ground truth:**
> Prescribed dose of 54 Gy in 30 fractions to the PTV with a simultaneous integrated boost to 60 Gy for the high-risk CTV.

**Whisper large-v3 (non-turbo):**
> Prescribed dose of 54 **GI** in 30 fractions to the PTV with a simultaneous integrated boost to 60 **GI** for the high-risk CTV.

**MedASR fp16:**
> prescribed dose of 54 **Gy** in 30 fractions to the PTV with a simultaneous **integradated** boost to 60 **gy** for the high risk CTV.

## What happened

**Whisper's failure mode: medical abbreviation substitution.**
- "Gy" (Gray, the dose unit) transcribed as "GI" — twice. "GI" is a more common medical abbreviation (gastrointestinal), which Whisper's language model prior weighted higher.
- Otherwise nails casing, punctuation, hyphenation.

**MedASR's failure mode: minor lexical artifacts.**
- Got "Gy" correct both times (one as "Gy", one as "gy" — capitalization inconsistency)
- Hallucinated a syllable: "integrated" → "integradated"
- Dropped the hyphen in "high-risk"
- All lowercase output

## What this tells us

This single sample matches the hypothesis Stuart articulated before I built the infrastructure:

> "If MedASR's edge on RAD-DICT is purely noise robustness, that's not domain specialization — Whisper would close the gap on clean audio. But if it's medical vocabulary competence, we should see a benefit on clean dense-medical content too."

**On clean dense-medical content, with no vocabulary biasing and no corrector, MedASR correctly transcribed the domain-specific dose unit while Whisper substituted a clinically different medical abbreviation.** That is consistent with MedASR having learned medical vocabulary distributions Whisper's general-web training did not.

**Caveats that matter for interpretation:**

1. **N=1.** One sample is not a trend. The real bake-off needs the full 24 dense-clinical fixtures × multiple voices to make any defensible claim.
2. **No vocabulary biasing on either model.** Whisper's `initial_prompt` with the RT vocabulary file would likely fix the "Gy→GI" error. A fairer comparison includes both biased and unbiased runs.
3. **Neither output went through the correction dictionary.** Our pipeline's post-processing tier would catch "GI" in `result.corrected_text` if "GI" were in the stop list (it is not — `GI` is a real RT-adjacent term). This reinforces that the corrector is not a safety net for substitution errors between real medical terms.
4. **MedASR's "integradated" artifact is a real error.** The model is not perfect; it made a lexical mistake that Whisper did not. A full run may show MedASR wins on domain vocabulary but loses on general word fluency.
5. **MedASR output is all-lowercase.** That's a known characteristic of the CTC + SentencePiece decoder. Downstream consumers need to handle casing.

## Infrastructure verification checklist

- [x] Backend registry loads both `mlx_whisper` and `medasr`
- [x] MedASR downloads weights via HuggingFace (gated repo, user has accepted terms)
- [x] MedASR loads via the vendored ainergiz library
- [x] MedASR greedy decoder produces output on real audio
- [x] `transcribe_with_backend()` composes backend + corrector cleanly
- [x] Whisper large-v3 non-turbo (not the degraded default) is the default model
- [x] 125 unit tests passing, mypy clean, pre-commit clean

## What's next

1. **Multi-sample bake-off** — run `run_multi_voice_e2e.py` against both backends on the 24 dense clinical fixtures. This is the real first comparison.
2. **Add vocabulary biasing for Whisper** — the fair comparison is biased Whisper vs raw MedASR, since MedASR has no equivalent prompting channel.
3. **Try MedASR with KenLM beam decoding** — the shipped `lm_6.kenlm` is the path to MedASR's headline 4.6% WER on RAD-DICT.
4. **Case-normalize before WER comparison** — done in `run_end_to_end.py::_normalize_for_wer`, so the casing difference does not affect metrics.
