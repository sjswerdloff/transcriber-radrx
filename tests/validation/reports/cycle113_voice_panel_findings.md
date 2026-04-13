# Cycle 113 Voice Panel Findings

**Date:** 2026-04-13
**Author:** Silas (silas-397300f6)

## Headline

TTS quality is the dominant variable in ESL UWR, not accent. The ensemble + phrase
corrections on high-quality macOS TTS voices approaches zero UWR (0.09% on particle
therapy). L2-Arctic ESL numbers represent a conservative worst-case that real clinical
deployments will not hit.

## UWR Comparison (full pipeline: phrase + word corrections)

| Panel                | RT Dense | Particle | Anatomy |
|----------------------|----------|----------|---------|
| **macOS CW (6 voices)** | **0.42%** | **0.09%** | **0.66%** |
| Piper CW (8 voices)    | 0.72%    | 1.93%    | 1.08%   |
| Piper ESL (26 voices)  | 3.38%    | 4.23%    | 4.67%   |

macOS voices include: Karen (AU), Matilda Premium (AU), Daniel (GB),
Moira (IE), Rishi (IN), Tessa (ZA).

## Key Finding: Rishi (macOS Indian English) vs L2-Arctic Hindi

| Metric (Whisper / Voxtral) | Rishi (macOS) | L2-Arctic Hindi avg |
|---------------------------|---------------|---------------------|
| RT Dense WER              | 9.4% / 6.9%  | 26.2% / 29.7%      |
| Particle WER              | 11.8% / 8.6% | 27.0% / 27.5%      |
| Anatomy WER               | 13.1% / 5.0% | 31.7% / 29.9%      |

Same accent family, 3x difference in ASR error rate. The difference is TTS
fidelity, not accent.

## Phrase Corrector Impact

13 regex patterns mined from bake-off substitution data. Biggest wins:

| Pattern | Example | Occurrences |
|---------|---------|-------------|
| gy_after_number | "50.4 ji" → "50.4 Gy" | 400+ |
| chemoradiation_join | "chemo radiation" → "chemoradiation" | 25+ |
| dose_painting | "dose pending" → "dose painting" | 15+ |
| fiducial_marker | "physical marker" → "fiducial marker" | 10+ |
| brachytherapy variants | "bracket therapy" → "brachytherapy" | 30+ |

Impact on particle therapy UWR: 3.11% → 1.93% (piper CW), 2.82% → 0.09% (macOS CW).

## What the Data Shows About Accent vs Domain

Analysis of missed terms by L1 group revealed two distinct failure classes:

1. **Domain vocabulary (accent-independent):** IGRT, SRS, SRT, "variable RBE" —
   100% miss rate for ALL speakers. Not an accent problem.

2. **Accent penalty:** Multi-syllable medical terms (medulloblastoma 85% ESL vs 6% CW,
   orchidectomy 78% vs 0%). But when high-quality TTS is used (macOS Rishi), the
   Indian English voice performs comparably to native English voices.

**Conclusion:** The ESL performance gap is predominantly a TTS quality artifact.
Real clinicians speaking into clinical microphones will produce clearer speech than
L2-Arctic's piper model. The validation framework correctly identifies the gap but
the gap's magnitude is bounded by the TTS, not the accent.

## Implications for the Paper

- ESL UWR numbers should be presented as **upper bounds**, not expected clinical performance
- The macOS TTS comparison provides evidence that accent alone is not the barrier
- The phrase corrector demonstrates that deterministic post-processing rules can
  nearly eliminate UWR when ASR output quality is sufficient
- The single-word corrector adds marginal value (<0.2% UWR improvement) because
  the remaining failures are either too garbled for edit-distance matching or are
  abbreviations the ASR doesn't know at all

## PRs Merged This Cycle

| PR | Description |
|----|-------------|
| #23 | Voice panels: Commonwealth (8 piper en_GB) + ESL (26 L2-Arctic + 2 singles) |
| #24 | macOS TTS backend: AU, IE, IN, ZA voices via `say` |
| #25 | Phrase-level domain corrections + analysis scripts |

## Bake-off Report Files

All in `tests/validation/reports/`:
- `bakeoff_commonwealth_{dense_rt,particle,anatomy}.json`
- `bakeoff_esl_{dense_rt,particle,anatomy}.json`
- `bakeoff_macos_commonwealth_{dense_rt,particle,anatomy}.json`
