# 6-Backend Bake-off: 16-Voice Panel, Clean Condition

**Date:** 2026-04-08 (cycle 111, option B)
**Fixtures:** 24 dense clinical photon (dense-0001 … dense-0024)
**Voices:** full 16-voice piper panel (8 UK + 8 US)
**Condition:** clean
**N per backend:** 384 (24 fixtures × 16 voices)
**System prompt:** none (neutral baseline)

## Headline

The cycle 110 / 111 clean baseline used **2 voices** (alan UK, lessac US)
which left voice-axis coverage severely under-sampled. Expanding to the
full 16-voice panel reveals:

1. **One ranking flip**: Cohere 2B and Granite-Speech 8B swap positions
   (3 ↔ 4). Granite 8B is revealed as unexpectedly voice-fragile; its
   2-voice number was luck.
2. **Voxtral Mini 3B is still #1 overall** by average WER across 16
   voices (0.1334 vs Whisper 0.1450), with the #1 margin narrowed but
   still clear.
3. **Whisper large-v3 is the voice-robustness winner** — lowest
   worst-case WER, smallest across-voice range. If deployment cares
   about *worst-case guarantee* rather than *mean performance*,
   Whisper is the conservative choice.
4. **MedASR and Granite 8B are voice-fragile in catastrophic ways** —
   each has at least one voice where WER exceeds 0.60 (essentially
   unusable).

## Full table: 2-voice vs 16-voice

| Rank (16v) | Backend | 2v WER | 16v WER | Δ WER | 2v Rec | 16v Rec | Δ Rec |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Voxtral Mini 3B | 0.1142 | **0.1334** | +1.92 | 0.5476 | 0.5050 | −4.26 |
| 2 | Whisper large-v3 | 0.1208 | 0.1450 | +2.42 | 0.3889 | 0.3532 | −3.57 |
| 3 | Cohere 2B | 0.1654 | 0.1815 | +1.61 | 0.3413 | 0.3343 | −0.70 |
| 4 | Granite-Speech 8B | 0.1242 | 0.1909 | **+6.67** | 0.5317 | 0.4286 | −10.31 |
| 5 | Granite-Speech 2B | 0.1660 | 0.1925 | +2.65 | 0.3889 | 0.3542 | −3.47 |
| 6 | MedASR fp16 | 0.1853 | 0.2482 | **+6.29** | 0.4524 | 0.3671 | −8.53 |

**Ranking flip:** Granite 8B was #3 on 2-voice (0.1242); on 16-voice it's
#4 (0.1909), swapped with Cohere which moves up from #4 to #3.

## Voice-robustness (the new axis)

Per-backend WER range across the 16 voices:

| Backend | Best WER (voice) | Worst WER (voice) | Range (pp) |
|---|---:|---:|---:|
| Whisper large-v3 | 0.1147 (jenny_dioco) | 0.1676 (southern_english_female-low) | **5.29** |
| Granite-Speech 2B | 0.1451 | 0.3065 (southern_english_female-low) | 16.14 |
| Voxtral Mini 3B | 0.0864 | 0.2806 (southern_english_female-low) | 19.42 |
| Cohere 2B | 0.1184 | 0.3625 (cori-high) | 24.41 |
| MedASR fp16 | 0.1827 | **0.6118** (southern_english_female-low) | 42.91 |
| Granite-Speech 8B | 0.1152 | **0.6228** (semaine-medium) | **50.76** |

Whisper's range is **5.29 pp** — nothing else is within 10 pp of that
consistency. Whisper was trained on hundreds of thousands of hours of
web audio spanning every realistic English accent, and the robustness
shows. Whisper's worst voice is still better than many backends'
average.

Granite-Speech 8B's range (50.76 pp, worst WER 0.6228 on
`en_GB-semaine-medium`) is the biggest surprise. The 2-voice cycle 110
result (0.1242) caught Granite 8B on two of its easier voices and
hid the underlying fragility entirely.

MedASR's 0.6118 on `en_GB-southern_english_female-low` — a 60% WER —
is clinically disqualifying for any deployment that might see UK
voices or lower-quality audio paths.

## What breaks each backend

Across the 16 voices, the "hardest" voice is specific to the backend:

- **Voxtral, Whisper, Granite 2B, MedASR** all worst on
  `en_GB-southern_english_female-low` (low-quality-tier UK female
  voice). This is a TTS-quality artefact more than a language
  artefact — the "low" quality piper tier has audible prosody
  distortions that stress most backends.
- **Granite 8B** worst on `en_GB-semaine-medium` (a different UK
  voice). Granite 8B also handles southern_english_female-low
  relatively well. Its fragility profile is genuinely orthogonal to
  everyone else's.
- **Cohere** worst on `en_GB-cori-high`. Cohere's failure mode
  concentrates on one specific UK voice.

Four of six backends have their worst voice in the UK panel. Three of
the four worst-case failures concentrate on the single `low`-quality
piper voice in the panel. This suggests two distinct voice-robustness
failure modes:

1. **Accent sensitivity** (cori, semaine breaking Cohere and Granite 8B)
2. **TTS-quality-tier sensitivity** (the `low` voice breaking 4/6 backends)

A deployment bake-off that only tested at `medium` and `high` piper
tiers would significantly over-report robustness.

## Voxtral Gy-recognition: now across 16 voices

Spot-check of the previous cycle's `Gy` mis-transcription finding:

**Overall: 272 Gy tokens in GT, 31.6% exact-miss, 25.7% case-insensitive-miss.**

Per-voice Gy miss rate varies wildly — not tracking the overall WER
ranking:

| Voice | Gy exact-miss rate |
|---|---:|
| en_US-ljspeech-high | 52.9% |
| en_US-hfc_male-medium | 52.9% |
| en_GB-northern_english_male-medium | 47.1% |
| en_GB-cori-high | 41.2% |
| en_US-norman-medium | 41.2% |
| ... | ... |
| en_US-amy-medium | 17.6% |
| en_GB-jenny_dioco-medium | 17.6% |
| en_GB-aru-medium | 17.6% |

**The 2-voice (alan + lessac) Gy miss rate (23.5–29.4%) was in the
middle of the distribution.** On the hardest voices, Voxtral misses
over half of the `Gy` tokens. The Gy recognition failure is **not
noise-driven** (cycle 111 showed flat miss rate across SNRs) and
**not voice-driven in the sense of "some voices break it"** — the
miss rate is universally non-trivial, and just worse on some voices.

This is a persistent Voxtral-specific clinical-vocabulary weakness
that needs a post-processor or phonetic rescue, independent of which
voice or noise level. It's not a 2-voice sampling artefact.

## Conclusions

1. **Voxtral Mini 3B remains the winner for mean WER across voices**
   but its #1 margin is narrower on 16-voice than 2-voice.
2. **Whisper large-v3 is the worst-case winner** — pick this if
   deployment cares about worst-case robustness over mean performance.
3. **Granite-Speech 8B's 2-voice result was significantly
   over-optimistic** — drop it down from #3 in the clean ranking.
4. **MedASR is disqualified for UK-voice deployment** — 60%+ WER on
   a common UK voice profile is not a marginal concern.
5. **Voxtral's Gy recognition weakness is a real, persistent
   clinical-vocabulary failure** independent of voice or noise.
   Fixing this should precede any deployment recommendation.
6. **2-voice bake-offs meaningfully under-sample the voice axis.**
   Going forward, voice-panel coverage should be part of any
   deployment-grade evaluation.

## Methodology notes

- Synthesis used the full 16-voice piper panel defined in
  `tests/validation/scripts/run_multi_voice_e2e.py::DEFAULT_VOICES`.
- Neutral baselines (no system prompt) for fair comparison with
  cycle 110's clean 2-voice results and cycle 111's noise sweeps.
- Reports:
  - `bakeoff_dense_3backend_16voice_clean.json` (classical)
  - `bakeoff_dense_audiollm_16voice_clean.json` (audio-LLM)
- Dense fixture set is 24 photon-therapy items. Particle therapy
  content (task #116) and safety-gate adversarials (#115) not yet
  added.

## Next

- Task #115 safety gate (Gy/GyE preservation metric) — still the
  deployment-decisive test.
- Task #114 anatomy-coverage fixtures.
- Task #116 proton/particle therapy fixtures.
- Consider a 16-voice × moderate-noise run to check whether
  voice-robustness and noise-robustness interact (does the UK/low-
  quality voice that breaks MedASR break it worse under noise?).
  Lower priority than closing the fixture gaps.
