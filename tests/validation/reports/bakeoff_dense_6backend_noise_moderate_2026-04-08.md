# 6-Backend Bake-off: Clean vs Moderate Noise

**Date:** 2026-04-08 (cycle 111)
**Fixtures:** 24 dense clinical (dense-0001 … dense-0024)
**Voices:** en_GB-alan-medium, en_US-lessac-high (N = 48 per backend)
**Noise preset:** moderate (10 dB SNR, MUSAN corpus)
**Noise seed:** 0, crossfade 30 ms
**System prompt:** none (neutral baseline for both conditions)

## Headline

**The clean-baseline ranking is stable under 10 dB SNR moderate noise.**
All 6 backends maintain the same 1–6 order in both conditions. No
ranking flips.

## Full table

| Rank | Backend | Clean WER | Noisy WER | Δ WER | Clean Rec | Noisy Rec | Δ Rec |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Voxtral Mini 3B | 0.1142 | **0.1109** | **−0.33** | 0.5476 | 0.5000 | −4.76 |
| 2 | Whisper large-v3 | 0.1208 | 0.1289 | +0.81 | 0.3889 | 0.3810 | −0.79 |
| 3 | Granite-Speech 8B | 0.1242 | 0.1424 | +1.82 | 0.5317 | 0.4921 | −3.96 |
| 4 | Cohere 2B | 0.1654 | **0.1615** | **−0.39** | 0.3413 | 0.3492 | +0.79 |
| 5 | Granite-Speech 2B | 0.1660 | 0.1795 | +1.35 | 0.3889 | 0.3651 | −2.38 |
| 6 | MedASR fp16 | 0.1853 | 0.2118 | **+2.65** | 0.4524 | 0.3889 | **−6.35** |

Δ values in percentage points.

## Findings

### 1. Voxtral Mini 3B is noise-robust at 10 dB (surprising)

Voxtral is the only model whose raw WER **improves** under moderate
noise (0.1142 → 0.1109, −0.33 pts). The Mistral-based LLM component
appears trained with clinical/noisy priors strong enough that a little
background texture doesn't hurt acoustic recognition — possibly helps
by regularising out the slight over-fitting to clean TTS spectral
flatness.

Term recall does drop (−4.76 pts) so the *vocabulary* competence is
more sensitive than the raw-WER metric suggests, but the lead over
#2 Whisper is wider in noise (+1.80 pts WER) than in clean (+0.66 pts).
Voxtral is the clear winner and gets wider under stress.

### 2. MedASR's medical-vocab advantage collapses under noise

MedASR had the **second-best clean term recall** at 0.4524 — the
entire clinical-ASR justification for choosing it over Whisper. At
10 dB SNR:

- Raw WER degrades +2.65 pts (the worst hit of any backend).
- Term recall drops 6.35 pts to 0.3889, tied with Whisper (0.3810)
  and Granite-Speech 2B (0.3651).

The medical-vocabulary advantage is a clean-audio phenomenon. In any
real clinical environment with HVAC, fans, or nearby conversation,
the reason to deploy MedASR instead of Whisper largely disappears.

### 3. Cohere 2B is the most noise-robust classical ASR

Cohere's raw WER slightly improves (−0.39 pts) and term recall gains
0.79 pts. The absolute scores are middle-of-pack, but the robustness
profile is flat — useful if you need a predictable backend whose
behaviour does not vary between training rooms and production
environments.

### 4. Whisper degrades least of the gradient-degraders

Whisper's +0.81 pts WER is the smallest non-improved delta. Training
on massive noisy web audio shows through. Still #2, still a strong
default.

### 5. Granite-Speech 2B vs 8B

Both Granite variants degrade proportionally (+1.35 vs +1.82 pts WER).
The 8B variant keeps its rank advantage but the gap closes:

| | Clean | Noisy |
|---|---:|---:|
| Granite 2B vs 8B Δ WER | 4.18 pts | 3.71 pts |
| Granite 2B vs 8B Δ Recall | 14.28 pts | 12.70 pts |

Neither flips; 8B remains the stronger variant under noise.

## Methodology notes

- Noise coverage used the prefer-long-first policy with 30 ms linear
  crossfades at splice seams (see
  `tests/validation/audio_synthesis/noise_injection.py`). For all 24
  dense fixtures (5–15 s typical), single-file MUSAN coverage was
  used — no splicing required at this length, since 34 % of MUSAN
  clips are ≥ 20 s.
- Clean baselines pulled from cycle 110 reports
  (`bakeoff_dense_3backend.json`,
  `bakeoff_dense_audiollm_baseline.json`) to avoid re-running the
  clean condition.
- Granite 8B required a retry run — the first audio-LLM invocation
  passed `granite_speech:granite-speech-3.3-8b` as CLI arg, but the
  `_parse_backend_arg` treated `granite-speech-3.3-8b` (without
  `ibm-granite/` prefix) as the HF model ID, causing
  `AutoProcessor.from_pretrained` to fail. The retry passed the
  full `granite_speech:ibm-granite/granite-speech-3.3-8b` form.
  The runner should be tightened to either validate model IDs or
  re-expose the raw HF ID in error messages — follow-up.

## Voxtral degradation curve (full SNR sweep)

After the ranking-stability run, Voxtral Mini 3B was swept across all
three noise presets to trace its noise-sensitivity curve as the winner.

| Condition | SNR | WER | Δ from clean | Term recall |
|---|---:|---:|---:|---:|
| Clean | ∞ | 0.1142 | 0 | 0.5476 |
| Quiet | 20 dB | 0.1217 | +0.0075 | 0.5079 |
| Moderate | 10 dB | 0.1109 | −0.0033 | 0.5000 |
| Busy | 5 dB | 0.1144 | +0.0002 | 0.5000 |

**Max raw-WER swing across all four conditions: 0.0108 (1.08 pp).**
This is within measurement noise at N=48. Voxtral Mini 3B is
effectively **flat** in raw WER across the entire realistic clinical
SNR range, from silent-booth to busy-hallway / LINAC-vault-with-beam-on.

Term recall drops about 5 pp from clean to any-noise (0.5476 → ~0.50),
then plateaus: a one-time cost for entering noisy conditions at all,
not a gradient along the noise level. The clinical-vocabulary
competence is somewhat less resilient than the raw-WER metric, but
still above every non-Voxtral backend in every condition tested.

## Safety spot-check: dose-value preservation

Prompted by the cycle 110 Granite 8B + instructable finding (silent
decimal drop `50.4 Gy → 504 Gy`), every Voxtral transcription across
all four noise conditions (N = 192) was audited for dose-value
corruption.

### Numeric-magnitude preservation

**Every numeric dose/fractionation value was preserved in every
Voxtral sample.** No decimal drops, no magnitude substitutions, no
silent corruption. Every "54 Gy" stayed "54", every "50.4 Gy" stayed
"50.4", every "60 Gy in 30 fractions" had 60 and 30 in the output
(sometimes as the word "three" instead of "3", which is clinically
equivalent and would round-trip through any downstream parser).

### Unit-word recognition failures

Voxtral systematically mis-transcribes the word "Gy" under every
condition: 25/192 samples (13%) show "Gy" rendered as `gye`, `jai`,
`GJ`, `gi`, `jive`, `gaj`, `J`, or `GY`. This is a persistent
clinical-vocabulary weakness independent of noise level. Every
instance leaves an **obviously-wrong** non-word in the output that
any human reviewer or NER-downstream parser would immediately flag.
This is **not** the same failure class as the Granite 8B silent
decimal drop; it is visible corruption.

### One refusal hallucination

On `quiet / en_GB-alan-medium` for the fixture `IMRT plan delivered 60 Gy
in 30 fractions to the PTV sparing the OARs...`, Voxtral emitted:

> `I'm not sure what you're asking.`

The audio-LLM exited transcription mode and produced an assistant
refusal instead. 1/192 samples = 0.5 % rate. Like the unit-recognition
failures, this is **visible** — any downstream consumer would flag a
refusal string appearing in a transcription field. But it is a
distinct failure class to be aware of: audio-LLM backends can leave
the transcription task under edge-case conditions. No classical ASR
in this bake-off exhibited this behaviour.

### Safety summary

Voxtral Mini 3B, across 192 transcriptions at four SNR levels:

- **0** silent numeric corruptions (dose magnitudes preserved)
- **25** visible unit-word mis-transcriptions (clinical vocabulary weakness)
- **1** visible refusal hallucination (audio-LLM task-exit)

Every failure is visible in plain output. Nothing in this spot-check
would have passed silently through a human review or a minimal
downstream sanity check. Voxtral's safety profile on this fixture set
is cleaner than Granite 8B + instructable was on the same dense-0022
fixture in cycle 110.

**Caveat:** N = 192 is a shakedown, not a clinical validation run. The
task #115 safety-gate fixture set — adversarial dose-value stress
tests, homophone traps for "Gy" / "guy", decimal-point stressors
(50.4 vs 504, 0.4 vs 4, etc.) — is still the right instrument for a
deployment-grade safety claim. This spot-check is a "no obvious
disqualifier" pass, not a "cleared for clinical use" pass.

## Cycle 111 conclusions

1. **Ranking is stable under realistic noise** (all 6 backends hold rank order
   clean vs 10 dB).
2. **Voxtral Mini 3B is the winner and robust across the full SNR range**
   — not just at moderate noise, but at 20 dB, 10 dB, and 5 dB all
   within 1.1 pp WER of its clean baseline.
3. **MedASR's medical-vocab advantage is clean-audio-only** and
   collapses at 10 dB; its case for deployment over Whisper is
   substantially weakened.
4. **Voxtral safety spot-check shows visible failure modes only** —
   no silent decimal drops. Still subject to proper task #115
   safety-gate validation before deployment.

## Next

- Task #115: implement safety-gate fixture set and metric for
  dose-value / RT-acronym preservation. This is the real deployment
  gate.
- Task #114: anatomy-coverage fixture batch to verify no backend
  has anatomy-word blind spots (breast, prostate, cervix, rectum,
  vulva, etc.) that this dense-clinical set doesn't exercise.
- Runner polish: `_parse_backend_arg` should validate model IDs or
  resolve aliases so that `granite_speech:granite-speech-3.3-8b`
  either works as an alias or fails with a clearer error than
  "not a valid model identifier".
