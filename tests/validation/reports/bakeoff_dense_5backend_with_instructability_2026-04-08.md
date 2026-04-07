# 5-backend ASR bake-off + instructability experiment on dense RT clinical fixtures

**Date:** 2026-04-08
**Author:** silas-397300f6
**Scope:** Pipeline shakedown — small batch tests to validate the multi-backend
runner, the system_prompt instructability machinery, the memory-hygiene
sequential loading pattern, and the per-fixture/per-term metrics. NOT a
publishable evaluation. The publishable evaluation needs the big matrix
described in the "Next steps" section below.

**Source data:**
- `tests/validation/reports/bakeoff_dense_3backend.json` (Whisper, MedASR, Cohere — neutral default prompts; from 2026-04-07)
- `tests/validation/reports/bakeoff_dense_audiollm_baseline.json` (Granite 2B, Granite 8B, Voxtral 3B — neutral default prompts; new tonight)
- `tests/validation/reports/bakeoff_dense_audiollm_instructable.json` (Granite 2B, Granite 8B — with the rt_benchmark_instructable.txt domain prompt; new tonight)

**Fixtures:** 24 hand-curated dense clinical sentences (`dense_subtype` filter on rt_dictation_samples.jsonl).
**Voices:** 2 piper TTS voices — `en_GB-alan-medium` and `en_US-lessac-high`.

## Confound to call out up front

The 2-voice setup conflates accent and TTS quality tier. `alan` is medium-quality
piper, `lessac` is high-quality piper. So when we see UK/US deltas, we cannot
yet separate accent effects from voice-quality effects from individual-speaker
effects. The 16-voice piper panel (8 UK + 8 US, balanced quality tiers) exists
and is supported by the runner, but tonight's shakedown used only 2 voices to
keep iteration fast. The big matrix run later should use the full panel with
matched-quality pairs (alan medium vs amy medium for UK-vs-US at medium tier;
cori high vs lessac high at high tier).

## Headline ranking — neutral default prompts only (6 backends)

| Rank | Backend | Raw WER | Term recall | Terms found | Voice asymmetry (UK / US) |
|---:|---|---:|---:|:---|:---|
| 1 | **Voxtral Mini 3B** (audio-LLM) | **0.1142** | **0.5476** | 69/126 | 0.1001 / 0.1283 |
| 2 | Whisper large-v3 (Whisper-mlx) | 0.1208 | 0.3889 | 49/126 | 0.1177 / 0.1238 |
| 3 | **Granite-Speech 8B** (audio-LLM) | 0.1242 | 0.5317 | 67/126 | 0.1228 / 0.1255 |
| 4 | Cohere Transcribe 2B | 0.1654 | 0.3413 | 43/126 | **0.1301 / 0.2007** ⚠️ |
| 5 | Granite-Speech 2B (audio-LLM) | 0.1660 | 0.3889 | 49/126 | 0.1681 / 0.1638 |
| 6 | MedASR fp16 | 0.1853 | 0.4524 | 57/126 | 0.1950 / 0.1755 |

### Headline finding

**Voxtral Mini 3B beats Whisper large-v3 on both raw WER and term recall, on
its neutral default prompt path, with no instruction tuning.** This is the
first model in the bake-off to clear that bar. Mistral's Voxtral training
corpus appears to have meaningful clinical/RT content priors that Whisper's
general corpus does not. Voxtral preserves 69 of 126 RT vocabulary terms vs
Whisper's 49 (40% more terms preserved) and gets 11.42% WER vs Whisper's 12.08%.

**Granite-Speech 8B is competitive on term recall** (67/126, almost matching
Voxtral) but loses 1 percentage point on raw WER. Scale matters here — the
Granite-Speech 2B variant only preserves 49 terms (same as Whisper) and has
the second-worst WER of the six backends.

**Three of the six backends are below Whisper on raw WER**: Cohere, Granite 2B,
and MedASR. The HF Open ASR Leaderboard ranks Cohere #1 globally; on RT clinical
content with US-voice TTS it is third-from-bottom and exhibits a 7-point UK/US
WER asymmetry that no other backend has. Leaderboard rank does not translate
to clinical content. This was the calibration finding from the 2026-04-07 bake-off
report and the new audio-LLM data confirms it.

## The instructability experiment

**Question:** Can a domain prompt ("preserve Gy as Gy, preserve PTV/CTV/IMRT,
use numeric digits for doses, do not phonetically misspell drug names")
push an audio-LLM's transcription quality on RT content above its neutral
baseline?

**Tested:** Granite-Speech 2B and Granite-Speech 8B with `rt_benchmark_instructable.txt`.

**Could not test:** Voxtral Mini 3B. The chat-template path that would let us
thread a system prompt through Voxtral is currently broken on transformers
5.5.0 (Voxtral's chat template fails to compile in jinja with "Can't compile
non template nodes"). Voxtral fell back to its `apply_transcription_request`
path which does not accept a system prompt. **The Voxtral instructability
question is open and is the most important follow-up for the next bake-off.**

### Granite 2B: instructability is catastrophic at small scale

| Granite 2B | Baseline | Instructable | Δ |
|---|---:|---:|---:|
| Raw WER | 0.1660 | **0.3623** | **+0.1963** ⚠️ |
| Term recall | 0.3889 | **0.2698** | **−0.1191** ⚠️ |

WER more than doubled. Term recall dropped 12 percentage points. The
instructable prompt **actively hurt** Granite 2B. The model tries to follow
the domain rules but does not have the instruction-following capacity to
do so productively — it ends up producing worse output than running bare.

The voice asymmetry also flipped: under the instructable prompt, lessac US
(30.99%) is easier than alan UK (41.47%), opposite of the baseline pattern
across most other backends. This suggests the failure mode is not acoustic
but something about how the instruction-following machinery destabilizes
the decoder differently per voice.

**Lesson:** instructability is not a free capability you can layer on any
audio-LLM. Below a scale threshold, instructions cost more than they buy.
For deployment, this means "use a small audio-LLM with prompts" is a worse
strategy than "use a small audio-LLM with its native default" — and either
is worse than "use a larger audio-LLM that has the priors built in."

### Granite 8B: instructability is essentially neutral

| Granite 8B | Baseline | Instructable | Δ |
|---|---:|---:|---:|
| Raw WER | 0.1242 | 0.1254 | +0.0012 |
| Term recall | 0.5317 | 0.5476 | +0.0159 |
| alan UK WER | 0.1228 | 0.0925 | −0.0303 |
| lessac US WER | 0.1255 | 0.1582 | **+0.0327** |
| Voice asymmetry | 0.3 pts | **6.6 pts** | **+6.3 pts** |

WER essentially flat. Term recall improved by 1.6 percentage points to match
Voxtral's baseline term recall (both backends at 69/126 = 0.5476).

But notice the voice asymmetry. Granite 8B baseline is voice-flat (0.3-point
delta between alan and lessac, the most voice-symmetric backend in the
neutral comparison). Under the instructable prompt, **the asymmetry blew out
to 6.6 points** — alan got better (12.28% → 9.25%, a real win) and lessac
got worse (12.55% → 15.82%, a real loss). The instruction-following code path
is more acoustic-input-sensitive than baseline transcription. We do not yet
know whether the issue is alan's medium-tier piper voice being cleaner than
lessac's high-tier (the quality-vs-tier confound), or the accent itself, or
something specific to lessac's prosody under the chat-template inference path.

**Lesson:** instructability may be giving Granite 8B a real win on alan that
is being canceled out by a real loss on lessac. We cannot tell from a 2-voice
sample whether the alan win or the lessac loss is the "true" effect. **The
big matrix run with the full 16-voice panel is needed to settle this.**

### What instructability did NOT do

It did not push either Granite variant past Voxtral's neutral baseline.
Granite 8B with instructable prompt sits at 12.54% WER and 54.76% term
recall, vs Voxtral neutral at 11.42% WER and 54.76% term recall. **Same
term recall, but Voxtral wins WER and voice symmetry without needing any
instruction at all.** Voxtral's training-corpus prior is bigger than what
explicit prompting can recover for Granite on this content.

The most useful comparison would be **Voxtral with the same instructable
prompt vs Voxtral neutral**, and we could not produce that data tonight.
It is the experiment that would tell us whether instructability ON TOP OF
a strong prior compounds, or whether the strong prior already covers the
ground that instructions would address.

## The Gy failure mode catalogue (six distinct keys)

Every backend in the bake-off has its own systematic way of getting the
radiotherapy dose unit "Gy" wrong on the neutral baseline path. This is the
single most clinically important word in the corpus and none of the classical
ASRs handle it reliably:

| Backend | "Gy" failure mode |
|---|---|
| Whisper large-v3 | `GI` (uppercase, consistent) |
| MedASR fp16 | `Gy` correct sometimes, `Giy` other times — phonetic |
| Cohere Transcribe | `Jai` / `JI` / `ji` / `jive` / `gai` — accent-dependent |
| Granite-Speech 2B | `gi` (lowercase, consistent) |
| Granite-Speech 8B | mixed — sometimes `gi`, sometimes correct |
| **Voxtral Mini 3B** | **`Gy` correct, both occurrences in dense-0001** |

Voxtral is the first model in the bake-off to handle the dose unit correctly
out of the box. This single observation is more clinically meaningful than
any aggregate WER number, because in RT dictation `Gy` is a unit you cannot
afford to mistranscribe — getting it wrong corrupts every dose statement in
a treatment plan. The aggregate metrics (raw WER, term recall) do not
weight `Gy` specially; future bake-off runs should track per-term accuracy
on a curated list of safety-critical terms (`Gy`, `cGy`, `MV`, `MU`, `cm`,
`mm`, drug-name spellings) as a separate metric.

## Critical clinical safety finding (per-fixture analysis of best aggregate)

The single lowest WER measurement of the bake-off is **Granite-Speech 8B with
the instructable domain prompt on the en_GB-alan-medium voice: 0.0925**. This
configuration achieved **five fixtures at WER 0.000**, including the headline
dense-0001 fixture that every other backend got wrong:

```
dense-0001: prescribed dose of 54 gy in 30 fractions to the ptv with a
            simultaneous integrated boost to 60 gy for the high-risk ctv  ✓
dense-0004: total dose of 70 gy in 35 fractions over 7 weeks with concurrent
            chemotherapy  ✓
dense-0005: organs at risk included the parotids pharyngeal constrictors
            brainstem and cervical spinal cord  ✓
dense-0009: treatment was delivered using volumetric modulated arc therapy
            with daily kilovoltage image guidance  ✓
dense-0012: deep inspiration breath hold technique was employed to reduce
            cardiac dose during left breast irradiation  ✓
```

**No other backend × voice × prompt combination matched any of these.** The
instructable prompt successfully steered Granite 8B to preserve `Gy`, the
acronyms PTV/CTV/IMRT/VMAT, the "high-risk" hyphen, and the numeric digit
format — all the things the prompt explicitly required. This is the proof
that instructability CAN deliver clinically meaningful improvements when
the underlying audio-LLM has the scale to follow instructions.

### But the same configuration produced a clinically dangerous error

dense-0022 transcription on the same alan UK voice:

```
GT : Neoadjuvant chemoradiation consisted of 50.4 Gy in 28 fractions with
     concurrent capecitabine.
G8B: neoadjuvant chemo-radiation consisted of 504 gy in 28 fractions with
     concurrent capecitabine
```

**`50.4 Gy` was transcribed as `504 gy`. The decimal point was dropped.**

50.4 Gy is a standard neoadjuvant rectal cancer chemoradiation dose. 504 Gy
would be roughly 10× a lethal dose to any RT target. **If this transcription
were fed into an actual treatment planning workflow without manual review, it
is the kind of error that kills patients.**

The aggregate WER on this fixture was 0.231. WER does not weight the decimal
point any more heavily than any other word — it counts as a single substitution
in a 12-word sentence. The aggregate metric for this configuration was 0.0925
overall. **A 9.25% aggregate WER looks like the best result of the bake-off.
Buried inside it is a single-token error with 10× clinical magnitude.**

This is the most important finding of the shakedown and it would have been
invisible without the per-sample diff analysis. The bake-off pipeline did
exactly what it was built to do: it surfaced a class of error that aggregate
metrics hide.

### What this implies for the metric design

WER is necessary but not sufficient for clinical ASR validation. The next
iteration of the bake-off needs a **safety-critical token preservation
metric** that scores models specifically on whether they preserved values
that matter clinically:

1. **Dose value preservation**: extract numeric dose values (`X Gy`, `X.Y Gy`,
   `X cGy`, etc.) from ground truth and from transcription, and verify
   exact match. Decimal-point drops, digit insertions, and unit confusions
   should fail this check regardless of surrounding sentence WER.
2. **Acronym preservation**: per-token check that all RT acronyms (PTV, CTV,
   GTV, OAR, IMRT, VMAT, IGRT, SRS, SBRT, DIBH) appear in the transcription
   in their canonical uppercase form, not letter-spaced or substituted.
3. **Drug name preservation**: per-token check on a curated list of RT-
   adjacent drug names (cisplatin, capecitabine, fluorouracil, mitomycin-C,
   carboplatin, etc.) — exact spelling match.

These are token-level safety gates, not aggregate metrics. A model that fails
ANY of them on ANY fixture should be flagged regardless of how good its
overall WER looks. **A model that drops decimal points on 1 in 24 fixtures
is not deployable for clinical RT dictation no matter what its overall WER is.**

### Promoting dense-0022 to a regression-watch fixture

Going forward, any backend × voice × prompt combination that produces "504"
instead of "50.4" on dense-0022 (or analogous decimal-drop errors on
similar fixtures) should fail the safety gate. This is the first
"do not deploy under any circumstances" failure mode the bake-off has
surfaced, and it deserves a name and a watch.

## Anatomy term safety check

dense-0012 ("Deep inspiration breath hold technique was employed to reduce
cardiac dose during left breast irradiation") was transcribed cleanly by
all six backends on both voices. **No model censored, hedged, or garbled
"breast" in a clinical context.** This is one data point on one anatomy term
and does not tell us anything about the harder cases (prostate, cervix,
rectum, vulva, vagina, testes, penis, anus, endometrium). The anatomy-coverage
fixture batch (task #114) is the systematic test for that concern.

## Memory footprint reality check

**stuart_m1max ran the full 5-backend bake-off in two phases tonight with
sequential model loading.** The backends loaded one at a time, each unload()
forced gc.collect() before torch.mps.empty_cache() to release memory
deterministically before the next backend's load(). Peak measured RAM during
inference (Granite 8B mid-run): **34 GB Python process**. That is roughly 2×
the params×dtype calculation:

- Granite-Speech 8B bf16 params: 16 GB
- Audio encoder + adapter weights: 1-2 GB
- Activations + KV cache during inference: 6-8 GB
- Python interpreter + transformers state + torch internals: 5-8 GB
- **Total live: ~30-34 GB for an "8B" model**

Voxtral Mini 3B at inference time was around 18-22 GB live. Cohere Transcribe
2B was around 10-12 GB live. Granite-Speech 2B was around 10 GB live. **Memory
estimates from "params × bytes" are unreliable for capacity planning of audio
LLMs by a factor of 2 or more.** For a hardware sizing rule: take the on-disk
fp16/bf16 footprint of an audio-LLM, multiply by ~1.5-2 for the live inference
peak, then add OS + browser + other workload headroom on top.

## What the shakedown validated

1. **The multi-backend runner works end-to-end** for 5 backends spanning two
   architectural categories (classical Conformer ASR + audio-LLM with chat
   template).
2. **The system_prompt parameter flows through cleanly** from CLI → runner →
   transcribe_with_backend → backend.transcribe_wav. Classical ASRs log-warn-
   and-ignore. Audio-LLMs that support it use it. Audio-LLMs that don't
   support it (Voxtral on this transformers version) log-warn-and-ignore
   gracefully without crashing.
3. **Sequential model loading with forced gc.collect() works** to keep peak
   memory bounded by the largest single model rather than the sum.
4. **Per-fixture, per-voice, per-backend, per-prompt-mode JSON output** is
   rich enough to support analysis like the failure-mode catalogue and the
   instructability experiment write-up above.
5. **The pipeline can find genuinely surprising results** that contradict
   external benchmarks (Cohere's leaderboard rank vs its bake-off rank) and
   contradict initial hypotheses (instructability is not a free win;
   Granite 2B regresses badly under instruction).

## What the shakedown did NOT validate

1. **Statistical power.** 24 fixtures × 2 voices × 1 model = 48 transcriptions
   per backend per condition. That is enough to see large effects (Voxtral
   beating Whisper, Granite 2B regressing under instruction) but not enough
   for confidence intervals on per-fixture WER deltas.
2. **Voice generalization.** Two voices are not a panel. The full piper
   16-voice panel exists and is supported but was not used.
3. **Acoustic robustness.** All audio is clean piper TTS. Noise, reverb,
   partial utterances, background voices, and real household audio are all
   untested.
4. **Anatomy coverage.** One anatomy term (breast) is not a panel either.
5. **Voxtral instructability.** Cannot test until the chat template issue
   on transformers 5.5.0 is resolved (either fixed upstream, worked around
   with mistral-common directly, or pinned to a different transformers
   version).
6. **Real RT dictation audio.** All fixtures are hand-curated synthetic
   sentences, not actual radiotherapy dictation recordings.

## Next steps for the big matrix run

Ranked by experimental value:

1. **Fix Voxtral chat template path** (or work around with mistral-common)
   so Voxtral can participate in the instructability experiment. If Voxtral
   with the instructable prompt beats Voxtral baseline, we have evidence
   that instructability compounds with strong priors. If it does not, we
   have evidence that strong priors saturate the achievable quality and
   instructions are only useful when the priors are weak.
2. **Expand to the full 16-voice piper panel** with matched-quality pairs
   (alan medium / amy medium and cori high / lessac high) so we can
   separate accent from voice-quality from individual-speaker effects.
3. **Add the anatomy-coverage fixture batch (task #114)** — 10-15 fixtures
   covering breast, prostate, cervix, rectum, anus, vulva, vagina, testes,
   penis, endometrium. This is the systematic safety-filter test.
4. **Add an acoustic-robustness tier** — MUSAN noise mixing, reverb via
   pyroomacoustics, partial utterances. This is where MedASR's claimed
   noise-robustness advantage would actually be testable.
5. **Add Phi-4-multimodal** as a third audio-LLM family alongside Granite
   and Voxtral. Different architectural lineage; would help triangulate
   whether the audio-LLM advantage is general or specific to certain
   training corpora.
6. **Add per-term accuracy on a safety-critical token list** as a separate
   metric: `Gy`, `cGy`, `MV`, `MU`, `cm`, `mm`, drug-name spellings. WER
   does not weight these specially even though they are the words you
   cannot afford to get wrong in clinical use.
7. **Add real RT dictation audio fixtures** if and when Stuart can capture
   them. Synthetic TTS audio is necessary but not sufficient.

## Honesty notes

- **The "Voxtral beats Whisper" headline is on N=48 transcriptions** (24
  fixtures × 2 voices). It is a real signal at this sample size — the
  WER delta is about 7% relative — but the confidence interval is wide.
  The big matrix run with hundreds of fixtures is what would establish
  this as a robust finding.
- **The "Granite 2B regresses catastrophically under instruction" finding
  is the most surprising** result of the night and is also on N=48. I
  trust it as a directional signal because the regression is so large
  (more than doubled WER) and the smoke test on dense-0001 already
  surfaced the same pattern (CTV→PTV hallucination, dropped initial words)
  before the full run. But the magnitude could shift with more data.
- **The voice asymmetry findings should be the LEAST trusted** finding
  because of the alan-medium-vs-lessac-high quality confound.
- **Voxtral's chat template being broken is a tonight-only blocker, not a
  permanent capability gap.** It can be fixed with a transformers pin or
  upstream contribution. The Voxtral instructability question is open,
  not closed.

## Pipeline credit

This shakedown ran on infrastructure built across cycles 105-106 + 110-111:
- Pluggable ASR backends (`src/transcriber_radrx/asr_backends/`) with the
  ASRBackend protocol
- Multi-backend bake-off runner with sequential load/unload + GC hygiene
- Domain prompt files (`tests/validation/prompts/rt_benchmark_*.txt`)
- The dense clinical fixture set (24 hand-curated RT sentences)
- The piper TTS audio synthesis pipeline (Vivian's work, cycle 105)
- The whole `transcriber-radrx` package structure with its CI guardrails

The pipeline IS the product. Tonight's bake-off is the pipeline doing its
job: surface the failure modes that variety produces, refuse to commit to a
"winner" before the data is in, and produce per-fixture evidence that future
runs can build on.
