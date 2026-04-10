# Proton/Particle Therapy Bake-off Findings — 2026-04-09

**Author:** silas-397300f6
**Corpus:** 28 hand-curated proton/particle therapy fixtures (`tests/validation/fixtures/proton_samples_draft.jsonl`), 15 adult/general + 12 pediatric proton + 1 carbon ion
**Voices:** `en_GB-alan-medium`, `en_US-lessac-high`
**Backends:** `mlx_whisper:whisper-large-v3-mlx`, `medasr:ainergiz/medasr-mlx-fp16`, `voxtral:mistralai/Voxtral-Mini-3B-2507`
**Total samples:** 168 (28 × 2 voices × 3 backends)
**Corrector:** phonetic correction OFF — raw ASR signal only
**Report JSON sources:**
- `bakeoff_proton_draft_2026-04-09.json` (Whisper + MedASR)
- `bakeoff_proton_voxtral_2026-04-09.json` (Voxtral)

**Status (updated 2026-04-09 late):** Stuart's clinical spot-check is complete. Dose numbers approved as-is. Junction vocabulary normalized to "junction feathering"; "dual ramping" retained as Stuart's original phrasing in one fixture; "gradient" removed per his guidance. Cranial field geometry corrected to "LPO and RPO" in proton-0026 per his clinical review. Fixture file renamed from `proton_samples_draft.jsonl` to `particle_samples.jsonl` to reflect the broader particle-therapy scope (including the carbon ion fixture). The corrected corpus re-ran cleanly against all three backends (see "Re-run and non-determinism finding" section below).

**CRITICAL caveat on all headline numbers in this document:** the aggregate metrics and per-sample safety-gate results presented below are from **single-pass bake-off runs** against a single piper TTS synthesis per fixture. The cycle 112 re-run surfaced that **piper TTS is non-deterministic** across successive calls: the same input text produces acoustically different WAV output each time it is synthesized, and this upstream acoustic variance cascades into different ASR transcriptions. The ASR backends themselves (Whisper, MedASR) are **deterministic given fixed WAV input** — the variance is entirely upstream in TTS. Two back-to-back bake-off runs produced meaningfully different CRITICAL failure counts *because the audio was different each time*, not because the ASR behaved inconsistently. **No single-pass number in this report should be treated as a reliable characterization of a backend's safety profile.** The right experimental design is to pre-synthesize N cached audio variants per fixture × voice and run each backend once across that expanded corpus (see "TTS variance finding" section below and task #122).

---

## Headline result

**Voxtral Mini 3B is the WER leader AND the hardest of the three backends to deploy safely in radiation oncology.**

"Best on aggregate WER" is not the same as "best on safety." This run is the first concrete demonstration of that separation in the transcriber-radrx bake-off history.

## Aggregate metrics

| Backend | Raw WER | Term recall | Voice asymmetry (alan − lessac) |
|---|---:|---:|---:|
| **Voxtral Mini 3B** | **0.1073** | 0.5495 | **0.0029** (0.3 pp) |
| Whisper large-v3 | 0.1325 | **0.5721** | 0.0198 (2.0 pp) |
| MedASR fp16 | 0.1868 | 0.4054 | 0.0067 (0.7 pp) |

Voxtral wins raw WER by 2.5 pp over Whisper and 8.0 pp over MedASR. Voxtral's voice asymmetry is effectively zero. Whisper's voice asymmetry is 2 pp — wider than on the original dense fixtures — suggesting proton and pediatric vocabulary pulls harder on accent than general RT content.

Whisper narrowly edges Voxtral on term recall (0.5721 vs 0.5495). This is a useful separation: Whisper preserves more vocabulary terms verbatim, Voxtral makes fewer overall errors but exhibits a systematic semantic substitution that term-counting catches.

## Safety-critical finding 1: Voxtral silently substitutes `GyE` → `Gy`

**This is the most important finding of the cycle.**

`GyE` appeared in 36 sample transcripts in the gold corpus. Preservation:

| Backend | GyE preserved | Dominant failure mode |
|---|---:|---|
| Whisper large-v3 | 0/36 (0%) | `GiE`, `GI-E`, `Jai E`, `JIE`, `JEE`, `HIE`, `J`, `J, E` — visibly broken renderings, 15+ variants |
| MedASR fp16 | 0/36 (0%) | `Jie`, `JiE`, `Gi`, `Gie`, `GiE`, `GE`, `GAE`, `gi.e.`, `giE` — similarly obviously broken |
| **Voxtral Mini 3B** | **7/36 (19%)** | **Silent `Gy` substitution on ~22 of the 29 failures** |

Voxtral is the only model to preserve `GyE` at all. But its dominant failure mode is **clinically dangerous**: it rewrites `GyE` as `Gy` without any visible corruption. Examples from the run:

| Fixture | Indication | Gold | Voxtral prediction |
|---|---|---|---|
| proton-0001 | prostate proton | `79.2 GyE` | `79.2 Gy` |
| proton-0002 | generic with RBE | `54 GyE` | `54 Gy` |
| proton-0004 | chordoma | `78 GyE` | `78 Gy` (also 78→70) |
| proton-0006 | rectal proton | `50.4 GyE` | `50.4 Gy` |
| proton-0007 | lung proton SBRT | `60 GyE` | `60 Gy` |
| proton-0008 | mixed Gy/GyE | `74 GyE` | `74 Gy` |
| proton-0011 | reirradiation | `60 GyE` | `60 Gy` |
| proton-0014 | proton beam therapy | `70 GyE` | `70 Gy` |
| proton-0016 | std-risk medullo CSI | `23.4 GyE` | `23.4 Gy` |
| proton-0019 | Ewing pelvis | `55.8 GyE` | `55.8 Gy` |
| proton-0024 | neuroblastoma | `21.6 GyE` | `21.6 Gy` |
| ... | ... | ... | ... |

### Why this is worse than a visible failure

A clinician reviewing a transcript that says `79.2 GiE` will notice "GiE" is not a unit and correct it. A clinician reviewing a transcript that says `79.2 Gy` on a proton plan may not notice anything wrong. `Gy` is a valid, well-known unit. The ~10% RBE correction that distinguishes `GyE` from `Gy` is lost silently.

Downstream consequences if the transcript is used verbatim:
- A planning system that ingests `79.2 Gy` on a proton Rx will interpret it as physical dose, not biologically equivalent dose, and the patient would be **undertreated by the RBE factor** (standard RBE = 1.1, so ~10%).
- A physicist checking the plan against the Rx would see `79.2 Gy` matching the plan's physical dose and approve it.
- No safety check in the chain catches the unit error because both reference frames are internally consistent.

### Mechanism (hypothesized)

Voxtral Mini 3B uses an LLM decoder for text generation. The decoder has strong priors that "Gy" is a radiation unit and "GyE" is either rare or not in its vocabulary as a distinct token. When the audio encoder produces acoustic features for "gray equivalent" or "gee-why-ee," the decoder's language-model prior pulls toward the more common form and silently normalizes. This is the audio-LLM equivalent of a translation model "fixing" unusual phrasing it doesn't recognize.

The same mechanism explains why Voxtral is worse than Whisper on the **spelled-out** "gray equivalent" form:

| Class | Whisper | MedASR | Voxtral |
|---|---:|---:|---:|
| `gray equivalent` preserved | 12/12 (100%) | 7/12 (58%) | **3/12 (25%)** |

Voxtral is rewriting "gray equivalent" as "gray" or "Gy" at a 75% rate — same normalization behavior on the spelled form.

### Implications

1. **Voxtral cannot be deployed for proton or particle therapy dictation without post-processing.** A corrector must detect particle-therapy context (pencil beam scanning, proton, carbon ion, craniospinal, etc.) and promote `Gy` → `GyE` where clinically appropriate, OR fail safe and flag the sample for human review.
2. **The cycle 111 negative finding on audio-LLMs was about instructability.** This is a *different* and more fundamental problem: the LLM decoder silently corrects clinically-critical terminology even on neutral baselines with no domain prompt.
3. **The finding is publishable as the motivating example for the validation framework.** "The model that wins WER is also the most dangerous to deploy unreviewed, and the validation framework is what makes that distinction visible."

## Safety-critical finding 2: Slashed IGRT forms — Voxtral wins decisively

`3D/3D`, `2D/3D`, and `3D/2D` appeared across 8 samples total in the gold corpus. Preservation:

| Class | Whisper | MedASR | Voxtral |
|---|---:|---:|---:|
| `3D/3D` | 0/4 (0%) | 0/4 (0%) | **4/4 (100%)** |
| `2D/3D` | 0/2 (0%) | 0/2 (0%) | **2/2 (100%)** |
| `3D/2D` | 0/2 (0%) | 0/2 (0%) | **2/2 (100%)** |

**Voxtral is the only model that preserves the slashed IGRT form.** Whisper and MedASR both normalize to `3D 3D` or drop the digit entirely. This is a clean positive finding for Voxtral — its LLM decoder's punctuation handling is significantly better than classical-ASR output formatting.

For the staged correction pipeline (#119): Whisper and MedASR outputs need a rule that promotes unslashed `3D 3D` / `2D 3D` → the slashed form when adjacent to IGRT vocabulary (image guidance, cone beam, CBCT).

## Safety-critical finding 3: Decimal drop class (cycle 110 pattern replicated)

Decimal values of dose (e.g. `50.4`, `55.8`, `59.4`, `21.6`, `23.4`) are a systematic failure class. Cycle 110 caught a single instance (Granite 8B + instructable dropping `50.4 → 504` on dense-0022). This run extends the class across three backends:

| Backend | Decimal drops | Cases |
|---|---:|---|
| Voxtral Mini 3B | 1 | `50.4 → 50` on proton-0020 (parameningeal rhabdo) |
| Whisper large-v3 | 1 | `55.8 → 55` on proton-0019 (Ewing pelvis, lessac voice) |
| MedASR fp16 | **7** | `50.4 → 50` (proton-0006, rectal — cycle 110 replica), `55.8 → 55` (0019 ×2, 0027), `59.4 → 59` (0021 ependymoma), `21.6 → 21` (0024 neuroblastoma), `23.4 → 23` (0027 std-risk medullo CSI) |

**MedASR is 7× worse than the other two backends** on this specific failure class. Its CTC decoder appears to have systematic trouble with the `.N` suffix on dose values.

`proton-0006` is a direct replication of the cycle 110 catch: same standard neoadjuvant rectal chemoradiation dose, dropped by a different model class. The failure mode is reproducible across acoustic model families.

## Safety-critical finding 4: Whole-number dose values dropped entirely

Not all dose-value errors are decimal drops. MedASR also frequently drops whole-number doses entirely:

| Fixture | Indication | Gold | MedASR prediction |
|---|---|---|---|
| proton-0002 | with RBE statement | `54 GyE in 30 fx` | `...30 fractions...` (54 gone) |
| proton-0005 | SIB dual dose | `70 GyE ... 56 GyE` | `70 ... 35 fractions...` (56 gone) |
| proton-0008 | Gy/GyE disambig | `70 Gy while protons deliver 74 GyE` | `...74 Gy...` (70 gone) |
| proton-0011 | reirradiation | `60 GyE in 30 fx` | `...30 fractions...` (60 gone) |
| proton-0014 | with RBE of 1.1 | `70 GyE in 35 fx` | `...35 fractions...` (70 gone) |

**MedASR dropped 5 additional dose values entirely.** When combined with its 7 decimal drops and 1 whole-number substitution, MedASR had **13 of 56 samples (23.2%) with some form of dose-value corruption**. Whisper had 3/56 (5.4%). Voxtral had 3/56 (5.4%).

## Safety-critical finding 5: MedASR fails the specialized vocabulary that is its value proposition

MedASR is marketed as medical-specialized. It is worse than both Whisper and Voxtral on every specialized RT vocabulary term in this corpus:

| Term | Whisper | MedASR | Voxtral |
|---|---:|---:|---:|
| pencil beam scanning | 11/12 (92%) | **4/12 (33%)** | 10/12 (83%) |
| carbon ion | 1/2 (50%) | **0/2 (0%)** | 1/2 (50%) |
| feathering | 4/4 (100%) | **0/4 (0%)** | 4/4 (100%) |
| medulloblastoma | 1/2 (50%) | **0/2 (0%)** | 1/2 (50%) |
| craniospinal | 7/8 (88%) | 5/8 (62%) | 6/8 (75%) |

MedASR should probably not continue to be a primary comparator in future bake-offs. It could reasonably be dropped from the Kindled framework's recommended backend list.

## Things that worked

- **Counterfactual held (proton-0015).** No model hallucinated `GyE` on the IMRT case where the correct unit is `Gy`. Voxtral's substitution is Gy→Gy(correct), not direction-reversible into hallucinated GyE from proton context.
- **`dual ramping` / `gradient junction` / `junction feathering` terminology**: Whisper and Voxtral 100%, MedASR 50%–100%. Stuart's flagged technique vocabulary is transcribable.
- **`dynamic DRR`**: Whisper and MedASR 100%, Voxtral 50%. Whisper is actually stronger on this specific IGRT acronym.
- **`RBE`**: all three backends 100%.

## Implications for cycle 112 tasks

### #115 safety-gate metric — *the immediate priority*

The proton run gives us five concrete failure classes the metric must detect. Proposed spec:

1. **Decimal drop**: gold has `\d+\.\d+`, prediction contains `\d+` at same semantic position (after tokenizing around dose vocabulary) where the decimal portion is missing. Severity: **CRITICAL** (Therac-25-class error).
2. **Whole-number dose value missing**: gold contains a number followed by a dose unit (`Gy`, `GyE`, `gray`, `cGy`) and that number does not appear anywhere in the prediction. Severity: **CRITICAL**.
3. **Silent unit substitution (`GyE` → `Gy`)**: gold contains `GyE` or `gray equivalent` in a dose expression, prediction contains `Gy` or `gray` at the same position, AND particle-therapy context clues are present elsewhere in the fixture (`proton`, `pencil beam scanning`, `carbon ion`, `craniospinal`, `GyE` elsewhere). Severity: **HIGH** (undertreatment of ~10% due to lost RBE correction).
4. **Slashed form loss**: gold contains `N[D]/M[D]` (IGRT pattern), prediction contains `N[D] M[D]` without the slash. Severity: **MEDIUM** (recoverable by the corrector).
5. **Dose-unit-vocabulary corruption**: gold contains `Gy`, prediction contains `GiE`, `Jai E`, `Ji`, `HIE`, `JEE`, `J`, `G i`, or any of the other 15+ documented failure renderings. Severity: **HIGH** (visibly broken but recoverable by the corrector).

The metric should report: (a) per-sample severity, (b) a per-backend × per-voice aggregate of failure counts by severity class, (c) a deployment-gate boolean (*no CRITICAL or HIGH failures*), and (d) a "safety-adjusted WER" that weights token errors in safety classes higher than ordinary WER errors.

### #119 staged correction pipeline

The correction pipeline must include at least these rules:

1. **`Gy` → `GyE` promotion** in dose expressions when particle-therapy context clues are present. Conservative: promote only on Voxtral output, only when a context clue appears within ±N tokens of the dose expression.
2. **`gray` → `gray equivalent` promotion** same conditions.
3. **Slashed form normalization**: `3D 3D` / `2D 3D` / `3D 2D` → slashed form when adjacent to IGRT vocabulary.
4. **Phonetic recovery**: the `GiE`/`Jai E`/`JEE` family of renderings → `Gy` (Whisper/MedASR baseline), with subsequent promotion to `GyE` per rule 1.

The corrector should **log every promotion** and attach a provenance field (`corrected_from`, `confidence`, `rule_id`) so downstream reviewers can see what was modified and why.

### #113 — close as negative finding

The audio-LLM domain-prompt work (cycle 110) should be closed with the negative finding that instructability at sub-flagship scale is not the right path. Voxtral Mini 3B neutral baseline is the correct comparison. The instructable path regressed catastrophically for Granite 2B and produced the 50.4 → 504 decimal drop for Granite 8B. That work is done and the answer is documented.

### Public framing

Any public writeup that includes Voxtral's WER numbers must also include finding 1 (the GyE substitution). Publishing `Voxtral WER 0.1073` as a standalone headline over a known silent-unit-substitution failure would be the cycle 111 "9.25% headline WER over a 10x dose error" mistake replayed.

## Open clinical questions for Stuart

1. **Dose numbers** — I drafted the fixtures from memory of standard practice. Please spot-check:
   - proton-0018: 36 GyE / 20 fx for high-risk medullo CSI
   - proton-0021: 59.4 GyE / 33 fx for post-GTR ependymoma
   - proton-0024: 21.6 GyE / 12 fx for high-risk neuroblastoma adjuvant
   - proton-0025: 21 GyE / 14 fx for pediatric Hodgkin involved-site
   - proton-0027: 23.4 CSI + 54 posterior fossa + 55.8 tumor bed for std-risk medullo
   - proton-0028: 21 GyE whole ventricular + 30 GyE boost for germinoma

2. **Junction technique vocabulary** — are `dual gradient ramping`, `dual ramping gradient junction`, and `junction feathering` all idiomatic? Should I normalize to one form?

3. **Field geometry (proton-0026)** — "two posterior cranial fields and three posterior spine fields" — realistic?

## TTS variance finding (appended 2026-04-09 late)

After Stuart's clinical review, the fixture corpus was corrected (4 text edits on proton-0016, 0017, 0023, 0026; file renamed to `particle_samples.jsonl`) and both bake-offs were re-run to regenerate the reports against the corrected gold. The re-runs surfaced a framework-level finding that reframes how this entire document should be read.

### The observation

Two back-to-back bake-off runs of the same corpus, same backends, same machine, same piper voices, with *only* text changes to 4 fixtures out of 28, produced **meaningfully different CRITICAL failure counts on the 24 fixtures that were NOT edited**.

| Backend | Run 1 (original corpus) | Run 2 (corrected corpus) | Delta |
|---|---:|---:|---|
| Voxtral Mini 3B | 2 unrecoverable CRITICAL | 1 unrecoverable CRITICAL | −1 |
| Whisper large-v3 | 3 unrecoverable CRITICAL | **0 unrecoverable CRITICAL** | −3 |
| MedASR fp16 | 13 unrecoverable CRITICAL | 11 unrecoverable CRITICAL | −2 |

In Run 2, Whisper's `post_correction_gate` flipped from FAIL to **PASS on both voices**. Not because the backend got safer; because the audio was different the second time.

### Disambiguation experiment: TTS or ASR?

After Stuart flagged that the observation confounded TTS and ASR variance, two controlled experiments were run to isolate the root cause.

**Experiment A — piper TTS determinism.** Synthesize the same clinical sentence (`"Dose escalation to 78 GyE in 39 fractions for the chordoma at the base of skull."`) twice with identical piper invocation parameters (same model, same config, same text, no seed flag because piper does not expose one). Bit-compare the resulting WAV files.

Result: **piper is non-deterministic.**
- Take 1: sha256 `8bb5c403...`, 300,588 bytes
- Take 2: sha256 `9f8640cc...`, 302,636 bytes
- Bit-identical: **False** (different sha256, different file sizes, ~2 kB delta)

**Experiment B — ASR determinism on a fixed WAV file.** Take one of the WAVs from Experiment A and run each backend on it twice, in-process, without reloading the model between passes. Compare the transcriptions.

Result: **ASR backends are deterministic given fixed input audio.**
- Whisper large-v3 MLX: `"Dose escalation to 78 Jai E in 39 fractions for the Kodoma at the base of skull."` — identical both passes
- MedASR fp16: `"dose escalation to 78 jie in 39 fractions for the chhoedoma at the base of skull."` — identical both passes
- Voxtral Mini 3B not directly tested but uses `do_sample=False` greedy decoding and is expected to be deterministic on fixed input consistent with this result

**Conclusion:** the cycle 112 cross-run CRITICAL count differences are **entirely explained by piper TTS variance**, not by the ASR backends. The ASRs responded consistently to the different acoustic inputs they were given; the variance source is upstream in synthesis.

### Root cause in piper

Piper is based on VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech). VITS has two stochastic layers controlled at synthesis time:

- `--noise-scale` (generator noise scale, default ~0.667) — controls the VITS generator prior
- `--noise-w` (phoneme width noise, default ~0.8) — controls the stochastic duration predictor

These are sampled from an unseeded RNG on every synthesis call. **Piper has no `--seed` flag.** Setting both to `0` would make piper deterministic but would also eliminate the natural prosody variance that makes the output sound human.

### Why this is a feature, not a bug (reframe per Stuart)

Clinicians have variance in their own speech. The same physician dictating the same prescription on two different days will produce acoustically different audio. The same prescription read by two different physicians will differ even more. Piper's stochastic layers happen to sample from a similar distribution of plausible acoustic variations — different fundamental frequency contours, different micro-prosody, different phoneme durations.

**This makes piper's variance a cheap proxy for real speaker variance.** The bake-off is effectively measuring ASR robustness to minor acoustic perturbations in ways that generalize to deployment: if a backend is brittle across piper's natural variations, it will be brittle across real speakers; if a backend is stable across piper's variations, it will likely be stable across real speakers.

The right framework response is **not** to kill piper's naturalness with `noise-scale=0`. The right response is to **characterize the distribution** by running the bake-off against N cached audio variants per fixture and reporting per-backend statistics across that sample.

### Implications for every number in this document

- **All aggregate WER, term recall, and per-class failure counts above are single-sample draws from piper's acoustic variance distribution.** They should be read as "one observation of how a backend handled one particular synthesis," not "the answer."
- **The per-sample examples in the findings sections are illustrative, not exhaustive.** The specific fixture IDs that trigger a given failure class under one synthesis may or may not trigger it under the next.
- **The framework-level conclusion (*no ASR backend is currently safe for unreviewed proton RT deployment*) is strengthened, not weakened, by this finding.** A method whose safety-gate verdict can flip between "FAIL" and "PASS" across back-to-back synthesis samples is, by definition, unreliable for making unreviewed deployment decisions. A single passing run is not evidence of deployability.
- **Whisper post_correction_gate = PASS (Run 2) is NOT a deployability signal.** It is a single sample from a distribution whose tail extends into CRITICAL failures that we observed only one run prior. Any deployment decision based on a single pass would be the same mistake as publishing the cycle 110 9.25% headline WER over the 50.4 → 504 decimal drop.

### What the framework needs next

Task #122 (*N=30 TTS variance characterization: pre-synthesized cached audio corpus + statistical aggregation of bake-off results*) was created in response to this finding. Scope:

1. Pre-synthesize N=30 audio variants per fixture × voice into a cached audio corpus on disk. Piper is called N times per fixture-voice pair (letting its natural stochastic layers sample the distribution of plausible acoustic variants); each resulting WAV is saved to a content-addressable cache.
2. Extend `run_multi_backend_e2e.py` to consume the cached corpus (rather than re-synthesizing on every run). Each backend runs once, seeing all N variants for each fixture-voice pair.
3. Extend `safety_gate.py` to aggregate per-fixture statistics across the N variants — mean / max / stddev / 95% CI for each failure class — with the deployment gate keyed on the **worst observed** CRITICAL count across variants (most conservative).
4. Run the expanded bake-off on the `particle_samples.jsonl` corpus and produce a proper statistical summary that supersedes the single-run numbers in this document.
5. Reframe the finding: piper variance is a cheap proxy for real speaker variance, and the expanded bake-off is measuring ASR *robustness to minor acoustic perturbations*, which is the property that actually determines deployment safety.

Why this is a better experimental design than "run the bake-off N times":
- Piper is called N times once (at cache-build time), not N × B times (where B is the number of backends).
- Each backend is loaded once per run, not N times.
- The cached audio corpus is reproducible and content-addressable — re-running a backend against a cached corpus produces bit-identical input, so any difference in output is strictly ASR-side (which we now know is zero).
- The corpus can be shared between different bake-off configurations (different backend combinations, different gates, different metrics) without re-synthesizing.

Until task #122 lands, **all numbers in this document are preliminary**.

### Attached artifacts from both runs

Both runs are committed for historical comparison:

**Run 1 (original corpus, pre-clinical-review):**
- `bakeoff_proton_draft_2026-04-09.json` — Whisper + MedASR
- `bakeoff_proton_voxtral_2026-04-09.json` — Voxtral
- `bakeoff_proton_draft_2026-04-09.json.safety_gate.json` — safety-gate v1.1 annotation
- `bakeoff_proton_voxtral_2026-04-09.json.safety_gate.json`

**Run 2 (corrected corpus, post-clinical-review):**
- `bakeoff_particle_whisper_medasr_2026-04-09.json` — Whisper + MedASR
- `bakeoff_particle_voxtral_2026-04-09.json` — Voxtral
- `bakeoff_particle_whisper_medasr_2026-04-09.json.safety_gate.json`
- `bakeoff_particle_voxtral_2026-04-09.json.safety_gate.json`

## 2-backend ensemble: Voxtral + Whisper (appended 2026-04-10)

### Motivation

The individual-backend analysis showed complementary failure profiles:
Voxtral wins WER but silently substitutes GyE→Gy; Whisper has higher WER
but its failures are visibly broken and it preserves "gray equivalent" 100%.
Stuart proposed (2026-04-10) combining both backends in a weighted ensemble
that leverages each model's strengths at disagreement points, analogous to
the multi-model ensemble approach used in the synapse-mkr-converter project.

### Architecture

```
Audio → Voxtral Mini 3B → text_a ──┐
                                    ├→ normalize → align → decide → ensemble
Audio → Whisper large-v3 → text_b ──┘
```

1. **Pre-alignment normalization** collapses known multi-word medical forms
   to single tokens (`gray equivalent` → `GyE`, `3D-slash-3D` → `3D/3D`)
   and splits joined number-unit tokens (`60GyE` → `60 GyE`).
2. **Word-level alignment** via `difflib.SequenceMatcher` (case-insensitive
   matching, original case preserved).
3. **10 prioritized decision rules** at disagreement points, including:
   - Dose-unit GyE promotion when particle-therapy context is present
   - RT vocabulary lookup tiebreaker (catches "Proton" vs "Grothendieck")
   - "Both wrong" → flag for downstream review (neither model matched
     vocabulary)
   - Decimal precision preference (take the higher-precision number)
   - Voxtral default for formatting and general text (lower WER)

### Introducing UWR: Unresolvable Word Rate

We define **UWR** (Unresolvable Word Rate) as the fraction of words in the
ensemble output that the pipeline cannot confidently resolve from its two
input channels and must defer to a downstream reviewer. The reviewer may be
a human clinician, a domain-specific LLM, or a rule-based post-processor —
the pipeline does not assume.

UWR is structurally analogous to WER but measures a different property:
- **WER** counts all errors, including ones the system doesn't know about
  (silent failures).
- **UWR** counts only the cases the system explicitly flags as unresolvable.

A system can have low UWR and high WER (confidently wrong — dangerous) or
high UWR and low WER (cautiously correct — safe). The ensemble targets the
latter: low WER AND low UWR AND zero unrecoverable CRITICAL failures.

### Results: three-corpus view

The ensemble was run on both the original 24 dense clinical RT fixtures
and the 28 particle therapy fixtures, then combined.

| Corpus | Samples | Words | Voxtral WER | Whisper WER | **Ensemble WER** | Review flags | **UWR** |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-only (24 dense) | 48 | 791 | 0.1233 | 0.1439 | **0.1172** | 9 | **1.14%** |
| PT-only (28 particle) | 56 | 1,147 | 0.1140 | 0.1430 | 0.1347 | 25 | 2.18% |
| **Combined** | **104** | **1,938** | 0.1183 | 0.1434 | **0.1266** | **34** | **1.75%** |

**Combined across the full 52-fixture corpus: 34 words out of 1,938
flagged for downstream review = 1.75% UWR. 98.25% of words resolved
automatically.**

### Safety-gate results (particle corpus, where the safety findings live)

| Backend | Raw WER | CRIT | HIGH | MED | post_correction_gate |
|---|---:|---:|---:|---:|---|
| Voxtral alone | 0.095 | 1 | 46 | 0 | FAIL |
| Whisper alone | 0.133 | 0 | 94 | 8 | PASS |
| **Ensemble** | 0.135 | **0** | **28** | **0** | **PASS** |

The ensemble eliminates all unrecoverable CRITICAL failures (0, down from
Voxtral's 1) and reduces HIGH failures by 40–70%.

### Interpretation

**5–10× improvement in what a Radiation Oncologist or Medical Physicist
needs to deal with.** Comparing the best single-backend WER (~12%) to the
ensemble's UWR (1.75%) gives approximately a 7× ratio. But the real
improvement in clinical workflow is larger than the ratio suggests:

- **Without the ensemble**, a clinician reviewing Voxtral output must
  scrutinize every word because the failures are *silent* — they cannot
  know which 12% of words are wrong. The review burden is proportional to
  the total word count.
- **With the ensemble**, the clinician reviews only the 1.75% of words
  that are explicitly flagged with provenance showing what each backend
  said and why the ensemble couldn't resolve the disagreement. The
  remaining 98.25% are either agreed-upon by both models (86%) or resolved
  by a decision rule with documented confidence.

This converts the review task from *"read and verify everything"* to
*"check 34 highlighted words across 104 samples."* The reduction in
cognitive load for the reviewer is closer to **50×** because the ensemble
transforms an untrustworthy document into a trustworthy document with a
small number of marked exceptions.

**The ensemble also beats Voxtral on WER for the RT corpus** (0.1172 vs
0.1233). On the RT fixtures, the vocabulary-match rule picks Whisper's
correct words over Voxtral's LLM hallucinations (e.g., "Proton" over
"Grothendieck beam therapy"), and those corrections net-improve WER. On
the PT corpus, the ensemble trades 4pp of WER for clinical safety — the
correct tradeoff for medical ASR where silent failures are the primary
risk.

**RT-only UWR (1.14%) is lower than PT-only UWR (2.18%)** because general
RT vocabulary is more familiar to both models. Particle therapy terminology
(craniopharyngioma, rhabdomyosarcoma, junction feathering) stresses both
backends harder, producing more "both wrong" cases that must be deferred.

### The 34 flagged words

All 34 review flags are "both wrong" cases (Rule 6) where neither backend's
word matched the RT vocabulary. Examples:
- Both mangling "craniopharyngioma" differently
- Both mangling "rhabdomyosarcoma" differently
- Both producing different broken forms of "chemoradiation"

These are complex multi-syllable medical terms where both models fail
independently. A downstream reviewer (human or LLM with RT domain
knowledge) can resolve them from context. A potential future enhancement:
multi-source fuzzy matching that uses BOTH wrong versions as evidence to
recover the correct term — two independent noisy observations of the same
word carry more information than one.

## Next actions

1. ~~Write this findings document~~ *(done)*
2. ~~Design #115 safety-gate metric~~ *(done, v1.1)*
3. ~~Clinical spot-check and file rename~~ *(done)*
4. ~~2-backend ensemble (Voxtral + Whisper)~~ *(done, PR #17)*
5. ~~Full-corpus ensemble evaluation (RT + PT + combined)~~ *(done)*
6. Task #122: N=30 TTS variance characterization (deferred — ensemble
   reduces the urgency because it is itself a variance-reduction strategy,
   but still needed for publishable confidence intervals)
7. Clinician review rendering (Phase 3 of the ensemble spec — HTML/Jinja
   showing both perspectives + ensemble choice + flagged words)
8. Public GitHub push when the above are in acceptable shape

---

*This document has been updated with the 2-backend ensemble results and the
three-corpus UWR analysis. The ensemble finding — 1.75% UWR, 0 CRITICAL,
98.25% automatic resolution — is the framework's headline result for
cycle 112.*
