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

**Status:** fixture corpus is still `_draft.jsonl` pending Stuart's clinical spot-check on dose values and technique terminology. Findings below are based on ASR measurement validity (gold is gold) and do not depend on whether every dose is exactly current-standard-of-care.

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

## Next actions

1. ~~Write this findings document~~ *(done)*
2. Design #115 safety-gate metric with the five failure classes as the spec
3. Delegate #115 mechanical implementation to a Sonnet agent
4. Wait for Stuart's clinical spot-check before promoting `proton_samples_draft.jsonl` to `proton_samples.jsonl`
5. Do NOT commit anything to git until clinical review is complete

---

*This document will be updated after Stuart's clinical review and after the #115 metric is applied against this run's JSON outputs to produce a quantified deployment-gate assessment for each backend.*
