# transcriber-radrx — Roadmap

**Living document.** Last updated: cycle 111 (April 2026). This roadmap
reflects the current open work identified by the cycle 110 and cycle 111
bake-offs. As items complete or as new gaps surface, this file is
updated. The git history is the changelog.

The items below are scoped at a level where an external reader should be
able to understand *what is being asked, why, and what would make an
answer defensible*. Internal implementation details live in the task
descriptions in the project's local task tracker; this document captures
the public shape of the open work.

---

## Fixture coverage gaps

The cycle 111 dense-clinical fixture set contains 24 hand-curated items
covering photon-therapy dose prescriptions, OAR constraints, IMRT/VMAT
treatment summaries, and a few acronym-dense stressors. That is enough
to produce meaningful differentiation between ASR backends, but it does
not cover several corners of real clinical content that matter for
deployment.

### Anatomy coverage

**Open question:** are any of the bake-off backends systematically
worse at recognising anatomy words, especially less common ones? The
current fixtures exercise parotids and cord (head/neck), rectum (GI),
and a few others — but there is no deliberate coverage of breast,
prostate, cervix, vulva, rectum, anus, testes, penis, endometrium,
ovaries, or the major bone anatomy used in RT targeting.

**What a defensible answer looks like:** a hand-curated fixture batch
of 20–30 items explicitly targeting anatomy recognition, with each
fixture tagged for the specific anatomy word(s) being stressed. A
bake-off run against these fixtures produces per-anatomy recall for
every backend, and any systematic blind spot is flagged for the
anatomy term catalog and the corrector.

### Safety-gate metric for dose-value preservation

**Open question:** word error rate does not catch the clinically
dangerous failures. The cycle 110 Granite-Speech 8B finding (a
headline 9.25 % WER on one voice, hiding a `50.4 Gy → 504 gy` silent
decimal drop that would propagate a ten-times-lethal dose) is the
motivating example. What is the right metric that WOULD catch this?

**What a defensible answer looks like:** a safety-gate metric that
scores a transcription as PASS or FAIL (not a continuous WER) based on
explicit preservation of safety-critical tokens: dose values, RT unit
words (`Gy`, `GyE`, `mGy`, `cGy`), drug names, anatomical targets,
numeric fractionation counts. The metric requires the transcription to
either preserve every safety-critical token exactly or explicitly flag
it as unresolved. An adversarial fixture set stresses the metric with
homophone traps (`50.4` vs `504`, `Gy` vs `guy`, `mg` vs `mcg` vs
`μg`), decimal-point stressors, and dose-value-at-boundary cases.
Any backend that silently fails a single safety case in this set is
disqualified from the corresponding deployment context.

### Proton and particle therapy fixtures

**Open question:** the entire dense-clinical fixture set is photon
therapy. Proton and other particle therapies use a distinct dose unit
(`GyE` — Gray Equivalent, the physical dose multiplied by relative
biological effectiveness) and a distinct vocabulary (pencil beam
scanning, spot weight, range uncertainty, robust optimization,
spread-out Bragg peak). Conflating `Gy` with `GyE` underdoses or
overclaims biological effectiveness by the RBE factor (~1.1 for
protons) — a clinically significant error specific to particle
contexts.

**What a defensible answer looks like:** a hand-curated particle-therapy
fixture batch (10–15 items) drafted by a radiation oncologist or medical
physicist with particle-therapy experience, plus explicit `Gy ↔ GyE`
safety cases as part of the safety-gate metric above. Every candidate
deployment in a particle-therapy context must pass the `Gy / GyE`
preservation test.

---

## Voice panel gaps

The cycle 111 16-voice panel covers 8 native UK English voices and 8
native US English voices. The ranking flip observed when going from
2 voices to 16 voices demonstrates that under-sampling the voice axis
produces misleading rankings. Expanding further is expected to produce
further refinements — or disqualifications.

### Commonwealth English

**Open question:** how do the bake-off backends perform on Australian,
New Zealand, Canadian, Irish, South African, and Indian English? These
are all common clinician accent distributions in specific regional
deployments, and none are covered by the current panel.

**What a defensible answer looks like:** at least 6–8 Commonwealth voices
(ideally covering AU, NZ, CA, IE, ZA, and en-IN-prestige) integrated
into the bake-off voice registry. A per-region bake-off with the cycle
111 mean-WER winner (Voxtral) and voice-robustness winner (Whisper)
produces per-voice WER and a report called out against the cycle 111
native-panel findings.

Piper TTS upstream does not distribute voice models for these
languages, so sourcing is a real gap — the candidates are commercial
cloud TTS (Azure, Google, AWS Polly all have the full set natively),
open-source voice cloning (Coqui XTTS-v2 or OpenVoice with reference
samples per accent), or real-human speaker recordings.

### ESL clinician voices

**Open question:** in US radiation oncology specifically, a significant
fraction of practicing radiation oncologists and medical physicists are
immigrants whose English is ESL. The accent distribution of the
*clinician* population differs dramatically from the accent distribution
of the general population. None of the cycle 111 voices are ESL. Any
deployment recommendation from cycle 111 is based on zero data about
how these backends behave on Indian-English, Chinese-English,
Spanish-English, Arabic-English, Korean-English, or Vietnamese-English
speech — the six L1 backgrounds most common in the practicing US RT
workforce.

**What a defensible answer looks like:** the L2-Arctic corpus (ISCA
Interspeech 2018, 24 non-native English speakers across the 6 L1
backgrounds above, 4 speakers per L1) is already packaged as a piper
voice model. Extending the bake-off runner to handle multi-speaker
piper models and to label each speaker with its L1 is mostly a wiring
task. A per-L1 bake-off produces the missing data with no external
sourcing. The Hindi-English subset (4 speakers) is the single most
clinically important cohort for US RadOnc deployment.

Additional single-speaker ESL piper voices are already downloaded
locally (`reza_ibrahim` — Persian-English) and can extend the L1
coverage beyond the L2-Arctic corpus.

**License note:** L2-Arctic is CC BY-NC 4.0 — validation and academic
use is fine; this precludes redistribution of the generated audio as
part of a commercial product.

---

## Correction pipeline (post-ASR)

Even the cycle 111 mean-WER winner (Voxtral) mis-transcribes the
fundamental RT unit word `Gy` in 25–30 % of its clean-audio samples.
No current ASR backend is safe on raw output for clinical dictation,
which means a post-ASR correction layer is a hard requirement for
deployment, not an optional enhancement. Three tasks cover the
correction architecture, the delivery formats for clinician review, and
the rule catalog.

### Staged correction pipeline + HTML rendering

**Open question:** how do you correct ASR output for clinical use in a
way that is auditable, bounded by safety properties the rule author
cannot violate, and reviewable by a clinician in their browser?

**What a defensible answer looks like:** a staged pipeline
(`raw → whitespace → punctuation → phonetic → grammar → terminology →
final`) where each stage emits both a new text and a diff-from-previous
with per-change rule attribution. The rule engine enforces a
safe-by-construction property: any rule that would modify, remove,
or reorder a protected clinical token (numeric values, dose
expressions, RT acronyms, drug names, anatomy terms from catalogs) is
rejected at application time by the engine, not by the rule author
remembering to check. An HTML renderer (Jinja-templated,
self-contained, inline CSS) produces word-level inline diffs with
hover-or-click popovers for rule provenance, suitable for clinician
review in Safari or Chrome.

**Concept art:** `docs/design/staged_correction_demo.html` is a
hand-written HTML page showing three examples of the target output
format, using real transcription text from cycle 110 and cycle 111.
It demonstrates the three pipeline behaviours the design needs to
support: safe rejection of unrecoverable corruption, safe refusal to
guess at clinically ambiguous cases, and clean multi-rule correction
with full provenance.

### MS Word `.docx` rendering with Track Changes

**Open question:** in air-gapped clinical IT environments or
email-based workflows, how is a correction review delivered to a
clinician in a format they already know how to review?

**What a defensible answer looks like:** a `.docx` renderer that
consumes the same per-sample data model as the HTML rendering above
and produces a Word document with native Track Changes markup
(`<w:ins>`, `<w:del>`), per-change comments containing rule
attribution, and protected-token highlighting. The clinician opens the
file in Word, reviews the changes through Word's built-in Accept /
Reject / Comment workflow, and saves the result. No browser, no
server, no custom software beyond Word itself.

### Web-based review server for networked deployments

**Open question:** in highly networked hospital IT environments or
cloud tenancies, how is a correction review delivered with persistent
state, multi-user coordination, audit trail, and integration with
institutional identity (SSO)?

**What a defensible answer looks like:** a small FastAPI web application
that serves the same Jinja templates as the static HTML renderer,
backed by a persistent database that tracks per-transcription review
state (pending / in-review / approved / rejected), per-change accept /
reject decisions with clinician attribution, and a full audit log of
who reviewed what when. Authentication and authorization are
pluggable — local username/password for development, OIDC/SSO for
enterprise. The three delivery formats (HTML, `.docx`, web server) all
consume the same per-sample data model, so the core pipeline is
implemented once.

### RT-specific grammar and terminology rule catalog

**Open question:** which corrections are actually clinically safe to
apply, and with what confidence?

**What a defensible answer looks like:** a rule catalog authored by or
reviewed by clinical experts, with each rule having explicit provenance
(author, review date, rationale), test fixtures (passing and failing
cases), a confidence score, and a guard specification narrowing when
the rule fires. An LLM may propose candidate rules but every rule must
be explicitly approved before entering the catalog. The first rules
in the catalog target the Voxtral `Gy` mis-transcription failure mode
(jai → Gy unambiguous, gye → Gy blocked by Gy/GyE ambiguity), the
common RT acronym phonetic failures (IMRT, IGRT, VMAT, CTV, PTV, GTV,
OAR, SBRT), and the anatomy term failures identified in cycle 110 and
111.

---

## Framework and process documentation

### Living validation framework document

**Open question:** what is the *process* we are proposing — the way of
thinking about clinical ASR validation that another group could pick
up and apply to their own deployment context?

**What a defensible answer looks like:** a `tests/validation/FRAMEWORK.md`
file that synthesises the cycle reports into an explicit methodology
document: the axes of variation, the decision rules for when to expand
scope and when to trim, the safety metrics, the delivery formats, the
explicit limitations. Maintained as a living document that tracks the
learning from each cycle, with a changelog at the top. Intended audience:
a clinician physicist, an engineering reviewer, or an academic reader
arriving at the project cold.

### Early-trim flag for the bake-off runner

**Open question:** when a backend fails a stage of the bake-off so
badly that continuing is a waste of compute (MedASR at 60 % WER on a
UK voice, for example), how is that trimming made explicit and
auditable rather than a judgment call after the fact?

**What a defensible answer looks like:** a CLI flag on the bake-off
runner that disqualifies a backend from subsequent conditions if its
worst-voice WER at the current condition exceeds a configurable
threshold. The disqualified backend appears in the final report as
"dropped at stage N: reason", creating a clear narrative arc of the
evaluation process and saving the compute that would otherwise run a
failing model through the particle therapy / safety / noise sweeps
unnecessarily.

---

## Running open questions we don't have tasks for yet

- Does the TTS voice quality tier matter more than the accent? Cycle
  111 found that four of the six backends hit their worst case on a
  single *low*-quality piper voice. A bake-off run at medium- and
  high-quality tiers only would significantly over-report robustness.
- Does the phonetic corrector (stage 2) help or hurt on ESL voices?
  The cycle 111 corrector is trained on native-US-English phonetics.
  Applying it to L2-Arctic Hindi-English speech could introduce
  errors as often as it removes them.
- Does adding noise on top of voice variance show interaction effects?
  The cycle 111 noise sweep used 2 voices; the voice panel used clean
  audio. The 16-voice × noise matrix is the missing quadrant.
- Is a safety-gate metric alone sufficient, or do we also need a
  dose-plausibility bound? "504 Gy" is safety-gate-detectable because
  it doesn't match "50.4 Gy" from ground truth, but even without
  ground truth it is dose-plausibility-detectable because 504 Gy is
  outside any clinical range. Two independent checks that catch
  different failure modes.

---

*Drafted by Silas (silas-397300f6) in cycle 111 for review by Stuart.
This document is living. If you are reading it and have questions,
disagreements, or an item you think belongs on the roadmap, we would
like to hear from you.*
