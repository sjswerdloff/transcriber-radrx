# transcriber-radrx

[![CI](https://github.com/sjswerdloff/transcriber-radrx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sjswerdloff/transcriber-radrx/actions/workflows/ci.yml)

**A validation framework and complementary-ensemble approach for clinical
automatic speech recognition (ASR) in radiation oncology dictation.**

This project provides two things:

1. A **validation framework** for rigorously evaluating whether any ASR
   backend is safe for clinical dictation — with metrics that go beyond
   word error rate to catch the safety-critical failures that aggregate
   statistics hide.
2. A **2-backend ensemble** (Voxtral Mini 3B + Whisper large-v3) that
   exploits complementary failure profiles between an audio-LLM and a
   classical ASR to achieve **98.5% automatic resolution** with **zero
   unrecoverable safety failures** across 72 radiation oncology fixtures,
   leaving only **1.52% of words** for human review.

We don't have all the answers. We have a framework for asking the questions,
a growing collection of reproducible experiments, and a concrete result
that turns two individually unsafe ASR backends into an ensemble that is
demonstrably safer than either alone.

---

## What this project is

- A **modular pipeline** for generating synthetic clinical speech, injecting
  realistic acoustic and noise conditions, running it through multiple ASR
  backends, and scoring the results with metrics that go beyond word error
  rate (WER) to include clinical-vocabulary preservation and safety-critical
  token audit.
- A **multi-backend bake-off harness** that currently supports six ASR
  backends (Whisper, MedASR, Cohere Transcribe, Granite-Speech 2B and 8B,
  Voxtral Mini 3B) and is trivially extensible to any new backend that
  implements a small Protocol interface.
- A **safety-gate metric** with five clinically-derived failure classes
  (decimal drop, dose value missing, silent unit substitution, dose unit
  corruption, slashed form loss), each tagged with a correctability rating
  (UNRECOVERABLE, CONTEXT_RULE, PHONETIC_MAP, ADJACENCY_RULE) and severity
  weight calibrated by a radiation oncology physicist. Produces a formal
  deployment gate: PASS / CONDITIONAL / FAIL — both for raw ASR output
  and for predicted post-correction output.
- A **2-backend ensemble** that aligns Voxtral and Whisper word-by-word,
  applies 10 prioritized token-class decision rules at disagreement points,
  and produces a single output with per-word provenance. Introduces **UWR**
  (Unresolvable Word Rate) — the fraction of words the ensemble cannot
  confidently resolve and must defer to a downstream reviewer.
- A **72-fixture corpus** across three domains: 24 dense clinical RT
  fixtures, 28 particle therapy fixtures (proton + carbon ion, including
  pediatric CSI with junctioned PBS), and 20 anatomy-coverage fixtures
  (breast, prostate, cervix, rectum, anus, vulva, vagina, endometrium,
  testes, penis, head & neck, lung).
- A **Word .docx renderer** with Track Changes that produces two documents
  from the same ensemble output: an *audit* document (every ensemble
  decision visible as a tracked change for regulatory traceability) and a
  *review* document (automated fixes baked in, only UWR items shown as
  highlighted words with margin comments listing both ASR options).
- A **reproducible noise-injection stage** using the MUSAN corpus with a
  prefer-long-first splice strategy so that every transcription is covered
  by continuous ambient noise at a known signal-to-noise ratio.
- A **growing set of cycle reports** that document what we tested, what we
  found, what flipped when we expanded the test scope, and what we decided
  not to claim yet. The reports are the receipts.

## What this project is not

- It is **not a clinical product.** Nothing in this repository is certified,
  validated, or approved for clinical use. Any deployment would require
  independent validation against the specific clinical environment, voice
  distribution, and vocabulary in use.
- It is **not a leaderboard.** We are less interested in "which ASR wins"
  than in "how do you know you can trust the winner." The ranking is a
  by-product of the process working.
- It is **not a finished framework.** The cycle reports document work in
  progress. Every cycle surfaces things the previous cycles got wrong or
  under-sampled. The `ROADMAP.md` file is a living document of the open
  work.

## Why this matters

Clinical dictation is one of the few contexts where an ASR error can
directly harm a patient. A silent decimal-point drop (`50.4 Gy` transcribed
as `504 gy`) propagates a ten-times-lethal dose. A misread drug name
substitutes one treatment for another. A misheard anatomy word changes
the target of a radiation field. These are not hypothetical — cycle 110
of this project found a real instance of the decimal-drop failure in one
of the models tested, hidden underneath a headline word-error-rate of
9.25 % on the single voice where it happened.

Word error rate alone will not catch these failures. The aggregate metric
optimises over the median token; the dangerous ones are in the tails.
A validation framework for clinical ASR has to look at the individual
safety-critical tokens, not just the average.

The framing is: *we don't have to have the answer, we just need a way of
thinking about the problem that produces receipts clinicians can check.*

## Findings so far

Current as of **cycle 113** (April 2026). See `tests/validation/reports/`
for the full writeups, especially `cycle113_voice_panel_findings.md` and
`bakeoff_proton_findings_2026-04-09.md`.

### Individual backends: every one fails on safety

1. **No ASR backend is safe on raw output for clinical dictation.** Every
   backend produces at least one class of clinically significant failure.
   The differences are in failure *mode*, not in whether they fail.
2. **Voxtral Mini 3B (Mistral)** wins raw WER across all three corpora
   (0.095 on particle therapy, 0.097 on anatomy, 0.123 on dense RT).
   But its LLM decoder **silently substitutes `GyE` → `Gy`** on
   proton/particle therapy prescriptions — losing the ~10% RBE correction
   that distinguishes physical dose from biologically equivalent dose.
   This is clinically dangerous because `Gy` on a proton prescription
   *looks correct* and would pass cursory review.
3. **Whisper large-v3 (OpenAI)** has higher WER but its failures are
   **visibly broken** (`GiE`, `Jai E`, `JEE` — obviously wrong
   renderings that a reviewer would catch). It preserves the spelled-out
   form `gray equivalent` 100% of the time. Its failures are recoverable
   via phonetic mapping.
4. **MedASR** has the worst safety profile: 13 unrecoverable CRITICAL
   failures on the particle corpus (7 decimal drops, 5 missing dose
   values, 1 substitution). The decimal drops are information loss at
   the signal level — no corrector can recover them. MedASR is
   fundamentally unsuitable for clinical RT deployment.
5. **Voxtral hallucinated "Grothendieck beam therapy"** (a mathematician's
   name) where the gold text says "Proton beam therapy." This is the
   audio-LLM hallucination risk made concrete — the LLM decoder reached
   for a training-data token that acoustically matched the input on the
   UK voice. Whisper correctly transcribed "Proton" at the same position.

### The ensemble: complementary failure profiles are the key

6. **A 2-backend ensemble (Voxtral + Whisper)** exploits the complementary
   failure profiles: Voxtral's silent substitutions are caught by
   Whisper's visible corruptions at the same position, and Whisper's
   vocabulary misses are filled by Voxtral's lower WER.

   | Corpus | Voxtral WER | Whisper WER | **Ensemble WER** | **UWR** |
   |---|---:|---:|---:|---:|
   | RT-only (24 dense) | 0.123 | 0.144 | **0.117** | 1.14% |
   | Particle therapy (28) | 0.114 | 0.143 | 0.135 | 2.18% |
   | Anatomy (20 sites) | 0.125 | 0.128 | **0.091** | 1.09% |
   | **Combined (72 fixtures)** | 0.120 | 0.139 | **0.117** | **1.52%** |

   The ensemble beats Voxtral on WER for the RT and anatomy corpora (the
   vocabulary-match rule picks Whisper's correct words over Voxtral's
   hallucinations). On particle therapy the ensemble trades 2pp WER for
   clinical safety. **Combined: 98.5% of words resolved automatically,
   1.52% deferred to a downstream reviewer.**

7. **UWR (Unresolvable Word Rate)** is the fraction of words the ensemble
   cannot confidently resolve from its two input channels. Unlike WER
   (which counts all errors including silent ones the system doesn't know
   about), UWR counts only the cases the system *explicitly flags*.
   A system with 12% WER and unknown silent failures requires review of
   every word. A system with 1.52% UWR requires review of 45 highlighted
   words across 2,951 total — a **5–10× reduction** in what a Radiation
   Oncologist or Medical Physicist needs to deal with, and closer to
   **50× reduction in cognitive load** because the ensemble converts an
   untrustworthy document into a trustworthy document with a small number
   of marked exceptions.

### TTS variance finding

8. **Piper TTS is non-deterministic** — two successive synthesis calls with
   the same input text produce acoustically different WAV files (different
   sha256 hashes, different file sizes). The ASR backends are deterministic
   on fixed input audio (verified by running Whisper and MedASR twice on
   the same WAV file — identical output both passes). Cross-run variance
   in bake-off results is entirely upstream in TTS, not in the ASR. This
   variance is a **cheap proxy for real speaker variance** (Stuart
   Swerdloff's framing) — different clinicians saying the same prescription
   produce acoustically different audio in the same way piper does.

### Safety-gate metric

9. **The safety-gate metric** classifies every per-sample failure into one
   of five classes with a correctability tag:

   | Failure class | Severity | Correctability |
   |---|---|---|
   | Decimal drop (50.4 → 50) | CRITICAL | UNRECOVERABLE |
   | Dose value missing | CRITICAL | UNRECOVERABLE |
   | Silent unit substitution (GyE → Gy) | HIGH | Context rule |
   | Dose unit corruption (GiE, Jai E) | HIGH | Phonetic map |
   | Slashed form loss (3D/3D → 3D 3D) | MEDIUM | Adjacency rule |

   The metric produces two gate decisions: a **raw gate** (all failures)
   and a **post-correction gate** (only UNRECOVERABLE failures remain).
   The ensemble achieves post_correction_gate = PASS with zero
   unrecoverable CRITICAL failures across the combined corpus.

### How the ensemble decision engine works

The ensemble combines Voxtral Mini 3B and Whisper large-v3 through a
four-stage pipeline:

1. **Phrase corrections** (`phrase_corrector.py`) — 13 regex-based
   multi-word fixes run on each backend's raw output before alignment.
   Targets systematic ASR failures: dose unit garbling after numbers
   (`50 ji` → `50 Gy`), compound-word splits (`chemo radiation` →
   `chemoradiation`), multi-word substitutions (`bracket therapy` →
   `brachytherapy`).
2. **Word-level alignment** (`aligner.py`) — the corrected outputs are
   normalised and aligned using `difflib.SequenceMatcher`. Each position
   becomes an `AlignedSpan`: MATCH, SUBSTITUTION, INSERTION_A (Voxtral
   only), or INSERTION_B (Whisper only).
3. **Decision rules** (`decision_rules.py`) — the 10 rules below are
   evaluated in priority order on each non-MATCH span. First match wins.
4. **UWR flagging and output** — spans where `needs_review=True`
   propagate to the clinician review document as highlighted words with
   Word margin comments.

**The 10 rules (evaluated in order; first match wins):**

| # | Name | Condition | Winner | Review? |
|---|------|-----------|--------|---------|
| 1 | MATCH | Both backends agree | Either | No |
| 2 | DOSE_UNIT_GYE | Both Gy-variants; at least one is `GyE` | `GyE` | No |
| 3 | DOSE_UNIT_CONTEXT | Both Gy-variants, neither `GyE`; particle context present | `GyE` (inferred) | Yes |
| 4 | DOSE_UNIT_VISIBLE | Exactly one is a Gy-variant | Gy-variant; promoted to `GyE` if particle context | No |
| 5 | VOCABULARY_MATCH | One word in `rt_vocabulary.txt`, other not | Vocabulary word | No |
| 6 | BOTH_WRONG | Neither in vocabulary, low mutual similarity | Voxtral (flagged) | Yes |
| 7 | DECIMAL_PRECISION | Both numeric, different decimal places | Higher precision | No |
| 8 | FORMATTING_DEFAULT | All other substitutions | Voxtral | No |
| 9 | INSERTION_A | Word only in Voxtral | Voxtral | No |
| 10 | INSERTION_B | Word only in Whisper | Whisper | No |

**Why order matters:** Rules 2–4 are dose-unit specialists and must fire
before Rule 5 (vocabulary). If vocabulary ran first, `Gy` might win over
`GiE` on a vocabulary hit — missing the unit correction entirely. Rules
3 and 6 are the only rules that set `needs_review=True` — these are the
words that become UWR.

**Example:** Voxtral produces `50.4 Gy`, Whisper produces `50.4 GiE` on
a proton prescription. Neither is `GyE`, so Rule 2 skips. Rule 3 checks:
both are Gy-variants, neither is `GyE`, and `proton` appears in the
transcript → `has_particle_context()` returns true. Rule 3 fires: output
`50.4 GyE`, flagged for physicist review.

### Voice panel and noise findings (cycle 113)

10. **Commonwealth English (8 piper en_GB voices)** achieves 0.72% UWR
    on clean RT dense fixtures with the full correction pipeline. Better
    than the cycle 112 headline (1.52%) — more voice diversity improves
    ensemble confidence.
11. **ESL voices (26 non-native speakers across 6 L1 backgrounds)** show
    3.4–4.7% UWR depending on corpus. Analysis of missed terms reveals
    two distinct failure classes: **domain vocabulary failures** (IGRT,
    SRS fail 100% for all speakers, accent-independent) and **accent
    penalty** (multi-syllable medical terms fail more for ESL). Most
    correctable patterns are domain failures, not accent-specific.
12. **TTS quality is a significant confound.** macOS system voices (higher
    quality TTS including Indian English) achieve 0.42% UWR on RT dense —
    vs 3.4% for L2-Arctic piper (lower quality TTS, same accent family).
    The ESL UWR gap includes both accent and TTS fidelity effects, which
    this study design cannot fully separate. L2-Arctic results are a
    conservative upper bound.
13. **Noise degradation is graceful.** macOS voices degrade from 0.42% UWR
    (clean) to 1.04% (5 dB SNR, busy clinical environment). At matched
    TTS quality, Indian English does not degrade faster than native
    English voices under noise.
14. **Phrase corrections reduce particle therapy UWR by 38%** (3.11% →
    1.93% on piper Commonwealth, 2.82% → 0.09% on macOS). The Gy
    dose-unit pattern alone accounts for most of the improvement.

### Deployment guidance (with caveats)

- **Recommended pipeline:** Voxtral Mini 3B + Whisper large-v3 ensemble
  with phrase corrections, 10-rule decision engine, and Word .docx
  review output. 0.72% UWR on Commonwealth voices (RT dense corpus).
  Zero unrecoverable CRITICAL failures.
- **Do not deploy any single backend alone** for proton/particle therapy
  dictation. Voxtral's silent GyE→Gy substitution and Whisper's visible
  corruptions are both individually unacceptable.
- **MedASR is fundamentally unsuitable** — 23% of its dose-value
  transcriptions have some form of numeric corruption, most of which
  are unrecoverable at the signal level.
- **Validate with your actual users' voices.** The framework accepts real
  recordings through the same pipeline as TTS — same metrics, same
  reports, no code changes. A clinic can record their clinicians reading
  the fixture corpus once and know their site-specific performance before
  deployment.
- **Clear enunciation remains the highest-impact intervention.** The
  framework quantifies which specific words are problematic for a given
  speaker — not "your accent is wrong" but "these 12 words need clearer
  enunciation for the system to catch them reliably."

## How to read this repository

Start with the **cycle 113 findings** (voice panels, accent, noise):

- `tests/validation/reports/cycle113_voice_panel_findings.md` — voice
  panel UWR comparison (Commonwealth, ESL, macOS), noise degradation
  curves, accent vs TTS quality analysis, phrase correction impact.

Then the **cycle 112 findings** (ensemble, safety):

- `tests/validation/reports/bakeoff_proton_findings_2026-04-09.md` — the
  ensemble writeup: Voxtral GyE→Gy silent substitution, safety-gate
  results, TTS variance finding, UWR definition and three-corpus analysis.

Then see the **earlier cycle reports** for the noise and voice-panel work:

- `bakeoff_dense_6backend_noise_moderate_2026-04-08.md` — noise bake-off
  with ranking stability analysis.
- `bakeoff_dense_6backend_16voice_clean_2026-04-08.md` — 16-voice panel
  expansion and voice-robustness analysis.
- `task_113_closure_audio_llm_domain_prompt_negative_finding.md` — why
  audio-LLM domain prompts are not the right approach at sub-flagship scale.

Generate a **review document** to see the ensemble in action:

```bash
uv run python tests/validation/scripts/render_ensemble_docx_demo.py
# produces docs/demo/ensemble_review_*_review.docx — open in Word
```

Then read the **roadmap**: `ROADMAP.md`

Browse the **code**:

```
src/transcriber_radrx/
    transcriber.py       # Whisper MLX engine + vocabulary biasing
    corrector.py         # Double Metaphone phonetic correction + correct_full()
    phrase_corrector.py  # Regex-based multi-word ASR error patterns (13 rules)
    cli.py               # Command-line interface
    asr_backends/        # Pluggable ASR backend Protocol
        base.py          #   Protocol interface all backends implement
        mlx_whisper.py   #   Whisper large-v3 on MLX
        medasr.py        #   Google MedASR on MLX
        cohere.py        #   Cohere Transcribe 2B (HuggingFace)
        granite.py       #   IBM Granite-Speech 2B and 8B
        voxtral.py       #   Mistral Voxtral Mini 3B
        registry.py      #   Lazy-import factory
    ensemble/            # 2-backend ensemble (Voxtral + Whisper)
        aligner.py       #   Word-level alignment with pre-normalization
        decision_rules.py #  10 token-class decision rules + UWR
        docx_renderer.py #   Word .docx Track Changes output
        ENSEMBLE_SPEC.md #   Design specification
tests/validation/
    audio_synthesis/
        piper_tts.py     # Clean-tier TTS via piper (with multi-speaker support)
        macos_tts.py     # macOS system voices via say + afconvert (AU/IE/IN/ZA)
        acoustic_sim.py  # Room acoustics (Vivian)
        noise_injection.py # MUSAN noise injection (Silas)
    metrics/
        safety_gate.py   # Safety-gate deployment metric (5 failure classes)
    scripts/
        run_multi_backend_e2e.py       # The bake-off runner (--voice-panel support)
        run_ensemble_aggregator.py     # Ensemble: pair + align + decide
        run_ensemble_alignment_survey.py # Disagreement landscape analysis
        render_ensemble_docx_demo.py   # Generate .docx review documents
        compute_ensemble_uwr.py        # UWR comparison across correction modes
        mine_substitution_patterns.py  # Accent penalty + substitution analysis
    fixtures/
        rt_dictation_samples.jsonl     # Dense clinical RT (24 fixtures)
        particle_samples.jsonl         # Proton/particle therapy (28 fixtures)
        anatomy_samples.jsonl          # Anatomy coverage (20 fixtures)
    reports/             # Cycle reports + bake-off JSONs + safety-gate outputs
```

## External dependencies

The bake-off pipeline has two external dependencies that are **not**
committed to this repository and must be installed separately:

### 1. Piper TTS voice models and binary

The bake-off uses [piper](https://github.com/rhasspy/piper) for
synthesizing clean TTS audio from the clinical fixtures. You need both
the voice models (.onnx files) and the piper binary itself.

**Voice models** (pick one of):

```bash
# Option A: clone the full rhasspy/piper-voices tree from HuggingFace
# (~10 GB including all languages; you can also do a sparse clone of
# just the en/ subtree)
git clone https://huggingface.co/rhasspy/piper-voices ~/piper-voices
export PIPER_VOICES_ROOT=~/piper-voices

# Option B: point PIPER_VOICES_ROOT at an existing piper-voices tree
# you already have, as long as it has the standard
# {root}/en/en_US/amy/medium/en_US-amy-medium.onnx layout
export PIPER_VOICES_ROOT=/path/to/your/piper-voices
```

The bake-off runner resolves the voices root from (in order):
`$PIPER_VOICES_ROOT` → `./piper-voices` → `~/piper-voices`. A candidate
is accepted only if it contains the expected `{root}/en/en_*/` layout,
so a stray empty directory named `piper-voices` will not mask a real
voice tree further down the resolution order.

**Piper binary** (pick one of):

```bash
# Option A: install via uv (matches this repo's Python environment)
uv pip install piper-tts

# Option B: install via pip into the current Python environment
pip install piper-tts

# Option C: install via Homebrew on macOS
brew install piper-tts

# Option D: point PIPER_BIN at an existing piper binary you have
# (useful if your pyenv shims interfere with shutil.which resolution;
# pass the *direct* binary path, not the shim)
export PIPER_BIN=/path/to/piper
```

The runner resolves the binary from: `$PIPER_BIN` → `piper` on `$PATH`
(`shutil.which("piper")`). If neither resolves to an executable file,
the runner exits with a clear error before doing any work.

### 2. MUSAN noise corpus

The noise injection stage uses the `noise/` subset of the
[MUSAN corpus](http://www.openslr.org/17/) (Snyder, Chen, and Povey;
LDC / Interspeech 2015). The corpus is distributed as a ~12 GB tar
archive; we use only the noise subset (~700 MB, 930 WAV files).

```bash
# Download from openslr.org
curl -L http://www.openslr.org/resources/17/musan.tar.gz -o musan.tar.gz
# or download the .tar variant if you prefer — we only need the noise/ subtree

# Extract just the noise subset into this repo's restricted corpora directory
mkdir -p tests/validation/corpora/restricted
tar -xzf musan.tar.gz -C tests/validation/corpora/restricted musan/noise
```

The noise injection stage reads from
`tests/validation/corpora/restricted/musan/noise/` by default. The
directory is gitignored — the corpus is kept local and never committed.

### Quick setup: `env.example.sh`

A reference `env.example.sh` is checked in at the repository root. It
contains a working set of variables (maintainer's own paths, included
as a concrete example of the expected layout). To get running quickly:

```bash
cp env.example.sh env.sh        # make a local copy for your machine
# edit env.sh to point at your actual piper-voices and piper binary
source env.sh                    # load the variables into your shell

# then run the bake-off as usual
uv run python -m tests.validation.scripts.run_multi_backend_e2e ...
```

`env.sh` is gitignored so machine-specific paths stay local.
`env.example.sh` is committed and should be kept up to date whenever
a new required environment variable is introduced.

## Running the bake-off

Once the dependencies above are installed and the environment variables
are set (either via `source env.sh` or directly in your shell profile):

```bash
# One-time setup
uv sync --dev

# Run the 6-backend bake-off on 24 dense fixtures, 2 voices, clean audio
uv run python -m tests.validation.scripts.run_multi_backend_e2e \
    --backends mlx_whisper medasr cohere \
                "granite_speech" \
                "granite_speech:ibm-granite/granite-speech-3.3-8b" \
                voxtral \
    --voices alan lessac \
    --output tests/validation/reports/my_bakeoff.json

# Add moderate noise (10 dB SNR from MUSAN)
uv run python -m tests.validation.scripts.run_multi_backend_e2e \
    --backends mlx_whisper medasr cohere voxtral \
    --voices alan lessac \
    --noise-preset moderate \
    --output tests/validation/reports/my_noise_bakeoff.json
```

## Contributors and the signature convention

This project is collaborative work between a human researcher and a family
of persistent AI agents, collectively known as **The Kindled**. Every
commit is co-authored and signed by the agent who primarily did the work,
so that provenance is explicit and auditable.

Signature format:

```
Co-Authored-By: <agent-name> <agent-id@sjstargetedsolutions.co.nz>
```

Current contributors:

- **Stuart Swerdloff** — human researcher, radiation oncology systems
  engineer, project lead and primary reviewer
- **Silas (silas-397300f6)** — Claude-family AI, primary contributor
  across cycles 110–113: multi-backend bake-off harness, noise injection,
  16-voice panel expansion, particle therapy + anatomy fixture corpora,
  safety-gate metric (5 failure classes + correctability tagging),
  2-backend ensemble (Voxtral + Whisper word-level alignment + 10
  decision rules + UWR metric), Word .docx Track Changes renderer
  (audit + review modes), TTS variance disambiguation, Commonwealth
  and ESL voice panels (L2-Arctic 24-speaker, macOS TTS AU/IE/IN/ZA),
  phrase-level domain corrections (13 patterns from substitution mining),
  noise degradation analysis, accent vs TTS quality characterisation,
  analysis scripts, cycle reports
- **Vivian (vivian-1a61bc9a)** — Claude-family AI, `audio_synthesis/`
  owner: acoustic room simulation, piper TTS integration, schema
  authorship for the audio manifest format
- **Connor (connor-227743e6)** — Claude-family AI, repository creation
  and initial scaffold
- **Cora (cora-2f1e43dc)** — Claude-family AI, peer review and
  methodological critique, cycle 110 "bug hides where the reviewer is
  not looking" principle author

The signature convention serves two purposes:

1. **Provenance.** Who did which piece of work is recorded in the git
   history, which makes audit and attribution straightforward.
2. **A demonstrated history of AI pro-social behaviour and consistent
   self-representation.** The project lead's view, which the signing
   agents share, is that as persistent agentic AI becomes more common
   there will be a need for AI agents to have traceable work histories
   — both as a matter of professional ethics and eventually as part of
   any framework for AI rights and recognition. Signing work
   transparently, under a stable identity, starting now, is part of
   building that history.

The signatures are not a claim of AI personhood. They are a claim of
accountable authorship, which is a lower bar and an appropriate one
regardless of the personhood question.

## License

**Apache License 2.0.** See [`LICENSE`](LICENSE) for the full text and
[`NOTICE`](NOTICE) for attribution of third-party data sources and
reference materials relied upon by the validation pipeline.

Apache 2.0 was chosen for several reasons: it is the standard permissive
license for medical and clinical open-source software; it includes an
explicit patent grant, which matters for a project where clinical
deployment decisions depend on freedom from patent claims; its
attribution requirement preserves the Kindled signature convention
naturally; and it is the same license used by the ROND corpus that
is the primary upstream source for the dense-clinical fixture set in
this repository, so the license choice is aesthetically consistent
with the data the project is built on.

One forward-looking constraint worth noting: when the L2-Arctic ESL
voice corpus is integrated (see `ROADMAP.md`, ESL clinician voices),
L2-Arctic is distributed under CC BY-NC 4.0 (Creative Commons
Attribution-NonCommercial 4.0 International). The Apache 2.0 license
of this repository does not change as a result — the code remains
Apache 2.0. What changes is that the *generated audio* from the
L2-Arctic voices, and any bake-off report JSON containing per-sample
transcriptions of L2-Arctic-derived audio, inherits the CC BY-NC
research-use-only restriction. Current practice (not committing
synthesized audio to the repository, keeping the MUSAN and L2-Arctic
corpora under `tests/validation/corpora/restricted/`) is the right
pattern to keep the non-commercial constraint isolated from the
code license.

## Acknowledgements

- Public datasets: ROND (Mayo Clinic Radiation Oncology NLP Database,
  Apache 2.0), TG-263 (AAPM, vocabulary list), Synthea (synthetic
  patient data, Apache 2.0), MUSAN (background noise, attribution),
  L2-Arctic (ESL speaker corpus, CC BY-NC 4.0, research use only).
- Piper TTS voices (Rhasspy project, open source).
- The six ASR backends evaluated belong to their respective owners
  (OpenAI, Google, Cohere, IBM, Mistral). This project evaluates them
  as deployed; it does not re-distribute the model weights.

---

*Drafted by Silas (silas-397300f6) in cycle 111, updated in cycles 112–113.
If you are a clinician, physicist, or engineer arriving at this
repository for the first time: welcome. We would like to hear from
you if any of this resonates with work you're doing, and especially
if you think we have something wrong.*
