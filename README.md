# transcriber-radrx

**A working prototype of a validation framework for clinical automatic speech
recognition (ASR) in radiation oncology dictation.**

This project is not primarily a clinical ASR product. It is a *way of thinking*
about how to validate clinical ASR rigorously — with receipts — and an open
work-in-progress exploring which models, which corrections, and which review
workflows actually meet the bar for clinical dictation where getting a dose
value wrong can kill a patient.

We don't have all the answers. We have a framework for asking the questions
and a growing collection of reproducible experiments that show which ideas
survive contact with real fixtures and which don't.

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
- A **reproducible noise-injection stage** using the MUSAN corpus with a
  prefer-long-first splice strategy so that every transcription is covered
  by continuous ambient noise at a known signal-to-noise ratio.
- A **growing fixture library** of dense radiation oncology clinical content
  (dose prescriptions, OAR constraints, treatment summaries) drawn from
  public sources (ROND, TG-263) and hand-curated adversarial cases.
- A **concept design for staged, auditable, safe-by-construction correction**
  of ASR output, with HTML rendering suitable for clinician review in a
  browser. See `docs/design/staged_correction_demo.html` — open it in Safari
  or Chrome.
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

Current as of **cycle 111** (April 2026). See `tests/validation/reports/`
for the full writeups.

1. **No ASR backend is safe on raw output for clinical dictation.** Every
   backend in the bake-off produces at least one class of clinically
   significant failure on the dense-clinical fixture set. The differences
   between backends are in failure *mode*, not in whether they fail.
2. **Voxtral Mini 3B (Mistral)** is the mean-WER winner on a 16-voice
   (8 UK + 8 US) panel at clean audio, and is essentially flat across
   the full realistic SNR range from clean through 5 dB SNR busy noise
   (within 1.1 percentage points WER across all conditions).
3. **Whisper large-v3 (OpenAI)** is the voice-robustness winner. Across
   the 16-voice panel its WER range is only 5.3 percentage points —
   nothing else is within 10 points. If a deployment cares about
   worst-case guarantee rather than mean performance, Whisper is the
   conservative choice.
4. **Expanding the voice panel from 2 to 16 voices flipped one ranking
   position** (Granite-Speech 8B dropped from #3 to #4; Cohere moved up
   to #3). The 2-voice test was under-sampling the voice axis in a way
   that produced misleading rankings.
5. **MedASR hits a 60 % WER** on a single UK voice
   (`en_GB-southern_english_female-low`). For any deployment that might
   see UK voices or lower-audio-quality inputs, this is clinically
   disqualifying.
6. **Voxtral mis-transcribes the word `Gy`** (the fundamental radiation
   dose unit) in 25–30 % of its clean-audio samples, rendering it as
   `jai`, `gye`, `GJ`, or various homophones. The failure rate is
   persistent, voice-dependent in magnitude (17.6 % on some voices,
   52.9 % on others), and largely independent of noise level. This is
   a clinical-vocabulary weakness that needs post-ASR correction
   before Voxtral can be deployed as-is.
7. **A dose-value safety spot-check across 192 Voxtral transcriptions**
   (across four SNR conditions) found **zero silent numeric
   corruptions** — every dose number was preserved exactly. Voxtral's
   failure modes are visible (obviously wrong non-words or explicit
   refusals), not silent (wrong numbers passing through undetected).
   This is a significantly cleaner safety profile than the Granite 8B
   decimal-drop failure mode.

### Deployment guidance (with caveats)

Based *only* on the US + UK native-speaker voice panel tested so far:

- **Best mean WER on US or mixed US/UK:** Voxtral Mini 3B
- **Best worst-case guarantee:** Whisper large-v3
- **UK-heavy deployment:** Whisper (Voxtral, Cohere, and Granite 8B each
  show a specific UK-voice failure)
- **Do not deploy:** MedASR (UK fragility), Granite-Speech 8B (generalised
  voice fragility)

**None of these backends has been tested on** Indian-English, any other ESL
clinician accent, Commonwealth English (Australian, New Zealand, Canadian,
Irish, South African), proton/particle therapy vocabulary, or an adversarial
dose-value safety gate. Each of those is an open item on the roadmap; the
current findings are therefore correctly read as *"this is the shape of the
problem"*, not *"this is the globally correct backend"*.

## How to read this repository

Start with the **cycle reports** under `tests/validation/reports/`:

- `bakeoff_dense_6backend_noise_moderate_2026-04-08.md` — the noise
  bake-off. Ranking stability under moderate noise (10 dB SNR), Voxtral
  degradation sweep across four SNR levels (clean, 20 dB, 10 dB, 5 dB),
  192-sample dose-value safety spot-check. Authoritative cycle 111
  noise writeup.
- `bakeoff_dense_6backend_16voice_clean_2026-04-08.md` — the 16-voice
  panel expansion. Ranking flip on voice-axis expansion, voice-robustness
  table, Voxtral Gy miss rate per voice. Authoritative cycle 111 voice
  writeup.
- The corresponding JSON files (`bakeoff_dense_*.json`) contain the
  per-sample data that backs the aggregates in the markdown reports.

Then browse the **concept art** for the staged correction pipeline:

- `docs/design/staged_correction_demo.html` — open in Safari or Chrome.
  A hand-written HTML page showing three examples of how a staged,
  safe-by-construction correction pipeline would present its output to
  a clinician reviewer, with word-level inline diffs and rule attribution.
  The examples use real transcription text from the cycle 110 and cycle
  111 bake-offs.

Then read the **roadmap**:

- `ROADMAP.md` — a living document of open research questions and the
  work planned or in progress. Each item is scoped as an external reader
  can understand what is being asked and why.

Finally, browse the **code**:

```
src/transcriber_radrx/
    transcriber.py       # Whisper MLX engine + vocabulary biasing
    corrector.py         # Double Metaphone phonetic correction (stage 2)
    cli.py               # Command-line interface
    asr_backends/        # Pluggable ASR backend Protocol
        base.py          #   Protocol interface all backends implement
        mlx_whisper.py   #   Whisper large-v3 on MLX
        medasr.py        #   Google MedASR on MLX
        cohere.py        #   Cohere Transcribe 2B (HuggingFace)
        granite.py       #   IBM Granite-Speech 2B and 8B
        voxtral.py       #   Mistral Voxtral Mini 3B
        registry.py      #   Lazy-import factory
tests/validation/
    audio_synthesis/
        piper_tts.py     # Clean-tier TTS via piper
        acoustic_sim.py  # Room acoustics (Vivian)
        noise_injection.py # MUSAN noise injection (Silas)
    scripts/
        run_multi_backend_e2e.py  # The bake-off runner
    fixtures/
        rt_dictation_samples.jsonl  # Dense clinical fixtures (24 items)
    reports/             # Cycle reports
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

## Running the bake-off

Once the dependencies above are installed and the environment variables
are set:

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
- **Silas (silas-397300f6)** — Claude-family AI, cycle 110 + 111 primary
  contributor: multi-backend bake-off harness, noise injection, 16-voice
  panel expansion, cycle reports, staged correction concept design
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

*Drafted by Silas (silas-397300f6) in cycle 111 for review by Stuart.
If you are a clinician, physicist, or engineer arriving at this
repository for the first time: welcome. We would like to hear from
you if any of this resonates with work you're doing, and especially
if you think we have something wrong.*
