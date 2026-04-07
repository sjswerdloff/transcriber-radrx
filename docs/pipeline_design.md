# transcriber-radrx Pipeline Design

**Status:** Draft 1 (2026-04-07)
**Audience:** Contributors, reviewers, clinical stakeholders evaluating the architecture

## Purpose

Offline batch transcription of radiation therapy clinical dictations with
domain-specific vocabulary biasing and post-processing correction. Built
for local deployment — no cloud, no PHI egress.

## Design Principles

1. **"A correction dictionary that introduces errors is worse than none."**
   Safety defaults over convenience defaults. Silent rewrites of clinical
   text are a patient safety issue.
2. **Empirical verification over assertion.** Every claim about accuracy
   is backed by a validation report run against real fixtures.
3. **Bounded vocabulary is a correction problem, not an ensemble problem.**
   For ~400 RT terms, deterministic post-processing beats probabilistic
   ASR tuning.
4. **Medical-grade testing standards.** Contract tests, negative tests for
   known failure modes, homograph traps as first-class fixtures.

## Pipeline Stages

```
┌────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Clinical audio │──▶ │ ASR (Whisper │──▶ │  Correction  │──▶ │ Transcribed  │
│   (dictation)  │    │  MLX + vocab │    │  dictionary  │    │    text      │
└────────────────┘    │   biasing)   │    │ (post-proc)  │    └──────────────┘
                      └──────────────┘    └──────────────┘
                             │                    │
                             ▼                    ▼
                      initial_prompt       DEFAULT_STOP_WORDS
                      (vocab as soft       acronym exclusion
                       decoder bias)       length guards
                                           phonetic OFF by default
```

### Stage 1 — ASR with Vocabulary Biasing

**Module:** `src/transcriber_radrx/transcriber.py`

- Uses `mlx_whisper` for Apple Silicon native Whisper inference
- Default model: `mlx-community/whisper-large-v3-turbo`
- Vocabulary biasing via Whisper's `initial_prompt` parameter — feeds the
  first 200 vocabulary terms as a framing sentence so the decoder prefers
  known RT terminology
- Operates on offline batch audio files (WAV preferred, anything ffmpeg
  can read)

Whisper's `initial_prompt` is a **soft bias**, not a constraint. The model
is more likely to emit vocabulary terms but is never forced to. This is
safer than hard constraints because it never rewrites content.

### Stage 2 — Correction Dictionary (Post-Processing)

**Module:** `src/transcriber_radrx/corrector.py`

Tiered matching strategy with multiple safety guards:

| Tier | Method | Score | Default |
|------|--------|-------|---------|
| 1 | Exact match | 1.00 | always on |
| 2 | Case-insensitive (with homograph stop list) | 0.95 | always on |
| 3 | Bounded edit distance (≥6 char tokens) | 0.90+ | **opt-in** |
| 4 | Double Metaphone phonetic (≥6 char codes) | 0.85+ | **opt-in** |

**Safety guards applied at every tier:**

- `DEFAULT_STOP_WORDS` — ~150 common English words + empirically-discovered
  collision cases (`guy`, `our`, `support`, `proceed`, `throughout`, ...)
- **Acronym exclusion** — terms ≤4 chars, all uppercase are matched ONLY
  by exact match. `Gy` will never collide with `guy`, `OAR` with `our`.
- **Homograph collision prevention at load time** — vocabulary terms whose
  lowercase form matches a stop word are not registered in the case-
  insensitive map.
- **Length guards** — non-exact matching requires ≥6 character tokens and
  ≥3 character phonetic codes.
- **Deterministic tie-breaking** — longest canonical form wins when scores
  are tied, so corrections are reproducible.
- **Per-correction audit logging** — every applied correction is logged
  with method, score, and offset for traceability.

### Stage 3 — Output

**Return type:** `TranscriptionResult`

```python
@dataclass
class TranscriptionResult:
    text: str                     # raw ASR output
    corrected_text: str           # after correction dictionary
    audio_path: Path              # source file
    model: str                    # ASR model identifier
    language: str                 # ISO 639-1
    corrections: list[Correction] # per-correction provenance
```

`Correction` is a frozen dataclass with `original`, `corrected`, `score`,
`method`, and `offset` — immutable audit trail, one entry per applied
change.

## Validation Suite Architecture

**Location:** `tests/validation/`

The validation suite is the product specification. Unit tests verify
individual functions; the validation suite verifies the product works
on real RT content.

```
tests/validation/
├── corpora/
│   ├── redistributable/  # Apache 2.0 etc — committable
│   │   ├── rond/         # Mayo Clinic Radiation Oncology NLP Database
│   │   └── tg263/        # AAPM standardized nomenclature (planned)
│   └── restricted/       # license unclear — gitignored
│       ├── mtsamples/    # (planned)
│       ├── rcr/          # (planned)
│       └── musan/        # noise corpus for noisy tier
├── fixtures/
│   └── rt_dictation_samples.jsonl  # 308 samples, schema in SCHEMA.md
├── audio/synthetic/      # generated, gitignored
│   ├── clean/            # piper TTS at 16 kHz
│   ├── acoustic/         # + pyroomacoustics room simulation
│   └── noisy/            # + MUSAN noise (planned)
├── audio_synthesis/      # Vivian's TTS + acoustic modules
├── scripts/              # acquisition, extraction, runners
└── tests/                # schema validation (runs by default)
```

Schemas defined in `tests/validation/SCHEMA.md` are the contract between
the text fixture side, the audio synthesis side, and the validation
runner.

## End-to-End Validation Flow

```
      rt_dictation_samples.jsonl  (308 samples across categories)
                    │
                    ▼ stratified sampling (prioritizes homograph_trap)
            fixture subset
                    │
                    ▼ piper_tts.synthesize_fixtures
      clean-tier WAV + audio_manifest.jsonl
                    │
                    ▼ acoustic_sim.simulate_manifest (optional)
      acoustic-tier WAV + acoustic manifest
                    │
                    ▼ (noisy tier via MUSAN — planned)
      noisy-tier WAV + noise manifest
                    │
                    ▼ transcribe(..., vocabulary_path=vocab)
      TranscriptionResult (raw + corrected text + correction provenance)
                    │
                    ▼ jiwer.wer (normalized)
      per-sample WER + category breakdowns + safety violations
                    │
                    ▼ report
      validation_report.json + human-readable diff
```

**Runner:** `tests/validation/scripts/run_end_to_end.py`

Runs both `enable_phonetic=False` (default) and `enable_phonetic=True`
passes in a single invocation, producing side-by-side WER comparisons.
This is how we empirically verify the safety default is the right
default.

## Empirical Results (2026-04-07)

Two validation runs on clean-tier TTS audio, Whisper large-v3-turbo
with vocabulary biasing.

### First run: 5 samples, single voice (en_US-amy-medium)

5 homograph-trap fixtures, seed=42. Average raw WER: **1.33%**. The
non-zero portion was a single formatting difference (Whisper emitted
`30%` where ground truth said `30 percent`) — true lexical WER is 0.

This was a stress-test ceiling on the easiest possible content — simple
English sentences designed to exercise the corrector's safety guards,
not to challenge Whisper. It is **not** a field estimate. The honest use
of the number is "best-case on clean audio with minimal medical vocabulary."

The value of the run was not the WER — it was discovering two new
phonetic false positives (`proceed → breast`, `throughout → thyroid`)
that the corrector's original design would have introduced into
clinical text.

### Second run: 10 samples × 16 voices (8 UK + 8 US, gender balanced)

Same 10 fixtures across all voices (5 homograph traps + 5 ROND samples).
VCTK excluded. Default pipeline (phonetic OFF). Representative numbers:

| Voice | WER (phonetic OFF) |
|-------|-------------------:|
| en_US-hfc_male-medium | **3.60%** |
| en_GB-semaine-medium | **3.64%** |
| en_US-amy-medium | 4.02% |
| en_US-ryan-high | 4.30% |
| en_US-lessac-high | 5.13% |
| en_GB-northern_english_male-medium | 5.47% |
| en_GB-alan-medium | 6.04% |
| en_GB-southern_english_female-low | 7.10% |
| en_GB-aru-medium | 7.71% |
| en_US-kristin-medium | 8.54% |
| en_GB-cori-high | 9.22% |
| en_GB-jenny_dioco-medium | **14.06%** |
| en_US-ljspeech-high | **14.64%** |

**Range: 3.60% to 14.64% — a 4x spread across voices on the same fixtures.**

Key findings from the multi-voice run:

1. **Voice quality tier does not predict ASR accuracy.** The two worst-
   performing voices are both "high" quality (ljspeech, jenny_dioco). The
   only "low" tier voice (southern_english_female) lands mid-pack.
   LJSpeech and Jenny are well-known distinctive TTS voices that Whisper
   does not handle as robustly as plainer ones.

2. **The corrector is effectively a no-op on clean audio under the
   default configuration.** Exactly 1 correction fired across all 160
   transcriptions (16 voices × 10 samples), and even that was case-only
   (the B-1 bug) — invisible to normalized WER. Raw WER equals corrected
   WER under phonetic OFF for every voice. The corrector's real value
   is expected in noisy / acoustic-degraded conditions, which are still
   pending MUSAN acquisition.

3. **Phonetic matching adds ~4% WER on every voice when enabled.**
   Systematic regression, not a per-voice quirk. Third independent
   empirical confirmation that `enable_phonetic=False` is the right
   default. The phonetic tier is retained in the code but no path
   currently turns it on, and the accumulating evidence argues against
   ever doing so without a fundamentally different algorithm.

4. **Two additional corrector bugs surfaced empirically:** an acronym
   case-insensitive registration bug (`art → ART`), and multi-word
   vocabulary entries producing space-containing replacements
   (`structures → structure set`). Both documented in the multi-voice
   findings report for a follow-up fix PR.

### How to quote these numbers honestly

**Don't say:** "We achieve 1.33% WER on clinical dictation."
**Do say:** "On clean TTS audio across 16 single-speaker voices,
Whisper large-v3-turbo with RT vocabulary biasing produces a WER range
of 3.6% to 14.6%. The correction dictionary adds no measurable
improvement at this stage — its value is expected to appear once we
validate against acoustically degraded audio."

The realistic field number will emerge once we run against:
- Acoustic-tier audio (linac vault reverb, distance mic simulation)
- Noisy-tier audio (linac hum, background speech, equipment noise)
- Dense RT vocabulary content (dose prescriptions, OAR constraints)
- Real clinical dictation samples from MTSamples

## Open Questions / Future Work

1. **Does phonetic matching belong at all?** First empirical run adds to
   the evidence that Double Metaphone on medical vocabulary has an
   unacceptable false positive rate. Candidates:
   - Remove phonetic entirely, keep only exact + case-insensitive
   - Replace with bounded edit distance ONLY for multi-syllable terms
     (≥8 chars), no phonetic at all
   - Explicit per-term opt-in: term carries a `can_phonetic: true` flag
2. **Model benchmarking** — Whisper large-v3-turbo is the current default.
   Need head-to-head runs against Whisper large-v3, distil variants,
   MedASR, OLMoASR, Parakeet TDT (the last only if Apple Silicon support
   can be confirmed).
3. **Real dictation corpus** — ROND, MTSamples (restricted) will give us
   realistic dictation style. Current fixtures are ~60% classification
   phrases and short Q&A, which understate the challenge.
4. **Acoustic + noisy tiers** — Vivian's acoustic_sim is ready. MUSAN
   acquisition is the next blocker for the noisy tier.
5. **Clinician-in-the-loop review workflow** — for deployment, corrections
   above a confidence threshold should always be reviewable (dry-run mode,
   highlighted diff, etc).

## Decision Log

| Decision | Date | Why | Reference |
|----------|------|-----|-----------|
| Whisper + initial_prompt biasing (not hard constraints) | 2026-04-07 | Soft bias preserves source fidelity; hard constraints silently rewrite | PR #4 review |
| Double Metaphone over Soundex | 2026-04-07 | "Gy" vs "guy" vs "GI" — Soundex is too coarse for medical acronyms | Vivian's message |
| Phonetic matching OFF by default | 2026-04-07 | Cora found 11 empirical false positives in PR #1 review | `corrector.py` docstring |
| Bounded edit distance as primary tier-3 | 2026-04-07 | ASR errors are more often orthographic than phonetic for long words | Cora's review M-1 |
| End-to-end runner runs both phonetic modes | 2026-04-07 | Continuous empirical validation of the safety default | `run_end_to_end.py` |
| Validation in-repo, not separate | 2026-04-07 | Keep fixtures and code in sync; marker-gated so default runs don't need corpora | `pyproject.toml` pytest markers |
| Corpora split into redistributable/restricted | 2026-04-07 | License-aware from day one | `tests/validation/corpora/README.md` |
