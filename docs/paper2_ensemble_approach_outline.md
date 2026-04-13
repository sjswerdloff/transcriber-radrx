# Paper 2: Complementary-Failure Ensemble ASR for Radiation Oncology Dictation

## Target
arXiv (cs.CL / eess.AS / physics.med-ph cross-list)

## Thesis
Two ASR backends with complementary failure profiles — one with medical
vocabulary priors (Voxtral), one with acoustic robustness (Whisper) —
can be combined with deterministic decision rules to resolve 95-99% of
words automatically, with the remaining UWR serving as both a deployment
readiness metric and a per-word feedback signal for speakers.

## Cites
Paper 1 (validation framework) for methodology, fixture corpora, voice panels.

## 1. Introduction
- Single-backend ASR is not safe for RT dictation (cycle 110 finding: 504 Gy)
- Neither Voxtral nor Whisper is safe alone — different failure modes
- Ensemble resolves complementary failures; UWR quantifies the residual
- UWR as bidirectional: measures system readiness AND guides speaker adaptation

## 2. Backend Characterisation
### 2.1 Whisper large-v3
- Acoustic robustness, noise tolerance
- Failure mode: domain vocabulary gaps (Gy → "j"/"gi", IMRT → "image")
- Failures are visibly broken — phonetically recoverable

### 2.2 Voxtral Mini 3B
- Audio-LLM with medical vocabulary priors from training corpus
- First model to beat Whisper on both WER and term recall (cycle 110)
- Failure mode: silent normalisation (GyE → Gy) — LLM decoder priors
- Failures look correct but are clinically wrong

### 2.3 Complementary Profiles
- Table: per-corpus WER and term recall for each backend
- Neither is safe alone; together they cover each other's weaknesses

## 3. Ensemble Architecture
### 3.1 Word-Level Alignment
- Pre-normalisation (GyE collapse, number-unit split, slashed forms)
- difflib SequenceMatcher alignment
- Case-insensitive matching preserving original case

### 3.2 Decision Rules (10 prioritised rules)
- MATCH, DOSE_UNIT_GYE, DOSE_UNIT_CONTEXT, DOSE_UNIT_VISIBLE
- VOCABULARY_MATCH, BOTH_WRONG (→ human review)
- DECIMAL_PRECISION, FORMATTING_DEFAULT, INSERTION_A, INSERTION_B
- Each rule has a source attribution and needs_review flag

### 3.3 Phrase-Level Corrections
- 13 regex patterns mined from bake-off substitution data
- Systematic ASR failures: Gy after numbers, compound joins, multi-word terms
- Applied before ensemble alignment
- Impact: particle therapy UWR 3.11% → 1.93% (piper CW), 2.82% → 0.09% (macOS CW)

### 3.4 UWR Metric
- Definition: fraction of words the ensemble cannot resolve automatically
- Complements WER: WER measures error magnitude, UWR measures review burden
- Per-sample UWR identifies which fixtures/voices are problematic

## 4. Results
### 4.1 UWR Across Voice Panels (full pipeline)
- Table: CW / ESL / macOS × RT dense / particle / anatomy
- Headline: 0.42% (macOS clean) to 4.67% (ESL anatomy)
- Comparison to cycle 112 baseline (1.52% on 2 voices)

### 4.2 Noise Robustness
- Table: clean → quiet → moderate → busy UWR
- Graceful degradation, all voices converge under noise

### 4.3 Correction Pipeline Contribution
- Table: raw → phrase-corrected → full pipeline UWR
- Phrase corrections: biggest impact on particle therapy (GyE patterns)
- Word-level corrector: marginal additional improvement
- Multi-word corrector: measured, <15% of review words addressable

### 4.4 Substitution Pattern Analysis
- Accent penalty table: ESL vs CW miss rates
- Two failure classes: domain vocabulary (accent-independent) vs accent penalty
- Most correctable patterns are domain failures, not accent-specific

## 5. Clinician Review Workflow
### 5.1 Track Changes .docx Rendering
- Audit mode: all changes visible for traceability
- Review mode: automated fixes baked in, only UWR items as Word comments
- Clinician resolves flagged words, approves the rest

### 5.2 UWR as Speaker Feedback
- Per-word flagging identifies which terms a specific speaker should enunciate
- Over time, the flagged word list shrinks as speaker adapts
- Not "your accent is wrong" but "these 12 words need clearer enunciation"
- Demosthenes: quantifying the oldest speech recognition technique

## 6. Discussion
### 6.1 Safety
- Safety-gate metric: CRITICAL failures (decimal drops, silent unit substitution)
- Zero unrecoverable CRITICAL failures across 72 fixtures in cycle 112
- The 504 Gy finding: aggregate WER hid a 10x lethal dose error

### 6.2 Practical Deployment
- Environmental interventions: noise control, beamforming microphones
- Speaker training via UWR feedback loop
- Site-specific validation: record your clinicians, run the framework (Paper 1)

### 6.3 Limitations
- TTS-based evaluation bounds, not deployment predictions
- Two-backend ensemble — additional backends could further reduce UWR
- Phrase corrections are RT-specific; other clinical domains need their own patterns
- No real clinician recordings yet

## 7. Conclusion
- Complementary-failure ensemble achieves <1% UWR on clean native speech
- UWR serves dual purpose: deployment readiness + speaker adaptation feedback
- Open-source, local-only, no cloud dependency — clinic data stays on premises

## References
