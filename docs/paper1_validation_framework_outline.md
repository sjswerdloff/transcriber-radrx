# Paper 1: TTS-Based Validation Framework for Clinical ASR in Radiation Oncology

## Target
JOSS (Journal of Open Source Software) or arXiv

## Thesis
A reproducible, open-source framework for evaluating ASR system performance
on radiation oncology dictation, using synthetic speech to characterise
system behaviour across accents, noise conditions, and clinical vocabulary
domains — before deployment with real clinicians.

## 1. Introduction
- Clinical transcription in RT: safety-critical domain, specialised vocabulary
- ASR adoption growing but validation methodology lags
- Gap: no standard way to evaluate ASR on RT-specific content across speaker populations
- Contribution: open-source framework, reusable fixture corpora, multi-voice evaluation

## 2. Related Work
- General ASR benchmarks (LibriSpeech, CommonVoice) — not domain-specific
- Medical ASR evaluation — scattered, usually single-backend, single-accent
- L2-Arctic corpus and accent-stratified evaluation in general domain
- TTS as a proxy for speaker variation in ASR testing

## 3. Framework Design
### 3.1 Fixture Corpora
- Three corpora: RT dense clinical (24), particle therapy (28), anatomy (20)
- JSONL schema with ground truth, vocabulary terms, clinical metadata
- Extensible — clinics add site-specific fixtures

### 3.2 Voice Panels
- Piper TTS: Commonwealth (8 en_GB), Default (16 en_GB + en_US)
- Piper TTS: ESL (24 L2-Arctic speakers across 6 L1 backgrounds + 2 singles)
- macOS system voices: AU, GB, IE, IN, ZA
- Multi-speaker model support (speaker_id for L2-Arctic, VCTK)
- Panel architecture is extensible — add voices without code changes

### 3.3 Acoustic Simulation
- MUSAN noise injection at calibrated SNR tiers (quiet 20dB, moderate 10dB, busy 5dB)
- Room acoustics via pyroomacoustics (linac vault, exam room, open office presets)
- Composable: noise + room combined

### 3.4 Metrics
- WER (standard)
- Per-term vocabulary recall (did specific RT terms survive?)
- Safety-gate metric: weighted failure classes for dose-critical tokens
- UWR (Unresolved Word Rate) for ensemble configurations

## 4. Validation Experiments
### 4.1 Accent Characterisation
- Commonwealth vs ESL voice panels on same fixtures
- Per-L1 miss rates: accent penalty table
- Finding: domain vocabulary failures are accent-independent (IGRT, SRS fail 100% for all)
- Finding: accent-specific failures concentrate on multi-syllable medical terms

### 4.2 TTS Quality as Confound
- macOS (high-quality) vs piper (lower-quality) for Indian English
- Same accent family, 3x WER difference
- Cannot cleanly separate accent from TTS fidelity
- L2-Arctic results are conservative upper bounds

### 4.3 Noise Degradation
- Clean → quiet → moderate → busy progression
- Graceful degradation: UWR 0.42% → 1.04% across 15 dB range
- Indian English voice does not degrade faster than native voices
- Implication: signal quality is addressable through environmental/hardware interventions

## 5. Discussion
### 5.1 The Oldest Speech Recognition Technique
- Clear enunciation remains the highest-impact intervention
- Framework quantifies which specific words and how much
- Site-specific validation with real clinician recordings: same pipeline, no code changes

### 5.2 Limitations
- TTS is not human speech: no disfluencies, variable volume, fatigue, Lombard effect
- Individual accent variation within L1 groups exceeds between-group averages
- Framework establishes system capability bounds, not deployment predictions

### 5.3 Recommendations for Deployment Validation
- Record your clinicians once, run through the framework
- Environmental noise measurement and mitigation
- Speaker feedback loop: system tells clinician which words to enunciate

## 6. Software Availability
- Repository, license, installation, fixture format, how to add voices

## References
