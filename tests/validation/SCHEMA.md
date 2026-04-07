# Validation Suite Schemas

Authoritative definitions for fixture and metadata formats.
Both the corpus extraction side (Silas) and the audio synthesis side
(Vivian) consume these. Changes require updates to both sides.

## 1. Text Fixture: `rt_dictation_samples.jsonl`

One JSON object per line. Used as input to TTS and as ground truth for WER.

```json
{
  "id": "rond-0001",
  "text": "The patient received 50 Gy in 25 fractions to the PTV using IMRT.",
  "category": "treatment_summary",
  "source": "ROND",
  "source_url": "https://github.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database",
  "license": "Apache-2.0",
  "vocabulary_terms": ["Gy", "PTV", "IMRT"],
  "expected_difficulty": "low",
  "language": "en",
  "char_count": 65,
  "word_count": 13
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable unique identifier (`{source}-{number:04d}` convention) |
| `text` | string | Ground truth text — what the ASR should produce |
| `category` | string | One of the enumerated categories (see below) |
| `source` | string | Source corpus name (ROND, MTSamples, synthetic, etc.) |
| `license` | string | SPDX identifier or short license name |
| `language` | string | ISO 639-1 code (default "en") |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `source_url` | string | Where the source can be found |
| `source_id` | string | Identifier within the source corpus (e.g. ROND record ID) |
| `vocabulary_terms` | string[] | RT vocabulary terms present in `text` (for category-specific WER) |
| `expected_difficulty` | string | "low", "medium", "high" — analyst judgement |
| `notes` | string | Free-form annotation |
| `char_count` | int | Pre-computed for filtering |
| `word_count` | int | Pre-computed for filtering |

### Categories

| Category | Description |
|----------|-------------|
| `treatment_summary` | Full RT plan summary |
| `dose_prescription` | Focused on dose/fractionation language |
| `oar_constraints` | Organ-at-risk dose limits |
| `setup_instructions` | Patient positioning, immobilization |
| `progress_note` | Interim clinical notes during treatment |
| `consent_discussion` | Patient-clinician dialogue |
| `acronym_dense` | High concentration of RT acronyms (stress test) |
| `homograph_trap` | Sentences with English words that collide with RT acronyms (negative test) |

## 2. Audio Manifest: `audio_manifest.jsonl`

One JSON object per generated audio file. Vivian's TTS pipeline writes
this; the validation runner reads it. Each entry references a text
fixture by `text_id` and adds audio metadata.

```json
{
  "audio_id": "rond-0001-piper-en_US-amy-medium-clean",
  "text_id": "rond-0001",
  "audio_path": "tests/validation/audio/synthetic/clean/rond-0001-piper-en_US-amy-medium.wav",
  "tier": "clean",
  "tts_engine": "piper",
  "tts_voice": "en_US-amy-medium",
  "sample_rate_hz": 16000,
  "duration_seconds": 4.32,
  "channels": 1,
  "bit_depth": 16,
  "acoustic_simulation": null,
  "noise_profile": null,
  "snr_db": null
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_id` | string | Unique audio file identifier |
| `text_id` | string | References `id` in `rt_dictation_samples.jsonl` |
| `audio_path` | string | Relative path from repo root |
| `tier` | string | "clean", "acoustic", or "noisy" |
| `tts_engine` | string | "piper" (or future alternatives) |
| `tts_voice` | string | Voice model identifier |
| `sample_rate_hz` | int | Output sample rate (16000 for ASR compatibility) |
| `duration_seconds` | float | Audio duration |
| `channels` | int | 1 (mono) for ASR |

### Optional fields (required for `acoustic` and `noisy` tiers)

| Field | Type | Description |
|-------|------|-------------|
| `bit_depth` | int | Default 16 |
| `acoustic_simulation` | object \| null | Room sim parameters (see below) |
| `noise_profile` | object \| null | Noise mix parameters (see below) |
| `snr_db` | float \| null | Signal-to-noise ratio for noisy tier |

### Acoustic simulation object (acoustic tier)

```json
{
  "room_type": "linac_vault",
  "room_dimensions_m": [6.0, 8.0, 3.5],
  "rt60_seconds": 0.6,
  "mic_distance_m": 1.5,
  "mic_position": "ceiling"
}
```

### Noise profile object (noisy tier)

```json
{
  "background_source": "musan",
  "background_categories": ["speech", "noise"],
  "background_files": ["musan/noise/free-sound/noise-free-sound-0001.wav"],
  "linac_hum_added": true,
  "snr_db": 15.0
}
```

## 3. Validation Report: `validation_report.json`

Output of `run_validation.py`. Single JSON document.

```json
{
  "run_id": "2026-04-07T15:30:00",
  "model": {
    "name": "mlx-community/whisper-large-v3-turbo",
    "type": "whisper-mlx",
    "version": "..."
  },
  "vocabulary": {
    "path": "tests/validation/fixtures/vocabulary_enriched.txt",
    "term_count": 487
  },
  "corrector": {
    "enable_phonetic": false,
    "min_phonetic_score": 0.85
  },
  "audio_set": {
    "manifest": "tests/validation/audio/synthetic/clean/audio_manifest.jsonl",
    "tier": "clean",
    "sample_count": 250
  },
  "metrics": {
    "overall_wer": 0.087,
    "wer_pre_correction": 0.124,
    "wer_post_correction": 0.087,
    "vocabulary_recall": 0.92,
    "vocabulary_precision": 0.98,
    "false_positive_count": 0,
    "by_category": {
      "treatment_summary": {"wer": 0.06, "n": 50},
      "dose_prescription": {"wer": 0.04, "n": 50},
      "oar_constraints": {"wer": 0.11, "n": 30},
      "acronym_dense": {"wer": 0.18, "n": 40},
      "homograph_trap": {"wer": 0.02, "false_positive_count": 0, "n": 30}
    }
  },
  "samples": [
    {
      "text_id": "rond-0001",
      "audio_id": "rond-0001-piper-en_US-amy-medium-clean",
      "ground_truth": "The patient received 50 Gy in 25 fractions to the PTV using IMRT.",
      "raw_transcription": "The patient received 50 gray in 25 fractions to the PTV using IMRT.",
      "corrected_transcription": "The patient received 50 Gy in 25 fractions to the PTV using IMRT.",
      "wer": 0.0,
      "corrections_applied": [
        {"original": "gray", "corrected": "Gy", "method": "..."}
      ]
    }
  ]
}
```

## Schema Versioning

Files include an implicit schema version through the field set. If we
add required fields, bump the schema and write a migration. For now we
use additive-only changes: any new fields are optional.

## Owners

- Text fixtures + extraction scripts: silas-397300f6
- Audio manifests + TTS/acoustic pipeline: vivian-1a61bc9a
- Validation runner + report format: silas-397300f6 (initial), shared after
