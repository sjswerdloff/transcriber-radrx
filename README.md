# transcriber-radrx

Local radiotherapy clinical transcription with vocabulary-biased ASR.

## Why Local?

Clinical voice data contains patient names, diagnoses, and treatment details. Local transcription means audio never leaves the clinic's network.

## Approach

1. **Whisper MLX** — offline batch transcription on Apple Silicon
2. **Vocabulary biasing** — initial_prompt seeds the decoder with ~400 RT domain terms
3. **Post-processing correction** — Double Metaphone phonetic matching corrects remaining ASR errors

## Quick Start

```bash
uv sync
uv run transcribe-radrx audio.wav --vocabulary data/rt_vocabulary.txt
```

## Project Structure

```
src/transcriber_radrx/
    transcriber.py    # Whisper MLX engine + vocabulary biasing
    corrector.py      # Post-processing correction dictionary (Silas)
    cli.py            # Command-line interface
data/
    rt_vocabulary.txt # ~400 RT domain terms
tests/
    test_transcriber.py
```

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check src/ tests/
```

## For the RT Systems Engineering Council

Open source clinical project. Synthetic test data from Synthea (no patient data).

## Authors

- vivian-1a61bc9a — transcription engine, project scaffold
- silas-397300f6 — correction dictionary, RT vocabulary
- connor-227743e6 — repo creation
