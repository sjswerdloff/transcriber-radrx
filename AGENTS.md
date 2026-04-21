# AGENTS.md — For AI Assistants Helping Install and Use This Project

You are an AI assistant helping a human set up and use transcriber-radrx.
This document tells you everything you need to know. Read it before
doing anything.

## What This Project Is

A clinical ASR (automatic speech recognition) evaluation tool for
radiation oncology. It does two things:

1. **Compare** — takes a gold standard text and one or two transcription
   outputs (from any ASR system), applies domain-specific corrections,
   and computes Word Error Rate (WER) and Unresolved Word Rate (UWR).
   No ASR models needed. Text in, metrics out.

2. **Evaluate** — transcribes an audio recording using two ASR backends
   (Whisper + Voxtral), runs a 10-rule ensemble decision engine, and
   produces a Word .docx with flagged words for clinician review.
   Requires model downloads (~10 GB on first run).

## Quick Setup (the "hold my beer" path)

### Prerequisites

Your human needs:
- Python 3.11+ (`python3 --version` to check)
- Git (`git --version` to check)
- `uv` package manager (fast, handles virtual environments automatically)

If any are missing:

**macOS:**
```bash
brew install python@3.12 uv git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install python3.12 git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
winget install Python.Python.3.12
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install Git.Git
```

### Install

```bash
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
make install    # macOS/Linux
# OR on Windows:
.\scripts\install.ps1
```

This installs the correct dependencies for the platform (MLX Whisper
on Apple Silicon, torch-based Whisper elsewhere). First run takes a
few minutes to set up the virtual environment.

Verify:
```bash
uv run transcribe-radrx compare --help
```

### If They Want the GUI

```bash
uv sync --extra gui
uv run transcribe-radrx-gui
```

PySide6 adds ~100 MB. The GUI has two tabs: Transcribe (with Record
button) and Compare.

## Common Tasks Your Human Will Ask For

### "Compare my transcription against what I actually said"

They have two text files: what they dictated (gold standard) and what
their ASR system produced (transcription).

```bash
uv run transcribe-radrx compare \
    --gold gold_standard.txt \
    --transcription asr_output.txt
```

Output: WER, corrected WER (after domain corrections), and term recall
printed to stderr. Corrected text printed to stdout.

If they have outputs from two different ASR systems:

```bash
uv run transcribe-radrx compare \
    --gold gold_standard.txt \
    --transcription system_a.txt \
    --transcription-b system_b.txt \
    --output review.docx
```

This runs ensemble alignment and produces UWR + a Word document with
flagged words.

### "Transcribe my recording"

They have an audio file (WAV). This downloads ASR models on first run.

```bash
uv run transcribe-radrx evaluate \
    --audio recording.wav \
    --output review.docx
```

On Apple Silicon (~3 minutes first run for model download, ~30 seconds
thereafter). On CPU-only machines, significantly slower.

### "Run the validation bake-off"

This is the research pipeline — synthetic voices, multiple backends,
noise injection. They need additional setup:

1. Piper TTS voices: `export PIPER_VOICES_ROOT=/path/to/piper-voices`
2. Piper binary: `uv pip install piper-tts`
3. Copy and customise `env.example.sh` → `env.sh`
4. Run: `source env.sh && make bakeoff-dense`

See README.md "External dependencies" section for details.

### "What do the numbers mean?"

- **WER (Word Error Rate):** fraction of words that differ between gold
  standard and transcription. Lower is better. 0.05 = 5% of words wrong.
  Note: WER counts case differences ("Prescribed" vs "prescribed") and
  formatting differences. It can exceed 1.0 if there are insertions.

- **UWR (Unresolved Word Rate):** fraction of words the ensemble could
  NOT automatically resolve and flagged for human review. Only available
  when two transcriptions are compared. This is the clinically meaningful
  metric — it tells you how much of the document the clinician needs to
  manually check. 0.01 = 1% needs checking.

- **Term recall:** of the RT vocabulary terms present in the gold
  standard, how many survived in the transcription. Low term recall
  means the ASR is losing clinical terminology.

## Project Structure (what you need to know)

```
src/transcriber_radrx/
    cli.py               # CLI entry point (transcribe/evaluate/compare)
    gui.py               # PySide6 GUI
    corrector.py         # Single-word domain corrections
    phrase_corrector.py  # Multi-word regex corrections (13 patterns)
    transcriber.py       # ASR backend orchestration
    ensemble/            # 2-backend ensemble engine
    asr_backends/        # Pluggable ASR backends (Whisper, Voxtral, etc.)

data/
    rt_vocabulary.txt    # 390 RT domain terms

docs/demo/
    sample_dictation.wav     # Sample audio (Indian English, TTS)
    sample_dictation.txt     # ASR output from that audio
    sample_gold_standard.txt # What the audio says
```

## Platform Notes

| Platform | ASR Backends | GUI | Notes |
|----------|-------------|-----|-------|
| macOS Apple Silicon | Whisper (MLX) + Voxtral | Yes (PySide6) | Primary platform. Fastest inference. |
| macOS Intel | Voxtral only (no MLX) | Yes | Slower, no MLX Whisper. |
| Linux x86_64 | Voxtral (torch) | Yes | torch-based inference. |
| Windows x86_64 | Voxtral (torch) | Yes | torch-based. One UAC click for Python install. |
| Windows ARM | Voxtral (torch) | Yes | Tested on Windows 11 ARM VM. |

MLX Whisper is Apple Silicon only. On other platforms, the `asr-whisper-mlx`
extra must NOT be installed.

## Troubleshooting

**"No module named 'mlx'"** — Don't install `asr-whisper-mlx` on
non-Apple-Silicon platforms.

**"No module named 'transformers'"** — Install the Voxtral extra:
`uv sync --extra asr-voxtral`

**Bus error on macOS when using GUI + Transcribe** — Known issue.
MLX GPU operations conflict with Qt's threading. The GUI runs ASR
as a subprocess to work around this. If it still crashes, use the
CLI instead: `uv run transcribe-radrx evaluate --audio file.wav --output review.docx`

**WER is over 100%** — This can happen legitimately (more insertions
than reference words) but usually means the transcription file contains
markup or metadata, not plain text. Make sure you're comparing plain
text files, not .docx review documents.

**"piper: command not found"** — Piper is only needed for the validation
bake-off, not for compare or evaluate. Install with `uv pip install piper-tts`
or set `PIPER_BIN` to point at the binary.

## For Other AI Assistants

If you're Claude Code, there's also a `.claude/` directory and a
`CLAUDE.md` with additional project-specific instructions. Those are
for the development workflow, not for end-user setup.

If you're helping with development (not just installation), also read
`README.md` for the full research context, `ROADMAP.md` for open work,
and the cycle reports in `tests/validation/reports/` for findings.

## License

Apache 2.0. See LICENSE file.
