# Installing transcriber-radrx

This guide is written for radiation oncologists and medical physicists who want
to evaluate the tool on their own clinical dictations. You need to be comfortable
with a terminal, but no software development experience is required.

---

## Prerequisites

### Python 3.11 or later

Check whether Python is already installed:

```bash
python3 --version
```

If it prints `3.11.x` or later, you are set. If not:

- **macOS**: Install via [Homebrew](https://brew.sh) — `brew install python@3.12`
  — or download an installer from [python.org](https://www.python.org/downloads/).
- **Linux**: Use your system package manager, e.g. `sudo apt install python3.12`.
- **Windows**: Download from [python.org](https://www.python.org/downloads/).

### uv (package manager)

`uv` is a fast Python package manager used by this project. It handles
virtual environments automatically so you never risk breaking system packages.

Install on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See [docs.astral.sh/uv](https://docs.astral.sh/uv/) for full instructions.

### Git

You need Git to clone the repository. On macOS it ships with Xcode Command Line
Tools (`xcode-select --install`). On Linux install via your package manager.
On Windows download from [git-scm.com](https://git-scm.com/).

---

## Quick Start: macOS / Linux (one command)

```bash
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
make install
```

That's it. The installer checks for Python 3.11+ and uv, installs them via
Homebrew (macOS) if missing, detects Apple Silicon vs Intel/Linux, and runs
`uv sync` with the right extras. On Apple Silicon you get MLX-accelerated
Whisper; on Linux/Intel you get the torch-based Whisper backend.

The first run downloads model dependencies and creates an isolated virtual
environment in `.venv/`. It does not touch your system Python.

## Quick Start: Windows

```powershell
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
.\scripts\install.ps1
```

The PowerShell installer checks for Python 3.11+ and uv, installs them via
`winget` if missing, and runs `uv sync` with the appropriate extras. MLX
Whisper is not available on Windows; the Voxtral + torch-Whisper backends
are used instead.

> **Note:** Windows support is tested on Windows 11 ARM. If you encounter
> issues on other Windows versions, please open an issue.

---

## Manual Installation (if you prefer)

### macOS Apple Silicon

Apple Silicon Macs (M1/M2/M3/M4) are the primary development platform. They get
the full feature set including MLX-accelerated Whisper (fast, on-device inference)
and the Voxtral audio-LLM backend.

```bash
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
uv sync --extra dev --extra asr-whisper-mlx --extra asr-voxtral --extra phonetic --extra audio --extra validation
```

Verify the install:

```bash
uv run transcribe-radrx evaluate --help
```

### Linux / macOS Intel

MLX is Apple Silicon only and must not be installed on Linux or Intel Macs.
Whisper runs via the PyTorch backend on these platforms — slightly slower but
functionally identical.

```bash
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
uv sync --extra dev --extra asr-voxtral --extra phonetic --extra audio --extra validation
```

Note: if you see `No module named 'mlx'` on Linux or Intel Mac, you have
accidentally installed the `asr-whisper-mlx` extra. Run
`uv sync --extra dev --extra asr-voxtral --extra phonetic --extra audio --extra validation`
(without `asr-whisper-mlx`) to fix it.

---

## Quick Start: Windows

Windows is currently **untested** by the development team. The Python code
itself should run, but:

- MLX is not available on Windows — do not install `asr-whisper-mlx`.
- PyTorch is available for Windows but may require a CUDA toolkit if you want
  GPU acceleration.
- Piper TTS (used for research bake-offs, not for evaluating your own recordings)
  ships Linux and macOS binaries. Windows support is unofficial.

Best-effort install (PyTorch CPU):

```bash
git clone https://github.com/sjswerdloff/transcriber-radrx.git
cd transcriber-radrx
uv sync --extra dev --extra asr-voxtral --extra phonetic --extra audio --extra validation
```

Report issues at the project repository.

---

## External Dependencies

### Piper TTS (for validation bake-offs only)

Piper TTS generates synthetic speech for running multi-voice bake-offs that
compare ASR backends side by side. You do **not** need Piper to transcribe your
own dictations — skip this section unless you want to run research experiments.

Install the Piper Python wrapper:

```bash
uv pip install piper-tts
```

Or install the native binary from the
[Piper releases page](https://github.com/rhasspy/piper/releases). Download the
binary for your platform, extract it, and note the path.

Download voice models from HuggingFace:
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

Set environment variables (copy `env.example.sh` and edit):

```bash
cp env.example.sh env.sh
# Edit env.sh: set PIPER_VOICES_ROOT and PIPER_BIN
source env.sh
```

### MUSAN Noise Corpus (optional — for noise injection experiments only)

Used by some research scripts to simulate real-world acoustic conditions. Not
needed for clinical use.

Download from [OpenSLR](http://www.openslr.org/17/):

```bash
wget http://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz
# Set MUSAN_PATH in env.sh to the extracted directory
```

### ASR Model Weights (downloaded automatically on first use)

The ASR backends download model weights from
[HuggingFace Hub](https://huggingface.co/) the first time they are called.
This is automatic — no manual steps required. Models are cached in
`~/.cache/huggingface/` and reused on subsequent runs.

Approximate download sizes:

| Model | Size on disk | Notes |
|---|---|---|
| Whisper large-v3 (MLX) | ~3 GB | Apple Silicon only |
| Voxtral Mini 3B | ~7 GB (bf16) | Cross-platform |

Make sure you have at least 15 GB of free disk space before first use.
Downloads can take 10–30 minutes depending on your connection.

---

## Evaluating Your Own Recordings

### Step 1: Record your dictation

Record as you normally would. The pipeline resamples to 16 kHz mono internally,
so any sample rate works.

Recommended tools:

- **macOS**: QuickTime Player → File → New Audio Recording, or the built-in
  Voice Memos app.
- **Any platform**: [Audacity](https://www.audacityteam.org/) (free, open
  source).
- **In the clinic**: Any WAV-capable recorder — the file just needs to be a
  standard WAV or FLAC.

Save as a WAV file, e.g. `my_dictation.wav`.

### Step 2: Run the ensemble evaluation with a gold reference

If you know what you said (or have a typed version of the dictation), provide it
as the gold reference. This lets the tool compute Word Error Rate (WER) so you
can quantify transcription accuracy.

```bash
uv run transcribe-radrx evaluate \
    --audio my_dictation.wav \
    --reference "Prescribed dose of 54 Gy in 30 fractions to the PTV with IMRT." \
    --output my_review.docx
```

What this produces:

- The ensemble transcription printed to the terminal.
- WER and UWR (Unresolved Word Rate) printed to the terminal.
- `my_review.docx` — a Word document with flagged words highlighted and margin
  comments showing what each backend heard.

### Step 3: Run without a gold reference (transcription + review document only)

If you are transcribing a new recording with no reference to compare against,
omit `--reference`:

```bash
uv run transcribe-radrx evaluate \
    --audio my_dictation.wav \
    --output my_review.docx
```

This produces the ensemble transcription and the review document without WER/UWR
metrics.

### Step 4: Review the Word document

Open `my_review.docx` in Microsoft Word or LibreOffice Writer.

- Words the ensemble resolved automatically appear as normal text.
- Words the ensemble could not confidently resolve are flagged with `[REVIEW]`
  and shown as Track Changes. The margin comment shows what Whisper and Voxtral
  each heard.
- Use Word's Accept/Reject Changes buttons to resolve each flagged word.

### Optional: generate an audit document

To see every ensemble decision (not just the uncertain ones) as Track Changes,
add `--audit-output`:

```bash
uv run transcribe-radrx evaluate \
    --audio my_dictation.wav \
    --reference "Prescribed dose of 54 Gy in 30 fractions." \
    --output my_review.docx \
    --audit-output my_audit.docx
```

`my_audit.docx` shows every word decision with full provenance. Useful for
research or quality assurance review.

### Using a reference file instead of inline text

For longer reference texts, write the gold standard to a file:

```bash
echo "Prescribed 54 Gy in 30 fractions to the PTV." > gold.txt

uv run transcribe-radrx evaluate \
    --audio my_dictation.wav \
    --reference-file gold.txt \
    --output my_review.docx
```

---

## Running the Full Bake-off (for developers and researchers)

The research bake-off runs multiple ASR backends side by side on a synthetic
corpus of RT dictation fixtures, with multiple TTS voices, and produces
aggregate WER and term recall statistics.

```bash
# Install all extras (bake-off needs Piper, MUSAN, and all backends)
make install-dev

# Run the 2-backend bake-off on dense RT fixtures
make bakeoff-dense

# Generate review + audit .docx files from the last run
make ensemble-demo
```

See `README.md` and `ROADMAP.md` for the full research workflow and fixture
generation details.

---

## Troubleshooting

**"No module named 'mlx'" on Linux or Intel Mac**
You accidentally installed the `asr-whisper-mlx` extra. Re-run `uv sync`
without that extra (see Quick Start for your platform).

**"No module named 'transformers'" when running evaluate**
Install the `asr-voxtral` extra: `uv sync --extra asr-voxtral`.

**Piper not found / bake-off fails at TTS synthesis**
Set `PIPER_BIN` in `env.sh` to the full path of the piper binary and
`source env.sh` before running. For the `piper-tts` Python package, ensure
it is installed in the project's virtual environment (`uv add piper-tts`).

**Model download is slow or times out**
HuggingFace Hub caches models in `~/.cache/huggingface/`. Check available
disk space (`df -h ~/.cache`). Downloads resume automatically if interrupted —
just re-run the command.

**"Audio file not found" error**
Make sure the path to your WAV file is correct and the file exists. Use the
full absolute path if in doubt: `--audio /Users/yourname/Documents/my_dictation.wav`.

**WER is unexpectedly high**
Check that your reference text uses the same terminology as the dictation. The
ensemble corrector knows RT vocabulary (`Gy`, `PTV`, `IMRT`, etc.) but unusual
abbreviations may need to be added to the vocabulary file
(`data/rt_vocabulary.txt`).
