#!/usr/bin/env bash
#
# env.example.sh — example environment variables for running the
# transcriber-radrx bake-off pipeline.
#
# Usage:
#
#   cp env.example.sh env.sh     # make a local copy for your machine
#   # edit env.sh to point at your actual piper-voices and piper binary
#   source env.sh                 # load the variables into your shell
#
# env.sh is gitignored so you can put machine-specific paths in it
# without committing them. The values below are the maintainer's
# working example for a macOS + pyenv + rhasspy/piper-voices setup
# and can be used verbatim on his machine; they are included as a
# concrete example of the expected layout for everyone else.
#
# All variables are optional — the bake-off runner will also resolve
# them from ./piper-voices and ~/piper-voices, and from 'piper' on
# $PATH, if nothing is set. See the "External dependencies" section
# of README.md for the full resolution order and installation
# instructions.

# Root of the rhasspy/piper-voices tree. The bake-off runner expects
# the HuggingFace layout: {root}/en/en_US/amy/medium/en_US-amy-medium.onnx
export PIPER_VOICES_ROOT="/Users/stuartswerdloff/PythonProjects/PiperTTS/piper-voices"

# Direct path to the piper binary. Point at the actual binary, not a
# pyenv shim — shims can resolve to a Python env that doesn't have
# piper installed under `uv run`, which surfaces as exit code 127.
export PIPER_BIN="/Users/stuartswerdloff/.pyenv/versions/piper311/bin/piper"

# Optional: MUSAN noise corpus location. Defaults to
# tests/validation/corpora/restricted/musan/noise/ relative to the
# repository root if not set.
#
# export MUSAN_NOISE_DIR="/path/to/musan/noise"
