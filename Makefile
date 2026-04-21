# transcriber-radrx Makefile
#
# Architecture-aware dependency management. Apple Silicon (arm64) gets
# all extras including mlx-whisper; Linux/Intel gets everything except
# mlx-whisper (which depends on mlx, Apple Silicon only).

ARCH := $(shell uname -m)

# Extras that work on all platforms
COMMON_EXTRAS := --extra dev --extra phonetic --extra audio --extra validation

# ASR backend extras that need torch/transformers (heavy but cross-platform)
ASR_EXTRAS := --extra asr-medasr --extra asr-cohere --extra asr-granite --extra asr-voxtral

# Apple Silicon only
ifeq ($(ARCH),arm64)
    PLATFORM_EXTRAS := --extra asr-whisper-mlx
else
    PLATFORM_EXTRAS :=
endif

ALL_EXTRAS := $(COMMON_EXTRAS) $(ASR_EXTRAS) $(PLATFORM_EXTRAS)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

.PHONY: install install-dev install-ci sync

## One-step install for clinicians (checks prerequisites, installs backends)
install:
	bash scripts/install.sh

## Install everything for local development (architecture-aware)
install-dev:
	uv sync $(ALL_EXTRAS)

## Install minimal deps for CI (lint + test, no heavy ASR backends)
install-ci:
	uv sync $(COMMON_EXTRAS)

## Install with GUI support (PySide6)
install-gui:
	uv sync $(ALL_EXTRAS) --extra gui

## Alias for install-dev
sync: install-dev

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: lint format type-check preflight

## Run ruff linter
lint:
	uv run ruff check src/ tests/

## Run ruff formatter
format:
	uv run ruff format src/ tests/

## Run mypy type checker
type-check:
	uv run mypy src/ tests/validation/audio_synthesis tests/validation/metrics

## Format + lint + type-check (run before committing)
preflight: format lint type-check

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test test-unit test-fast

## Run full test suite
test:
	uv run python -m pytest tests/ -q

## Run tests excluding slow integration tests
test-fast:
	uv run python -m pytest tests/ -q -m "not slow"

## Run only unit tests (no validation tests)
test-unit:
	uv run python -m pytest tests/ -q --ignore=tests/validation/

# ---------------------------------------------------------------------------
# Bake-off helpers
# ---------------------------------------------------------------------------

.PHONY: bakeoff-dense bakeoff-particle bakeoff-anatomy ensemble-demo

## Run the 2-backend bake-off on dense RT fixtures
bakeoff-dense:
	uv run python -m tests.validation.scripts.run_multi_backend_e2e \
		--backends mlx_whisper voxtral \
		--fixtures tests/validation/fixtures/rt_dictation_samples.jsonl \
		--dense-only --voices alan lessac

## Run the 2-backend bake-off on particle therapy fixtures
bakeoff-particle:
	uv run python -m tests.validation.scripts.run_multi_backend_e2e \
		--backends mlx_whisper voxtral \
		--fixtures tests/validation/fixtures/particle_samples.jsonl \
		--dense-only --voices alan lessac

## Run the 2-backend bake-off on anatomy fixtures
bakeoff-anatomy:
	uv run python -m tests.validation.scripts.run_multi_backend_e2e \
		--backends mlx_whisper voxtral \
		--fixtures tests/validation/fixtures/anatomy_samples.jsonl \
		--dense-only --voices alan lessac

## Generate .docx review documents (audit + review modes)
ensemble-demo:
	uv run python tests/validation/scripts/render_ensemble_docx_demo.py

## Example: run ensemble evaluation on a recording with a gold reference.
## Replace MY_DICTATION.wav and the --reference text with your own.
evaluate-example:
	uv run transcribe-radrx evaluate \
		--audio MY_DICTATION.wav \
		--reference "Prescribed dose of 54 Gy in 30 fractions to the PTV with IMRT." \
		--output my_review.docx

# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

.PHONY: info

## Show detected architecture and extras that will be installed
info:
	@echo "Architecture: $(ARCH)"
	@echo "Common extras: $(COMMON_EXTRAS)"
	@echo "ASR extras: $(ASR_EXTRAS)"
	@echo "Platform extras: $(PLATFORM_EXTRAS)"
	@echo ""
	@echo "Run 'make install-dev' to install all extras for this platform."
	@echo "Run 'make preflight' before committing."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help

## Show this help
help:
	@echo "transcriber-radrx Makefile targets:"
	@echo ""
	@echo "  install       One-step install for clinicians (macOS/Linux)"
	@echo "  install-dev   Install all deps for local development (arch-aware)"
	@echo "  install-ci    Install minimal deps for CI (no heavy ASR backends)"
	@echo "  sync          Alias for install-dev"
	@echo ""
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter"
	@echo "  type-check    Run mypy"
	@echo "  preflight     format + lint + type-check"
	@echo ""
	@echo "  test          Run full test suite"
	@echo "  test-fast     Run tests excluding slow markers"
	@echo "  test-unit     Run only unit tests"
	@echo ""
	@echo "  bakeoff-dense     Run bake-off on dense RT fixtures"
	@echo "  bakeoff-particle  Run bake-off on particle therapy fixtures"
	@echo "  bakeoff-anatomy   Run bake-off on anatomy fixtures"
	@echo "  ensemble-demo     Generate .docx review documents"
	@echo "  evaluate-example  Example: run evaluate on MY_DICTATION.wav"
	@echo ""
	@echo "  info          Show architecture and extras"
	@echo "  help          Show this help"
