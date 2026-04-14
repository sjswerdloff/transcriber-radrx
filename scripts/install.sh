#!/usr/bin/env bash
# install.sh — One-step installation for transcriber-radrx
#
# Usage:
#   make install          # via Makefile (recommended)
#   bash scripts/install.sh   # directly
#
# What this does:
#   1. Checks for Python 3.11+ and uv
#   2. On macOS: offers to install missing prerequisites via Homebrew
#   3. Installs the project with platform-appropriate ASR backend extras
#   4. Verifies the installation works
#
# This installs what's needed for the clinician workflow:
#   transcribe-radrx evaluate --audio dictation.wav --output review.docx
#
# For the developer/validation bake-off workflow (piper TTS, MUSAN noise,
# multi-voice panels), use `make install-dev` instead.
#
# Authors: silas-397300f6

set -euo pipefail

# Colours (if terminal supports them)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Colour
else
    RED='' GREEN='' YELLOW='' NC=''
fi

info()  { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[install]${NC} $*"; }
error() { echo -e "${RED}[install]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"

case "$PLATFORM" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    *)      error "Unsupported platform: $PLATFORM. Use install.ps1 for Windows."; exit 1 ;;
esac

info "Platform: $OS ($ARCH)"

# ---------------------------------------------------------------------------
# Check Python 3.11+
# ---------------------------------------------------------------------------

check_python() {
    if ! command -v python3 &>/dev/null; then
        return 1
    fi
    local version
    version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local major minor
    major="${version%%.*}"
    minor="${version##*.}"
    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 11 ]; }; then
        warn "Python $version found, but 3.11+ is required."
        return 1
    fi
    info "Python $version found."
    return 0
}

# ---------------------------------------------------------------------------
# Check uv
# ---------------------------------------------------------------------------

check_uv() {
    if command -v uv &>/dev/null; then
        info "uv found: $(uv --version)"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# macOS: install via Homebrew
# ---------------------------------------------------------------------------

macos_install_prereqs() {
    local need_python=false
    local need_uv=false

    check_python || need_python=true
    check_uv || need_uv=true

    if [ "$need_python" = false ] && [ "$need_uv" = false ]; then
        return 0
    fi

    # Check for Homebrew
    if ! command -v brew &>/dev/null; then
        error "Homebrew is not installed. Install it from https://brew.sh and re-run."
        error "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    if [ "$need_python" = true ]; then
        info "Installing Python 3.12 via Homebrew..."
        brew install python@3.12
    fi

    if [ "$need_uv" = true ]; then
        info "Installing uv via Homebrew..."
        brew install uv
    fi
}

# ---------------------------------------------------------------------------
# Linux: check prerequisites (don't auto-install system packages)
# ---------------------------------------------------------------------------

linux_check_prereqs() {
    local ok=true

    if ! check_python; then
        error "Python 3.11+ is required. Install via your package manager:"
        error "  Ubuntu/Debian: sudo apt install python3.12"
        error "  Fedora: sudo dnf install python3.12"
        error "  Arch: sudo pacman -S python"
        ok=false
    fi

    if ! check_uv; then
        warn "uv not found. Installing via official installer..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Reload PATH
        export PATH="$HOME/.local/bin:$PATH"
        if ! check_uv; then
            error "uv installation failed. Install manually: https://docs.astral.sh/uv/"
            ok=false
        fi
    fi

    if [ "$ok" = false ]; then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Install the project
# ---------------------------------------------------------------------------

install_project() {
    local extras="--extra dev --extra phonetic --extra audio --extra validation"

    # ASR backends (cross-platform)
    extras="$extras --extra asr-voxtral"

    # Apple Silicon only: add MLX Whisper
    if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        extras="$extras --extra asr-whisper-mlx"
        info "Apple Silicon detected — including MLX Whisper backend."
    else
        info "Non-Apple-Silicon — MLX Whisper excluded (Voxtral + torch-Whisper available)."
    fi

    info "Installing transcriber-radrx with extras: $extras"
    # shellcheck disable=SC2086
    uv sync $extras

    info "Installation complete."
}

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

verify_install() {
    info "Verifying installation..."
    if uv run transcribe-radrx evaluate --help &>/dev/null; then
        info "Verification passed: 'transcribe-radrx evaluate --help' works."
    else
        error "Verification failed: 'transcribe-radrx evaluate --help' did not work."
        error "Check the output above for errors."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

info "transcriber-radrx installer"
echo ""

case "$OS" in
    macos) macos_install_prereqs ;;
    linux) linux_check_prereqs ;;
esac

install_project
verify_install

echo ""
info "Ready to go. Try:"
info ""
info "  transcribe-radrx evaluate --audio your_dictation.wav --output review.docx"
info ""
info "For the developer/validation workflow, run 'make install-dev' instead."
