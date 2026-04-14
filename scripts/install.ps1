# install.ps1 — One-step installation for transcriber-radrx on Windows
#
# Usage (from the repository root in PowerShell):
#   .\scripts\install.ps1
#
# What this does:
#   1. Checks for Python 3.11+ and uv
#   2. Offers to install missing prerequisites via winget
#   3. Installs the project with appropriate ASR backend extras
#   4. Verifies the installation works
#
# This installs what's needed for the clinician workflow:
#   transcribe-radrx evaluate --audio dictation.wav --output review.docx
#
# Note: MLX Whisper is Apple Silicon only and is not available on Windows.
# The Whisper backend will use the torch-based implementation instead.
#
# Authors: silas-397300f6

$ErrorActionPreference = "Stop"

function Write-Info { Write-Host "[install] " -ForegroundColor Green -NoNewline; Write-Host $args }
function Write-Warn { Write-Host "[install] " -ForegroundColor Yellow -NoNewline; Write-Host $args }
function Write-Err  { Write-Host "[install] " -ForegroundColor Red -NoNewline; Write-Host $args }

# ---------------------------------------------------------------------------
# Check Python 3.11+
# ---------------------------------------------------------------------------

function Test-Python {
    try {
        $version = & python --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                Write-Info "Python $major.$minor found."
                return $true
            }
            Write-Warn "Python $major.$minor found, but 3.11+ is required."
            return $false
        }
    } catch {
        return $false
    }
    return $false
}

# ---------------------------------------------------------------------------
# Check uv
# ---------------------------------------------------------------------------

function Test-Uv {
    try {
        $null = & uv --version 2>&1
        Write-Info "uv found."
        return $true
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------
# Install missing prerequisites
# ---------------------------------------------------------------------------

function Install-Prerequisites {
    $needPython = -not (Test-Python)
    $needUv = -not (Test-Uv)

    if (-not $needPython -and -not $needUv) {
        return
    }

    # Check for winget
    $hasWinget = $false
    try {
        $null = & winget --version 2>&1
        $hasWinget = $true
    } catch {}

    if ($needPython) {
        if ($hasWinget) {
            Write-Info "Installing Python 3.12 via winget..."
            & winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            # Refresh PATH
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        } else {
            Write-Err "Python 3.11+ is required. Install from https://www.python.org/downloads/"
            Write-Err "Or install winget (App Installer from Microsoft Store) for automatic installation."
            exit 1
        }
    }

    if ($needUv) {
        Write-Info "Installing uv via official installer..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";$env:USERPROFILE\.local\bin"
        if (-not (Test-Uv)) {
            Write-Err "uv installation failed. Install manually: https://docs.astral.sh/uv/"
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# Install the project
# ---------------------------------------------------------------------------

function Install-Project {
    $extras = @(
        "--extra", "dev",
        "--extra", "phonetic",
        "--extra", "audio",
        "--extra", "validation",
        "--extra", "asr-voxtral"
    )

    Write-Info "Windows detected - MLX Whisper excluded (Apple Silicon only)."
    Write-Info "Installing transcriber-radrx..."

    & uv sync @extras

    Write-Info "Installation complete."
}

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

function Test-Installation {
    Write-Info "Verifying installation..."
    try {
        $null = & uv run transcribe-radrx evaluate --help 2>&1
        Write-Info "Verification passed."
    } catch {
        Write-Err "Verification failed. Check the output above for errors."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Info "transcriber-radrx installer (Windows)"
Write-Host ""

Install-Prerequisites
Install-Project
Test-Installation

Write-Host ""
Write-Info "Ready to go. Try:"
Write-Info ""
Write-Info "  uv run transcribe-radrx evaluate --audio your_dictation.wav --output review.docx"
Write-Info ""
Write-Info "For the developer/validation workflow, see INSTALLING.md."
