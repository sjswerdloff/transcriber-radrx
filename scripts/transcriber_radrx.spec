# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for transcriber-radrx GUI (compare-only bundle).

Bundles the PySide6 GUI with the compare subcommand — text-only
transcription comparison, correction pipeline, WER/UWR computation,
and .docx review output. No ASR backends, no torch, no model downloads.

Usage:
    pyinstaller scripts/transcriber_radrx.spec

Output:
    dist/transcriber-radrx/transcriber-radrx.exe  (--onedir mode)

Authors: silas-397300f6
"""

import sys
from pathlib import Path

block_cipher = None

# Repository root (spec file is in scripts/)
repo_root = Path(SPECPATH).parent

# Data files to bundle
datas = [
    # RT vocabulary file — needed for corrections and term recall
    (str(repo_root / "data" / "rt_vocabulary.txt"), "data"),
    # Sample demo files for self-test
    (str(repo_root / "docs" / "demo" / "sample_gold_standard.txt"), "docs/demo"),
    (str(repo_root / "docs" / "demo" / "sample_dictation.txt"), "docs/demo"),
]

# Exclude heavy packages that aren't needed for compare-only
excludes = [
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "mlx",
    "mlx_whisper",
    "soundfile",
    "librosa",
    "scipy",
    "numpy",
    "pyroomacoustics",
    "peft",
    "huggingface_hub",
    # Test frameworks
    "pytest",
    "mypy",
    "ruff",
    "pre_commit",
]

a = Analysis(
    [str(repo_root / "src" / "transcriber_radrx" / "gui.py")],
    pathex=[str(repo_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "transcriber_radrx.cli",
        "transcriber_radrx.corrector",
        "transcriber_radrx.phrase_corrector",
        "transcriber_radrx.ensemble",
        "transcriber_radrx.ensemble.aligner",
        "transcriber_radrx.ensemble.decision_rules",
        "transcriber_radrx.ensemble.docx_renderer",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="transcriber-radrx",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,  # TODO: add an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="transcriber-radrx",
)
