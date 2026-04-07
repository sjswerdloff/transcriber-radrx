"""ASR backend package.

Backends are pluggable speech-to-text engines that share a common interface
(`ASRBackend`). Each backend is an optional extra in pyproject.toml so users
install only what they need.

Registry:
    from transcriber_radrx.asr_backends import get_backend
    backend = get_backend("mlx_whisper", model_id="mlx-community/whisper-large-v3")
    text = backend.transcribe_wav(Path("clip.wav"), initial_prompt="medical terms")

To add a new backend:
    1. Create a new module `transcriber_radrx/asr_backends/<name>.py`
    2. Implement a class that satisfies the ASRBackend protocol
    3. Register it in BACKENDS below
    4. Add an optional dependency group in pyproject.toml

Authors: silas-397300f6
"""

from __future__ import annotations

from transcriber_radrx.asr_backends.base import ASRBackend, ASRBackendError, UnsupportedFeatureError
from transcriber_radrx.asr_backends.registry import BACKENDS, get_backend, register_backend

__all__ = [
    "BACKENDS",
    "ASRBackend",
    "ASRBackendError",
    "UnsupportedFeatureError",
    "get_backend",
    "register_backend",
]
