"""Backend registry.

Maps short backend names to factory functions. Factories are lazy — they
import the backend module only when the backend is requested, so users
don't pay the dependency cost for backends they don't use.

Authors: silas-397300f6
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber_radrx.asr_backends.base import ASRBackend

# Factory type: takes keyword args and returns a loaded-ready backend
BackendFactory = Callable[..., "ASRBackend"]


def _make_mlx_whisper(**kwargs: object) -> ASRBackend:
    from transcriber_radrx.asr_backends.mlx_whisper import MlxWhisperBackend

    return MlxWhisperBackend(**kwargs)  # type: ignore[arg-type]


def _make_medasr(**kwargs: object) -> ASRBackend:
    from transcriber_radrx.asr_backends.medasr import MedASRBackend

    return MedASRBackend(**kwargs)  # type: ignore[arg-type]


def _make_cohere(**kwargs: object) -> ASRBackend:
    from transcriber_radrx.asr_backends.cohere import CohereBackend

    return CohereBackend(**kwargs)  # type: ignore[arg-type]


def _make_granite_speech(**kwargs: object) -> ASRBackend:
    from transcriber_radrx.asr_backends.granite import GraniteSpeechBackend

    return GraniteSpeechBackend(**kwargs)  # type: ignore[arg-type]


def _make_voxtral(**kwargs: object) -> ASRBackend:
    from transcriber_radrx.asr_backends.voxtral import VoxtralBackend

    return VoxtralBackend(**kwargs)  # type: ignore[arg-type]


BACKENDS: dict[str, BackendFactory] = {
    "mlx_whisper": _make_mlx_whisper,
    "medasr": _make_medasr,
    "cohere": _make_cohere,
    "granite_speech": _make_granite_speech,
    "voxtral": _make_voxtral,
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a new backend factory at runtime (primarily for tests)."""
    BACKENDS[name] = factory


def get_backend(name: str, **kwargs: object) -> ASRBackend:
    """Look up and instantiate a backend by name.

    Args:
        name: Backend short name (e.g. "mlx_whisper", "medasr").
        **kwargs: Backend-specific constructor arguments (model_id, etc.).

    Returns:
        An ASRBackend instance. Not yet loaded — call `.load()` to warm it.

    Raises:
        KeyError: If the backend name is not registered.
    """
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        msg = f"Unknown ASR backend: {name!r}. Available: {available}"
        raise KeyError(msg)
    return BACKENDS[name](**kwargs)
