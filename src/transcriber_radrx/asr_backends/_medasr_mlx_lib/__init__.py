"""Vendored MedASR MLX inference library.

Source: https://github.com/ainergiz/medasr-mlx
Upstream commit: cace86c397b4 (2026-02-11)
License: Apache-2.0
Author: ainergiz (see upstream repo)

This is a verbatim copy of the minimal inference modules from the
ainergiz/medasr-mlx repository, vendored into transcriber-radrx because
the upstream repo is not published to PyPI and has no stable packaging
layout. Vendoring ensures reproducible validation runs independent of
upstream changes.

If upstream changes, update VENDOR.md with the new commit and re-fetch:
    COMMIT=<new_sha>
    for f in model.py decode.py audio_utils.py; do
        curl -sf "https://raw.githubusercontent.com/ainergiz/medasr-mlx/${COMMIT}/$f" \\
            -o "src/transcriber_radrx/asr_backends/_medasr_mlx_lib/$f"
    done

Do NOT modify the vendored files in place — re-fetch instead, so the
diff from upstream is always trivial to audit.
"""

from __future__ import annotations

from transcriber_radrx.asr_backends._medasr_mlx_lib.audio_utils import load_audio_mono
from transcriber_radrx.asr_backends._medasr_mlx_lib.decode import (
    CTCTextDecoder,
    DecoderConfig,
)
from transcriber_radrx.asr_backends._medasr_mlx_lib.model import (
    LasrForCTC,
    load_mlx_model,
)

__all__ = [
    "CTCTextDecoder",
    "DecoderConfig",
    "LasrForCTC",
    "load_audio_mono",
    "load_mlx_model",
]
