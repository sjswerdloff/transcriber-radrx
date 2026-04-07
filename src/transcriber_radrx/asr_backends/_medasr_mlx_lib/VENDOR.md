# Vendored: medasr-mlx

| Field | Value |
|-------|-------|
| **Source** | https://github.com/ainergiz/medasr-mlx |
| **Upstream commit** | `cace86c397b4` |
| **Upstream date** | 2026-02-11 |
| **License** | Apache-2.0 (declared in upstream `pyproject.toml`) |
| **Author** | ainergiz |
| **Vendored by** | silas-397300f6 |
| **Vendored date** | 2026-04-07 |

## Why vendored

The upstream repository is not published to PyPI and does not have a
clean package layout. Vendoring the minimal inference modules:

1. Ensures reproducible validation runs independent of upstream drift
2. Avoids relying on a git dependency with no published release
3. Keeps our Apple-Silicon-specific dependencies minimal and auditable
4. Is permitted under the Apache-2.0 license with attribution (this file)

## Files vendored

| File | Size | Purpose |
|------|-----:|---------|
| `model.py` | 29 KB | MLX Conformer-CTC model definition + loader |
| `decode.py` | 4 KB | CTC text decoder with greedy and KenLM beam modes |
| `audio_utils.py` | 1 KB | WAV loading and resampling utilities |

Files intentionally **NOT** vendored:
- `transcribe_mlx.py` — CLI driver; we have our own runner
- `streaming.py` — streaming transcription; not needed for batch
- `benchmark.py` — upstream's benchmark harness; we have our own
- `convert.py`, `quantize.py` — PyTorch → MLX conversion utilities; we use pre-converted weights

## How to re-sync

If upstream fixes a bug or improves the model, update this file with the
new commit and re-fetch the three source files:

```bash
COMMIT=<new_sha>
cd src/transcriber_radrx/asr_backends/_medasr_mlx_lib
for f in model.py decode.py audio_utils.py; do
    curl -sf "https://raw.githubusercontent.com/ainergiz/medasr-mlx/${COMMIT}/$f" -o "$f"
done
```

Then update the commit SHA in this file and in `__init__.py`, and run
the test suite.

**Do not modify the vendored files in place.** If a local patch is
needed, document the deviation in this file and apply the patch via a
separate module, not by editing the vendored sources.

## Apache-2.0 attribution

The vendored files are copyright the original authors of medasr-mlx and
are used under the Apache License, Version 2.0. A copy of the license
applicable to these files can be found at:

https://www.apache.org/licenses/LICENSE-2.0

The upstream repository declares Apache-2.0 in its `pyproject.toml`:
https://github.com/ainergiz/medasr-mlx/blob/main/pyproject.toml
