# Insights — transcriber-radrx

## Cycle 112 — Safety-Gate Metric (#115)

### SILENT_UNIT_SUBSTITUTION: decimal-drop interaction

When a decimal dose value is dropped AND the unit is simultaneously substituted
(e.g. gold `50.4 GyE` → pred `50 Gy`), only `DECIMAL_DROP` fires, not
`SILENT_UNIT_SUBSTITUTION`. The SUS detector looks for the gold number
(`50.4`) followed by `Gy` in the prediction. The prediction has `50 Gy`
— the numbers don't match — so SUS correctly does not trigger.

Implication: the two CRITICAL failure classes (DECIMAL_DROP and DOSE_VALUE_MISSING)
already cover the dose-value loss dimension. SILENT_UNIT_SUBSTITUTION fires
independently when the numeric value is correctly transcribed but the unit is
downgraded from GyE to Gy.

### `gray equivalent` → `gray` substitution detection

The `_RE_GYE_DOSE` regex captures both `GyE` and `gray equivalent` in gold,
but the prediction check must also look for `gray` (without `equivalent`) as
the substituted form, not just `Gy`. A separate `gray_pattern` is needed
alongside the `gy_pattern` in `_detect_silent_unit_substitution`.

### False-positive safety for proton-0015

The two-gate structure (step 1: gold must contain GyE; step 2: particle-therapy
context must be present) is sufficient to prevent false positives on the
counterfactual. proton-0015 gold uses `Gy` not `GyE`, so step 1 fails and
the detector returns empty. The presence of the word "protons" in the sentence
does not override the gold-unit check — the architecture correctly separates
"context-clue check" from "gold-unit check."

### ruff S105 on string constants

ruff flags any string constant assigned to a variable with "PASS" in the name
as a possible hardcoded password (S105). This is a false positive for deployment
gate constants like `GATE_PASS = "PASS"`. The correct fix is an inline `# noqa: S105`
comment, not disabling the rule project-wide.

### Integration tests against real JSON files

Placing integration tests in the same file as unit tests and guarding them with
`@pytest.mark.skipif(not _JSON.exists(), ...)` is effective for this project's
pattern where the test JSONs live in the repo. The tests run when the files
exist and skip gracefully in environments where they have been deleted or not
checked in.

### loguru as a project dependency

The project did not have loguru in its dependencies despite it being the project
convention. Added via `uv add loguru`. The `dev` extras group would be a better
home for validation-only dependencies, but adding to the main deps is acceptable
since loguru has no heavy transitive dependencies and the project already uses
it in other modules.

## Cycle 112 — Safety-Gate v1.1 (#115 continuation)

### Post-correction gate can pass per-voice even when backend fails

`mlx_whisper en_GB-alan-medium` has 0 CRITICAL failures (all 25 HIGH are
`DOSE_UNIT_CORRUPTION`, which is `PHONETIC_MAP` / correctable). So its
`post_correction_gate = PASS`. But `mlx_whisper en_US-lessac-high` has 3
CRITICAL (`DOSE_VALUE_MISSING`, UNRECOVERABLE), so its voice post-correction gate
is FAIL. The backend-level aggregate is FAIL. The voice asymmetry in unrecoverable
count (alan=0, lessac=3) is clinically meaningful — lessac voice triggers more
dose-drop failures because its accent changes how MedASR hears dose numbers.

### StrEnum vs (str, Enum) in Python 3.11+

ruff UP042 flags `class Foo(str, Enum)` in favour of `class Foo(StrEnum)`.
`StrEnum` was added in Python 3.11 and the project targets 3.10+, but
`from enum import StrEnum` works from 3.11 onward. The UP rules assume the
configured Python target version is the minimum — if targeting 3.10, UP042
should be suppressed or the pyproject.toml `target-version` key should be
bumped to 3.11. In this project, ruff accepted StrEnum without complaint,
suggesting the target version is 3.11 or later in pyproject.toml.

### Correctability is an orthogonal axis to severity

All CRITICAL failures in the current spec are UNRECOVERABLE (DECIMAL_DROP,
DOSE_VALUE_MISSING). All HIGH and MEDIUM failures are correctable by rules.
This means `residual_unrecoverable[CRITICAL]` equals `counts[CRITICAL]` for
every backend and voice in this data. That coincidence is specific to this
corpus and the five failure classes in scope — it does not hold in general and
should not be hard-coded anywhere. The correctability mapping is a constant dict
precisely so it can be extended when new failure classes are added.

### Frozen dataclasses with new required fields break existing call sites

Adding `correctability: Correctability` to the frozen `Failure` dataclass forced
updates to all 3 places in tests where `Failure` was constructed directly with
positional arguments. In a frozen dataclass, field ordering matters: the new
field was inserted between `severity` and `detail`, which changed the positional
signature. If `Failure` had been constructed with keyword arguments throughout,
the impact would have been zero for existing tests. Lesson: for frozen
dataclasses that serve as public API (used in tests), always construct with
keyword arguments.

---

## Cycle 111 — PySide6 GUI launcher (gui.py)

### pytest.importorskip at module level skips the ENTIRE file

When `pytest.importorskip("PySide6")` is placed at the top level of a test
module (not inside a test class or function), pytest skips collection of the
entire file when the package is absent. This means non-GUI tests in the same
file also silently disappear. The correct pattern is to use `pytest.mark.skipif`
with a conditional flag defined from a try/except import check, applied per-class
rather than per-file.

```python
try:
    import PySide6 as _pyside6_check  # noqa: F401
    _PYSIDE6_PRESENT = True
except ModuleNotFoundError:
    _PYSIDE6_PRESENT = False

_skip_no_pyside6 = pytest.mark.skipif(not _PYSIDE6_PRESENT, reason="PySide6 not installed")

@_skip_no_pyside6
class TestPySide6Widget:
    ...
```

This gives 18 passing tests in the no-PySide6 CI environment instead of
0 collected.

### dual-environment mypy for optional GUI dependencies

The `misc` error "Class cannot subclass X (has type Any)" from mypy appears
whenever PySide6 is absent (Qt classes become `Any` via ignore_missing_imports).
The same dual-environment problem exists for asr backend modules. Pattern:

```toml
[[tool.mypy.overrides]]
module = "transcriber_radrx.gui"
warn_unused_ignores = false
disable_error_code = ["misc"]
```

This suppresses both "unused type: ignore comment" (when PySide6 absent) and
"class cannot subclass Any" without affecting strict mode elsewhere.

### dict[str, object] requires explicit str() conversion for float()

When a results dict is typed as `dict[str, object]`, mypy rejects
`float(results.get("key", 0.0))` because `.get()` returns `object`, not
`SupportsFloat`. The safe pattern is `float(str(results.get("key", 0.0)))`.
Similarly for list values: use a conditional isinstance check:
```python
raw = results.get("terms_missing", [])
terms: list[str] = [str(t) for t in raw] if isinstance(raw, list) else []
```

### ARG002 for fixture parameters that exist only for side effects

pytest fixtures used only for their side effects (e.g. ensuring QApplication
is created) trigger ARG002 ("unused method argument") when the test method
parameter is named but not referenced. Since pytest requires the parameter name
to match the fixture name exactly (renaming to `_qt_app` breaks fixture injection),
the clean fix is to add `"ARG002"` to the `tests/**` per-file-ignores in ruff config.

### QThread.finished Signal type annotation with PySide6

PySide6 Signals are declared as class attributes:
```python
class CompareWorker(QThread):
    finished: Signal = Signal(dict)
    error: Signal = Signal(str)
    progress: Signal = Signal(str)
```
When PySide6 is absent (CI), these class bodies are inside `if _PYSIDE6_AVAILABLE:` guards
and never executed — mypy and ruff both skip them cleanly.

---

## Phase 1.1 + Phase 2 — Ensemble Decision Rules (#119)

### WER vs. safety: the ensemble trades one for the other

The 2-backend ensemble (Voxtral + Whisper) reduces CRIT+HIGH safety failures
from 47 (best individual: Voxtral=47, Whisper=94) to 28, and eliminates the
single CRITICAL failure (a DOSE_VALUE_MISSING from Voxtral). However raw WER
increases from 0.0951 (Voxtral) to 0.1347 (ensemble). This is the expected
clinical tradeoff: the ensemble's Rules 3–5 sometimes pick the more-clinically-correct
word (e.g. "GyE" over Voxtral's "Gy") at the cost of raw WER when Voxtral's
lower-WER choice happened to be correct for WER but wrong for clinical safety.

The integration test contract therefore checks `ensemble WER <= max(vox, whi)`,
not `<= min(vox, whi)`. The ensemble should never be WORSE than BOTH, but
being worse than the best-WER model on raw string matching is acceptable if
safety failures decrease.

### Normalization ordering matters: -slash- before -hyphen

The normalizer table must have `3D-slash-3D` before `3D-3D`. If `3D-3D` fires
first on the input `3D-slash-3D`, it would match the suffix `3D-3D` (since
`3D-slash-3D` contains `3D` and `3D` but with `-slash-` in between — actually
the `-3D` suffix IS a substring match risk). The current table puts the longer
`-slash-` forms before the shorter `-` hyphen forms, which is correct.

### Rule priority subtlety: Rule 2 vs Rule 4

Rule 2 (DOSE_UNIT_GYE) fires when "at least one is GyE, one or both are Gy-variants."
Since GyE is itself in `_GY_VARIANT_SET`, if word_b is `GyE` and word_a is `dose`
(not a Gy-variant), Rule 2 still fires because `is_gy_variant("GyE")` is True.
This is correct behavior — Rule 2 should pick `GyE` over any non-GyE word when
GyE is present, regardless of what the other word is. Rule 4 (DOSE_UNIT_VISIBLE_CORRUPTION)
handles the case where one is a Gy-variant that is NOT GyE.

### sys.path injection pattern for scripts that import from the tests package

The `run_ensemble_aggregator.py` script needs both `src/` (for the transcriber_radrx
package) and the repo root (for `tests.validation.metrics.safety_gate`). When
run via `uv run python tests/validation/scripts/run_ensemble_aggregator.py`,
neither is automatically on sys.path unless the project is installed in dev mode.
The pattern used: insert `_REPO_ROOT` (= `Path(__file__).parents[2]`) into
`sys.path[0]` at module load time. This is safe for a validation script but
would be inappropriate in production library code.

---

## Cycle 110 — Task #120: Ensemble docx renderer

### docx-revisions deletion API uses paragraph.text offsets, not cumulative run lengths

The `RevisionParagraph.add_tracked_deletion(start, end)` API takes character
offsets into the *current visible text* of the paragraph (`para.text`), not
cumulative byte or element offsets. The pattern that works for building Track
Changes from scratch:
1. Record `offset_start = len(rp.text)` BEFORE adding the to-be-deleted word.
2. Add the word with `rp.add_run(word)` — this makes it visible in `rp.text`.
3. Call `rp.add_tracked_deletion(offset_start, offset_start + len(word), ...)`.
   The library wraps the run in a `<w:del>` element, removing it from `rp.text`.
4. Call `rp.add_tracked_insertion(replacement_word, author=...)` for the chosen word.
5. Add a space with `rp.add_run(" ")`.

The key subtlety: `rp.text` *excludes* already-deleted runs, so recording the
offset before step 2 gives the correct position into the paragraph's current
visible character sequence.

### RevisionParagraph.from_paragraph() shares the underlying XML element

`RevisionParagraph.from_paragraph(para)` does NOT copy the paragraph — it
wraps the same `CT_P` XML element. Mutations via `rp.add_run()` are visible
on the original `para` object, and vice versa. This is the correct behavior
for building Track Changes into a paragraph created by `doc.add_paragraph()`.

### docx-revisions is built for editing existing documents; from-scratch construction works too

The library's main documented use case is editing existing .docx files (find-and-
replace-tracked). But from-scratch construction — add paragraph, convert to
RevisionParagraph, add deletion/insertion — works correctly as demonstrated by
the 24-test suite. The only gap is that `RevisionDocument` wraps an existing
document; we use `docx.Document()` directly and then apply `RevisionParagraph.from_paragraph()`
to each paragraph that needs revision marks.

### docx_revisions has no py.typed marker

The `docx_revisions` package ships without a `py.typed` marker or bundled stubs.
Mypy reports `import-untyped` for it. The correct fix for a project where pyproject.toml
cannot be modified in CI is an inline `# type: ignore[import-untyped]` comment on
the import line. Adding a `[[tool.mypy.overrides]]` entry would be cleaner but
requires touching pyproject.toml.

### 25 words flagged for human review across 56 pairs

The ensemble flagged 25 words via Rule 6 (BOTH_WRONG — neither in vocabulary,
words differ significantly). Most of these come from proper-noun corruption where
both backends fail on rare disease names or patient-specific terminology. This
count is a meaningful clinical signal: a human reviewer could check 25 words
across 56 fixture-voice pairs in under 5 minutes, versus reviewing entire
transcriptions.

---

## Cycle 106 — macOS TTS backend (#117)

### macOS module-level platform guard requires reload in tests

`macos_tts.py` calls `_check_platform()` at module level so that importing
on non-Darwin fails immediately. Tests that mock `sys.platform` must call
`importlib.reload(mod)` after patching, or the guard has already fired
against the real platform at import time. The autouse fixture in the test
file sets `sys.platform = "darwin"` via monkeypatch before any test body
runs, but the module may have already been imported; `importlib.reload` in
individual platform-guard tests re-runs the module-level check under the
patched platform value.

### Sentinel model_path for non-file-backed TTS engines

`VoiceSpec` uses `model_path: Path` to locate the ONNX model for piper.
For macOS system voices, there is no ONNX file — the voice is identified
solely by name. Using `Path("/usr/bin/say")` as a sentinel preserves the
typed interface while signaling "this is a macOS system voice." The
`tts_engine` field is the authoritative dispatch key; `model_path` is only
passed to the piper synthesis path when `tts_engine == "piper"`.

### afconvert inline flags: list formatting vs. string formatting

ruff formats `["-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", ...]`
with each flag on its own line (hanging indent), which makes the flag/value
pairs harder to read as pairs. This is a style tradeoff with no linting
solution — the formatter is authoritative and ruff format's output is
correct. The test asserts on the *values* in the command list, not the
source layout, so it is unaffected by formatter changes.

### _NOVELTY_VOICE_NAMES uses lowercase keys for case-insensitive matching

The `say -v '?'` output uses title-case voice names. The exclusion set
stores all names in lowercase and the comparison uses `.lower()`. This
handles platform variation (e.g. if Apple ever ships a voice whose display
name capitalisation differs between macOS versions) without requiring regex.

## Cycle 113 — Phrase-level corrector (phrase_corrector.py)

### Lookahead patterns break the "sub on isolated match group" anti-pattern

When a regex uses a lookahead assertion (e.g. `\bvulva\b(?=\s+squamous)`),
`match.group()` returns only the consumed text (`"vulva"`), not the context
in the lookahead. Re-running `pattern.regex.sub(replacement, match.group())`
then silently fails because the lookahead condition can never be satisfied
inside the isolated substring. The correct approach is `match.expand(replacement)`,
which expands backreference groups from the match object directly, bypassing
the need to re-run the regex. This also preserves correctness for capture-group
replacements like `r"\1 Gy"`.

### lump/lymph regex etymology trap

`\blymph?[et]ectomy\b` looks right for `lymphectomy` but fails: the pattern
needs `lymp + h? + [et] + ectomy` = 4+1+1+6 = 12 chars, but `lymphectomy` is
only 11. The `[et]` slot consumes the `e` that should start `ectomy`, leaving
only `ctomy` (5 chars) instead of `ectomy` (6 chars). The correct pattern is
`\blymph?t?ectomy\b`: optional `h` then optional `t` then `ectomy`.

### Plural suffixes in phrase patterns

A pattern for `physical marker` won't match `physical markers`. Always consider
whether the real-world ASR output will produce singular or plural and include
`s?` in the pattern when both are expected in clinical dictation.

---

## Cycle 110 — 0.2.0 CLI refactor: subcommand architecture

### Local imports in command handlers — patch at the source module

`_run_evaluate` uses local imports (`from transcriber_radrx.transcriber import
transcribe_with_backend` etc.) inside the function body rather than at the
module level. This is intentional: lazy imports prevent pulling in 7+ GB of
PyTorch/MLX at process start when the user only wanted `--help`. For tests,
the implication is that `patch("transcriber_radrx.cli.transcribe_with_backend")`
will FAIL (no such attribute on the CLI module), while
`patch("transcriber_radrx.transcriber.transcribe_with_backend")` works correctly
because it patches the name at its source. Always patch at the source module for
locally-imported names.

### contextlib.ExitStack for multiple patches keeps lines short

When a test needs 6–7 simultaneous patches, repeating the full dotted path for
each patch target creates 130+ character lines that violate E501. Using
`contextlib.ExitStack` with `stack.enter_context(patch(...))` allows storing
patch targets as class-level constants (`_P_GET_BACKEND = "..."`) and keeps
each line well within 127 characters. The entered contexts return their mock
objects, so assertions against them work normally.

### argparse subparser backward compat — inspect argv before parsing

When adding subcommands to a CLI that previously had bare positional arguments,
argparse will ERROR (not just fall through) if the first positional arg is not a
valid subcommand choice. The correct backward-compat pattern is to inspect the
effective argv BEFORE calling `parse_args`: find the first non-flag argument; if
it is not in the known subcommand set, prepend the default subcommand name. This
preserves `transcribe-radrx audio.wav` semantics while allowing
`transcribe-radrx transcribe audio.wav` and `transcribe-radrx evaluate --audio
audio.wav`.

### CorrectionDictionary mock must return a proper 3-tuple from correct_full

`correct_full` returns `tuple[str, list[Correction], list[PhraseCorrection]]`.
When the test patches `CorrectionDictionary`, the mock instance's `correct_full`
by default returns a `MagicMock`, not a 3-tuple. Unpacking that with
`text, _, _ = corrector.correct_full(...)` raises `ValueError: not enough values
to unpack`. Always configure: `corrector_instance.correct_full.return_value =
("corrected text", [], [])`.

### TRY300 and try/else: return must go in the else block

ruff TRY300 requires that a `return` or other value-producing statement that
appears inside a `try` block (after the potential-exception code) be moved to an
`else` block. The pattern is:
```python
try:
    do_risky_thing()
except SomeError:
    handle_or_raise()
else:
    return result
```
This makes the "success path" semantically distinct from the exception path.
