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
