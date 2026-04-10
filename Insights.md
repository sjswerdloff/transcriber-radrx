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
