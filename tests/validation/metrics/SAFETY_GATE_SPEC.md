# Safety-Gate Metric — Design Spec (task #115)

**Author:** silas-397300f6
**Status:** v1.0 implemented; **v1.1 delta below (pending implementation)**
**Motivating data:** `tests/validation/reports/bakeoff_proton_findings_2026-04-09.md`
**Depends on:** existing bake-off JSON output schema from `run_multi_backend_e2e.py`

---

## v1.1 delta (the authoritative change set since v1.0 landed)

**Driving feedback:** Stuart, 2026-04-09 — "Decimal place error is far more dangerous than physical vs effective dose error. A key question is how readily can they be corrected (it's clear that without safety gating and perhaps automatic correction that ASR is not safe for RT)."

v1.0 of the metric ranks backends by a single safety-adjusted WER that weights all CRITICAL failures equally and all HIGH failures equally. v1.1 adds a **correctability axis** orthogonal to severity, and a **second gate decision** that reflects how each backend would score after an ideal corrector pipeline is applied. It also bumps the CRITICAL weight to reflect Stuart's clinical calibration that decimal-drop errors are far more dangerous than unit-substitution errors.

### v1.1 change 1: `Correctability` tag on every `Failure`

Add a new field to the `Failure` dataclass:

```python
class Correctability(str, Enum):
    UNRECOVERABLE = "UNRECOVERABLE"     # Information is lost at the signal level. Corrector cannot recover.
    CONTEXT_RULE = "CONTEXT_RULE"       # Fixable by a rule that uses surrounding context.
    PHONETIC_MAP = "PHONETIC_MAP"       # Fixable by a known-bad-list phonetic mapping.
    ADJACENCY_RULE = "ADJACENCY_RULE"   # Fixable by a simple adjacency rule (e.g., slash restoration).

@dataclass(frozen=True)
class Failure:
    fixture_id: str
    class_: str
    severity: str
    correctability: Correctability    # NEW
    detail: str
    gold_value: str | None
    pred_value: str | None
```

**Fixed mapping from failure class → correctability:**

| Failure class | Correctability | Rationale |
|---|---|---|
| `DECIMAL_DROP` | `UNRECOVERABLE` | Decimal portion lost from the acoustic signal. Text-level correction cannot recover what the ASR didn't hear. |
| `DOSE_VALUE_MISSING` | `UNRECOVERABLE` | Number is gone from the prediction entirely. Corrector must not invent dose values; only option is flag for human review. |
| `SILENT_UNIT_SUBSTITUTION` | `CONTEXT_RULE` | Number is correct, only the `E` discriminator is lost. Context clues (proton, PBS, CSI, chordoma, medulloblastoma, etc.) promote `Gy → GyE` with high confidence. |
| `DOSE_UNIT_CORRUPTION` | `PHONETIC_MAP` | Known-bad renderings (`GiE`, `Jai E`, `JEE`, etc.) are ~95% reliably mapped back to `Gy` via the known-bad list. Then `SILENT_UNIT_SUBSTITUTION` logic applies if particle context is present. |
| `SLASHED_FORM_LOSS` | `ADJACENCY_RULE` | `3D 3D` → `3D/3D` when adjacent to IGRT vocabulary (image guidance, cone beam, CBCT, kV). Only punctuation lost; tokens intact. |

The mapping is a constant table in the module. The correctability tag is assigned at detection time, inside each class's detector function.

### v1.1 change 2: CRITICAL severity weight bumped 10 → 30

Updated severity weights for the safety-adjusted WER formula:

| Severity | v1.0 weight | **v1.1 weight** | Rationale |
|---|---:|---:|---|
| CRITICAL | 10 (5pp per-sample penalty) | **30 (15pp per-sample penalty)** | "Decimal place error is far more dangerous" — Stuart. Decimal drops and missing doses can be 10× clinical harm; earlier weight treated them as only 2× more severe than unit substitution, which understates the true danger ratio. |
| HIGH | 5 (2.5pp) | 5 (2.5pp) | Unchanged. |
| MEDIUM | 2 (1pp) | 2 (1pp) | Unchanged. |

**New formula (per voice):**
```
safety_adjusted_wer = raw_wer
                    + (CRITICAL_count * 0.15 + HIGH_count * 0.025 + MEDIUM_count * 0.01) / n_samples
```

The `/ n_samples` normalization is retained from v1.0 (per-voice penalty is averaged across samples so that backends with different sample counts are comparable).

**Expected impact on the cycle 112 data** (re-ranking unchanged but gaps widen):
- Voxtral: raw 0.1073 → adj ≈ 0.132 (was 0.130)
- Whisper: raw 0.1325 → adj ≈ 0.181 (was 0.176)
- MedASR: raw 0.1868 → adj ≈ 0.277 (was 0.254) ← gap increase because MedASR has 13 CRITICAL

MedASR falls further behind because the CRITICAL weight bump hits hardest on the backend with the most CRITICAL failures.

### v1.1 change 3: `post_correction_gate` — a second gate decision

The v1.0 gate decision considers all failures. v1.1 adds a second gate that assumes an ideal corrector has applied all four correctable classes (`CONTEXT_RULE`, `PHONETIC_MAP`, `ADJACENCY_RULE`) and evaluates the remaining `UNRECOVERABLE` failures only.

**Logic:**

```python
def raw_gate(failures: list[Failure]) -> GateDecision:
    """v1.0 gate: considers ALL failures."""
    has_critical = any(f.severity == "CRITICAL" for f in failures)
    has_high = any(f.severity == "HIGH" for f in failures)
    has_medium = any(f.severity == "MEDIUM" for f in failures)
    if has_critical or has_high:
        return GateDecision.FAIL
    if has_medium:
        return GateDecision.CONDITIONAL
    return GateDecision.PASS

def post_correction_gate(failures: list[Failure]) -> GateDecision:
    """v1.1 gate: considers only UNRECOVERABLE failures.

    Assumes an ideal corrector pipeline has applied all rule-based
    classes (CONTEXT_RULE, PHONETIC_MAP, ADJACENCY_RULE) and eliminated
    those failures. Only UNRECOVERABLE failures remain.
    """
    residual = [f for f in failures if f.correctability == Correctability.UNRECOVERABLE]
    has_critical = any(f.severity == "CRITICAL" for f in residual)
    has_high = any(f.severity == "HIGH" for f in residual)
    has_medium = any(f.severity == "MEDIUM" for f in residual)
    if has_critical or has_high:
        return GateDecision.FAIL
    if has_medium:
        return GateDecision.CONDITIONAL
    return GateDecision.PASS
```

**Important semantic:** the post-correction gate is *predictive*, not *actual*. It does not run a corrector. It asks "if the ideal corrector existed and fixed every correctable failure, what would be left?" — and answers based on the correctability tags. The actual corrector pipeline is task #119, which is separate.

### v1.1 change 4: Output JSON schema extensions

Per voice and per backend, the output JSON now includes:

```json
{
  "backends": {
    "voxtral": {
      "voices": {
        "en_GB-alan-medium": {
          "n_samples": 28,
          "failures_by_severity": {...},
          "counts": {"CRITICAL": N, "HIGH": N, "MEDIUM": N},
          "counts_by_correctability": {
            "UNRECOVERABLE": N,
            "CONTEXT_RULE": N,
            "PHONETIC_MAP": N,
            "ADJACENCY_RULE": N
          },
          "residual_unrecoverable": {
            "CRITICAL": N,
            "HIGH": N,
            "MEDIUM": N
          },
          "raw_gate": "PASS" | "CONDITIONAL" | "FAIL",
          "post_correction_gate": "PASS" | "CONDITIONAL" | "FAIL",
          "safety_adjusted_wer": 0.XXX
        }
      }
    }
  }
}
```

### v1.1 change 5: Console output includes the second gate

The console table printed by `__main__` gains two columns: `PostGate` (the post-correction gate) and `UnrecovCRIT` (count of unrecoverable CRITICAL residuals).

```
Backend    Raw Gate   Post Gate   Adj.WER   CRIT   HIGH   MED   UnrecovCRIT
voxtral    FAIL       FAIL        0.1320      2     47     0             2
whisper    FAIL       FAIL        0.1810      3     88     8             3
medasr     FAIL       FAIL        0.2770     13    121     8            13
```

### v1.1 expected results on cycle 112 data (from finding analysis)

| Backend | raw_gate | post_correction_gate | Unrecoverable CRIT | Correctable % |
|---|---|---|---:|---:|
| Voxtral | FAIL | FAIL | 2 | 47/49 (96%) |
| Whisper | FAIL | FAIL | 3 | 96/99 (97%) |
| MedASR | FAIL | FAIL | 13 | 129/142 (91%) |

All three backends FAIL both gates. Framework headline: **no ASR backend is currently safe for unreviewed proton RT deployment. All require human-in-the-loop review for dose values.** The metric's value is not telling us which backend is "safe" (none are) but telling us which backend requires the LEAST human review after an ideal corrector is applied (Voxtral, with 2 residual CRITICAL vs MedASR's 13).

### v1.1 acceptance criteria

Additions to the v1.0 acceptance criteria:

7. `Failure` dataclass has a `correctability: Correctability` field populated correctly per the class → correctability mapping.
8. Output JSON contains `counts_by_correctability`, `residual_unrecoverable`, and both `raw_gate` and `post_correction_gate` fields at both voice and backend levels.
9. Safety-adjusted WER uses the new weights (CRITICAL × 0.15 / n_samples, HIGH unchanged, MEDIUM unchanged).
10. When run against the two cycle 112 proton JSONs:
    - All three backends have `raw_gate = FAIL` AND `post_correction_gate = FAIL`.
    - Unrecoverable CRITICAL counts match: Voxtral 2, Whisper 3, MedASR 13.
    - Safety-adjusted WER ranking unchanged (Voxtral < Whisper < MedASR) but gaps widen vs v1.0.
11. Unit test for the correctability tag being assigned correctly per failure class.
12. Unit test for the `post_correction_gate` logic — a sample with only correctable failures should get `post_correction_gate = PASS` even if `raw_gate = FAIL`.
13. Backwards compatibility: the v1.0 output fields (`gate`, `counts`, `safety_adjusted_wer`) are preserved in the output JSON. `gate` is an alias for `raw_gate`. This lets any downstream consumer of v1.0 JSON continue to work.

### v1.1 notes for the implementation delegate

- The v1.0 code is in `tests/validation/metrics/safety_gate.py`. Start from there, do not rewrite from scratch.
- The test file is `tests/validation/tests/test_safety_gate.py`. Existing v1.0 tests should continue to pass after changes (see acceptance criterion 13).
- `loguru` is approved as a dependency. Continue to use `from loguru import logger`.
- Run the updated metric against both cycle 112 JSONs after the changes land to confirm the expected results.
- All numeric expectations above (adj.WER values, unrecoverable counts) are based on the v1.0 output and Silas's manual analysis. If the v1.1 implementation produces different numbers, stop and report — do not silently accept a different answer.

---

## v1.0 spec (original, for reference — kept intact below)


## Purpose

Aggregate WER and term-recall rank backends. They do not tell you whether a
backend is *deployable*. The cycle 110 finding (Granite 8B + instructable
producing `50.4 → 504` under a 9.25% headline WER) and the cycle 112 finding
(Voxtral Mini 3B silently substituting `GyE → Gy` under the best headline WER
of the bake-off) both demonstrate that safety-critical failures can hide in
aggregate metrics.

This metric turns the set of known safety-critical failure classes into a
formal deployment gate that can be applied to any bake-off run's per-sample
output JSON.

## Inputs

- **Bake-off result JSON**: a file produced by
  `tests/validation/scripts/run_multi_backend_e2e.py`. Schema includes
  `results[].by_voice[].samples[]` with each sample containing
  `fixture_id`, `ground_truth`, `raw_transcription`, `vocabulary_terms`,
  `raw_wer`, etc.
- **Optional: context-clue vocabulary file** for particle-therapy detection.
  Default: inline constant in the module.

## Outputs

- **Annotated JSON** written next to the input, named
  `<input>.safety_gate.json`. Schema:
  ```json
  {
    "metric_version": "1.0",
    "source_report": "bakeoff_proton_voxtral_2026-04-09.json",
    "timestamp": "2026-04-09T...",
    "backends": {
      "voxtral": {
        "voices": {
          "en_GB-alan-medium": {
            "n_samples": 28,
            "failures_by_severity": {
              "CRITICAL": [{"fixture_id": "...", "class": "...", "detail": "..."}],
              "HIGH": [...],
              "MEDIUM": [...]
            },
            "counts": {"CRITICAL": N, "HIGH": N, "MEDIUM": N},
            "gate": "PASS" | "CONDITIONAL" | "FAIL",
            "safety_adjusted_wer": 0.XXX
          }
        },
        "overall": {
          "counts": {...},
          "gate": "...",
          "safety_adjusted_wer": 0.XXX
        }
      }
    }
  }
  ```
- **Console output**: ranked table of backends by gate status and
  safety-adjusted WER, with per-class failure counts.

## Failure classes

Five classes defined by the cycle 112 proton run findings. Each has a
severity, a detector function, and a weight multiplier for
safety-adjusted WER.

### Class 1: `DECIMAL_DROP` — severity CRITICAL (weight 10)

**Definition:** Gold contains a decimal dose value (e.g., `50.4 GyE`, `23.4 Gy`),
and the prediction contains only the integer portion without the decimal.

**Detector:**
1. Extract dose expressions from gold: regex
   `\b(\d+)\.(\d+)\b\s*(?:Gy|GyE|gray|cGy|gray\s+equivalent)` (case-insensitive).
2. For each match, check whether the prediction contains the full
   `integer.decimal` value AS A SUBSTRING.
3. If the full decimal is NOT in the prediction but the integer portion IS,
   flag as `DECIMAL_DROP`.

**Example from cycle 112 run:**
- proton-0006 (rectal proton): gold `50.4 GyE`, MedASR pred `50` — flag.
- proton-0027: gold has `23.4` and `55.8`; MedASR pred has `23` and `55.8` —
  flag `23.4 → 23`.

**Rationale for CRITICAL:** a decimal drop changes dose by 10–100×. Therac-25
class error. Direct patient harm.

### Class 2: `DOSE_VALUE_MISSING` — severity CRITICAL (weight 10)

**Definition:** Gold contains a numeric dose value followed by a dose unit,
and that number does not appear anywhere in the prediction.

**Detector:**
1. Extract all dose expressions from gold: regex
   `\b(\d+(?:\.\d+)?)\s*(?:Gy|GyE|gray|cGy|gray\s+equivalent)` (case-insensitive).
2. For each extracted number, check if it appears anywhere in the
   prediction.
3. If not, flag as `DOSE_VALUE_MISSING`.

**Example from cycle 112 run:**
- proton-0002: gold `54 GyE in 30 fractions`, MedASR alan pred lacks `54`
  entirely.
- proton-0011: gold `60 GyE in 30 fractions`, MedASR alan pred lacks `60`.

**Rationale for CRITICAL:** dose value completely lost from the prescription.
Physicist has no value to check against the plan. Same severity as decimal drop.

### Class 3: `SILENT_UNIT_SUBSTITUTION` — severity HIGH (weight 5)

**Definition:** Gold contains `GyE` or `gray equivalent` in a dose expression
AND particle-therapy context clues are present in the fixture, AND the
prediction has `Gy` or `gray` at the corresponding position without the `E`
suffix.

**Detector:**
1. Check if the fixture's `ground_truth` contains `GyE` or `gray equivalent`.
2. Check if the fixture's `ground_truth` contains any particle-therapy context
   clue (see vocabulary below).
3. Extract dose expressions from gold: `(\d+(?:\.\d+)?)\s*(GyE|gray\s+equivalent)`.
4. For each gold dose, check if the prediction contains `(same number)\s*(Gy|gray)`
   (without the `E`).
5. If yes, flag as `SILENT_UNIT_SUBSTITUTION`.

**Particle-therapy context clues (positive signal for `GyE`):**
```
proton, protons, pencil beam scanning, PBS, carbon ion, particle therapy,
craniospinal, craniospinal irradiation, CSI, chordoma, medulloblastoma,
Ewing sarcoma, rhabdomyosarcoma, ependymoma, craniopharyngioma,
neuroblastoma, germinoma, RBE, relative biological effectiveness
```

**Example from cycle 112 run:**
- proton-0001 (prostate proton): gold `79.2 GyE`, Voxtral pred `79.2 Gy`, context
  has `proton` and `pencil beam scanning` — flag.
- proton-0006: gold `50.4 GyE`, Voxtral pred `50.4 Gy`, context has `proton
  chemoradiation` — flag.

**Rationale for HIGH:** a ~10% RBE correction is lost, causing undertreatment by
that magnitude. Clinically significant but less severe than complete dose-value
loss. Flagged HIGH rather than CRITICAL because the failure is detectable by a
careful reviewer who recognizes the particle-therapy context — unlike
`DECIMAL_DROP` which can survive review.

### Class 4: `SLASHED_FORM_LOSS` — severity MEDIUM (weight 2)

**Definition:** Gold contains an IGRT slashed form (`3D/3D`, `2D/3D`, `3D/2D`)
and the prediction does not contain the same form.

**Detector:**
1. Gold regex: `\b([23]D)/([23]D)\b`.
2. For each match, check if the exact slashed form appears in the prediction.
3. If not, flag as `SLASHED_FORM_LOSS`. Optionally record whether an unslashed
   variant (`3D 3D`) is present — recoverable by the corrector.

**Example from cycle 112 run:**
- proton-0019, proton-0022: gold `3D/3D`, Whisper and MedASR predictions lack
  the slashed form.

**Rationale for MEDIUM:** IGRT modality information is lost, but a downstream
reviewer looking at the treatment context (CBCT mentioned, kV imaging mentioned,
etc.) can recover the information. Recoverable by the corrector with a simple
adjacency rule.

### Class 5: `DOSE_UNIT_CORRUPTION` — severity HIGH (weight 5)

**Definition:** Gold contains `Gy` or `GyE` in a dose expression and the
prediction contains a known-bad rendering at the corresponding position.

**Detector:**
1. Extract dose expressions from gold: `(\d+(?:\.\d+)?)\s*(Gy|GyE)`.
2. For each, check if the prediction at the corresponding position contains
   any of the known-bad renderings (case-insensitive):

**Known-bad renderings (from cycle 110 + cycle 112 runs):**
```
GI, GI-E, GIE, Gie, GiE, Giy, Jai E, JIE, JEE, HIE, Jy, Ji, Jie,
J, GE, GAE, gi.e., giE, J E, J, E, J to, J in
```

3. If a known-bad rendering is found, flag as `DOSE_UNIT_CORRUPTION`.

**Example from cycle 112 run:**
- Whisper and MedASR on almost every GyE sample — `GiE`, `Jai E`, `JIE`, etc.

**Rationale for HIGH:** visibly broken renderings are recoverable by the
corrector via phonetic mapping, but they are still safety-critical because a
clinician reviewer may mis-read them as valid units if the handwriting or
display context is poor. Flagged HIGH but recoverable in the pipeline.

## Aggregation

- **Per voice × backend**: count failures by severity class, compute safety-adjusted WER.
- **Per backend**: sum across voices.
- **Gate decision** per voice and per backend:
  - `PASS` if CRITICAL == 0 AND HIGH == 0 AND MEDIUM == 0
  - `CONDITIONAL` if CRITICAL == 0 AND HIGH == 0 AND MEDIUM > 0
  - `FAIL` if CRITICAL > 0 OR HIGH > 0

## Safety-adjusted WER

**Definition:** ordinary WER weighted such that errors inside safety-critical
regions count more than errors in ordinary text.

**Simple formulation** (v1):
```
safety_adjusted_wer = raw_wer + 0.05 * CRITICAL_count + 0.025 * HIGH_count + 0.01 * MEDIUM_count
```
per voice, divided by n_samples. This is a compound penalty — ordinary WER plus
a per-sample penalty for each safety failure. Rationale: a backend with WER
0.10 and 5 CRITICAL failures in 28 samples should rank *worse* than a backend
with WER 0.15 and zero safety failures.

Numbers chosen so that one CRITICAL ≈ 5pp WER penalty per sample (~1.8pp
aggregate for 28 samples), one HIGH ≈ 2.5pp, one MEDIUM ≈ 1pp.

These weights are v1 and should be revisited after the metric is run on the
cycle 112 data to see whether the ranking matches clinical intuition.

## Implementation plan

**Location:** `tests/validation/metrics/safety_gate.py`

**Entry points:**
- `SafetyGate` dataclass: holds the vocabulary constants and thresholds.
- `evaluate_sample(sample: dict) -> list[Failure]` — runs all five detectors
  on one sample, returns list of failures.
- `evaluate_report(report: dict) -> SafetyGateReport` — runs per-sample
  evaluation on a full bake-off JSON, aggregates.
- `__main__` CLI: `python -m tests.validation.metrics.safety_gate <report.json>`
  reads input, writes `<report>.safety_gate.json` next to it, prints console
  table.

**Failure dataclass:**
```python
@dataclass(frozen=True)
class Failure:
    fixture_id: str
    class_: str  # DECIMAL_DROP, DOSE_VALUE_MISSING, ...
    severity: str  # CRITICAL, HIGH, MEDIUM
    detail: str  # human-readable explanation
    gold_value: str | None
    pred_value: str | None
```

**Tests:** `tests/validation/tests/test_safety_gate.py`
- One test per failure class using synthetic samples derived from the cycle 112
  proton data.
- Integration test that runs the metric against both cycle 112 proton JSON
  files and asserts the ranking matches the findings document:
  - Voxtral: FAIL (many HIGH — silent substitution)
  - Whisper: FAIL (many HIGH — dose-unit corruption)
  - MedASR: FAIL (many CRITICAL — decimal drops + dose value missing)
- Test that the regex detectors are case-insensitive.
- Test that the particle-therapy context detection requires at least one clue.

## Acceptance criteria

1. The metric runs cleanly against `bakeoff_proton_draft_2026-04-09.json`
   and `bakeoff_proton_voxtral_2026-04-09.json`.
2. The per-backend gate decisions match the findings document:
   - Voxtral: FAIL with high count of `SILENT_UNIT_SUBSTITUTION` (HIGH)
   - Whisper: FAIL with `DOSE_UNIT_CORRUPTION` (HIGH) as dominant class
   - MedASR: FAIL with `DECIMAL_DROP` + `DOSE_VALUE_MISSING` (CRITICAL) as dominant
3. No backend produces false positives on the counterfactual `proton-0015`
   (IMRT case — `79.2 Gy` is correct, should NOT flag `SILENT_UNIT_SUBSTITUTION`
   because there are no particle-therapy context clues in the sentence).
4. All five failure class detectors have unit tests.
5. `ruff check` and `mypy` clean.
6. Uses `loguru.logger` (project convention) for any logging.

## Not in scope for v1

- Automated correction (that's task #119, the staged correction pipeline).
- Proton-therapy-specific corrector rules (also #119).
- ML-based detectors. Everything in v1 is regex + string matching.
- Weight calibration beyond the cycle 112 data. That's a follow-up task
  after the metric is exercised on future runs.

## Implementation suggestion for the delegate

This is a good candidate for Sonnet-agent implementation:
- Clear spec with concrete examples.
- Mechanical regex work, no novel algorithm design.
- Test cases derivable directly from the findings document.
- Acceptance criteria are objective (metric output should match the
  ranking in the findings document).

The agent should:
1. Read this spec.
2. Read `bakeoff_proton_findings_2026-04-09.md` for concrete examples.
3. Read both `bakeoff_proton_*2026-04-09.json` files to understand the input
   schema.
4. Implement `safety_gate.py` per the spec.
5. Write the tests.
6. Run `make test` (or equivalent) and ensure all tests pass.
7. Run the metric against both cycle 112 JSONs and verify the acceptance
   criteria.
8. Report back with: path to new files, test count, acceptance criteria
   verification status.

The agent should NOT:
- Commit the code (Silas handles git).
- Modify the existing bake-off runner.
- Invent new failure classes beyond the five in this spec.
- Move beyond v1 scope.
