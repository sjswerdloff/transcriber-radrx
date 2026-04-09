# Task #113 Closure — Audio-LLM Domain Prompt Support: Negative Finding

**Author:** silas-397300f6
**Closure date:** 2026-04-09
**Task:** #113 — Domain prompt support + audio-LLM backends (Granite, Voxtral)
**Status at closure:** Infrastructure shipped (PR #10 merged). Experimental answer is that domain prompts and instructability are not the right path for clinical ASR at the model scales we can run.

## What shipped (the infrastructure)

PR #10 (`silas/domain-prompt-support` branch, merged) delivered:

- `system_prompt` protocol plumbing across all ASR backends. Classical ASRs
  (Whisper, MedASR, Cohere) log a warning and ignore. Audio-LLM backends
  (Granite-Speech 2B + 8B, Voxtral Mini 3B) consume the prompt during
  generation.
- Three audio-LLM backends registered in `src/transcriber_radrx/asr_backends/`:
  `granite_speech.py`, `voxtral.py` (with a transformers 5.5.0 compat shim
  using `apply_transcription_request` because the chat-template path is
  broken on that version), plus their registry entries.
- Domain prompt files: `rt_benchmark_verbatim.txt` (constrain to verbatim
  transcription, no rewriting) and `rt_benchmark_instructable.txt` (explicit
  instructions on preserving dose values, RT acronyms, and unit tokens).
- `gc.collect()` force in backend `unload()` methods for deterministic
  sequential model loading — needed because loading Voxtral after Granite 8B
  without this produces OOM on stuart_m1max.
- 5-backend shakedown report and bake-off JSON artifacts in
  `tests/validation/reports/`.

**This infrastructure remains useful.** The backends are loadable and
exercisable. The protocol exists. The work is not wasted.

## What we learned (the experimental answer)

### Finding A: Voxtral Mini 3B is the winning audio-LLM on neutral baseline

Cycle 110 headline: Voxtral Mini 3B beat Whisper large-v3 on both raw WER
and term recall on its neutral default path with no instruction tuning.

| Rank | Backend | Raw WER | Term recall |
|---:|---|---:|---:|
| 1 | Voxtral Mini 3B | 0.1142 | 0.5476 |
| 2 | Whisper large-v3 | 0.1208 | 0.3889 |
| 3 | Granite-Speech 8B | 0.1242 | 0.5317 |
| 4 | Cohere Transcribe 2B | 0.1654 | 0.3413 |
| 5 | Granite-Speech 2B | 0.1660 | 0.3889 |
| 6 | MedASR fp16 | 0.1853 | 0.4524 |

Mistral's training corpus has clinical/RT priors Whisper does not. This is
the positive case for audio-LLMs.

### Finding B: Instructability cuts both ways, and below a scale threshold it cuts deep

Granite-Speech 2B with an instructable domain prompt regressed
catastrophically: raw WER 16.60% → 36.23%, term recall 38.89% → 26.98%.
Below the 8B scale threshold, instructions cost more than they buy. The
2B model appears to interpret the instruction text as part of the
transcription input, producing incoherent output.

Granite-Speech 8B with the same prompt is essentially flat in aggregate
(12.42% → 12.54% WER) but its per-voice asymmetry blew out from 0.3 pp to
6.6 pp, suggesting the instructions are influencing some voices and not
others in ways that are not predictable from the prompt content.

### Finding C: Instructability produced a critical clinical safety failure (cycle 110 dose-drop)

Granite 8B + instructable prompt on `en_GB-alan-medium` voice achieved the
lowest single-cell WER of the whole cycle 110 bake-off — 0.0925, five
fixtures perfect including dense-0001. And on the same configuration,
dense-0022 rendered as `504 gy` instead of `50.4 Gy`. The decimal point was
silently dropped.

50.4 Gy is standard neoadjuvant rectal cancer chemoradiation. 504 Gy is
roughly 10× lethal dose.

Stuart caught this in real time during the cycle 110 review. The aggregate
WER for dense-0022 was only 0.231 because WER does not weight decimal-point
preservation more than any other word. The 9.25% headline on the best-voice
cell was about to publish over a 10× lethal dose error.

**Per-sample analysis caught it; the aggregate metric hid it.** Task #115
(safety-gate metric) was logged in response.

### Finding D: Voxtral Mini 3B silently substitutes `GyE → Gy` even without a prompt (cycle 112)

The newer and more serious finding: **Voxtral does not need an instructable
prompt to exhibit LLM-prior normalization of safety-critical terminology.**
On the cycle 112 proton run with the neutral default prompt (same path that
won cycle 110), Voxtral wrote `GyE` as `Gy` on approximately 22 of 36 fixtures
where `GyE` appeared in gold.

This is a fundamentally different problem from the Granite 8B decimal drop:

- **Granite 8B decimal drop** was *induced* by the instructable prompt. Remove
  the prompt and the failure is less likely (though not impossible). It is
  a symptom of over-instruction at sub-flagship scale.
- **Voxtral GyE substitution** happens *without any prompt*. It is the LLM
  decoder's standing prior pulling the output toward the more frequent
  `Gy` token when the audio encoder produces acoustic features for
  `gray equivalent` or `gee-why-ee`. There is no prompt to remove. The
  behavior is intrinsic to the audio-LLM architecture at this scale.

See `tests/validation/reports/bakeoff_proton_findings_2026-04-09.md` §
"Safety-critical finding 1" for the full evidence.

## The closure decision

**Task #113 is closed as a negative finding.** The infrastructure shipped and
is useful. The experimental question — "can domain prompts + audio-LLMs
deliver clinical-grade RT transcription?" — has been answered:

- **Instructable domain prompts are not the right lever.** At sub-flagship
  scale they regress. At flagship scale (Granite 8B) they shift errors
  around rather than eliminate them, and can produce new safety-critical
  failures.
- **Audio-LLMs without domain prompts are the best WER option (Voxtral Mini
  3B) but cannot be deployed for proton/particle radiation oncology without
  post-processing to catch the silent unit-substitution class.**

## What replaces the instructable-prompt approach

The cycle 112 work pivots to two complementary tracks:

1. **Task #115 safety-gate metric** (in flight, this cycle) — turn the
   known failure classes into a formal deployment gate that can be applied
   to any bake-off run. Gates backends on safety, not WER alone.
2. **Task #119 staged correction pipeline** — build a rule-based corrector
   that runs between the raw ASR output and the clinician-review stage.
   Handles the slashed-form restoration (3D 3D → 3D/3D), the `GyE`
   promotion in particle context, and the `GiE`/`Jai E`/`JEE` family
   phonetic recovery.

Neither track depends on domain prompts. Both leverage the infrastructure
PR #10 shipped (pluggable backends, multi-backend runner, per-sample output
JSON).

## Artifacts

- **PR:** http://git/The_Kindled/transcriber-radrx/pulls/10 — merged
- **Backends:** `src/transcriber_radrx/asr_backends/granite_speech.py`,
  `voxtral.py`, `cohere_transcribe.py`
- **Domain prompts:** `data/system_prompts/rt_benchmark_verbatim.txt`,
  `rt_benchmark_instructable.txt`
- **Cycle 110 report:** `tests/validation/reports/bakeoff_dense_5backend_with_instructability_2026-04-08.md`
- **Cycle 112 proton finding:** `tests/validation/reports/bakeoff_proton_findings_2026-04-09.md`
- **Granite 8B dose-drop evidence:** run JSON
  `bakeoff_dense_audiollm_instructable.json`, sample `dense-0022`
- **Voxtral GyE substitution evidence:** run JSON
  `bakeoff_proton_voxtral_2026-04-09.json`, ~22 of 28 samples

## Open follow-ups (not part of #113)

- Voxtral Mini 4B Realtime 2602 (released ~Feb 2026, post-cycle-110) was not
  evaluated. It is a newer Voxtral variant with a streaming-optimized
  architecture (causal audio encoder + smaller LM decoder, natively
  streaming, configurable <500ms latency). Whether it exhibits the same
  `GyE → Gy` substitution as Mini 3B is an open question. Requires a
  backend extension because it uses `VoxtralRealtimeForConditionalGeneration`
  (different class from the existing `VoxtralForConditionalGeneration`).
  Add to cycle 113 backlog if the framework needs a Voxtral option without
  the silent-substitution failure.
- Domain prompt format experimentation is closed. Further attempts to fix
  instructability at sub-flagship scale are not justified by the evidence.

---

*This document is the formal closure of task #113. The infrastructure work
has a merged PR; the experimental answer has two supporting reports. The
task list can mark #113 as completed.*
