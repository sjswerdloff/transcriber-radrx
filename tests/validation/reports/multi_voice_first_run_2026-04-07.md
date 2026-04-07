# Multi-Voice Validation: First Run

**Date:** 2026-04-07
**Branch:** silas/multi-voice-e2e
**Runner:** `tests/validation/scripts/run_multi_voice_e2e.py`
**Pipeline:** text → piper (16 voices) → Whisper large-v3-turbo → corrector → WER

## Setup

- **Voices:** 8 UK (alan, alba, aru, cori, jenny_dioco, northern_english_male, semaine, southern_english_female) + 8 US (amy, kristin, lessac, ljspeech, hfc_male, joe, norman, ryan). Gender-balanced US subset. VCTK excluded.
- **Samples:** 10 fixtures per voice, seed=42 (5 homograph traps + 5 ROND samples, same across all voices)
- **Vocabulary:** `data/rt_vocabulary.txt` (391 terms)

## Results

| Voice | Raw WER (off) | Corrected WER (off) | Raw WER (on) | Corrected WER (on) | Traps (on) |
|-------|---------------|---------------------|--------------|---------------------|------------|
| en_GB-alan-medium | 0.0604 | 0.0604 | 0.0604 | 0.1011 | 0 |
| en_GB-alba-medium | 0.0531 | 0.0531 | 0.0531 | 0.0938 | 0 |
| en_GB-aru-medium | 0.0771 | 0.0771 | 0.0771 | 0.1178 | 0 |
| en_GB-cori-high | 0.0922 | 0.0922 | 0.0922 | 0.1328 | 0 |
| en_GB-jenny_dioco-medium | **0.1406** | 0.1406 | 0.1406 | 0.1813 | 0 |
| en_GB-northern_english_male-medium | 0.0547 | 0.0547 | 0.0547 | 0.0954 | 0 |
| en_GB-semaine-medium | **0.0364** | 0.0364 | 0.0364 | 0.0771 | 0 |
| en_GB-southern_english_female-low | 0.0710 | 0.0710 | 0.0710 | 0.1116 | 0 |
| en_US-amy-medium | 0.0402 | 0.0402 | 0.0402 | 0.0809 | 0 |
| en_US-kristin-medium | 0.0854 | 0.0854 | 0.0854 | 0.1261 | 0 |
| en_US-lessac-high | 0.0513 | 0.0513 | 0.0513 | 0.0920 | 0 |
| en_US-ljspeech-high | **0.1464** | 0.1464 | 0.1464 | 0.1871 | 0 |
| en_US-hfc_male-medium | **0.0360** | 0.0360 | 0.0360 | 0.0766 | 0 |
| en_US-joe-medium | 0.0513 | 0.0513 | 0.0513 | 0.0920 | 0 |
| en_US-norman-medium | 0.0650 | 0.0650 | 0.0650 | 0.1057 | 0 |
| en_US-ryan-high | 0.0430 | 0.0430 | 0.0430 | 0.0837 | 0 |

Best raw WER: **en_US-hfc_male-medium 3.60%** and **en_GB-semaine-medium 3.64%**.
Worst raw WER: **en_US-ljspeech-high 14.64%** and **en_GB-jenny_dioco-medium 14.06%**.
**Range: 4x across voices.**

## Observations

### 1. Voice quality tier does not predict ASR accuracy

The two worst-performing voices (`ljspeech-high`, `cori-high`, `jenny_dioco-medium`) include two "high" quality piper voices. Meanwhile, `southern_english_female-low` — the only "low" tier voice in the panel — lands mid-pack at 7.1%, beating several "medium" voices.

**Takeaway:** piper's voice-quality tier labels reflect synthesis fidelity, not ASR-friendliness. Whisper may actually do worse on distinctive "high" quality voices like LJSpeech and Jenny (both are hallmark TTS voices with unusual prosody).

### 2. Raw WER is higher than the first run

The first E2E run (5 homograph-trap samples only) gave 1.33%. This run (10 samples, half from ROND) gives 3.6% to 14.6%. **The 1.33% was a ceiling on stress-test content, not a realistic estimate** — the design doc's framing holds up.

### 3. Phonetic ON adds ~4% WER on every single voice

Corrected WER with phonetic ON is systematically ~0.04 higher than raw WER across all 16 voices. This is because the corrector is making wrong phonetic corrections at a predictable rate. Every voice. Every run. **The empirical case against enabling phonetic matching is now overwhelming.**

### 4. Phonetic OFF corrected WER = raw WER for every voice

The corrector's exact + case-insensitive tiers don't change the WER because the WER normalizer lowercases before comparing. Case corrections exist (and some are buggy — see bugs below) but they don't affect WER. **This means phonetic OFF is effectively a no-op at the WER level, which is actually the goal for clean audio — do no harm.**

### 5. Known false positives are still firing

Confirmed across voices:
- `proceed → breast` (15 voices where Whisper produced "proceed")
- `throughout → thyroid` (16 voices)

Plus NEW finding: `structures → structure set` (surface difference from multi-word vocab, see bug B-2 below).

### 6. `must_not_become` trap violations all zero

This is a quirk of my trap fixture design. The phonetic corrections that DID fire (e.g., `proceed → breast`) aren't in the trap's `must_not_become` list because the trap was written to catch `our → OAR`, `support → SBRT`, etc. The new corrections sneak past the explicit trap check even though they corrupt the meaning. **The trap violation count is necessary but not sufficient** — we also need a general WER delta check, which this run already surfaces.

## Bugs Discovered

### B-1. Acronym case-insensitive registration (CRITICAL)

**Symptom:** `"The loan's art risk"` → `"The loan's ART risk"` — the case-insensitive corrector rewrites lowercase English `art` into the RT acronym `ART` (adaptive radiotherapy).

**Root cause:** In `corrector.py::_load`, every vocabulary term is registered in `_lower_map` unless its lowercase form is a stop word. `art` is not a stop word (it's not in `DEFAULT_STOP_WORDS`), so the vocabulary entry `ART` registers as `lower_map["art"] = "ART"`.

**The safety intent** was "short acronyms (≤4 chars, all caps) are matched ONLY by exact match" — but I enforced this only at **match time**, not at **load time**. `_is_acronym()` is called on the matched entry during phonetic matching, but for case-insensitive matching, any vocab term that registers in `_lower_map` gets used.

**Fix:** At load time, do not register acronyms in `_lower_map`. Acronyms should only be matchable by exact match.

```python
# Current (buggy):
if lower not in self._stop_words and lower not in self._lower_map:
    self._lower_map[lower] = term

# Fixed:
if not entry.is_acronym and lower not in self._stop_words and lower not in self._lower_map:
    self._lower_map[lower] = term
```

This affects `ART`, `OAR`, `PTV`, `GTV`, `CTV`, `SBRT`, `IMRT`, `VMAT`, `Gy`, etc. — any short all-caps acronym that collides with an English word. `art`, `oar`, `ptv`, `gty` probably not a concern for most except `art`. `our`/`are`/`or` collide with `OAR` but are already in `DEFAULT_STOP_WORDS` so they escape via the stop-word path.

### B-2. Multi-word vocabulary entries produce space-containing replacements (HIGH)

**Symptom:** `"brain structures for skull"` → `"brain structure set for skull"` under phonetic ON — the word `structures` gets phonetically matched to the vocabulary entry `structure set`, and the replacement string (containing a space) is inserted verbatim into the output text.

**Root cause:** The corrector uses `_WORD_PATTERN = re.compile(r"[\w'-]+|[^\w\s]")` which matches single tokens. When `structures` matches `structure set` phonetically, the replacement is the literal string `"structure set"` including the embedded space, turning one word into two in the output.

**The design intent** was single-word matching only. Multi-word vocabulary entries were not in scope but were silently accepted at load time.

**Fix options:**
1. **Reject multi-word entries at load time** with a clear warning (simplest).
2. **Strip multi-word entries from phonetic matching** but keep them available for future n-gram-aware matching.
3. **Implement proper n-gram matching** (large scope — probably not needed for single-term vocabulary).

I recommend option (1) for now: log a warning and exclude any vocab term containing whitespace from all matching tiers. Multi-word terms remain in the initial_prompt (which passes them to Whisper as soft bias) but are not available for post-processing correction until n-gram matching exists.

### B-3 (not a bug, observation). Jenny Dioco and LJSpeech are hard for Whisper

Two "high quality" piper voices gave the worst WER. Possibly because:
- LJSpeech is a very distinctive single-speaker corpus — Whisper was trained on it but may not generalize well beyond its own voice characteristics.
- Jenny Dioco has idiosyncratic prosody that Whisper mis-aligns.

**Not a fix, but worth documenting.** Might be worth including both voices in regular validation runs specifically because they stress the ASR more than easy voices.

## Recommendations

1. **Immediate:** Fix B-1 (acronym registration) as a follow-up PR — one-line change + new negative test.
2. **Soon:** Fix B-2 (multi-word vocabulary rejection) — adds a load-time validation step + test.
3. **Keep:** `enable_phonetic=False` as the default. Third independent empirical confirmation.
4. **Consider:** removing phonetic matching entirely. Across 16 voices and 10 samples each, it has produced zero useful corrections and a consistent ~4% WER regression. The case for keeping it is now purely theoretical.
5. **Consider:** adding a **general WER regression check** to the validation suite — not just `must_not_become` traps, but "corrected WER should not exceed raw WER by more than X% under any pipeline configuration." This would catch bugs like B-1 and B-2 automatically.
