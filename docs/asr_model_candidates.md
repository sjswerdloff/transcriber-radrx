# ASR Model Candidates for Evaluation

Radiation therapy dictation differs from diagnostic radiology dictation.
Models trained on radiology reports (chest X-ray, CT findings) may not
handle RT-specific terminology (beam arrangements, dose fractionation,
OAR constraints). All candidates must be tested against RT content.

## Candidates

### Whisper (via MLX)
- **Size**: large-v3 (~1.5B), large-v3-turbo (~800M), distil variants
- **Apple Silicon**: Proven via mlx-whisper
- **Vocabulary biasing**: initial_prompt (soft bias, not hard constraint)
- **License**: MIT
- **Notes**: Most mature option, large community. Default in current scaffold.

### Google MedASR
- **Size**: 105M parameters (Conformer)
- **Apple Silicon**: Needs evaluation (originally CUDA)
- **Medical training**: ~5,000 hours de-identified physician dictations
- **Performance**: 4.6% WER on radiology dictation (58% fewer errors than Whisper large-v3 on chest X-ray)
- **License**: Free for research and commercial use
- **Notes**: Trained on diagnostic radiology/internal medicine/family medicine. **RT is a different domain** — dose fractionation language, beam terminology, OAR constraints are NOT in diagnostic dictation. Must benchmark against RT content specifically.
- **HuggingFace**: google/medasr

### OLMoASR (Allen Institute)
- **Size**: TBD
- **Apple Silicon**: Needs evaluation
- **License**: Fully open (Allen Institute)
- **Notes**: Cyril researched this. Competitive with Whisper. Check ONNX/MLX availability.

### Parakeet TDT (NVIDIA)
- **Size**: 0.6B (v3)
- **Apple Silicon**: sherpa-onnx supports it but **hotword boosting does NOT work** with TDT models (GitHub issue #2753, greedy search only)
- **Notes**: Works on Intel MacBook Pro (kitchen station). Apple Silicon needs testing. The hotword limitation is significant for RT vocabulary.

## Evaluation Plan

1. Select RT-specific test content from ROND and MTSamples
2. Generate synthetic audio via TTS
3. Run each model against same audio set
4. Measure WER overall and per-category (abbreviations, dose units, anatomical terms, drug names)
5. Apply correction dictionary post-processing
6. Measure corrected WER — the metric that matters is post-correction accuracy

## Key Question

For a bounded ~400-term RT vocabulary, does the ASR model matter as much
as the correction dictionary? A mediocre ASR + excellent post-processing
may outperform a medical ASR with no post-processing. Test both paths.
