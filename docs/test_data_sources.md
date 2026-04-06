# Test Data Sources for RT Transcription

## RT-Specific Text Corpora

### ROND — Radiation Oncology NLP Database (Best Single Resource)
- **Source**: Mayo Clinic
- **License**: Apache 2.0
- **Content**: 20,160 instruction-tuning pairs including patient-clinician conversations with RT-specific language (treatment planning, doses, anatomy, side effects)
- **GitHub**: Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database
- **Paper**: arxiv.org/abs/2401.10995

### MTSamples — Medical Transcription Samples
- **Content**: 5,043 transcribed medical reports across 40 specialties
- **RT-relevant reports**: HDR brachytherapy, IMRT, IMRT simulation, 3D simulation, breast RT followup, chemoradiotherapy, conformal simulation
- **Structured CSV**: salgadev/medical-nlp on GitHub (from Kaggle)
- **Also**: nlpie/nlptab-corpus — 120 synthetic clinical notes from MTSamples

## Vocabulary and Nomenclature Sources

### AAPM TG-263 Nomenclature
- Standardized OAR names, target volume naming conventions
- Downloadable spreadsheet filterable by anatomic group
- Essential for the correction dictionary

### RCR Radiotherapy Dose Fractionation (4th Edition, 2024)
- 22 tumour-site chapters with evidence-based fractionation schedules
- Rich dose/fractionation language: "50 Gy in 25 fractions", etc.

### CORSAIR Dose Constraints
- "All-in-One" practical summary of OAR dose constraints
- Constraint language: "V20 < 35%", "Dmax < 54 Gy", "mean dose < 26 Gy"

### RTAnnot — Annotation Guidelines
- GitHub: RTParse/RTAnnot
- Guidelines for text-level annotation of RT treatment detail

## Synthetic Audio Generation

### United-Syn-Med Methodology
- ~790,000 TTS-generated audio files (~5,486 hours) with medical terminology
- Pipeline: authoritative text → LLM sentence generation → TTS audio
- Directly applicable to our pipeline
- HuggingFace: united-we-care/United-Syn-Med
- Paper: arxiv.org/abs/2412.00055

### Our Approach
- Use ROND + MTSamples RT reports as seed text
- Generate audio via TTS (Kindled voices or standard TTS)
- Three validation tiers: clean, acoustic chain, noise stress
- Synthea 1121-patient data for medical vocabulary stress testing (conditions, medications — not RT-specific but tests general medical term handling)

## DICOM RT Datasets (Clinical Metadata)

### RADCURE (TCIA)
- 3,346 head-and-neck cancer patients
- CT, RTSTRUCT, RTPLAN, RTDOSE + clinical metadata

### RT-HaND
- 2,895 head-and-neck patients, DICOM RT + clinical data
- GitHub: GSTT-Radiotherapy-Physics/RT-HaND
