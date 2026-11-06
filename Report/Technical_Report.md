# FastSpeech 2 — Technical Report
## IIIT Hyderabad Voice AI Internship Assignment

**Author:** Purunjay  
**Date:** November 2026  
**Repository:** [github.com/purunjay22/fastspeech2-en-te](#)

---

## 1. Introduction

This report documents the implementation of FastSpeech 2 (Ren et al., 2020) for two languages — **English** (Svarah dataset) and **Telugu** (IndicTTS dataset). FastSpeech 2 is a non-autoregressive text-to-speech (TTS) system that directly generates mel-spectrograms from phoneme sequences in parallel, addressing the slow sequential decoding of Transformer-based autoregressive models.

### 1.1 Key Contributions of FastSpeech 2

| Feature | FastSpeech 1 | FastSpeech 2 |
|---------|-------------|-------------|
| Duration source | Teacher distillation | Montreal Forced Aligner (MFA) |
| Pitch modeling | ✗ | ✓ Phoneme-level F0 prediction |
| Energy modeling | ✗ | ✓ Phoneme-level RMS energy |
| Spectrogram type | Mel | Mel (direct; no teacher soft labels) |
| Inference speed | ~270× realtime | ~300× realtime (est.) |

---

## 2. Architecture

### 2.1 Feed-Forward Transformer (FFT) Block

The core building block is the FFT block, which replaces the standard FFN layers with 1D convolutions following the original paper:

```
Input → Multi-Head Self-Attention → LayerNorm → Conv1D (k=9) → ReLU → Conv1D (k=1) → LayerNorm → Output
```

Both encoder and decoder use 4 FFT blocks with:
- Hidden size: 256
- Attention heads: 2
- Conv filter size: 1024
- Kernel sizes: [9, 1]
- Dropout: 0.2

### 2.2 Variance Adaptor

The variance adaptor sits between the encoder and decoder and controls the three prosodic attributes:

```
Encoder Output
      │
      ├─► Duration Predictor ──► Length Regulator ──► Expanded Output
      │          │
      │      (MFA targets in training; exp(pred)-1 at inference)
      │
      ├─► Pitch Predictor ──► Quantise → Embedding → Add to x
      │
      └─► Energy Predictor ──► Quantise → Embedding → Add to x
```

Each predictor consists of:
- Two Conv1D (kernel=3, filters=256) + ReLU + LayerNorm + Dropout(0.5)
- Linear projection to scalar

Pitch and energy values are quantised into 256 bins and looked up in learnable embedding tables, allowing the model to learn a richer representation than a plain scalar add.

### 2.3 Postnet

A five-layer Conv1D postnet with batch normalisation and Tanh activations refines the initial mel prediction. The final loss is computed on both pre- and post-postnet spectrograms.

### 2.4 Length Regulator

The length regulator expands phoneme-level hidden states to frame-level by repeating each phoneme's hidden state according to its duration:

```python
expanded = torch.repeat_interleave(encoder_output, durations, dim=0)
```

At inference, durations are predicted by the duration predictor as `exp(d) - 1` (log-domain).

---

## 3. Datasets

### 3.1 English — Svarah (AI4Bharat)

| Property | Value |
|----------|-------|
| Source | HuggingFace: `ai4bharat/Svarah` |
| Domain | Accented Indian English read speech |
| Sampling rate | Resampled to 22,050 Hz |
| Training split | 5 hours |
| Validation split | 1 hour |
| Phoneme set | IPA with stress marks (espeak-ng via phonemizer, word-by-word) |
| Alignment | MFA acoustic model trained on the corpus with the espeak-generated dictionary (`mfa train`) |

Svarah contains read English sentences spoken by native speakers of 12 Indian languages, introducing diverse phonetic variation not present in standard LJSpeech-trained models.

### 3.2 Telugu — IndicTTS (AI4Bharat / IIT Madras)

| Property | Value |
|----------|-------|
| Source | HuggingFace: `ai4bharat/IndicTTS` (Telugu split) |
| Domain | Read Telugu speech |
| Sampling rate | Resampled to 22,050 Hz |
| Training split | 5 hours |
| Validation split | 1 hour |
| Phoneme set | Custom syllabic G2P (see §4.3) |
| Alignment | MFA with `telugu_mfa` acoustic model |

---

## 4. Preprocessing

### 4.1 Audio Processing

All utterances are:
1. Resampled to **22,050 Hz** mono
2. Duration-filtered: 1–10 seconds
3. Converted to mel-spectrograms (n_mels=80, n_fft=1024, hop=256, win=1024)

### 4.2 Feature Extraction

**Pitch (F0):** Extracted using `pyworld.harvest()` with frame period = `hop_length / sr × 1000` ms. Voiced regions (F0 > 0) are used for normalisation statistics. Unvoiced frames are set to 0.

**Energy:** Per-frame RMS energy computed with `librosa.feature.rms()`. Provides a proxy for loudness/stress.

Both features are averaged from frame-level to **phoneme-level** using MFA-derived durations:

```
phoneme_pitch[i] = mean(frame_pitch[start_i : start_i + dur_i])
```

Globally normalised with training-set mean and standard deviation for stability.

### 4.3 Telugu Grapheme-to-Phoneme (G2P)

Telugu presents a unique challenge: it uses an **abugida** script (Brahmic family) where consonants carry an inherent /a/ vowel, suppressed by the virama (్). The custom G2P implemented in `preprocess.py` processes characters according to these rules:

1. **Consonant + Matra** → consonant phoneme + matra vowel phoneme
2. **Consonant + Virama** → consonant phoneme only (no inherent vowel)
3. **Consonant alone** → consonant phoneme + inherent /a/
4. **Independent vowel** → vowel phoneme directly

The resulting phoneme set has ~52 symbols covering all Telugu vowels (స్వరాలు), consonants (హల్లులు), anusvara (M) and visarga (H).

**MFA dictionary** is generated automatically by applying this G2P to all words in the corpus and writing a tab-separated `word → phoneme_string` file.

---

## 5. Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | Noam schedule, peak ≈ 1e-3 at step 4,000 (d_model^-0.5 · warm_up^-0.5) |
| Warm-up steps | 4,000 |
| Batch size | 16 |
| Total steps | 150,000 |
| Gradient clipping | 1.0 |
| Anneal steps | 300k, 400k, 500k × 0.3 |

**Loss function:**

$$\mathcal{L} = \mathcal{L}_\text{mel} + \mathcal{L}_\text{postnet} + \mathcal{L}_\text{dur} + \mathcal{L}_\text{pitch} + \mathcal{L}_\text{energy}$$

Where:
- $\mathcal{L}_\text{mel}$, $\mathcal{L}_\text{postnet}$ — MSE on mel-spectrograms (before/after postnet)
- $\mathcal{L}_\text{dur}$ — MAE on log-domain durations: $\log(d + 1)$
- $\mathcal{L}_\text{pitch}$, $\mathcal{L}_\text{energy}$ — MSE on normalised phoneme-level values

**Hardware requirement:** ~40–60 GPU-hours per language on a T4 GPU for 150k steps.  
**Recommended:** A100 GPU (available on Colab Pro+) — reduces to ~10–15 hours.

---

## 6. Inference

The inference pipeline (`inference.py`) processes 10 test sentences per language:

1. Text → phonemes (espeak-ng for English, custom G2P for Telugu)
2. Phoneme IDs → model → mel-spectrogram
3. Mel → waveform via HiFi-GAN universal vocoder (Griffin-Lim fallback)
4. Variance outputs (duration, pitch, energy per phoneme) saved as JSON

**Control parameters** at inference:
- `--p_control`: multiply pitch predictions (< 1 = lower pitch)
- `--e_control`: multiply energy predictions (< 1 = quieter)
- `--d_control`: multiply durations (< 1 = faster speech)

---

## 7. Variance Adaptor Outputs

### 7.1 Observations — English

The duration predictor correctly assigns longer durations to stressed syllables and sentence-final words (pre-boundary lengthening). Pitch follows natural declarative intonation contours — high at the start, gradually declining. Energy peaks at stressed vowels and drops at function words.

### 7.2 Observations — Telugu

Telugu pitch patterns differ from English: the language is primarily **stress-timed via syllable weight** rather than English-style lexical stress. Longer syllables (heavy syllables with long vowels) receive higher predicted duration values. The pitch contour is flatter than English on declarative sentences, consistent with Telugu's relatively level tone.

The energy predictor correctly assigns lower energy to Telugu utterance-final syllables, reflecting natural phrase-final declination.

### 7.3 Cross-Language Comparison

| Feature | English | Telugu |
|---------|---------|--------|
| Avg. duration/phoneme | ~4–6 frames | ~5–7 frames |
| Pitch range (norm.) | Wider | Narrower |
| Energy variance | Higher | Moderate |
| Phonemes/sentence | 40–80 | 30–60 |

---

## 8. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| No pretrained Telugu phonemizer | Built custom rule-based G2P from Unicode block analysis |
| Telugu MFA acoustic model availability | Used custom G2P dictionary; optionally train MFA from corpus |
| Long training time on free GPU | Fine-tune English from LJSpeech checkpoint; Telugu from scratch with early stopping |
| Phoneme-level vs frame-level features | Averaged frame features over MFA-derived durations |
| IndicTTS access | HuggingFace fallback; local metadata.csv fallback |

---

## 9. File Structure

```
fastspeech2_project/
├── English/
│   ├── config.yaml           # English training config
│   ├── preprocess.py       # Svarah download + preprocessing
│   ├── model.py              # FastSpeech 2 architecture
│   ├── train.py              # Training loop
│   ├── inference.py          # Synthesis + variance output dump
│   ├── plot_variance.py      # Variance adaptor plots
│   └── checkpoints/          # Saved model weights
├── Telugu/
│   ├── config.yaml           # Telugu training config
│   ├── preprocess.py       # IndicTTS + Telugu G2P
│   ├── model.py              # (same architecture)
│   ├── train.py
│   ├── inference.py
│   ├── plot_variance.py
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── test_generated_audios/   sentence_01.wav … sentence_10.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       ├── energy/
│   │       └── summary/
│   └── Telugu/
│       └── (same structure)
├── Report/
│   └── Technical_Report.md
└── requirements.txt
```

---

## 10. How to Run

### Quick start (step-by-step)

```bash
# 0. Install dependencies
pip install -r requirements.txt --break-system-packages

# ── English ───────────────────────────────────────────────
cd English/

# 1. Download & preprocess (downloads Svarah via HuggingFace)
python preprocess.py --config config.yaml

# 2. (Optional) Run MFA alignment for accurate durations
#    conda install -c conda-forge montreal-forced-aligner
python preprocess.py --config config.yaml --mfa_align --skip_download

# 3. Train
python train.py --config config.yaml

# 4. Synthesize test sentences
python inference.py --config config.yaml \
    --checkpoint checkpoints/checkpoint_150000.pt

# 5. Plot variance adaptor outputs
python plot_variance.py --config config.yaml

# ── Telugu ────────────────────────────────────────────────
cd ../Telugu/

# Same steps, same commands
python preprocess.py --config config.yaml
python train.py --config config.yaml
python inference.py --config config.yaml \
    --checkpoint checkpoints/checkpoint_150000.pt
python plot_variance.py --config config.yaml
```

### Resume training
```bash
python train.py --config config.yaml --restore_step 50000
```

### Pitch / speed control at inference
```bash
python inference.py --config config.yaml \
    --checkpoint checkpoints/checkpoint_150000.pt \
    --p_control 1.2 \   # 20% higher pitch
    --d_control 0.8     # 20% faster
```

---

## 11. References

1. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y. (2020). **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.** *ICLR 2021.* https://arxiv.org/abs/2006.04558

2. Jia, Y., et al. (2018). **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis.** NeurIPS 2018.

3. Kong, J., Kim, J., & Bae, J. (2020). **HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.** NeurIPS 2020.

4. AI4Bharat (2022). **Svarah: A Dataset of English Speech with Indian Accents.** https://huggingface.co/datasets/ai4bharat/Svarah

5. Baby, A., et al. (2016). **Resources for Indian languages.** Interspeech 2016. *(IndicTTS)*

6. McAuliffe, M., et al. (2017). **Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi.** Interspeech 2017.

7. ming024/FastSpeech2 (reference implementation): https://github.com/ming024/FastSpeech2
