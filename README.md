# FastSpeech 2 — English & Telugu TTS
### IIIT Hyderabad Voice AI Internship Assignment

Non-autoregressive Text-to-Speech using FastSpeech 2 for two languages:
- 🇬🇧 **English** — Svarah dataset (AI4Bharat)
- 🇮🇳 **Telugu** — IndicTTS dataset (IIT Madras / AI4Bharat)

---

## Repository Structure

```
fastspeech2_project/
├── English/
│   ├── config.yaml           # Hyperparameters & paths
│   ├── preprocess.py         # Download Svarah + MFA preprocessing
│   ├── features.py           # Shared feature/alignment helpers
│   ├── model.py              # FastSpeech 2 architecture (PyTorch)
│   ├── train.py              # Training loop (Noam LR + checkpointing)
│   ├── inference.py          # Synthesis for the 5 assignment test sentences
│   ├── plot_variance.py      # Variance adaptor plots (duration/pitch/energy)
│   └── checkpoints/          # (populated after training)
│
├── Telugu/
│   ├── config.yaml           # Telugu-specific config
│   ├── preprocess.py       # IndicTTS + custom Telugu G2P
│   ├── model.py              # Same architecture
│   ├── train.py
│   ├── inference.py
│   └── plot_variance.py
│
├── Results/
│   ├── English/
│   │   ├── training_loss.png             ← written by train.py
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/        ← E1, E2a, E2b, E3, E4 .wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/                 ← bar charts per sentence
│   │       ├── pitch/                    ← line plots per sentence
│   │       ├── energy/                   ← area plots per sentence
│   │       └── summary/                  ← 3-panel combined plot per sentence
│   │   (variance_adaptor_data/           ← raw predictions JSON + mels)
│   └── Telugu/
│       └── (same structure)
│
├── Report/
│   └── Technical_Report.md
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt --break-system-packages

# espeak-ng (backend for the English phonemizer)
brew install espeak-ng          # macOS  (Linux: apt-get install espeak-ng)

# MFA (for forced alignment — improves duration quality significantly)
conda install -c conda-forge montreal-forced-aligner

# (Optional) HiFi-GAN vocoder — otherwise Griffin-Lim is used.
# Put generator checkpoint + config.json + models.py + env.py from
# https://github.com/jik876/hifi-gan in <Lang>/pretrained/hifigan/
```

Python ≥ 3.9 | PyTorch ≥ 2.0 | CUDA recommended

---

## Quick Start

### English

```bash
cd English/

# Step 1: Download Svarah + extract features
python preprocess.py --config config.yaml

# Step 2: (Recommended) MFA alignment for real phoneme durations.
# Trains MFA on the corpus with our own G2P dictionary, then re-extracts features.
# Without it, durations are spread uniformly across phonemes.
python preprocess.py --config config.yaml --mfa_align --skip_download

# Step 3: Train
python train.py --config config.yaml

# Step 4: Generate speech for the test sentences (E1, E2a, E2b, E3, E4)
python inference.py --config config.yaml \
    --checkpoint checkpoints/model_final_fp16.pt

# Step 5: Plot duration / pitch / energy for each sentence
python plot_variance.py --config config.yaml
```

### Telugu

```bash
cd Telugu/
# Same commands — the config and G2P handle the language differences
python preprocess.py --config config.yaml
python preprocess.py --config config.yaml --mfa_align --skip_download
python train.py --config config.yaml
python inference.py --config config.yaml \
    --checkpoint checkpoints/model_final_fp16.pt
python plot_variance.py --config config.yaml
```

Training writes `logs/train_loss.csv` / `logs/val_loss.csv` and redraws
`Results/<Lang>/training_loss.png` and `validation_loss.png` at every validation step
(`python train.py --config config.yaml --plot_only` redraws them from the logs).
At the end it exports `checkpoints/model_final_fp16.pt` (~60 MB, weights only — under
GitHub's 100 MB limit); the full `checkpoint_<step>.pt` files (~350 MB, with optimizer
state) are for resuming and should not be committed.

`model.py`, `train.py`, `inference.py`, `features.py` and `plot_variance.py` are
identical in both folders — keep them in sync when editing.

---

## Model Architecture

```
Phoneme IDs
    │
    ▼
[Embedding + Positional Encoding]
    │
    ▼
[Encoder — 4× FFT Blocks]
    │
    ├──► Duration Predictor ──► (durations for LR)
    │
    ├──► Pitch Predictor  ──► Quantise ──► Embedding ──► Add
    │
    └──► Energy Predictor ──► Quantise ──► Embedding ──► Add
                                                           │
                                                           ▼
                                              [Decoder — 4× FFT Blocks]
                                                           │
                                                           ▼
                                                    [Linear → Mel]
                                                           │
                                                           ▼
                                                       [Postnet]
                                                           │
                                                           ▼
                                                   Mel Spectrogram
                                                           │
                                                     [HiFi-GAN]
                                                           │
                                                        Waveform
```

Each FFT Block: Multi-Head Self-Attention → LayerNorm → Conv1D(k=9) → ReLU → Conv1D(k=1) → LayerNorm

---

## Telugu G2P

Telugu uses a Brahmic abugida script. The custom G2P in `Telugu/preprocess.py` handles:

| Script feature | Rule |
|---------------|------|
| Consonant + inherent /a/ | C → `c a` |
| Consonant + matra | C + ◌ → `c matra_vowel` |
| Consonant + virama (్) | C + ◌ → `c` (no vowel) |
| Independent vowel | V → `v` |
| Anusvara (ం) | → `M` (after the vowel, e.g. కం → `k a M`) |
| Visarga (ః) | → `H` |

Phoneme set, automatically written to `text/phoneme_set_telugu.txt`.

---

## Inference Controls

```bash
python inference.py \
    --p_control 1.2 \    # 20% higher pitch
    --e_control 0.9 \    # 10% quieter
    --d_control 0.85     # 15% faster speech
```

---

## References

- Ren et al. (2021) — [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- Kong et al. (2020) — [HiFi-GAN](https://arxiv.org/abs/2010.05646)
- AI4Bharat — [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- IIT Madras — [IndicTTS](https://www.iitm.ac.in/donlab/tts)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) — reference implementation
