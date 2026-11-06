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
│   ├── prep_dataset.py       # Download Svarah + MFA preprocessing
│   ├── model.py              # FastSpeech 2 architecture (PyTorch)
│   ├── train.py              # Training loop (Noam LR + checkpointing)
│   ├── inference.py          # Synthesis for 10 test sentences
│   ├── plot_variance.py      # Variance adaptor plots (duration/pitch/energy)
│   └── checkpoints/          # (populated after training)
│
├── Telugu/
│   ├── config.yaml           # Telugu-specific config
│   ├── prep_dataset.py       # IndicTTS + custom Telugu G2P
│   ├── model.py              # Same architecture
│   ├── train.py
│   ├── inference.py
│   └── plot_variance.py
│
├── Results/
│   ├── English/
│   │   ├── test_generated_audios/        ← .wav files for 10 test sentences
│   │   └── variance_adaptor_plots/
│   │       ├── duration/                 ← bar charts per sentence
│   │       ├── pitch/                    ← line plots per sentence
│   │       ├── energy/                   ← area plots per sentence
│   │       └── summary/                  ← 3-panel combined plot per sentence
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

# MFA (for forced alignment — improves duration quality significantly)
conda install -c conda-forge montreal-forced-aligner
```

Python ≥ 3.9 | PyTorch ≥ 2.0 | CUDA recommended

---

## Quick Start

### English

```bash
cd English/

# Step 1: Download Svarah + extract features
python prep_dataset.py --config config.yaml

# Step 2: (Optional) MFA alignment for accurate durations
python prep_dataset.py --config config.yaml --mfa_align --skip_download

# Step 3: Train
python train.py --config config.yaml

# Step 4: Generate speech for 10 test sentences
python inference.py --config config.yaml \
    --checkpoint checkpoints/checkpoint_150000.pt

# Step 5: Plot duration / pitch / energy for each sentence
python plot_variance.py --config config.yaml
```

### Telugu

```bash
cd Telugu/
# Same commands — the config and G2P handle the language differences
python prep_dataset.py --config config.yaml
python train.py --config config.yaml
python inference.py --config config.yaml \
    --checkpoint checkpoints/checkpoint_150000.pt
python plot_variance.py --config config.yaml
```

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
    ├──► Duration Predictor ──► Length Regulator
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

Telugu uses a Brahmic abugida script. The custom G2P in `Telugu/prep_dataset.py` handles:

| Script feature | Rule |
|---------------|------|
| Consonant + inherent /a/ | C → `c a` |
| Consonant + matra | C + ◌ → `c matra_vowel` |
| Consonant + virama (్) | C + ◌ → `c` (no vowel) |
| Independent vowel | V → `v` |
| Anusvara (ం) | → `M` |
| Visarga (ః) | → `H` |

52-token phoneme set, automatically written to `text/phoneme_set_telugu.txt`.

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
