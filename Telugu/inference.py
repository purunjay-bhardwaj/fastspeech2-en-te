"""
inference.py — FastSpeech 2 Inference Script
==============================================
Loads a trained checkpoint, synthesizes speech for all test sentences,
saves .wav files and mel-spectrograms.

Also collects raw variance adaptor outputs (duration, pitch, energy)
for the subsequent plotting script.

Usage:
    # English
    python inference.py --config config.yaml --checkpoint checkpoints/checkpoint_150000.pt

    # Telugu (from Telugu/ folder)
    python inference.py --config config.yaml --checkpoint checkpoints/checkpoint_150000.pt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

import yaml

from model import FastSpeech2


# ── Test Sentences ───────────────────────────────────────────────────────────

ENGLISH_TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis has advanced remarkably in recent years.",
    "Deep learning models can now produce natural sounding voices.",
    "FastSpeech two generates mel spectrograms in parallel.",
    "The variance adaptor controls pitch duration and energy.",
    "Natural language processing enables machines to understand speech.",
    "She sells sea shells by the sea shore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
    "To be or not to be, that is the question.",
    "Artificial intelligence is transforming the world of technology.",
]

TELUGU_TEST_SENTENCES = [
    "నేను తెలుగు మాట్లాడతాను.",                        # I speak Telugu.
    "మీరు ఎలా ఉన్నారు?",                               # How are you?
    "ఈ నగరం చాలా అందంగా ఉంది.",                        # This city is very beautiful.
    "మీ పేరు ఏమిటి?",                                   # What is your name?
    "అతను పుస్తకం చదువుతున్నాడు.",                      # He is reading a book.
    "ఆకాశం నీలంగా ఉంది.",                               # The sky is blue.
    "సంగీతం ఆత్మకు పోషణ.",                              # Music is nourishment for the soul.
    "జ్ఞానం అన్నింటికంటే గొప్ప సంపద.",                 # Knowledge is the greatest wealth.
    "ఈ రోజు వాతావరణం చాలా బాగుంది.",                   # The weather is very nice today.
    "కృత్రిమ మేధస్సు భవిష్యత్తును మారుస్తోంది.",        # AI is changing the future.
]


# ── Text Processing ───────────────────────────────────────────────────────────

def get_english_phonemes(text: str) -> str:
    """Convert English text to phoneme string using phonemizer."""
    try:
        import phonemizer
        backend = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=False, with_stress=True
        )
        result = backend.phonemize(
            [text],
            separator=phonemizer.separator.Separator(phone=" ", word="|", syllable="")
        )[0]
        return " ".join(result.replace("|", " ").split())
    except Exception as e:
        print(f"Phonemizer warning: {e} — using character fallback")
        return text.lower()


def get_telugu_phonemes(text: str) -> str:
    """Convert Telugu text to phoneme string."""
    # Import the G2P from prep_dataset in the Telugu folder
    import sys
    tel_dir = Path(__file__).parent.parent / "Telugu"
    if str(tel_dir) not in sys.path:
        sys.path.insert(0, str(tel_dir))
    from prep_dataset import telugu_g2p, telugu_cleaners
    return telugu_g2p(telugu_cleaners(text))


# ── HiFi-GAN Vocoder ─────────────────────────────────────────────────────────

def load_hifigan(checkpoint_path: str, device: torch.device):
    """
    Load HiFi-GAN generator from a checkpoint.
    Falls back to Griffin-Lim if checkpoint not found.
    """
    if not Path(checkpoint_path).exists():
        print(f"HiFi-GAN checkpoint not found at {checkpoint_path}. "
              f"Falling back to Griffin-Lim.")
        return None

    # Try loading from torch hub or local
    try:
        import sys
        sys.path.append(str(Path(checkpoint_path).parent))
        from models import Generator
        from env import AttrDict
        import json as _json
        cfg_path = Path(checkpoint_path).parent / "config.json"
        h = AttrDict(_json.loads(cfg_path.read_text()))
        generator = Generator(h).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(state.get("generator", state))
        generator.eval()
        generator.remove_weight_norm()
        return generator
    except Exception as e:
        print(f"HiFi-GAN load failed: {e}. Using Griffin-Lim.")
        return None


def mel_to_wav_griffinlim(mel_db: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert mel-spectrogram to waveform using Griffin-Lim."""
    ac   = cfg["audio"]
    mel  = librosa.db_to_power(mel_db)
    wav  = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=ac["sampling_rate"],
        n_fft=ac["n_fft"],
        hop_length=ac["hop_length"],
        win_length=ac["win_length"],
        fmin=ac["fmin"],
        fmax=ac["fmax"],
        n_iter=60,
    )
    return wav


# ── Main ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_vocab(vocab_path: str = "text/phoneme_vocab.json") -> dict:
    p = Path(vocab_path)
    if p.exists():
        return json.loads(p.read_text())
    # Build minimal vocab on the fly
    return {}


def phones_to_ids(phones: str, vocab: dict) -> torch.LongTensor:
    ids = [vocab.get(p, 0) for p in phones.split()]
    return torch.LongTensor(ids).unsqueeze(0)   # (1, T)


def synthesize(model, src_ids: torch.Tensor, cfg: dict, device: torch.device,
               p_control: float = 1.0, e_control: float = 1.0,
               d_control: float = 1.0):
    """Run model inference and return outputs."""
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.to(device)
        (mel_before, mel_after,
         dur_pred, pitch_pred, energy_pred,
         mel_len, _) = model(
            src_ids,
            p_control=p_control, e_control=e_control, d_control=d_control,
        )
    return {
        "mel_before":  mel_before[0].cpu().numpy(),   # (T, n_mels)
        "mel_after":   mel_after[0].cpu().numpy(),
        "duration":    torch.exp(dur_pred[0]).cpu().numpy() - 1,  # (T_phone,)
        "pitch":       pitch_pred[0].cpu().numpy(),
        "energy":      energy_pred[0].cpu().numpy(),
        "mel_len":     mel_len[0].item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--language",   default=None,
                        help="english | telugu (default: read from config)")
    parser.add_argument("--p_control",  type=float, default=1.0,
                        help="Pitch control multiplier")
    parser.add_argument("--e_control",  type=float, default=1.0,
                        help="Energy control multiplier")
    parser.add_argument("--d_control",  type=float, default=1.0,
                        help="Duration control multiplier")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    lang   = args.language or cfg.get("language", "english")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Language: {lang} | Device: {device}")

    # Load model
    vocab = load_vocab()
    if not vocab:
        raise RuntimeError("phoneme_vocab.json not found. Run prep_dataset.py first.")

    vocab_size = len(vocab)
    model = FastSpeech2(cfg, vocab_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded model from {args.checkpoint} (step {state.get('step', '?')})")

    # Vocoder
    hifigan = load_hifigan(cfg["vocoder"]["checkpoint"], device)

    # Output dirs
    audio_dir = Path(cfg["paths"]["audio_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    var_dir   = Path(cfg["paths"]["plots_dir"]).parent / "variance_adaptor_data"
    var_dir.mkdir(parents=True, exist_ok=True)

    # Test sentences
    sentences = TELUGU_TEST_SENTENCES if lang == "telugu" else ENGLISH_TEST_SENTENCES
    get_phones = get_telugu_phonemes if lang == "telugu" else get_english_phonemes

    all_variance_data = []

    print(f"\nSynthesizing {len(sentences)} test sentences...")
    for i, sentence in enumerate(sentences):
        print(f"  [{i+1:02d}/{len(sentences)}] {sentence[:60]}...")

        # Text → phonemes → ids
        phones = get_phones(sentence)
        if not phones.strip():
            print(f"    ⚠ Empty phoneme string, skipping.")
            continue
        src_ids = phones_to_ids(phones, vocab)

        # Inference
        out = synthesize(model, src_ids, cfg, device,
                         args.p_control, args.e_control, args.d_control)

        mel = out["mel_after"].T    # (n_mels, T) for librosa

        # Waveform
        if hifigan is not None:
            mel_t = torch.FloatTensor(mel).unsqueeze(0).to(device)
            with torch.no_grad():
                wav = hifigan(mel_t).squeeze().cpu().numpy()
        else:
            wav = mel_to_wav_griffinlim(mel, cfg)

        # Save audio
        wav_path = audio_dir / f"sentence_{i+1:02d}.wav"
        sf.write(str(wav_path), wav, cfg["audio"]["sampling_rate"])

        # Save mel
        mel_path = audio_dir / f"sentence_{i+1:02d}_mel.npy"
        np.save(mel_path, out["mel_after"])

        # Store variance data for plotting
        var_data = {
            "sentence_id":  i + 1,
            "text":         sentence,
            "phones":       phones.split(),
            "duration":     out["duration"].tolist(),
            "pitch":        out["pitch"].tolist(),
            "energy":       out["energy"].tolist(),
            "mel_len":      out["mel_len"],
        }
        all_variance_data.append(var_data)
        print(f"    ✓ Saved: {wav_path.name}  (mel_len={out['mel_len']})")

    # Save all variance data as JSON for plotting script
    json_path = var_dir / "variance_outputs.json"
    json_path.write_text(json.dumps(all_variance_data, indent=2, ensure_ascii=False))
    print(f"\nVariance data saved → {json_path}")
    print(f"Audio files saved  → {audio_dir}")
    print("Inference complete ✓")


if __name__ == "__main__":
    main()
