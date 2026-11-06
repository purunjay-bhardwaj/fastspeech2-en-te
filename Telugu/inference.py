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

# Test sentences from the assignment brief — text preserved exactly as provided
# (including the deliberate "rhyth m" and "అడ విలో" splits).
ENGLISH_TEST_SENTENCES = [
    ("E1",  "The antidisestablishmentarianism movement gained considerable momentum "
            "during the parliamentary proceedings."),
    ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3",  "The rhyth m of the orchestra changed unexpectedly."),
    ("E4",  "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should "
            "arrive on time."),
]

TELUGU_TEST_SENTENCES = [
    ("T1",  "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
    ("T2a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
    ("T2b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
    ("T3",  "పులి అడ విలో నిశ్శబ్దంగా నడిచింది."),
    ("T4",  "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, "
            "సమయానికి సిద్ధంగా ఉంచండి."),
]


# ── Text Processing ───────────────────────────────────────────────────────────

def load_text_frontend(lang: str):
    """
    Return `text_to_phones` from the language's preprocess.py, so inference uses
    exactly the same G2P that produced the training phonemes / MFA dictionary.
    """
    import importlib.util
    import sys
    folder = Path(__file__).resolve().parent.parent / ("Telugu" if lang == "telugu" else "English")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"preprocess_{lang}", folder / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.text_to_phones


# ── HiFi-GAN Vocoder ─────────────────────────────────────────────────────────

def load_hifigan(checkpoint_path: str, device: torch.device):
    """
    Load HiFi-GAN generator from a checkpoint.
    Expects the jik876/hifi-gan code (models.py, env.py) and config.json in the
    same folder as the generator checkpoint. Falls back to Griffin-Lim otherwise.
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


def mel_to_wav_griffinlim(log_mel: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert a natural-log magnitude mel-spectrogram to waveform using Griffin-Lim."""
    ac   = cfg["audio"]
    mel  = np.exp(log_mel)
    wav  = librosa.feature.inverse.mel_to_audio(
        mel,
        power=1.0,
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
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def phones_to_ids(phones: list, vocab: dict):
    """Map phones to ids, dropping (and reporting) phones unseen in training."""
    known   = [p for p in phones if p in vocab]
    unknown = sorted({p for p in phones if p not in vocab})
    if unknown:
        print(f"    ⚠ dropping phones not in training vocab: {unknown}")
    return known, torch.LongTensor([vocab[p] for p in known]).unsqueeze(0)   # (1, T)


def synthesize(model, src_ids: torch.Tensor, cfg: dict, device: torch.device,
               p_control: float = 1.0, e_control: float = 1.0,
               d_control: float = 1.0, speaker_id: int = 0):
    """Run model inference and return outputs."""
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.to(device)
        (mel_before, mel_after,
         dur_pred, pitch_pred, energy_pred,
         mel_len, _) = model(
            src_ids,
            p_control=p_control, e_control=e_control, d_control=d_control,
            speakers=torch.LongTensor([speaker_id]).to(device),
        )
    return {
        "mel_before":  mel_before[0].cpu().numpy(),   # (T, n_mels)
        "mel_after":   mel_after[0].cpu().numpy(),
        # frames actually used by the length regulator (same rounding as model.py)
        "duration":    torch.round((torch.exp(dur_pred[0]) - 1) * d_control)
                            .clamp(min=0).cpu().numpy(),             # (T_phone,)
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
    parser.add_argument("--speaker",    default=None,
                        help="Speaker name from text/speakers.json "
                             "(default: the speaker with the most training data)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    lang   = args.language or cfg.get("language", "english")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Language: {lang} | Device: {device}")

    # Load model
    vocab = load_vocab()
    if not vocab:
        raise RuntimeError("phoneme_vocab.json not found. Run preprocess.py first.")

    get_phones = load_text_frontend(lang)

    vocab_size = len(vocab)
    state = torch.load(args.checkpoint, map_location=device)
    n_speakers = (state["model"]["speaker_emb.weight"].shape[0]
                  if "speaker_emb.weight" in state["model"] else 1)
    model = FastSpeech2(cfg, vocab_size, n_speakers).to(device)
    model.load_state_dict(state["model"])

    spk_path = Path("text/speakers.json")
    speakers = json.loads(spk_path.read_text(encoding="utf-8")) if spk_path.exists() else {}
    if args.speaker is not None:
        if args.speaker not in speakers:
            raise ValueError(f"Unknown speaker {args.speaker!r}; options: {list(speakers)[:20]}")
        speaker_id = speakers[args.speaker]
    else:
        speaker_id = 0
    spk_name = next((k for k, v in speakers.items() if v == speaker_id), "default")
    print(f"Speaker: {spk_name} (id {speaker_id} of {n_speakers})")
    model.eval()
    print(f"Loaded model from {args.checkpoint} (step {state.get('step', '?')})")

    # Vocoder
    hifigan = load_hifigan(cfg["vocoder"]["checkpoint"], device)

    # Output dirs
    audio_dir = Path(cfg["paths"]["audio_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    var_dir   = Path(cfg["paths"]["plots_dir"]).parent / "variance_adaptor_data"
    mel_dir   = var_dir / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)
    var_dir.mkdir(parents=True, exist_ok=True)

    # Test sentences
    sentences = TELUGU_TEST_SENTENCES if lang == "telugu" else ENGLISH_TEST_SENTENCES

    all_variance_data = []

    print(f"\nSynthesizing {len(sentences)} test sentences...")
    for i, (sid, sentence) in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {sid}: {sentence}")

        # Text → phonemes → ids
        phones, src_ids = phones_to_ids(get_phones(sentence), vocab)
        print(f"    phones: {' '.join(phones)}")
        if not phones:
            print(f"    ⚠ Empty phoneme sequence, skipping.")
            continue

        # Inference
        out = synthesize(model, src_ids, cfg, device,
                         args.p_control, args.e_control, args.d_control, speaker_id)

        mel = out["mel_after"][:out["mel_len"]].T    # (n_mels, T) log-mel

        # Waveform
        if hifigan is not None:
            mel_t = torch.FloatTensor(mel).unsqueeze(0).to(device)
            with torch.no_grad():
                wav = hifigan(mel_t).squeeze().cpu().numpy()
        else:
            wav = mel_to_wav_griffinlim(mel, cfg)

        # Save audio
        wav_path = audio_dir / f"{sid}.wav"
        sf.write(str(wav_path), np.clip(wav, -1.0, 1.0), cfg["audio"]["sampling_rate"])

        # Save mel
        mel_path = mel_dir / f"{sid}_mel.npy"
        np.save(mel_path, out["mel_after"])

        # Store variance data for plotting
        var_data = {
            "sentence_id":  sid,
            "text":         sentence,
            "phones":       phones,
            "duration":     out["duration"].tolist(),
            "pitch":        out["pitch"].tolist(),
            "energy":       out["energy"].tolist(),
            "mel_len":      out["mel_len"],
            "speaker":      spk_name,
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
