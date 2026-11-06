"""
prep_dataset.py  —  English (Svarah) Dataset Preprocessing Pipeline
======================================================================
Steps:
  1. Load & filter Svarah dataset (HuggingFace: ai4bharat/Svarah)
  2. Resample audio to 22050 Hz
  3. Deterministic split: 5 h train / 1 h val
  4. (--mfa_align) Build MFA corpus + espeak dictionary, train MFA & align
  5. Extract log-mel, phoneme durations, phoneme-level pitch & energy
  6. Compute pitch/energy stats (train split) for normalisation
  7. Write train.txt / val.txt filelists and text/phoneme_vocab.json

Usage:
    python prep_dataset.py --config config.yaml                 # download + features
    python prep_dataset.py --config config.yaml --mfa_align --skip_download
"""

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

import features as F


# ── Text Cleaning & Phonemization ────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def english_cleaners(text: str) -> str:
    """Basic English text normalization."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 \'\-\,\.\!\?]", "", text)
    return re.sub(r"\s+", " ", text).strip()


_backend = None

def _get_backend():
    global _backend
    if _backend is None:
        import phonemizer.backend
        _backend = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=False, with_stress=True)
    return _backend


def phonemize_words(words: List[str]) -> List[List[str]]:
    """Phonemize each word independently (keeps dictionary == inference G2P)."""
    if not words:
        return []
    from phonemizer.separator import Separator
    out = _get_backend().phonemize(
        words, separator=Separator(phone=" ", word="", syllable=""), strip=True)
    return [o.split() for o in out]


@lru_cache(maxsize=None)
def _word_phones(word: str) -> tuple:
    return tuple(phonemize_words([word])[0])


def text_to_phones(text: str) -> List[str]:
    """Raw English text → list of phoneme tokens (used by prep and inference)."""
    phones = []
    for w in F.mfa_words(english_cleaners(text)):
        phones.extend(_word_phones(w))
    return phones


def build_lexicon(entries) -> dict:
    words = sorted({w for *_, text in entries for w in F.mfa_words(text)})
    print(f"Phonemizing {len(words)} unique words for the MFA dictionary ...")
    return dict(zip(words, phonemize_words(words)))


# ── Svarah Dataset Loader ────────────────────────────────────────────────────

def load_svarah(raw_dir: Path, cfg: dict):
    """
    Load Svarah from HuggingFace datasets (all available splits).
    Returns list of (speaker_id, utterance_id, wav_path, text).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading Svarah dataset (this may take a while)...")
    dsd = load_dataset("ai4bharat/Svarah")
    print(f"Available splits: {list(dsd.keys())}")

    ac = cfg["audio"]
    sr = ac["sampling_rate"]
    wav_dir = raw_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries, i = [], 0
    for split_name, ds in dsd.items():
        for sample in tqdm(ds, desc=f"Extracting Svarah [{split_name}]"):
            uid = f"svarah_{i:06d}"
            i += 1
            text = english_cleaners(sample.get("text") or sample.get("sentence") or "")
            if not text:
                continue

            wav_path = wav_dir / f"{uid}.wav"
            if not wav_path.exists():
                arr, orig_sr = F.audio_to_array(sample["audio"])
                if orig_sr != sr:
                    arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")

            dur = F.get_duration(str(wav_path))
            if not (ac["min_duration"] <= dur <= ac["max_duration"]):
                continue
            spk = str(sample.get("speaker_id", "spk0"))
            entries.append((spk, uid, str(wav_path), text))

    print(f"Loaded {len(entries)} valid utterances.")
    return entries


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mfa_align", action="store_true",
                        help="Train MFA on the corpus and align (requires MFA installed)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Reuse the entry manifest from a previous run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir  = Path(cfg["raw_data_dir"])
    prep_dir = Path(cfg["preprocessed_data_dir"])
    prep_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_dir / "entries.json"

    # 1. Load dataset
    if args.skip_download:
        entries = F.load_entries(manifest)
        print(f"Loaded {len(entries)} entries from {manifest}")
    else:
        entries = load_svarah(raw_dir, cfg)
        F.save_entries(entries, manifest)

    # 2. Split (deterministic)
    train, val = F.split_entries(entries, train_hours=5.0, val_hours=1.0)

    # 3–4. MFA
    if args.mfa_align:
        corpus_dir = raw_dir / "mfa_corpus"
        dict_path  = Path(cfg["mfa"]["dictionary"])
        F.prepare_mfa_corpus(train + val, corpus_dir)
        F.write_mfa_dictionary(build_lexicon(train + val), dict_path)
        F.run_mfa_train(corpus_dir, dict_path, prep_dir / "textgrids",
                        prep_dir / "mfa_acoustic_model.zip", cfg["mfa"]["num_jobs"])

    # 5. Features + filelists
    collect = {}
    phones  = F.process_split(train, "train", cfg, prep_dir, text_to_phones, collect)
    phones += F.process_split(val,   "val",   cfg, prep_dir, text_to_phones)

    # 6–7. Stats & vocab
    F.write_stats(collect, prep_dir)
    F.write_vocab(phones)
    print("Preprocessing done ✓")


if __name__ == "__main__":
    main()
