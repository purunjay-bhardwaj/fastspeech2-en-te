"""
preprocess.py  —  English (Svarah) Dataset Preprocessing Pipeline
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
    python preprocess.py --config config.yaml                 # download + features
    python preprocess.py --config config.yaml --mfa_align --skip_download
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


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Cardinal number → English words (0 … 999 999)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    rest = n % 1000
    return number_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_to_words(rest))


def year_to_words(n: int) -> str:
    """2026 → 'twenty twenty six', 1999 → 'nineteen ninety nine', 2005 → 'two thousand five'."""
    hi, lo = divmod(n, 100)
    if 2000 <= n <= 2009:
        return number_to_words(n)
    if lo == 0:
        return number_to_words(hi) + " hundred"
    return number_to_words(hi) + " " + (("oh " + _ONES[lo]) if lo < 10 else number_to_words(lo))


def normalize_numbers(text: str) -> str:
    """Expand times, AM/PM, years and other numbers into words."""
    def _time(m):
        h, mnt = int(m.group(1)), int(m.group(2))
        out = number_to_words(h)
        if mnt:
            out += " " + (("oh " + _ONES[mnt]) if mnt < 10 else number_to_words(mnt))
        else:
            out += " o'clock" if not m.group(3) else ""
        if m.group(3):
            out += " " + " ".join(m.group(3).replace(".", "").lower())   # "a m"
        return out

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?(?=\W|$)", _time, text)
    text = re.sub(r"\b(\d{1,2})\s*([AaPp]\.?[Mm]\.?)(?=\W|$)",
                  lambda m: number_to_words(int(m.group(1))) + " "
                  + " ".join(m.group(2).replace(".", "").lower()), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: year_to_words(int(m.group(1))), text)
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)                       # 1,000 → 1000
    text = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), text)
    return text


def english_cleaners(text: str) -> str:
    """English text normalization (numbers → words, lowercase, strip symbols)."""
    text = normalize_numbers(text.strip())
    text = text.lower()
    text = re.sub(r"[^a-z0-9 \'\-\,\.\!\?]", " ", text)
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
    """
    Raw English text → list of phoneme tokens (used by prep and inference).
    The sentence is phonemized as a whole so espeak can use context
    (e.g. heteronyms such as 'conduct' noun vs verb); falls back to
    word-by-word phonemization if that fails.
    """
    words = F.mfa_words(english_cleaners(text))
    if not words:
        return []
    try:
        from phonemizer.separator import Separator
        out = _get_backend().phonemize(
            [" ".join(words)], separator=Separator(phone=" ", word=" ", syllable=""),
            strip=True)[0]
        phones = out.split()
        if phones:
            return phones
    except Exception as e:
        print(f"Sentence-level phonemization failed ({e}); using word-by-word.")
    phones = []
    for w in words:
        phones.extend(_word_phones(w))
    return phones


def build_lexicon(entries) -> dict:
    words = sorted({w for *_, text in entries for w in F.mfa_words(text)})
    print(f"Phonemizing {len(words)} unique words for the MFA dictionary ...")
    return dict(zip(words, phonemize_words(words)))


# ── Svarah Dataset Loader ────────────────────────────────────────────────────

def load_svarah(raw_dir: Path, cfg: dict):
    """
    Load Svarah from HuggingFace (all available splits — the dataset ships a single
    'test' split of ~9.6 h from 117 speakers). The dataset is GATED: accept its terms
    on huggingface.co and log in (`huggingface-cli login`) first.
    Returns list of (speaker_id, utterance_id, wav_path, text).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading Svarah dataset (this may take a while)...")
    dsd = load_dataset(cfg.get("hf_dataset", "ai4bharat/Svarah"))
    print(f"Available splits: {list(dsd.keys())}")

    ac = cfg["audio"]
    sr = ac["sampling_rate"]
    wav_dir = raw_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries, i = [], 0
    for split_name, ds in dsd.items():
        print(f"[{split_name}] columns: {ds.column_names}")
        first = ds[0]
        text_col = F.find_column(first, ["text", "transcript", "transcription", "sentence",
                                         "normalized_text", "raw_text"])
        spk_col  = F.find_column(first, ["speaker_id", "speaker", "spk_id", "client_id",
                                         "speaker_name"])
        dur_col  = F.find_column(first, ["duration", "duration (s)", "audio_duration"])
        if text_col is None:
            raise KeyError(f"No transcript column found in {ds.column_names}")
        print(f"  using text='{text_col}', speaker='{spk_col}', duration='{dur_col}'")

        for sample in tqdm(ds, desc=f"Extracting Svarah [{split_name}]"):
            uid = f"svarah_{i:06d}"
            i += 1
            text = english_cleaners(str(sample.get(text_col) or ""))
            if not text:
                continue
            if dur_col and sample.get(dur_col) is not None and not (
                    ac["min_duration"] <= float(sample[dur_col]) <= ac["max_duration"]):
                continue   # skip without decoding audio

            wav_path = wav_dir / f"{uid}.wav"
            if not wav_path.exists():
                arr, orig_sr = F.audio_to_array(sample["audio"])
                if orig_sr != sr:
                    arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")

            dur = F.get_duration(str(wav_path))
            if not (ac["min_duration"] <= dur <= ac["max_duration"]):
                continue
            spk = str(sample.get(spk_col)) if spk_col else "spk0"
            entries.append((spk, uid, str(wav_path), text))

    print(f"Loaded {len(entries)} valid utterances "
          f"({sum(F.get_duration(e[2]) for e in entries)/3600:.2f} h, "
          f"{len({e[0] for e in entries})} speakers).")
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
    sp = cfg.get("split", {})
    train, val = F.split_entries(entries, sp.get("train_hours", 5.0),
                                 sp.get("val_hours", 1.0), sp.get("seed", 1234))

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
    F.write_speakers(train)
    print("Preprocessing done ✓")


if __name__ == "__main__":
    main()
