"""
preprocess.py  —  Telugu (IndicTTS) Dataset Preprocessing Pipeline
======================================================================
Steps:
  1. Download/load IndicTTS Telugu from AI4Bharat or iitm-tts mirrors
  2. Resample to 22050 Hz
  3. Deterministic split: 5 h train / 1 h val
  4. Telugu G2P (rule-based) → MFA dictionary; (--mfa_align) train MFA & align
  5. Extract log-mel, phoneme durations, phoneme-level pitch & energy
  6. Compute pitch/energy stats (train split)
  7. Write filelists and text/phoneme_vocab.json

Usage:
    python preprocess.py --config config.yaml
    python preprocess.py --config config.yaml --mfa_align --skip_download
"""

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

import features as F


# ── Telugu G2P (Grapheme-to-Phoneme) ────────────────────────────────────────

# Telugu vowel nucleus map (simplified IPA representation for MFA)
_TELUGU_VOWELS = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii',
    'ఉ': 'u', 'ఊ': 'uu', 'ఋ': 'ru', 'ఎ': 'e',
    'ఏ': 'ee', 'ఐ': 'ai', 'ఒ': 'o', 'ఓ': 'oo',
    'ఔ': 'au',
}

_TELUGU_CONSONANTS = {
    'క': 'k', 'ఖ': 'kh', 'గ': 'g', 'ఘ': 'gh', 'ఙ': 'ng',
    'చ': 'ch', 'ఛ': 'chh', 'జ': 'j', 'ఝ': 'jh', 'ఞ': 'ny',
    'ట': 'tt', 'ఠ': 'tth', 'డ': 'dd', 'ఢ': 'ddh', 'ణ': 'nn',
    'త': 't', 'థ': 'th', 'ద': 'd', 'ధ': 'dh', 'న': 'n',
    'ప': 'p', 'ఫ': 'ph', 'బ': 'b', 'భ': 'bh', 'మ': 'm',
    'య': 'y', 'ర': 'r', 'ల': 'l', 'వ': 'v',
    'శ': 'sh', 'ష': 'ssh', 'స': 's',
    'హ': 'h', 'ళ': 'll', 'ఱ': 'rr', 'ఴ': 'rrh',
}

# Dependent vowel signs (replace the inherent /a/)
_TELUGU_MATRAS = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu',
    'ృ': 'ru', 'ె': 'e', 'ే': 'ee', 'ై': 'ai',
    'ొ': 'o', 'ో': 'oo', 'ౌ': 'au',
}

# Signs that follow a vowel (they do NOT replace the inherent /a/)
_TELUGU_POST_VOWEL = {
    'ం': 'M',   # anusvara
    'ః': 'H',   # visarga
}

_VIRAMA = '్'   # halant: suppresses the inherent /a/


def telugu_g2p(text: str) -> str:
    """
    Rule-based Telugu G2P. Returns a space-separated phoneme string
    (words are concatenated without a boundary token).
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        text = IndicNormalizerFactory().get_normalizer("te").normalize(text)
    except ImportError:
        pass

    phones = []
    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else None

        if c in _TELUGU_CONSONANTS:
            phones.append(_TELUGU_CONSONANTS[c])
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _TELUGU_MATRAS:
                phones.append(_TELUGU_MATRAS[nxt])
                i += 2
                continue
            phones.append('a')          # inherent vowel (also before ం / ః)
            i += 1
        elif c in _TELUGU_VOWELS:
            phones.append(_TELUGU_VOWELS[c])
            i += 1
        elif c in _TELUGU_POST_VOWEL:
            phones.append(_TELUGU_POST_VOWEL[c])
            i += 1
        elif c in _TELUGU_MATRAS:       # stray matra (shouldn't happen)
            phones.append(_TELUGU_MATRAS[c])
            i += 1
        else:                           # spaces, unknown symbols
            i += 1

    return " ".join(phones)


_TE_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది",
            "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను",
            "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TE_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]


def telugu_number(n: int) -> str:
    """Cardinal number → Telugu words (0 … 99 999)."""
    if n < 20:
        return _TE_ONES[n]
    if n < 100:
        return _TE_TENS[n // 10] + ("" if n % 10 == 0 else " " + _TE_ONES[n % 10])
    if n < 1000:
        h, rest = divmod(n, 100)
        if rest == 0:
            return "వంద" if h == 1 else f"{_TE_ONES[h]} వందలు"
        head = "నూట" if h == 1 else f"{_TE_ONES[h]} వందల"
        return f"{head} {telugu_number(rest)}"
    th, rest = divmod(n, 1000)
    if rest == 0:
        return "వెయ్యి" if th == 1 else f"{telugu_number(th)} వేలు"
    head = "వెయ్యి" if th == 1 else f"{telugu_number(th)} వేల"
    return f"{head} {telugu_number(rest)}"


def telugu_ordinal(n: int) -> str:
    """12 → పన్నెండవ (drop the final vowel sign, add వ)."""
    word = telugu_number(n)
    if word.endswith(("ు", "ి")):
        word = word[:-1]
    return word + "వ"


def normalize_telugu_numbers(text: str) -> str:
    """Expand times (10:45), ordinals (12వ) and cardinals (2026) into Telugu words."""
    def _time(m):
        h, mnt = int(m.group(1)), int(m.group(2))
        return telugu_number(h) + ("" if mnt == 0 else " " + telugu_number(mnt))
    text = re.sub(r"(\d{1,2}):(\d{2})", _time, text)
    text = re.sub(r"(\d+)\s*వ(?![ా-ౌ్])", lambda m: telugu_ordinal(int(m.group(1))), text)
    text = re.sub(r"\d+", lambda m: telugu_number(int(m.group(0))), text)
    return text


def telugu_cleaners(text: str) -> str:
    """Expand numbers, then remove non-Telugu characters (keep spaces)."""
    text = normalize_telugu_numbers(text)
    # Keep Telugu Unicode block: U+0C00–U+0C7F, plus space
    text = re.sub(r"[^ఀ-౿\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_to_phones(text: str) -> List[str]:
    """Raw Telugu text → list of phoneme tokens (used by prep and inference)."""
    phones = []
    for w in F.mfa_words(telugu_cleaners(text)):
        phones.extend(telugu_g2p(w).split())
    return phones


def build_lexicon(entries) -> dict:
    words = {w for *_, text in entries for w in F.mfa_words(text)}
    return {w: telugu_g2p(w).split() for w in words}


def g2p_phoneme_set() -> set:
    """Every phoneme the Telugu G2P can emit."""
    return (set(_TELUGU_VOWELS.values()) | set(_TELUGU_CONSONANTS.values())
            | set(_TELUGU_MATRAS.values()) | set(_TELUGU_POST_VOWEL.values()))


def write_phoneme_set() -> None:
    """Write all Telugu phonemes used to a text file."""
    all_phones = (set(_TELUGU_VOWELS.values()) | set(_TELUGU_CONSONANTS.values())
                  | set(_TELUGU_MATRAS.values()) | set(_TELUGU_POST_VOWEL.values()))
    all_phones |= set(F.SPECIAL_TOKENS)
    path = Path("text/phoneme_set_telugu.txt")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(sorted(all_phones)), encoding="utf-8")
    print(f"Phoneme set ({len(all_phones)} tokens) → {path}")


# ── IndicTTS Loader ───────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_indictts_telugu(raw_dir: Path, cfg: dict):
    """
    Load IndicTTS Telugu dataset.
    Expected local structure (fallback):
        raw_dir/
            wavs/   *.wav
            metadata.csv    (filename|text, no header)

    Download from: https://www.iitm.ac.in/donlab/tts/database.php
    or via HuggingFace: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
    """
    ac  = cfg["audio"]
    sr  = ac["sampling_rate"]
    min_dur, max_dur = ac["min_duration"], ac["max_duration"]

    wav_dir = raw_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # -- Try HuggingFace first (SPRINGLab/IndicTTS_Telugu: columns audio, text,
    #    gender, "duration (s)"; 2 speakers (male/female), ~8.7 h, 48 kHz)
    try:
        from datasets import load_dataset
        name = cfg.get("hf_dataset", "SPRINGLab/IndicTTS_Telugu")
        print(f"Attempting to load {name} via HuggingFace...")
        ds = load_dataset(name, split="train")
        print(f"columns: {ds.column_names}")
        first = ds[0]
        text_col = F.find_column(first, ["text", "sentence", "transcript"])
        spk_col  = F.find_column(first, ["speaker_id", "speaker", "gender"])
        dur_col  = F.find_column(first, ["duration (s)", "duration"])
        spk_names = None
        if spk_col and hasattr(ds.features[spk_col], "names"):
            spk_names = ds.features[spk_col].names        # ClassLabel → "female"/"male"
        print(f"  using text='{text_col}', speaker='{spk_col}' {spk_names or ''}, "
              f"duration='{dur_col}'")

        entries = []
        for i, sample in enumerate(tqdm(ds, desc="Extracting IndicTTS-Telugu")):
            text = telugu_cleaners(str(sample.get(text_col) or ""))
            if not text:
                continue
            if dur_col and sample.get(dur_col) is not None and not (
                    min_dur <= float(sample[dur_col]) <= max_dur):
                continue   # skip without decoding audio

            wav_path = wav_dir / f"telugu_{i:06d}.wav"
            if not wav_path.exists():
                arr, orig_sr = F.audio_to_array(sample["audio"])
                if orig_sr != sr:
                    arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")

            dur = F.get_duration(str(wav_path))
            if min_dur <= dur <= max_dur:
                spk = sample.get(spk_col, "te_spk0") if spk_col else "te_spk0"
                if spk_names is not None and isinstance(spk, int):
                    spk = spk_names[spk]
                entries.append((f"te_{spk}", f"telugu_{i:06d}", str(wav_path), text))

        print(f"Loaded {len(entries)} valid utterances from HuggingFace "
              f"({sum(F.get_duration(e[2]) for e in entries)/3600:.2f} h).")
        return entries

    except Exception as e:
        print(f"HuggingFace load failed ({e}); falling back to local files.")

    # -- Local fallback
    metadata_path = raw_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not load IndicTTS Telugu. "
            f"Place wavs in {wav_dir} and metadata in {metadata_path} "
            f"(format: filename|text, no header)."
        )

    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            fname, text = parts[0], telugu_cleaners("|".join(parts[1:]))
            wav_path = wav_dir / f"{fname}.wav"
            if not text or not wav_path.exists():
                continue
            # resample local files if needed
            if sf.info(str(wav_path)).samplerate != sr:
                arr, _ = librosa.load(str(wav_path), sr=sr, mono=True)
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            dur = F.get_duration(str(wav_path))
            if min_dur <= dur <= max_dur:
                entries.append(("te_spk0", fname, str(wav_path), text))

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

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg["raw_data_dir"])
    prep_dir = Path(cfg["preprocessed_data_dir"])
    prep_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_dir / "entries.json"

    write_phoneme_set()

    # 1. Load
    if args.skip_download:
        entries = F.load_entries(manifest)
        print(f"Loaded {len(entries)} entries from {manifest}")
    else:
        entries = load_indictts_telugu(raw_dir, cfg)
        F.save_entries(entries, manifest)

    # 2. Split (deterministic)
    sp = cfg.get("split", {})
    train, val = F.split_entries(entries, sp.get("train_hours", 5.0),
                                 sp.get("val_hours", 1.0), sp.get("seed", 1234))

    # 3–4. MFA corpus + dictionary + alignment
    dict_path = Path(cfg["mfa"]["dictionary"])
    F.write_mfa_dictionary(build_lexicon(train + val), dict_path)
    if args.mfa_align:
        corpus_dir = raw_dir / "mfa_corpus"
        F.prepare_mfa_corpus(train + val, corpus_dir)
        F.run_mfa_train(corpus_dir, dict_path, prep_dir / "textgrids",
                        prep_dir / "mfa_acoustic_model.zip", cfg["mfa"]["num_jobs"])

    # 5. Features + filelists
    collect = {}
    phones  = F.process_split(train, "train", cfg, prep_dir, text_to_phones, collect)
    phones += F.process_split(val,   "val",   cfg, prep_dir, text_to_phones)

    # 6–7. Stats & vocab
    F.write_stats(collect, prep_dir)
    # include every G2P phoneme so rare ones at test time are never dropped
    F.write_vocab(phones + sorted(g2p_phoneme_set()))
    F.write_speakers(train)
    print("Preprocessing done ✓")


if __name__ == "__main__":
    main()
