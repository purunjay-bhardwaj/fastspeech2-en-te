"""
prep_dataset.py  —  Telugu (IndicTTS) Dataset Preprocessing Pipeline
======================================================================
Steps:
  1. Download/load IndicTTS Telugu from AI4Bharat or iitm-tts mirrors
  2. Resample to 22050 Hz
  3. Split: 5 h train / 1 h val
  4. Telugu G2P: Indic script → IPA/Devanagari phonemes using indic-nlp
  5. Build MFA corpus + pronunciation dictionary
  6. Extract mel, pitch, energy features
  7. Write filelists

Usage:
    python prep_dataset.py --config config.yaml [--mfa_align]
"""

import argparse
import os
import re
import json
import random
import shutil
import subprocess
import yaml
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


# ── Telugu G2P (Grapheme-to-Phoneme) ────────────────────────────────────────

# Telugu vowel nucleus map (simplified IPA representation for MFA)
_TELUGU_VOWELS = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii',
    'ఉ': 'u', 'ఊ': 'uu', 'ఋ': 'ru', 'ఎ': 'e',
    'ఏ': 'ee', 'ఐ': 'ai', 'ఒ': 'o', 'ఓ': 'oo',
    'ఔ': 'au', 'అం': 'am', 'అః': 'ah',
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

_TELUGU_MATRAS = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu',
    'ృ': 'ru', 'ె': 'e', 'ే': 'ee', 'ై': 'ai',
    'ొ': 'o', 'ో': 'oo', 'ౌ': 'au',
    'ం': 'M',   # anusvara
    'ః': 'H',   # visarga
    '్': '',    # halant / virama (suppresses inherent 'a')
}


def telugu_g2p(text: str) -> str:
    """
    Naive Telugu G2P using character-level rules.
    Returns space-separated phoneme string.

    For production: replace with indic-nlp G2P or IndicG2P.
    """
    try:
        # Try using indic_unified_parser if available
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer("te")
        text = normalizer.normalize(text)
    except ImportError:
        pass

    phones = []
    i = 0
    chars = list(text)

    while i < len(chars):
        c = chars[i]

        # Consonant
        if c in _TELUGU_CONSONANTS:
            phones.append(_TELUGU_CONSONANTS[c])
            # check for virama (halant) or matra next
            if i + 1 < len(chars):
                nxt = chars[i + 1]
                if nxt == '్':          # halant: no inherent vowel
                    i += 2
                    continue
                elif nxt in _TELUGU_MATRAS:
                    m = _TELUGU_MATRAS[nxt]
                    if m:
                        phones.append(m)
                    i += 2
                    continue
                else:
                    phones.append('a')   # inherent vowel
            else:
                phones.append('a')
            i += 1

        # Independent vowel
        elif c in _TELUGU_VOWELS:
            phones.append(_TELUGU_VOWELS[c])
            i += 1

        # Matra without preceding consonant (shouldn't happen, skip)
        elif c in _TELUGU_MATRAS:
            m = _TELUGU_MATRAS[c]
            if m:
                phones.append(m)
            i += 1

        # ASCII digit or space
        elif c == ' ':
            phones.append('|')   # word boundary
            i += 1

        # Skip unknown
        else:
            i += 1

    # collapse repeated separators
    result = " ".join(p for p in phones if p)
    return result


def build_telugu_mfa_dict(entries, dict_path: Path) -> None:
    """Generate MFA pronunciation dictionary for Telugu."""
    all_words: dict = {}
    for _, _, _, text in entries:
        for word in text.split():
            if word not in all_words:
                phones = telugu_g2p(word)
                if phones.strip():
                    all_words[word] = phones

    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dict_path, "w", encoding="utf-8") as f:
        for word, phones in sorted(all_words.items()):
            f.write(f"{word}\t{phones}\n")
    print(f"Telugu MFA dictionary: {len(all_words)} entries → {dict_path}")


def write_phoneme_set(prep_dir: Path) -> None:
    """Write all Telugu phonemes used to a text file."""
    all_phones = set(_TELUGU_VOWELS.values()) | set(_TELUGU_CONSONANTS.values())
    all_phones |= {v for v in _TELUGU_MATRAS.values() if v}
    # add special tokens
    all_phones |= {"_", "^", "$", "|", "M", "H"}
    phoneme_set_path = Path("text/phoneme_set_telugu.txt")
    phoneme_set_path.parent.mkdir(exist_ok=True)
    phoneme_set_path.write_text("\n".join(sorted(all_phones)), encoding="utf-8")
    print(f"Phoneme set ({len(all_phones)} tokens) → {phoneme_set_path}")


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def telugu_cleaners(text: str) -> str:
    """Remove non-Telugu characters (keep spaces)."""
    # Keep Telugu Unicode block: U+0C00–U+0C7F, plus space
    text = re.sub(r"[^ఀ-౿\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Helpers (shared with English script) ────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def extract_mel(wav: np.ndarray, cfg: dict) -> np.ndarray:
    ac = cfg["audio"]
    mel = librosa.feature.melspectrogram(
        y=wav, sr=ac["sampling_rate"], n_fft=ac["n_fft"],
        hop_length=ac["hop_length"], win_length=ac["win_length"],
        n_mels=ac["n_mels"], fmin=ac["fmin"], fmax=ac["fmax"],
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_pitch(wav: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    try:
        import pyworld as pw
        wav64 = wav.astype(np.float64)
        frame_ms = (hop_length / sr) * 1000
        f0, _ = pw.harvest(wav64, sr, frame_period=frame_ms)
        return f0.astype(np.float32)
    except ImportError:
        return librosa.yin(wav, fmin=65, fmax=600, hop_length=hop_length).astype(np.float32)


def extract_energy(wav: np.ndarray, cfg: dict) -> np.ndarray:
    ac = cfg["audio"]
    rms = librosa.feature.rms(
        y=wav, frame_length=ac["win_length"], hop_length=ac["hop_length"]
    )[0]
    return rms.astype(np.float32)


# ── IndicTTS Loader ───────────────────────────────────────────────────────────

def load_indictts_telugu(raw_dir: Path, cfg: dict):
    """
    Load IndicTTS Telugu dataset.
    Expected structure after download:
        raw_dir/
            wavs/   *.wav
            metadata.csv    (filename|text)   OR   lines.txt

    Download from: https://www.iitm.ac.in/donlab/tts/database.php
    or via AI4Bharat: https://huggingface.co/datasets/ai4bharat/IndicTTS

    If HuggingFace is accessible, this function attempts to load it
    automatically. Otherwise it falls back to the local file structure.
    """
    ac  = cfg["audio"]
    sr  = ac["sampling_rate"]
    min_dur = ac["min_duration"]
    max_dur = ac["max_duration"]

    wav_dir = raw_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # -- Try HuggingFace first
    try:
        from datasets import load_dataset
        print("Attempting to load IndicTTS Telugu via HuggingFace...")
        ds = load_dataset("ai4bharat/IndicTTS", "te", split="train", trust_remote_code=True)
        entries = []
        for i, sample in enumerate(tqdm(ds, desc="Extracting IndicTTS-Telugu")):
            raw_text = sample.get("text", sample.get("sentence", ""))
            text = telugu_cleaners(raw_text)
            if not text:
                continue

            wav_path = wav_dir / f"telugu_{i:06d}.wav"
            if not wav_path.exists():
                audio = sample["audio"]
                arr = np.array(audio["array"], dtype=np.float32)
                orig_sr = audio["sampling_rate"]
                if orig_sr != sr:
                    arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")

            dur = get_duration(str(wav_path))
            if min_dur <= dur <= max_dur:
                spk = sample.get("speaker_id", "te_spk0")
                entries.append((str(spk), f"telugu_{i:06d}", str(wav_path), text))

        print(f"Loaded {len(entries)} valid utterances from HuggingFace.")
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
            fname, text = parts[0], "|".join(parts[1:])
            text = telugu_cleaners(text)
            if not text:
                continue
            wav_path = wav_dir / f"{fname}.wav"
            if not wav_path.exists():
                continue
            dur = get_duration(str(wav_path))
            if min_dur <= dur <= max_dur:
                entries.append(("te_spk0", fname, str(wav_path), text))

    return entries


# ── MFA Alignment ────────────────────────────────────────────────────────────

def prepare_mfa_corpus(entries, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for spk, uid, wav_path, text in tqdm(entries, desc="Building MFA corpus"):
        spk_dir = corpus_dir / spk
        spk_dir.mkdir(exist_ok=True)
        shutil.copy2(wav_path, spk_dir / f"{uid}.wav")
        (spk_dir / f"{uid}.lab").write_text(text, encoding="utf-8")


def run_mfa(corpus_dir: Path, dict_path: Path, output_dir: Path,
            acoustic_model: str, num_jobs: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa", "align",
        str(corpus_dir), str(dict_path), acoustic_model,
        str(output_dir),
        "--num_jobs", str(num_jobs), "--clean",
    ]
    print("Running MFA:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mfa_align", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg["raw_data_dir"])
    prep_dir = Path(cfg["preprocessed_data_dir"])
    prep_dir.mkdir(parents=True, exist_ok=True)

    # Write phoneme set
    write_phoneme_set(prep_dir)

    # 1. Load
    if not args.skip_download:
        entries = load_indictts_telugu(raw_dir, cfg)
    else:
        wav_dir = raw_dir / "wavs"
        meta    = raw_dir / "metadata.csv"
        entries = []
        with open(meta, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                fname, text = parts[0], parts[1]
                wav_path = wav_dir / f"{fname}.wav"
                if wav_path.exists():
                    entries.append(("te_spk0", fname, str(wav_path), telugu_cleaners(text)))

    # 2. Split
    random.shuffle(entries)
    train, val = [], []
    train_secs = val_secs = 0.0
    for e in entries:
        dur = get_duration(e[2])
        if train_secs / 3600 < 5.0:
            train.append(e); train_secs += dur
        elif val_secs / 3600 < 1.0:
            val.append(e); val_secs += dur
        else:
            break
    print(f"Train: {len(train)} utts ({train_secs/3600:.2f} h)")
    print(f"Val:   {len(val)} utts ({val_secs/3600:.2f} h)")

    # 3. Build MFA corpus + dictionary
    corpus_dir = raw_dir / "mfa_corpus"
    dict_path  = Path(cfg["mfa"]["dictionary"])
    prepare_mfa_corpus(train + val, corpus_dir)
    build_telugu_mfa_dict(train + val, dict_path)

    # 4. MFA alignment
    if args.mfa_align:
        run_mfa(corpus_dir, dict_path, prep_dir / "textgrids",
                cfg["mfa"]["acoustic_model"], cfg["mfa"]["num_jobs"])

    # 5. Feature extraction
    mel_dir    = prep_dir / "mels"
    pitch_dir  = prep_dir / "pitches"
    energy_dir = prep_dir / "energies"
    for d in [mel_dir, pitch_dir, energy_dir]:
        d.mkdir(exist_ok=True)

    all_pitches, all_energies = [], []

    def process_split(split_entries, split_name: str):
        lines = []
        for spk, uid, wav_path, text in tqdm(split_entries, desc=f"Features [{split_name}]"):
            wav, _ = librosa.load(wav_path, sr=cfg["audio"]["sampling_rate"], mono=True)

            mel    = extract_mel(wav, cfg)
            pitch  = extract_pitch(wav, cfg["audio"]["sampling_rate"], cfg["audio"]["hop_length"])
            energy = extract_energy(wav, cfg)

            T = mel.shape[1]
            pitch  = pitch[:T]  if len(pitch)  >= T else np.pad(pitch,  (0, T - len(pitch)))
            energy = energy[:T] if len(energy) >= T else np.pad(energy, (0, T - len(energy)))

            phones = telugu_g2p(text)

            np.save(mel_dir    / f"{uid}.npy", mel)
            np.save(pitch_dir  / f"{uid}.npy", pitch)
            np.save(energy_dir / f"{uid}.npy", energy)

            all_pitches.append(pitch[pitch > 0])
            all_energies.append(energy)

            lines.append(f"{uid}|{spk}|{phones}|{text}")

        out = prep_dir / f"{split_name}.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out}")

    process_split(train, "train")
    process_split(val,   "val")

    # 6. Stats
    all_p = np.concatenate(all_pitches)
    all_e = np.concatenate(all_energies)
    stats = {
        "pitch_mean":  float(np.mean(all_p)),
        "pitch_std":   float(np.std(all_p)),
        "pitch_min":   float(np.min(all_p)),
        "pitch_max":   float(np.max(all_p)),
        "energy_mean": float(np.mean(all_e)),
        "energy_std":  float(np.std(all_e)),
        "energy_min":  float(np.min(all_e)),
        "energy_max":  float(np.max(all_e)),
    }
    (prep_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Preprocessing done ✓")


if __name__ == "__main__":
    main()
