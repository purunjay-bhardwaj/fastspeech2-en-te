"""
features.py — Shared preprocessing helpers (identical copy in English/ and Telugu/)
====================================================================================
  - Acoustic features: log-mel (HiFi-GAN compatible), F0, energy
  - MFA corpus / dictionary / alignment and TextGrid → frame durations
  - Phoneme-level averaging of pitch & energy
  - Reproducible train/val split, entry manifest, vocabulary, stats
"""

import json
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# MFA labels for silence / unknown speech — all mapped to a single "sp" token
SIL_PHONES = {"", "sil", "sp", "spn", "<eps>"}
SPECIAL_TOKENS = ["_", "^", "$", "sp"]   # "_" = padding → id 0
VOCAB_PATH = Path("text/phoneme_vocab.json")
SPEAKERS_PATH = Path("text/speakers.json")

Entry = Tuple[str, str, str, str]       # (speaker, uid, wav_path, text)


# ── Audio helpers ────────────────────────────────────────────────────────────

def get_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def audio_to_array(audio) -> Tuple[np.ndarray, int]:
    """HF `datasets` audio → (mono float32 array, sr). Handles dict and torchcodec decoders."""
    if isinstance(audio, dict):
        arr, sr = np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    else:  # datasets>=4 returns a torchcodec AudioDecoder
        samples = audio.get_all_samples()
        arr, sr = samples.data.numpy().astype(np.float32), int(samples.sample_rate)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr, sr


def fit_length(x: np.ndarray, T: int) -> np.ndarray:
    """Trim or edge-pad a 1-D frame sequence to exactly T frames."""
    if len(x) >= T:
        return x[:T]
    return np.pad(x, (0, T - len(x)), mode="edge") if len(x) else np.zeros(T, x.dtype)


def extract_mel(wav: np.ndarray, cfg: dict) -> np.ndarray:
    """Natural-log magnitude mel (n_mels × T), the format HiFi-GAN expects."""
    import librosa
    ac = cfg["audio"]
    mel = librosa.feature.melspectrogram(
        y=wav, sr=ac["sampling_rate"], n_fft=ac["n_fft"],
        hop_length=ac["hop_length"], win_length=ac["win_length"],
        n_mels=ac["n_mels"], fmin=ac["fmin"], fmax=ac["fmax"], power=1.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)


def extract_pitch(wav: np.ndarray, sr: int, hop_length: int, n_frames: int) -> np.ndarray:
    """Frame-level F0 in Hz (0 = unvoiced), aligned to the mel frames."""
    try:
        import pyworld as pw
        wav64 = wav.astype(np.float64)
        f0, t = pw.dio(wav64, sr, frame_period=hop_length / sr * 1000)
        f0 = pw.stonemask(wav64, f0, t, sr)
    except ImportError:
        import librosa
        f0, voiced, _ = librosa.pyin(wav, fmin=65, fmax=600, sr=sr, hop_length=hop_length)
        f0 = np.where(voiced, np.nan_to_num(f0), 0.0)
    return fit_length(f0.astype(np.float32), n_frames)


def interpolate_unvoiced(f0: np.ndarray) -> Optional[np.ndarray]:
    """Linearly interpolate F0 across unvoiced frames. None if (almost) nothing is voiced."""
    voiced = f0 > 0
    if voiced.sum() < 2:
        return None
    idx = np.arange(len(f0))
    return np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)


def extract_energy(wav: np.ndarray, cfg: dict, n_frames: int) -> np.ndarray:
    import librosa
    ac = cfg["audio"]
    rms = librosa.feature.rms(y=wav, frame_length=ac["n_fft"], hop_length=ac["hop_length"])[0]
    return fit_length(rms.astype(np.float32), n_frames)


def average_by_duration(values: np.ndarray, durations: Sequence[int]) -> np.ndarray:
    """Average frame-level values to phoneme level using integer frame durations."""
    out, idx = [], 0
    for d in durations:
        d = int(d)
        if d > 0:
            out.append(float(np.mean(values[idx: idx + d])))
        else:  # zero-length phoneme: take the value at the boundary
            out.append(float(values[min(idx, len(values) - 1)]))
        idx += d
    return np.array(out, dtype=np.float32)


# ── Durations ────────────────────────────────────────────────────────────────

def read_textgrid_phones(path: Path) -> Optional[List[Tuple[float, float, str]]]:
    """Parse the 'phones' tier of a long-format (MFA default) TextGrid."""
    text = path.read_text(encoding="utf-8")
    for tier in re.split(r"item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "([^"]*)"', tier)
        if name and name.group(1) == "phones":
            ivs = re.findall(
                r'xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "([^"]*)"', tier)
            return [(float(a), float(b), t.strip()) for a, b, t in ivs] or None
    return None


def durations_from_textgrid(intervals, sr: int, hop: int,
                            n_frames: int) -> Tuple[List[str], List[int]]:
    """Convert phone intervals to (phones, frame durations) summing to n_frames."""
    phones, durs = [], []
    for start, end, p in intervals:
        p = "sp" if p in SIL_PHONES else p
        d = int(round(end * sr / hop)) - int(round(start * sr / hop))
        if phones and p == "sp" and phones[-1] == "sp":
            durs[-1] += d
            continue
        phones.append(p)
        durs.append(d)

    diff = n_frames - sum(durs)
    if diff >= 0:
        durs[-1] += diff
    else:  # alignment slightly longer than the mel: shave frames off the end
        i = len(durs) - 1
        while diff < 0 and i >= 0:
            take = min(durs[i], -diff)
            durs[i] -= take
            diff += take
            i -= 1
    return phones, durs


def uniform_durations(n_phones: int, n_frames: int) -> List[int]:
    """Fallback when no alignment exists: spread frames evenly over phonemes."""
    base = np.full(n_phones, n_frames // n_phones, dtype=np.int64)
    base[: n_frames % n_phones] += 1
    return base.tolist()


def find_textgrid(textgrid_dir: Path, spk: str, uid: str) -> Optional[Path]:
    for p in (textgrid_dir / spk / f"{uid}.TextGrid", textgrid_dir / f"{uid}.TextGrid"):
        if p.exists():
            return p
    return None


# ── MFA ──────────────────────────────────────────────────────────────────────

def mfa_words(text: str) -> List[str]:
    """Tokenise a transcript the same way for the .lab files and the dictionary."""
    return re.sub(r"[.,!?;:\"()\[\]\-]", " ", text).split()


def prepare_mfa_corpus(entries: Sequence[Entry], corpus_dir: Path) -> None:
    """corpus_dir/<speaker>/<uid>.wav + <uid>.lab"""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for spk, uid, wav_src, text in tqdm(entries, desc="Building MFA corpus"):
        spk_dir = corpus_dir / spk
        spk_dir.mkdir(exist_ok=True)
        dst = spk_dir / f"{uid}.wav"
        if not dst.exists():
            shutil.copy2(wav_src, dst)
        (spk_dir / f"{uid}.lab").write_text(" ".join(mfa_words(text)), encoding="utf-8")


def write_mfa_dictionary(lexicon: Dict[str, List[str]], dict_path: Path) -> None:
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dict_path, "w", encoding="utf-8") as f:
        for word, phones in sorted(lexicon.items()):
            if phones:
                f.write(f"{word}\t{' '.join(phones)}\n")
    print(f"MFA dictionary: {len(lexicon)} words → {dict_path}")


def run_mfa_train(corpus_dir: Path, dict_path: Path, textgrid_dir: Path,
                  model_out: Path, num_jobs: int = 4) -> bool:
    """
    Train an acoustic model on our own corpus + dictionary and export alignments.
    Using our own dictionary guarantees TextGrid phones == the G2P phones used at
    inference (a pretrained ARPAbet model would produce a different phone set).
    """
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mfa", "train", str(corpus_dir), str(dict_path), str(model_out),
           "--output_directory", str(textgrid_dir),
           "--num_jobs", str(num_jobs), "--clean"]
    print("Running MFA:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Don't lose the whole preprocessing run: fall back to uniform durations.
        print(f"⚠ MFA failed ({e}). Continuing WITHOUT alignments — durations will be "
              f"uniform. Fix MFA and re-run with --mfa_align --skip_download.")
        return False


# ── Entries / split ──────────────────────────────────────────────────────────

def save_entries(entries: Sequence[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(e) for e in entries], ensure_ascii=False, indent=1),
                    encoding="utf-8")


def load_entries(path: Path) -> List[Entry]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run once without --skip_download.")
    return [tuple(e) for e in json.loads(path.read_text(encoding="utf-8"))]


def split_entries(entries: Sequence[Entry], train_hours: float = 5.0,
                  val_hours: float = 1.0, seed: int = 1234):
    """Deterministic split (same result on every run)."""
    entries = sorted(entries, key=lambda e: e[1])
    random.Random(seed).shuffle(entries)
    train, val = [], []
    train_secs = val_secs = 0.0
    for e in entries:
        dur = get_duration(e[2])
        if train_secs < train_hours * 3600:
            train.append(e); train_secs += dur
        elif val_secs < val_hours * 3600:
            val.append(e); val_secs += dur
        else:
            break
    print(f"Train: {len(train)} utts ({train_secs/3600:.2f} h)")
    print(f"Val:   {len(val)} utts ({val_secs/3600:.2f} h)")
    return train, val


# ── Feature extraction for a split ───────────────────────────────────────────

def process_split(entries: Sequence[Entry], split_name: str, cfg: dict, prep_dir: Path,
                  text_to_phones: Callable[[str], List[str]],
                  collect: Optional[dict] = None) -> List[str]:
    """
    Writes per-utterance mels/, pitches/, energies/, durations/ (.npy; pitch & energy
    are phoneme-level, raw Hz / RMS) and <split>.txt. Returns all phones seen.
    """
    import librosa
    ac = cfg["audio"]
    sr, hop = ac["sampling_rate"], ac["hop_length"]
    textgrid_dir = prep_dir / "textgrids"
    dirs = {k: prep_dir / k for k in ("mels", "pitches", "energies", "durations")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    lines, all_phones = [], set()
    n_aligned = n_skipped = 0
    for spk, uid, wav_path, text in tqdm(entries, desc=f"Features [{split_name}]"):
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        mel = extract_mel(wav, cfg)
        T = mel.shape[1]

        tg = find_textgrid(textgrid_dir, spk, uid)
        intervals = read_textgrid_phones(tg) if tg else None
        if intervals:
            phones, durs = durations_from_textgrid(intervals, sr, hop, T)
            n_aligned += 1
        else:
            phones = text_to_phones(text)
            if not phones:
                n_skipped += 1
                continue
            durs = uniform_durations(len(phones), T)

        f0 = interpolate_unvoiced(extract_pitch(wav, sr, hop, T))
        if f0 is None:
            n_skipped += 1
            continue
        energy = extract_energy(wav, cfg, T)

        pitch_ph  = average_by_duration(f0, durs)
        energy_ph = average_by_duration(energy, durs)

        np.save(dirs["mels"]      / f"{uid}.npy", mel)
        np.save(dirs["pitches"]   / f"{uid}.npy", pitch_ph)
        np.save(dirs["energies"]  / f"{uid}.npy", energy_ph)
        np.save(dirs["durations"] / f"{uid}.npy", np.asarray(durs, dtype=np.int64))

        if collect is not None:
            collect.setdefault("pitch", []).append(pitch_ph)
            collect.setdefault("energy", []).append(energy_ph)
        all_phones.update(phones)
        lines.append(f"{uid}|{spk}|{' '.join(phones)}|{text}")

    out_path = prep_dir / f"{split_name}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}: {len(lines)} utts "
          f"({n_aligned} MFA-aligned, {len(lines) - n_aligned} uniform durations, "
          f"{n_skipped} skipped)")
    if lines and n_aligned == 0:
        print("  ⚠ No TextGrids found — durations are uniform. Run with --mfa_align "
              "for real phoneme durations.")
    return sorted(all_phones)


def write_stats(collect: dict, prep_dir: Path) -> None:
    p = np.concatenate(collect["pitch"])
    e = np.concatenate(collect["energy"])
    stats = {
        "pitch_mean":  float(p.mean()), "pitch_std":  float(p.std()),
        "pitch_min":   float(p.min()),  "pitch_max":  float(p.max()),
        "energy_mean": float(e.mean()), "energy_std": float(e.std()),
        "energy_min":  float(e.min()),  "energy_max": float(e.max()),
    }
    (prep_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Stats saved to {prep_dir / 'stats.json'}")


def write_vocab(phones: Sequence[str], path: Path = VOCAB_PATH) -> Dict[str, int]:
    symbols = SPECIAL_TOKENS + sorted(set(phones) - set(SPECIAL_TOKENS))
    vocab = {p: i for i, p in enumerate(symbols)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Vocabulary ({len(vocab)} symbols) → {path}")
    return vocab


def write_speakers(train_entries: Sequence[Entry], path: Path = SPEAKERS_PATH) -> Dict[str, int]:
    """Speaker → id map, ordered by amount of training data (id 0 = largest speaker)."""
    from collections import Counter
    counts = Counter(e[0] for e in train_entries)
    speakers = {spk: i for i, (spk, _) in enumerate(counts.most_common())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(speakers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Speakers ({len(speakers)}) → {path}  (most data: {next(iter(speakers), None)})")
    return speakers


def find_column(sample: dict, candidates: Sequence[str]) -> Optional[str]:
    """First matching column name in a HF sample (case-insensitive)."""
    lower = {k.lower(): k for k in sample.keys()}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None
