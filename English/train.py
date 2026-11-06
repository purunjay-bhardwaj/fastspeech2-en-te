"""
train.py — FastSpeech 2 Training Script
=========================================
Supports both English and Telugu; call from the respective language folder.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --restore_step 50000
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import yaml
from tqdm import tqdm

from model import FastSpeech2


# ── Dataset ───────────────────────────────────────────────────────────────────

class TTSDataset(Dataset):
    """
    Reads filelists produced by preprocess.py.
    Format: uid|speaker|phones|raw_text
    """

    def __init__(self, filelist_path: str, prep_dir: str, cfg: dict, vocab: dict,
                 speakers: dict):
        self.entries   = [l for l in Path(filelist_path).read_text(encoding="utf-8")
                          .strip().split("\n") if l.strip()]
        self.prep_dir  = Path(prep_dir)
        self.cfg       = cfg
        self.vocab     = vocab
        self.speakers  = speakers

        # Load normalisation stats
        stats_path = self.prep_dir / "stats.json"
        if stats_path.exists():
            self.stats = json.loads(stats_path.read_text())
        else:
            print(f"⚠ {stats_path} not found — pitch/energy will not be normalised.")
            self.stats = {}

    def phones_to_ids(self, phones_str: str):
        return [self.vocab[p] for p in phones_str.split() if p in self.vocab]

    def _normalise(self, x: torch.Tensor, key: str) -> torch.Tensor:
        std = self.stats.get(f"{key}_std", 0.0)
        if std > 0:
            x = (x - self.stats[f"{key}_mean"]) / std
        return x

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        parts = self.entries[idx].split("|")
        uid   = parts[0]
        ids   = torch.LongTensor(self.phones_to_ids(parts[2]))

        mel       = torch.FloatTensor(np.load(self.prep_dir / "mels" / f"{uid}.npy")).T  # (T, n_mels)
        durations = torch.LongTensor(np.load(self.prep_dir / "durations" / f"{uid}.npy"))
        pitch     = torch.FloatTensor(np.load(self.prep_dir / "pitches" / f"{uid}.npy"))
        energy    = torch.FloatTensor(np.load(self.prep_dir / "energies" / f"{uid}.npy"))

        assert len(ids) == len(durations) == len(pitch) == len(energy), \
            f"{uid}: phone/feature length mismatch — re-run preprocess.py"

        # mel must have exactly sum(durations) frames
        T_mel = int(durations.sum())
        mel = mel[:T_mel]

        return {
            "ids":       ids,
            "mel":       mel,
            "pitch":     self._normalise(pitch,  "pitch"),
            "energy":    self._normalise(energy, "energy"),
            "durations": durations,
            "uid":       uid,
            "speaker":   self.speakers.get(parts[1], 0),
        }


def load_or_build_speakers(cfg: dict) -> dict:
    """text/speakers.json is written by preprocess.py; rebuild from train list if missing."""
    path = Path("text/speakers.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from collections import Counter
    counts = Counter(l.split("|")[1] for l in
                     Path(cfg["train_filelist"]).read_text(encoding="utf-8").splitlines()
                     if l.count("|") >= 2)
    speakers = {s: i for i, (s, _) in enumerate(counts.most_common())}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(speakers, indent=2, ensure_ascii=False), encoding="utf-8")
    return speakers


def load_or_build_vocab(cfg: dict) -> dict:
    """text/phoneme_vocab.json is written by preprocess.py; rebuild if missing."""
    vocab_file = Path("text/phoneme_vocab.json")
    if vocab_file.exists():
        return json.loads(vocab_file.read_text(encoding="utf-8"))
    special = ["_", "^", "$", "sp"]           # "_" (padding) must be id 0
    phones = set()
    for fl in (cfg["train_filelist"], cfg["val_filelist"]):
        for line in Path(fl).read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                phones |= set(parts[2].split())
    vocab = {p: i for i, p in enumerate(special + sorted(phones - set(special)))}
    vocab_file.parent.mkdir(exist_ok=True)
    vocab_file.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    return vocab


def collate_fn(batch):
    """Pad to max lengths within batch."""
    ids_lens  = [b["ids"].size(0)  for b in batch]
    mel_lens  = [b["mel"].size(0)  for b in batch]
    max_id    = max(ids_lens)
    max_mel   = max(mel_lens)
    n_mels    = batch[0]["mel"].size(1)

    src_ids   = torch.zeros(len(batch), max_id,  dtype=torch.long)
    mels      = torch.zeros(len(batch), max_mel, n_mels)
    pitches   = torch.zeros(len(batch), max_id)
    energies  = torch.zeros(len(batch), max_id)
    durations = torch.zeros(len(batch), max_id,  dtype=torch.long)
    src_masks = torch.ones(len(batch),  max_id,  dtype=torch.bool)
    mel_masks = torch.ones(len(batch),  max_mel, dtype=torch.bool)

    for i, b in enumerate(batch):
        L  = b["ids"].size(0)
        Lm = b["mel"].size(0)
        src_ids[i, :L]     = b["ids"]
        mels[i, :Lm]       = b["mel"]
        pitches[i, :L]     = b["pitch"][:L]
        energies[i, :L]    = b["energy"][:L]
        durations[i, :L]   = b["durations"]
        src_masks[i, :L]   = False   # False = not masked
        mel_masks[i, :Lm]  = False

    return {
        "src_ids":    src_ids,
        "mels":       mels,
        "pitches":    pitches,
        "energies":   energies,
        "durations":  durations,
        "src_masks":  src_masks,
        "mel_masks":  mel_masks,
        "ids_lens":   torch.LongTensor(ids_lens),
        "mel_lens":   torch.LongTensor(mel_lens),
        "speakers":   torch.LongTensor([b["speaker"] for b in batch]),
    }


# ── Loss ─────────────────────────────────────────────────────────────────────

class FastSpeech2Loss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        w = cfg["training"]["loss_weights"]
        self.w_mel  = w.get("mel_loss",      1.0)
        self.w_post = w.get("postnet_loss",  1.0)
        self.w_dur  = w.get("duration_loss", 1.0)
        self.w_pit  = w.get("pitch_loss",    1.0)
        self.w_eng  = w.get("energy_loss",   1.0)

    def forward(self, preds: tuple, targets: tuple, masks: tuple):
        mel_before, mel_after, dur_pred, pitch_pred, energy_pred = preds
        mel_gt, dur_gt, pitch_gt, energy_gt = targets
        src_mask, mel_mask = masks

        mel_valid = ~mel_mask
        src_valid = ~src_mask
        mel_gt = mel_gt[:, :mel_before.size(1)]

        mel_loss    = self.mse(mel_before[mel_valid], mel_gt[mel_valid]) * self.w_mel
        post_loss   = self.mse(mel_after[mel_valid],  mel_gt[mel_valid]) * self.w_post
        dur_loss    = self.mae(dur_pred[src_valid],
                               torch.log(dur_gt.float() + 1)[src_valid]) * self.w_dur
        pitch_loss  = self.mse(pitch_pred[src_valid],  pitch_gt[src_valid])  * self.w_pit
        energy_loss = self.mse(energy_pred[src_valid], energy_gt[src_valid]) * self.w_eng

        total = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss
        return total, {
            "mel":    mel_loss.item(),
            "post":   post_loss.item(),
            "dur":    dur_loss.item(),
            "pitch":  pitch_loss.item(),
            "energy": energy_loss.item(),
        }


# ── LR Scheduler (NoamLR from "Attention is All You Need") ──────────────────

def noam_lambda(step: int, d_model: int, warm_up: int,
                anneal_steps=(), anneal_rate: float = 1.0) -> float:
    """Noam schedule. With learning_rate = 1.0 the peak LR is
    d_model^-0.5 * warm_up^-0.5 (≈ 9.9e-4 for d=256, warm_up=4000)."""
    step = max(step, 1)
    lr = (d_model ** -0.5) * min(step ** -0.5, step * warm_up ** -1.5)
    for s in anneal_steps:
        if step > s:
            lr *= anneal_rate
    return lr


# ── Training Loop ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Loss logging & plots ─────────────────────────────────────────────────────

LOSS_KEYS = ["total", "mel", "post", "dur", "pitch", "energy"]


def append_log(path: Path, step: int, losses: dict) -> None:
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write("step," + ",".join(LOSS_KEYS) + "\n")
        f.write(f"{step}," + ",".join(f"{losses[k]:.6f}" for k in LOSS_KEYS) + "\n")


def read_log(path: Path):
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data) if data.size else None


def plot_losses(log_dir: Path, out_dir: Path, language: str) -> None:
    """Write training_loss.png and validation_loss.png (total + each component)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = read_log(log_dir / "train_loss.csv")
    val   = read_log(log_dir / "val_loss.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data, other in [("training", train, val), ("validation", val, train)]:
        if data is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        ax.plot(data["step"], data["total"], label=f"{name} total", color="#1f4e79", lw=1.6)
        if other is not None:   # overlay the other curve to show the train/val gap
            ax.plot(other["step"], other["total"], label=("validation" if name == "training"
                    else "training") + " total", color="#aaaaaa", lw=1.0, ls="--")
        ax.set_title(f"{language.capitalize()} — {name.capitalize()} loss (total)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_yscale("log")
        ax.grid(alpha=0.3); ax.legend()
        ax = axes[1]
        for k in LOSS_KEYS[1:]:
            ax.plot(data["step"], data[k], label=k, lw=1.2)
        ax.set_title(f"{name.capitalize()} loss components")
        ax.set_xlabel("Step"); ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_loss.png", dpi=150)
        plt.close(fig)
    print(f"  ↳ Loss plots → {out_dir}")


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step}, path)


def export_model(path: Path, model, step: int) -> None:
    """Weights-only fp16 checkpoint (~60 MB) for submission; loadable by inference.py."""
    torch.save({"model": {k: (v.half() if v.is_floating_point() else v)
                          for k, v in model.state_dict().items()}, "step": step}, path)
    print(f"  ↳ Exported submission checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=0,
                        help="Step to resume from; -1 = newest checkpoint")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only redraw loss plots from logs/*.csv")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    log_dir = Path(cfg["paths"]["log_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    if args.plot_only:
        plot_losses(log_dir, out_dir, cfg.get("language", ""))
        return
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    prep_dir = cfg["preprocessed_data_dir"]

    # Datasets
    vocab    = load_or_build_vocab(cfg)
    speakers = load_or_build_speakers(cfg)
    print(f"Vocabulary: {len(vocab)} symbols | Speakers: {len(speakers)}")
    train_ds = TTSDataset(cfg["train_filelist"], prep_dir, cfg, vocab, speakers)
    val_ds   = TTSDataset(cfg["val_filelist"],   prep_dir, cfg, vocab, speakers)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2)

    vocab_size = len(vocab)
    model      = FastSpeech2(cfg, vocab_size, len(speakers)).to(device)
    criterion  = FastSpeech2Loss(cfg)

    tc = cfg["training"]
    optimizer  = Adam(model.parameters(), lr=tc["learning_rate"],
                      betas=tc["betas"], eps=tc["eps"],
                      weight_decay=tc["weight_decay"])
    scheduler  = LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lambda(
            step, cfg["model"]["encoder_hidden"], tc["warm_up_step"],
            tc.get("anneal_steps", []), tc.get("anneal_rate", 1.0),
        ),
    )

    # Restore checkpoint
    ckpt_dir  = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    step = args.restore_step
    if step < 0:   # -1 → resume from the newest checkpoint, if any
        found = sorted(ckpt_dir.glob("checkpoint_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        step = int(found[-1].stem.split("_")[1]) if found else 0
    log_dir.mkdir(parents=True, exist_ok=True)
    if step == 0:   # fresh run: start new loss logs
        for f in ("train_loss.csv", "val_loss.csv"):
            (log_dir / f).unlink(missing_ok=True)

    if step > 0:
        ckpt = ckpt_dir / f"checkpoint_{step}.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"Restored checkpoint at step {step}")
        else:
            print(f"Checkpoint not found: {ckpt}, starting from scratch.")
            step = 0

    model.train()
    total_steps = tc["total_steps"]
    log_step    = tc["log_step"]
    save_step   = tc["save_step"]
    val_step    = tc["val_step"]

    loader_iter = iter(train_loader)
    t0 = time.time()
    running = {k: 0.0 for k in LOSS_KEYS}
    n_running = 0

    print(f"Starting training from step {step} → {total_steps}")
    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        for k in ["src_ids", "mels", "pitches", "energies", "durations",
                  "src_masks", "mel_masks", "speakers"]:
            batch[k] = batch[k].to(device)

        # Forward
        (mel_before, mel_after, dur_pred, pitch_pred, energy_pred,
         mel_len, mel_mask_out) = model(
            batch["src_ids"],
            src_mask=batch["src_masks"],
            mel_mask=batch["mel_masks"],
            duration_target=batch["durations"],
            pitch_target=batch["pitches"],
            energy_target=batch["energies"],
            speakers=batch["speakers"],
        )

        loss, loss_dict = criterion(
            preds=(mel_before, mel_after, dur_pred, pitch_pred, energy_pred),
            targets=(batch["mels"], batch["durations"], batch["pitches"], batch["energies"]),
            masks=(batch["src_masks"], batch["mel_masks"]),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_thresh"])
        optimizer.step()
        scheduler.step()
        step += 1
        running["total"] += loss.item()
        for k in LOSS_KEYS[1:]:
            running[k] += loss_dict[k]
        n_running += 1

        # Logging
        if step % log_step == 0:
            elapsed = time.time() - t0
            lr      = optimizer.param_groups[0]["lr"]
            msg = (f"Step {step:>7d} | loss {loss.item():.4f} | "
                   f"mel {loss_dict['mel']:.3f} | dur {loss_dict['dur']:.3f} | "
                   f"pit {loss_dict['pitch']:.3f} | eng {loss_dict['energy']:.3f} | "
                   f"lr {lr:.2e} | {elapsed:.0f}s")
            print(msg)
            append_log(log_dir / "train_loss.csv", step,
                       {k: v / n_running for k, v in running.items()})
            running = {k: 0.0 for k in LOSS_KEYS}
            n_running = 0
            t0 = time.time()

        # Validation
        if step % val_step == 0 and len(val_ds) > 0:
            model.eval()
            val_losses = {k: [] for k in LOSS_KEYS}
            with torch.no_grad():
                for vb in val_loader:
                    for k in ["src_ids", "mels", "pitches", "energies", "durations",
                              "src_masks", "mel_masks", "speakers"]:
                        vb[k] = vb[k].to(device)
                    (mb, ma, dp, pp, ep, _, _) = model(
                        vb["src_ids"], src_mask=vb["src_masks"],
                        mel_mask=vb["mel_masks"], duration_target=vb["durations"],
                        pitch_target=vb["pitches"], energy_target=vb["energies"],
                        speakers=vb["speakers"],
                    )
                    vl, vd = criterion(
                        (mb, ma, dp, pp, ep),
                        (vb["mels"], vb["durations"], vb["pitches"], vb["energies"]),
                        (vb["src_masks"], vb["mel_masks"]),
                    )
                    val_losses["total"].append(vl.item())
                    for k in LOSS_KEYS[1:]:
                        val_losses[k].append(vd[k])
            val_mean = {k: float(np.mean(v)) for k, v in val_losses.items()}
            append_log(log_dir / "val_loss.csv", step, val_mean)
            print(f"  ↳ Val loss: {val_mean['total']:.4f}")
            plot_losses(log_dir, out_dir, cfg.get("language", ""))
            model.train()

        # Checkpoint
        if step % save_step == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step)
            print(f"  ↳ Saved checkpoint: {ckpt_path}")
            # keep only the newest N full checkpoints (each ~350 MB)
            keep = tc.get("keep_last_checkpoints", 2)
            old_ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"),
                               key=lambda p: int(p.stem.split("_")[1]))
            for p in old_ckpts[:-keep] if keep > 0 else []:
                p.unlink()

    # Final save
    ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, step)
    export_model(ckpt_dir / "model_final_fp16.pt", model, step)
    plot_losses(log_dir, out_dir, cfg.get("language", ""))
    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
