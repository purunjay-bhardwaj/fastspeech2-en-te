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
    Reads filelists produced by prep_dataset.py.
    Format: uid|speaker|phones|raw_text
    """

    def __init__(self, filelist_path: str, prep_dir: str, cfg: dict):
        self.entries   = Path(filelist_path).read_text().strip().split("\n")
        self.prep_dir  = Path(prep_dir)
        self.cfg       = cfg

        # Load phoneme vocabulary
        vocab_file = Path("text/phoneme_vocab.json")
        if vocab_file.exists():
            self.vocab = json.loads(vocab_file.read_text())
        else:
            self.vocab = self._build_vocab()
            vocab_file.parent.mkdir(exist_ok=True)
            vocab_file.write_text(json.dumps(self.vocab, indent=2, ensure_ascii=False))

        # Load normalisation stats
        stats_path = self.prep_dir / "stats.json"
        if stats_path.exists():
            self.stats = json.loads(stats_path.read_text())
        else:
            self.stats = {}

    def _build_vocab(self):
        """Build phoneme vocab from all filelists in prep_dir."""
        phones = set(["_", "^", "$"])   # pad, bos, eos
        for entry in self.entries:
            parts = entry.split("|")
            if len(parts) >= 3:
                phones |= set(parts[2].split())
        vocab = {p: i for i, p in enumerate(sorted(phones))}
        return vocab

    def phones_to_ids(self, phones_str: str):
        return [self.vocab.get(p, 0) for p in phones_str.split()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        parts = self.entries[idx].split("|")
        uid   = parts[0]
        # phones
        ids   = self.phones_to_ids(parts[2]) if len(parts) > 2 else [0]
        ids   = torch.LongTensor(ids)

        # mel
        mel_path = self.prep_dir / "mels" / f"{uid}.npy"
        mel = torch.FloatTensor(np.load(mel_path)).T  # (T, n_mels)

        # pitch
        p_path = self.prep_dir / "pitches" / f"{uid}.npy"
        pitch  = torch.FloatTensor(np.load(p_path))
        if self.stats.get("pitch_std", 1.0) > 0:
            pitch = (pitch - self.stats.get("pitch_mean", 0)) / self.stats.get("pitch_std", 1)

        # energy
        e_path = self.prep_dir / "energies" / f"{uid}.npy"
        energy = torch.FloatTensor(np.load(e_path))
        if self.stats.get("energy_std", 1.0) > 0:
            energy = (energy - self.stats.get("energy_mean", 0)) / self.stats.get("energy_std", 1)

        # dummy duration (equal spread)
        T_phone = len(ids)
        T_mel   = mel.size(0)
        dur_per_phone = T_mel // max(T_phone, 1)
        durations = torch.LongTensor([dur_per_phone] * T_phone)
        # adjust last to match total
        diff = T_mel - durations.sum().item()
        if diff > 0:
            durations[-1] += diff

        # trim pitch/energy to T_phone (phoneme-level)
        pitch  = pitch[:T_phone]
        energy = energy[:T_phone]

        return {
            "ids":       ids,
            "mel":       mel,
            "pitch":     pitch,
            "energy":    energy,
            "durations": durations,
            "uid":       uid,
        }


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

        # mask out padding
        def apply_mask(x, mask):
            return x.masked_fill(mask.unsqueeze(-1) if x.dim() == 3 else mask, 0.0)

        mel_before = apply_mask(mel_before, mel_mask)
        mel_after  = apply_mask(mel_after,  mel_mask)
        mel_gt     = apply_mask(mel_gt,     mel_mask)
        dur_pred   = dur_pred.masked_fill(src_mask,   0.0)
        pitch_pred = pitch_pred.masked_fill(src_mask, 0.0)
        energy_pred= energy_pred.masked_fill(src_mask,0.0)

        mel_loss    = self.mse(mel_before, mel_gt) * self.w_mel
        post_loss   = self.mse(mel_after,  mel_gt) * self.w_post
        dur_loss    = self.mae(dur_pred, torch.log(dur_gt.float() + 1)) * self.w_dur
        pitch_loss  = self.mse(pitch_pred, pitch_gt)   * self.w_pit
        energy_loss = self.mse(energy_pred, energy_gt) * self.w_eng

        total = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss
        return total, {
            "mel":    mel_loss.item(),
            "post":   post_loss.item(),
            "dur":    dur_loss.item(),
            "pitch":  pitch_loss.item(),
            "energy": energy_loss.item(),
        }


# ── LR Scheduler (NoamLR from "Attention is All You Need") ──────────────────

def noam_lambda(step: int, d_model: int, warm_up: int) -> float:
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warm_up ** -1.5)


# ── Training Loop ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_vocab_size(prep_dir: str) -> int:
    vocab_path = Path("text/phoneme_vocab.json")
    if vocab_path.exists():
        return len(json.loads(vocab_path.read_text()))
    # estimate from phoneme sets
    return 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    prep_dir = cfg["preprocessed_data_dir"]

    # Datasets
    train_ds = TTSDataset(cfg["train_filelist"], prep_dir, cfg)
    val_ds   = TTSDataset(cfg["val_filelist"],   prep_dir, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2)

    vocab_size = len(train_ds.vocab)
    model      = FastSpeech2(cfg, vocab_size).to(device)
    criterion  = FastSpeech2Loss(cfg)

    tc = cfg["training"]
    optimizer  = Adam(model.parameters(), lr=tc["learning_rate"],
                      betas=tc["betas"], eps=tc["eps"],
                      weight_decay=tc["weight_decay"])
    scheduler  = LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lambda(
            step, cfg["model"]["encoder_hidden"], tc["warm_up_step"]
        ),
    )

    # Restore checkpoint
    ckpt_dir  = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    step = args.restore_step

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

    print(f"Starting training from step {step} → {total_steps}")
    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        for k in ["src_ids", "mels", "pitches", "energies", "durations",
                  "src_masks", "mel_masks"]:
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

        # Anneal LR
        if step in tc.get("anneal_steps", []):
            for g in optimizer.param_groups:
                g["lr"] *= tc.get("anneal_rate", 0.3)

        # Logging
        if step % log_step == 0:
            elapsed = time.time() - t0
            lr      = optimizer.param_groups[0]["lr"]
            msg = (f"Step {step:>7d} | loss {loss.item():.4f} | "
                   f"mel {loss_dict['mel']:.3f} | dur {loss_dict['dur']:.3f} | "
                   f"pit {loss_dict['pitch']:.3f} | eng {loss_dict['energy']:.3f} | "
                   f"lr {lr:.2e} | {elapsed:.0f}s")
            print(msg)
            t0 = time.time()

        # Validation
        if step % val_step == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    for k in ["src_ids", "mels", "pitches", "energies", "durations",
                              "src_masks", "mel_masks"]:
                        vb[k] = vb[k].to(device)
                    (mb, ma, dp, pp, ep, _, _) = model(
                        vb["src_ids"], src_mask=vb["src_masks"],
                        mel_mask=vb["mel_masks"], duration_target=vb["durations"],
                        pitch_target=vb["pitches"], energy_target=vb["energies"],
                    )
                    vl, _ = criterion(
                        (mb, ma, dp, pp, ep),
                        (vb["mels"], vb["durations"], vb["pitches"], vb["energies"]),
                        (vb["src_masks"], vb["mel_masks"]),
                    )
                    val_losses.append(vl.item())
            print(f"  ↳ Val loss: {np.mean(val_losses):.4f}")
            model.train()

        # Checkpoint
        if step % save_step == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step":      step,
            }, ckpt_path)
            print(f"  ↳ Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step":      step,
    }, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
