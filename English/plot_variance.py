"""
plot_variance.py — Variance Adaptor Output Visualizations
===========================================================
Generates one plot per sentence per variance type (duration, pitch, energy).
Reads the JSON produced by inference.py.

Output structure:
    Results/<Language>/variance_adaptor_plots/
        duration/   E1_duration.png  ...
        pitch/      E1_pitch.png     ...
        energy/     E1_energy.png    ...
        summary/    E1_all.png       ...

Usage:
    python plot_variance.py --config config.yaml
    python plot_variance.py --config ../Telugu/config.yaml  # from English/ folder too
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml


# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "duration": "#4C72B0",
    "pitch":    "#DD8452",
    "energy":   "#55A868",
    "bg":       "#F8F8F8",
    "grid":     "#E0E0E0",
    "text":     "#2C2C2C",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def wrap_title(text: str, width: int = 60) -> str:
    """Wrap long sentence for plot titles."""
    if len(text) <= width:
        return text
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def phone_tick_labels(phones: list, max_ticks: int = 20) -> tuple:
    """Subsample phone labels if too many to display legibly."""
    n = len(phones)
    if n <= max_ticks:
        return list(range(n)), phones
    step = max(1, n // max_ticks)
    idxs = list(range(0, n, step))
    return idxs, [phones[i] for i in idxs]


# ── Individual Plot Functions ─────────────────────────────────────────────────

def plot_duration(phones: list, durations: list, out_path: Path,
                  sentence_id: str, text: str) -> None:
    """Bar chart of predicted phoneme durations."""
    n   = len(phones)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["bg"])

    x   = np.arange(n)
    dur = np.array(durations[:n], dtype=float)

    bars = ax.bar(x, dur, color=PALETTE["duration"], alpha=0.85, edgecolor="white", linewidth=0.5)

    # value labels on bars if not too many
    if n <= 30:
        for bar, d in zip(bars, dur):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{d:.1f}", ha="center", va="bottom", fontsize=7, color=PALETTE["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Predicted Duration (frames)")
    ax.set_title(f"{sentence_id} — Duration Predictor\n{wrap_title(text)}")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_pitch(phones: list, pitch: list, out_path: Path,
               sentence_id: str, text: str) -> None:
    """Line plot of predicted phoneme-level pitch (F0)."""
    n = len(phones)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["bg"])

    x = np.arange(n)
    p = np.array(pitch[:n], dtype=float)

    ax.fill_between(x, p, alpha=0.2, color=PALETTE["pitch"])
    ax.plot(x, p, color=PALETTE["pitch"], linewidth=1.8, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.2,
            markeredgecolor=PALETTE["pitch"])

    tick_idxs, tick_labels = phone_tick_labels(phones)
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Predicted Pitch (normalized)")
    ax.set_title(f"{sentence_id} — Pitch Predictor\n{wrap_title(text)}")
    ax.grid(color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_energy(phones: list, energy: list, out_path: Path,
                sentence_id: str, text: str) -> None:
    """Filled area plot of predicted phoneme-level energy."""
    n = len(phones)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["bg"])

    x = np.arange(n)
    e = np.array(energy[:n], dtype=float)

    ax.fill_between(x, e, alpha=0.35, color=PALETTE["energy"])
    ax.plot(x, e, color=PALETTE["energy"], linewidth=1.8, marker="s",
            markersize=3.5, markerfacecolor="white", markeredgewidth=1.2,
            markeredgecolor=PALETTE["energy"])

    tick_idxs, tick_labels = phone_tick_labels(phones)
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Predicted Energy (normalized)")
    ax.set_title(f"{sentence_id} — Energy Predictor\n{wrap_title(text)}")
    ax.grid(color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_all_three(phones: list, durations: list, pitch: list, energy: list,
                   out_path: Path, sentence_id: str, text: str) -> None:
    """3-row summary figure: duration, pitch, energy for one sentence."""
    n = len(phones)
    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.4), 11), sharex=True)
    fig.suptitle(
        f"Variance Adaptor Outputs — {sentence_id}\n{wrap_title(text, 70)}",
        fontsize=13, fontweight="bold", color=PALETTE["text"]
    )
    fig.patch.set_facecolor("white")

    x    = np.arange(n)
    dur  = np.array(durations[:n], dtype=float)
    pit  = np.array(pitch[:n],    dtype=float)
    eng  = np.array(energy[:n],   dtype=float)

    # -- Duration
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    ax.bar(x, dur, color=PALETTE["duration"], alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Duration\n(frames)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # -- Pitch
    ax = axes[1]
    ax.set_facecolor(PALETTE["bg"])
    ax.fill_between(x, pit, alpha=0.2, color=PALETTE["pitch"])
    ax.plot(x, pit, color=PALETTE["pitch"], linewidth=1.5, marker="o",
            markersize=3, markerfacecolor="white", markeredgewidth=1.0,
            markeredgecolor=PALETTE["pitch"])
    ax.set_ylabel("Pitch\n(normalized)")
    ax.grid(color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # -- Energy
    ax = axes[2]
    ax.set_facecolor(PALETTE["bg"])
    ax.fill_between(x, eng, alpha=0.3, color=PALETTE["energy"])
    ax.plot(x, eng, color=PALETTE["energy"], linewidth=1.5, marker="s",
            markersize=3, markerfacecolor="white", markeredgewidth=1.0,
            markeredgecolor=PALETTE["energy"])
    ax.set_ylabel("Energy\n(normalized)")
    ax.grid(color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    tick_idxs, tick_labels = phone_tick_labels(phones)
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    plt.close(fig)


# ── Aggregate Distribution Plots ─────────────────────────────────────────────

def plot_aggregate_distributions(all_data: list, out_dir: Path, language: str) -> None:
    """Histogram of duration/pitch/energy across ALL test sentences."""
    all_dur = [d for s in all_data for d in s["duration"]]
    all_pit = [p for s in all_data for p in s["pitch"]]
    all_eng = [e for s in all_data for e in s["energy"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Variance Adaptor — Aggregate Distributions ({language.capitalize()})",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("white")

    for ax, data, label, color in zip(
        axes,
        [all_dur, all_pit, all_eng],
        ["Duration (frames)", "Pitch (normalized)", "Energy (normalized)"],
        [PALETTE["duration"], PALETTE["pitch"], PALETTE["energy"]],
    ):
        ax.set_facecolor(PALETTE["bg"])
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label.split("(")[0].strip())
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        # stat annotation
        mu, sd = np.mean(data), np.std(data)
        ax.axvline(mu, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.text(0.97, 0.95, f"μ={mu:.2f}\nσ={sd:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / f"aggregate_distributions_{language}.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  ↳ Aggregate distribution plot: {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--data_json", default=None,
                        help="Path to variance_outputs.json from inference.py")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    language = cfg.get("language", "english")
    plots_dir = Path(cfg["paths"]["plots_dir"])

    # Subdirectories
    dur_dir     = plots_dir / "duration"
    pit_dir     = plots_dir / "pitch"
    eng_dir     = plots_dir / "energy"
    summary_dir = plots_dir / "summary"
    for d in [dur_dir, pit_dir, eng_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load variance data
    if args.data_json:
        json_path = Path(args.data_json)
    else:
        json_path = (plots_dir.parent / "variance_adaptor_data" / "variance_outputs.json")

    if not json_path.exists():
        raise FileNotFoundError(
            f"Variance data not found at {json_path}. "
            f"Run inference.py first."
        )

    all_data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Loaded variance data for {len(all_data)} sentences.")

    for item in all_data:
        sid     = item["sentence_id"]
        text    = item["text"]
        phones  = item["phones"]
        dur     = item["duration"]
        pitch   = item["pitch"]
        energy  = item["energy"]

        print(f"  Plotting {sid}: {text[:50]}...")

        plot_duration(phones, dur,
                      dur_dir / f"{sid}_duration.png", sid, text)
        plot_pitch(phones, pitch,
                   pit_dir / f"{sid}_pitch.png", sid, text)
        plot_energy(phones, energy,
                    eng_dir / f"{sid}_energy.png", sid, text)
        plot_all_three(phones, dur, pitch, energy,
                       summary_dir / f"{sid}_all.png", sid, text)

    # Aggregate distribution
    plot_aggregate_distributions(all_data, summary_dir, language)

    print(f"\nAll plots saved to: {plots_dir}")
    print("Plot generation complete ✓")


if __name__ == "__main__":
    main()
