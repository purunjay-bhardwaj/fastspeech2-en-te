"""
model.py — FastSpeech 2 Architecture
======================================
Implements:
  - Feed-Forward Transformer (FFT) blocks
  - Encoder & Decoder
  - Variance Adaptor (Duration / Pitch / Energy predictors + LR)
  - Postnet
  - Full FastSpeech2 model

Reference: Ren et al. (2020) "FastSpeech 2: Fast and High-Quality
           End-to-End Text to Speech"
           https://arxiv.org/abs/2006.04558
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        # not saved in state_dict, so it can be grown for long utterances
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float()
                        * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build(2 * x.size(1)).to(self.pe.device)
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)


def get_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True = padding."""
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h   = num_heads
        self.q   = nn.Linear(d_model, d_model)
        self.k   = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = q.shape

        def split(x): return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split(self.q(q)), split(self.k(k)), split(self.v(v))

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out(out)


# ── Feed-Forward Block (Conv1D-based as in original FS2) ─────────────────────

class FeedForwardBlock(nn.Module):
    """Two Conv1D layers (kernel sizes [9,1] in the original paper)."""
    def __init__(self, d_model: int, d_ff: int, kernel_sizes: Tuple[int, int] = (9, 1),
                 dropout: float = 0.1):
        super().__init__()
        k1, k2 = kernel_sizes
        p1, p2 = (k1 - 1) // 2, (k2 - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_ff, k1, padding=p1)
        self.conv2 = nn.Conv1d(d_ff, d_model, k2, padding=p2)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        out = x.transpose(1, 2)        # → (B, d_model, T)
        out = F.relu(self.conv1(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out.transpose(1, 2)     # → (B, T, d_model)


# ── FFT Block = MHA + LayerNorm + FF + LayerNorm ─────────────────────────────

class FFTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 kernel_sizes: Tuple[int, int] = (9, 1), dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff    = FeedForwardBlock(d_model, d_ff, kernel_sizes, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── Variance Predictor (shared for duration / pitch / energy) ────────────────

class VariancePredictor(nn.Module):
    """
    Two Conv1D + ReLU + LN blocks, then linear projection to scalar.
    Predicts duration / pitch / energy per phoneme.
    """
    def __init__(self, d_model: int, d_inner: int, kernel_size: int,
                 dropout: float = 0.5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.linear = nn.Linear(d_inner, 1)
        self.d_inner = d_inner
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ln1   = nn.LayerNorm(d_inner)
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_inner, d_inner, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ln2   = nn.LayerNorm(d_inner)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, d_model)
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.ln1(out)
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.ln2(out)
        out = self.linear(out).squeeze(-1)    # (B, T)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


# ── Length Regulator ──────────────────────────────────────────────────────────

class LengthRegulator(nn.Module):
    """Expand encoder output according to phoneme durations."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor,
                target_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, T_phone, d_model)
            durations:  (B, T_phone)  integer durations per phoneme
            target_len: optional maximum output length
        Returns:
            expanded:   (B, T_mel, d_model)
            mel_len:    (B,)
        """
        B = x.size(0)
        expanded_list = []
        mel_lengths = []

        for i in range(B):
            dur = durations[i]           # (T_phone,)
            reps = dur.long().clamp(min=0)
            expanded = torch.repeat_interleave(x[i], reps, dim=0)  # (T_mel, d)
            expanded_list.append(expanded)
            mel_lengths.append(expanded.size(0))

        # pad to max length in batch
        max_len = max(mel_lengths) if target_len is None else target_len
        max_len = max(max_len, 1)
        out = torch.zeros(B, max_len, x.size(-1), device=x.device, dtype=x.dtype)
        for i, e in enumerate(expanded_list):
            L = min(e.size(0), max_len)
            out[i, :L] = e[:L]

        mel_lengths = [min(l, max_len) for l in mel_lengths]
        return out, torch.tensor(mel_lengths, device=x.device)


# ── Variance Adaptor ──────────────────────────────────────────────────────────

class VarianceAdaptor(nn.Module):
    """
    Contains:
      - Duration Predictor → Length Regulator
      - Pitch Predictor (phoneme-level)   → embedding lookup
      - Energy Predictor (phoneme-level)  → embedding lookup
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = m["encoder_hidden"]
        p = m["duration_predictor_filter_size"]

        self.duration_predictor = VariancePredictor(
            d, p, m["duration_predictor_kernel_size"],
            m["duration_predictor_dropout"]
        )
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = VariancePredictor(
            d, m["pitch_predictor_filter_size"],
            m["pitch_predictor_kernel_size"],
            m["pitch_predictor_dropout"]
        )
        self.energy_predictor = VariancePredictor(
            d, m["energy_predictor_filter_size"],
            m["energy_predictor_kernel_size"],
            m["energy_predictor_dropout"]
        )

        n_bins = cfg["variance"]["pitch_n_bins"]
        # Learnable quantisation embeddings
        self.pitch_embedding  = nn.Embedding(n_bins, d)
        self.energy_embedding = nn.Embedding(cfg["variance"]["energy_n_bins"], d)

        self.register_buffer("pitch_bins",  torch.linspace(-5, 5, n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(-5, 5, cfg["variance"]["energy_n_bins"] - 1))

    def _quantise(self, values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(values, bins)

    def forward(self,
                x: torch.Tensor,
                src_mask: Optional[torch.Tensor],
                mel_mask: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                p_control: float = 1.0,
                e_control: float = 1.0,
                d_control: float = 1.0,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Returns: (output, duration_pred, pitch_pred, energy_pred, mel_len, mel_mask)
        """
        dur_pred   = self.duration_predictor(x, src_mask)
        pitch_pred = self.pitch_predictor(x, src_mask)
        enrg_pred  = self.energy_predictor(x, src_mask)

        # Phoneme-level pitch / energy embeddings (GT in training, predicted at inference)
        if pitch_target is not None:
            pitch_q = self._quantise(pitch_target, self.pitch_bins)
        else:
            pitch_q = self._quantise(pitch_pred * p_control, self.pitch_bins)
        if energy_target is not None:
            energy_q = self._quantise(energy_target, self.energy_bins)
        else:
            energy_q = self._quantise(enrg_pred * e_control, self.energy_bins)
        x = x + self.pitch_embedding(pitch_q) + self.energy_embedding(energy_q)

        # Length regulation (phonemes → mel frames)
        if duration_target is not None:   # training: GT durations
            max_len = mel_mask.size(1) if mel_mask is not None else None
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:                             # inference: predicted durations
            dur_rounded = torch.round((torch.exp(dur_pred) - 1) * d_control).clamp(min=0).long()
            if src_mask is not None:
                dur_rounded = dur_rounded.masked_fill(src_mask, 0)
            # guarantee at least one frame per utterance
            empty = dur_rounded.sum(dim=1, keepdim=True) == 0
            dur_rounded = torch.where(empty, torch.ones_like(dur_rounded), dur_rounded)
            x, mel_len = self.length_regulator(x, dur_rounded)

        if mel_mask is None:
            mel_mask = get_mask_from_lengths(mel_len, x.size(1))

        return x, dur_pred, pitch_pred, enrg_pred, mel_len, mel_mask


# ── Postnet ───────────────────────────────────────────────────────────────────

class Postnet(nn.Module):
    """Five Conv1D + BN + Tanh blocks to refine mel output."""

    def __init__(self, n_mels: int = 80, n_channels: int = 512,
                 kernel_size: int = 5, n_layers: int = 5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        layers = []
        for i in range(n_layers):
            in_ch  = n_mels     if i == 0           else n_channels
            out_ch = n_mels     if i == n_layers - 1 else n_channels
            act    = nn.Tanh()  if i < n_layers - 1 else nn.Identity()
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                act,
                nn.Dropout(0.5),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_mels)
        return x + self.net(x.transpose(1, 2)).transpose(1, 2)


# ── FastSpeech 2 ──────────────────────────────────────────────────────────────

class FastSpeech2(nn.Module):
    """
    Full FastSpeech 2 model.
    Input:  phoneme token ids  (B, T_phone)
    Output: mel-spectrogram    (B, T_mel, n_mels) + variance predictions
    """

    def __init__(self, cfg: dict, vocab_size: int, n_speakers: int = 1):
        super().__init__()
        m   = cfg["model"]
        d   = m["encoder_hidden"]
        ff  = m["fft_conv1d_filter_size"]
        ks  = tuple(m["fft_conv1d_kernel_size"])
        n_mels = cfg["audio"]["n_mels"]

        self.phoneme_embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_enc       = PositionalEncoding(d, cfg["model"]["max_seq_len"],
                                                 m["encoder_dropout"])

        self.encoder = nn.ModuleList([
            FFTBlock(d, m["encoder_head"], ff, ks, m["encoder_dropout"])
            for _ in range(m["encoder_layer"])
        ])

        # Multi-speaker datasets (Svarah: 117 speakers, IndicTTS-Te: 2): a learned
        # speaker embedding is added to the encoder output, as in ming024/FastSpeech2.
        self.speaker_emb = nn.Embedding(n_speakers, d) if n_speakers > 1 else None

        self.variance_adaptor = VarianceAdaptor(cfg)

        self.decoder = nn.ModuleList([
            FFTBlock(d, m["decoder_head"], ff, ks, m["decoder_dropout"])
            for _ in range(m["decoder_layer"])
        ])

        self.mel_linear = nn.Linear(d, n_mels)
        self.postnet     = Postnet(n_mels)

    def forward(self,
                src_ids: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                mel_mask: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None,
                pitch_target: Optional[torch.Tensor] = None,
                energy_target: Optional[torch.Tensor] = None,
                p_control: float = 1.0,
                e_control: float = 1.0,
                d_control: float = 1.0,
                speakers: Optional[torch.Tensor] = None,
                ):
        """
        Returns:
            mel_before:   (B, T_mel, n_mels)   — before postnet
            mel_after:    (B, T_mel, n_mels)   — after postnet
            dur_pred:     (B, T_phone)
            pitch_pred:   (B, T_phone)
            energy_pred:  (B, T_phone)
            mel_len:      (B,)
            mel_mask:     (B, T_mel)
        """
        # Encoder
        x = self.phoneme_embed(src_ids)
        x = self.pos_enc(x)
        for layer in self.encoder:
            x = layer(x, src_mask)

        if self.speaker_emb is not None:
            if speakers is None:
                speakers = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = x + self.speaker_emb(speakers).unsqueeze(1)

        # Variance Adaptor
        (x, dur_pred, pitch_pred, energy_pred,
         mel_len, mel_mask) = self.variance_adaptor(
            x, src_mask, mel_mask,
            duration_target, pitch_target, energy_target,
            p_control, e_control, d_control,
        )

        # Decoder
        x = self.pos_enc(x)
        for layer in self.decoder:
            x = layer(x, mel_mask)

        mel_before = self.mel_linear(x)
        mel_after  = self.postnet(mel_before)

        return mel_before, mel_after, dur_pred, pitch_pred, energy_pred, mel_len, mel_mask
