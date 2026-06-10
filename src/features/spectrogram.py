"""Log-mel spectrogram feature transform (torchaudio).

Turns a 1-D waveform tensor into a (1, n_mels, time) log-mel spectrogram ready
for a CNN. Built as an nn.Module so it can live on GPU and inside a model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.config import load_config


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024,
                 hop_length=256, fmin=50, fmax=8000, to_db=True):
        super().__init__()
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0) if to_db else None

    @classmethod
    def from_config(cls, path="configs/default.yaml"):
        cfg = load_config(path)
        f = cfg.features
        return cls(sample_rate=cfg.audio["sample_rate"],
                   n_mels=f["n_mels"], n_fft=f["n_fft"],
                   hop_length=f["hop_length"], fmin=f["fmin"],
                   fmax=f["fmax"], to_db=f.get("to_db", True))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.melspec(waveform)
        if self.to_db is not None:
            spec = self.to_db(spec)
        return spec
