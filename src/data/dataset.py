"""PyTorch Dataset over the window manifest.

Reads one split's CSV (train/val/test.csv) and lazily loads each window directly
from the resampled clip using soundfile's seek (no per-window files on disk).
Short tail windows are zero-padded to the fixed length. An optional feature
transform (e.g. LogMelSpectrogram) is applied on the fly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.config import load_config


class AcousticWindowDataset(Dataset):
    def __init__(self, manifest_csv, feature=None,
                 config="configs/default.yaml", transform=None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        cfg = load_config(config)
        self.sr = cfg.audio["sample_rate"]
        self.win = int(round(cfg.audio["window_seconds"] * self.sr))
        self.feature = feature        # waveform -> feature tensor
        self.transform = transform    # optional waveform augmentation (Phase 3)

    def __len__(self):
        return len(self.df)

    def _load_window(self, row) -> np.ndarray:
        start = int(row["start_sample"])
        stop = int(row["end_sample"])
        audio, _ = sf.read(row["interim_path"], start=start, stop=stop,
                           dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < self.win:
            audio = np.pad(audio, (0, self.win - len(audio)))
        return audio[: self.win]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = self._load_window(row)
        wav = torch.from_numpy(audio).float()
        if self.transform is not None:
            wav = self.transform(wav)
        x = self.feature(wav) if self.feature is not None else wav
        y = int(row["label"])
        return x, y

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for an imbalanced loss."""
        counts = self.df["label"].value_counts().sort_index()
        w = counts.sum() / (len(counts) * counts)
        return torch.tensor(w.values, dtype=torch.float32)


if __name__ == "__main__":
    from src.features.spectrogram import LogMelSpectrogram
    ds = AcousticWindowDataset("data/manifests/train.csv",
                               feature=LogMelSpectrogram.from_config())
    x, y = ds[0]
    print("feature shape:", tuple(x.shape), "label:", y, "n=", len(ds))
