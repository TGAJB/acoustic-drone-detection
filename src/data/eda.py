"""Phase 1, step 4 - sanity-check EDA.

Print dataset statistics and save example log-mel spectrograms for both classes.
Drones show distinctive horizontal harmonic bands (rotor blade-pass frequency +
harmonics) that you should be able to see by eye - a quick way to catch
mislabelled or silent clips before modelling.

Usage:
    python -m src.data.eda --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.data.dataset import AcousticWindowDataset
from src.features.spectrogram import LogMelSpectrogram


def print_stats(df: pd.DataFrame, win_s: float) -> None:
    print("=" * 56)
    print(f"windows total      : {len(df)}")
    print(f"recordings total   : {df['source_clip_id'].nunique()}")
    print("\nwindows per class:")
    print(df["label_name"].value_counts().to_string())
    print("\nrecordings per class:")
    print(df.groupby("label_name")["source_clip_id"].nunique().to_string())
    print("\nwindows per dataset:")
    print(df["dataset"].value_counts().to_string())
    print(f"\napprox audio (windowed): {len(df) * win_s / 3600.0:.2f} h")
    if "split" in df.columns:
        print("\nwindows per split / class:")
        print(df.groupby(["split", "label_name"]).size()
                .unstack(fill_value=0).to_string())
    print("=" * 56)


def save_examples(cfg, df: pd.DataFrame, out_dir: Path, n_per_class=3) -> None:
    feature = LogMelSpectrogram.from_config()
    manifest = Path(cfg.paths["manifest_dir"]) / "windows.csv"
    ds = AcousticWindowDataset(manifest, feature=feature)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label_name in df["label_name"].unique():
        idxs = df.index[df["label_name"] == label_name][:n_per_class]
        for i, idx in enumerate(idxs):
            x, y = ds[int(idx)]
            spec = x.squeeze(0).numpy()
            plt.figure(figsize=(4, 3))
            plt.imshow(spec, origin="lower", aspect="auto", cmap="magma")
            plt.title(f"{label_name} (label={y})")
            plt.xlabel("time frame"); plt.ylabel("mel bin")
            plt.colorbar(label="dB"); plt.tight_layout()
            fp = out_dir / f"spec_{label_name}_{i}.png"
            plt.savefig(fp, dpi=110); plt.close()
            print(f"[eda] saved {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-per-class", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)

    manifest = Path(cfg.paths["manifest_dir"]) / "windows.csv"
    df = pd.read_csv(manifest)
    print_stats(df, cfg.audio["window_seconds"])
    save_examples(cfg, df, Path(cfg.paths["output_dir"]) / "eda",
                  n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()
