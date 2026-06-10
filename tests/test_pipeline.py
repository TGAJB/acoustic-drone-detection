"""End-to-end smoke test of the Phase 1 pipeline on synthetic audio.

Run from the repo root:
    python tests/make_synthetic.py
    python -m src.data.prepare  --config configs/default.yaml
    python -m src.data.manifest --config configs/default.yaml
    python -m src.data.split    --config configs/default.yaml
    pytest -q                     # or: python tests/test_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config

CFG = load_config("configs/default.yaml")
MANI = Path(CFG.paths["manifest_dir"])


def test_manifest_exists_and_nonempty():
    df = pd.read_csv(MANI / "windows.csv")
    assert len(df) > 0
    assert {"start_sample", "end_sample", "label", "source_clip_id"}.issubset(df.columns)


def test_window_length_consistent():
    df = pd.read_csv(MANI / "windows.csv")
    win = round(CFG.audio["window_seconds"] * CFG.audio["sample_rate"])
    lengths = df["end_sample"] - df["start_sample"]
    assert (lengths <= win).all()
    assert (lengths == win).mean() > 0.5


def test_no_group_leakage_across_splits():
    df = pd.read_csv(MANI / "windows.csv")
    assert "split" in df.columns, "run src.data.split first"
    assert (df.groupby("source_clip_id")["split"].nunique() == 1).all()


def test_all_splits_present():
    df = pd.read_csv(MANI / "windows.csv")
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_dataset_feature_shape():
    import torch  # noqa
    from src.data.dataset import AcousticWindowDataset
    from src.features.spectrogram import LogMelSpectrogram

    ds = AcousticWindowDataset(MANI / "train.csv",
                               feature=LogMelSpectrogram.from_config())
    x, y = ds[0]
    assert x.dim() == 3 and x.shape[0] == 1
    assert x.shape[1] == CFG.features["n_mels"]
    assert y in (0, 1)


if __name__ == "__main__":
    test_manifest_exists_and_nonempty()
    test_window_length_consistent()
    test_no_group_leakage_across_splits()
    test_all_splits_present()
    test_dataset_feature_shape()
    print("ALL SMOKE TESTS PASSED")
