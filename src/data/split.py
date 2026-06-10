"""Phase 1, step 3 - grouped train/val/test split.

THE important step. We split by `source_clip_id` (the recording), never by
individual window. If windows from one recording leaked across splits, the model
would memorise that recording's background noise and report inflated scores that
collapse on real data.

Usage:
    python -m src.data.split --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config


def grouped_split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    group_col = cfg.split["group_col"]
    test_size = cfg.split["test_size"]
    val_size = cfg.split["val_size"]
    stratify_on = cfg.split.get("stratify", True)
    seed = cfg.seed

    groups = (df.groupby(group_col)["label"]
                .agg(lambda s: s.iloc[0])
                .reset_index())
    g_ids = groups[group_col].values
    g_lab = groups["label"].values
    strat = g_lab if stratify_on else None

    train_val_ids, test_ids = train_test_split(
        g_ids, test_size=test_size, random_state=seed, stratify=strat)

    remainder = 1.0 - test_size
    val_rel = val_size / remainder
    strat_tv = (groups.set_index(group_col).loc[train_val_ids, "label"].values
                if stratify_on else None)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_rel, random_state=seed, stratify=strat_tv)

    assign = {gid: "train" for gid in train_ids}
    assign.update({gid: "val" for gid in val_ids})
    assign.update({gid: "test" for gid in test_ids})
    df = df.copy()
    df["split"] = df[group_col].map(assign)
    return df


def report_leakage(df: pd.DataFrame, group_col: str) -> None:
    per_group_splits = df.groupby(group_col)["split"].nunique()
    leaked = per_group_splits[per_group_splits > 1]
    if len(leaked):
        raise AssertionError(
            f"LEAKAGE: {len(leaked)} groups span multiple splits: "
            f"{list(leaked.index[:5])} ...")
    print("[split] leakage check passed - every recording lives in one split.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)

    manifest_dir = Path(cfg.paths["manifest_dir"])
    df = pd.read_csv(manifest_dir / "windows.csv")

    df = grouped_split(df, cfg)
    report_leakage(df, cfg.split["group_col"])

    df.to_csv(manifest_dir / "windows.csv", index=False)
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        part.to_csv(manifest_dir / f"{name}.csv", index=False)

    summary = (df.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    print("[split] windows per split / class:")
    print(summary.to_string())
    n_groups = df.groupby("split")[cfg.split["group_col"]].nunique()
    print("\n[split] recordings per split:")
    print(n_groups.to_string())


if __name__ == "__main__":
    main()
