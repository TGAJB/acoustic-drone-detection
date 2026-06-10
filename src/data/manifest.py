"""Phase 1, step 2 - build the window manifest.

For every resampled clip in data/interim/, compute fixed-length windows
(start/end sample offsets) and write one row per window to
data/manifests/windows.csv. We store offsets rather than slicing audio to disk,
so the Dataset reads windows lazily and storage stays small.

Usage:
    python -m src.data.manifest --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import load_config


def windows_for_clip(n_samples: int, win: int, hop: int):
    """Yield (start, end) sample offsets covering the clip, including a final
    tail window (padded later) so the end of a recording is never dropped."""
    if n_samples <= 0:
        return
    if n_samples <= win:
        yield 0, n_samples
        return
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop
    if start < n_samples and (n_samples - start) > hop // 2:
        yield start, n_samples


def build_manifest(cfg) -> pd.DataFrame:
    interim_dir = Path(cfg.paths["interim_dir"])
    manifest_dir = Path(cfg.paths["manifest_dir"])
    sr = cfg.audio["sample_rate"]
    win = int(round(cfg.audio["window_seconds"] * sr))
    hop = int(round(cfg.audio["hop_seconds"] * sr))
    min_rms = float(cfg.audio.get("min_rms", 0.0))
    label_map = cfg.labels

    if not interim_dir.exists():
        raise FileNotFoundError(
            f"{interim_dir} not found. Run `python -m src.data.prepare` first.")

    rows = []
    clips = [p for p in interim_dir.rglob("*.wav")]
    if not clips:
        raise RuntimeError(f"No resampled clips under {interim_dir}.")

    for clip in tqdm(clips, desc="windowing"):
        rel = clip.relative_to(interim_dir)
        label_name = rel.parts[0].lower()
        if label_name not in label_map:
            print(f"[skip] {clip}: unknown label '{label_name}'")
            continue
        dataset = rel.parts[1] if len(rel.parts) > 2 else "unknown"
        source_clip_id = rel.with_suffix("").as_posix()

        info = sf.info(clip)
        n_samples = info.frames
        audio = None
        if min_rms > 0:
            audio, _ = sf.read(clip, dtype="float32")

        for start, end in windows_for_clip(n_samples, win, hop):
            rms = np.nan
            if min_rms > 0:
                seg = audio[start:end]
                rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
                if rms < min_rms:
                    continue
            rows.append({
                "interim_path": clip.as_posix(),
                "start_sample": start,
                "end_sample": end,
                "label": label_map[label_name],
                "label_name": label_name,
                "dataset": dataset,
                "source_clip_id": source_clip_id,
                "rms": rms,
            })

    df = pd.DataFrame(rows)
    df.insert(0, "window_id", range(len(df)))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / "windows.csv"
    df.to_csv(out, index=False)

    print(f"[manifest] {len(df)} windows from {len(clips)} clips -> {out}")
    print(df["label_name"].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    build_manifest(cfg)


if __name__ == "__main__":
    main()
