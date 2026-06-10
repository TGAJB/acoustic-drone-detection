"""Phase 1, step 1 - resample & standardise.

Walk data/raw/<label>/<dataset>/<clip>.* , load each clip, convert to mono and
resample to the target sample rate, then write a 16 kHz wav to data/interim/
preserving the relative path. Raw audio is never modified.

Usage:
    python -m src.data.prepare --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

from src.config import load_config

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aif", ".aiff"}
VALID_LABELS = {"drone", "negative"}


def find_audio_files(raw_dir: Path):
    """Yield (path, label) for every audio file under raw/<label>/..."""
    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.lower()
        if label not in VALID_LABELS:
            print(f"[skip] '{label_dir.name}' is not a valid label dir "
                  f"({VALID_LABELS}); ignoring.")
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                yield path, label


def prepare(cfg) -> int:
    raw_dir = Path(cfg.paths["raw_dir"])
    interim_dir = Path(cfg.paths["interim_dir"])
    sr = cfg.audio["sample_rate"]

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Create data/raw/<label>/<dataset>/ "
            f"and drop audio in, where <label> is 'drone' or 'negative'.")

    files = list(find_audio_files(raw_dir))
    if not files:
        raise RuntimeError(f"No audio found under {raw_dir}. Expected "
                           f"data/raw/drone/... and data/raw/negative/...")

    n_done = 0
    for path, _label in tqdm(files, desc="resampling"):
        rel = path.relative_to(raw_dir).with_suffix(".wav")
        out_path = interim_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        y, _ = librosa.load(path, sr=sr, mono=cfg.audio["mono"])
        sf.write(out_path, y, sr, subtype="PCM_16")
        n_done += 1

    print(f"[prepare] wrote {n_done} clips to {interim_dir} at {sr} Hz")
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    prepare(cfg)


if __name__ == "__main__":
    main()
