"""Phase 1, step 0 - fetch a small, balanced starter dataset into data/raw/.

Pulls two public datasets straight from GitHub (no account / API key needed):

  * Drone positives  -> saraalemadi/DroneAudioDataset (Parrot Bebop + Membo).
        The dataset ships as 1-second segments whose filenames encode the
        original recording, e.g.  B_S2_D1_067-bebop_000_.wav , _001_, _002_ ...
        We group segments by recording and concatenate them back into one clip
        per recording. That keeps `source_clip_id` = one real recording, so the
        grouped train/val/test split cannot leak adjacent windows across splits.

  * Negatives        -> karoldvl/ESC-50 (environmental sound, 5 s clips).
        A diverse subset of outdoor / urban categories plus the hard buzzing
        confusers (helicopter, airplane). Each ESC-50 clip is already one
        recording, so it maps to one `source_clip_id` directly.

Output layout (consumed by src.data.prepare):
    data/raw/drone/saraalemadi/<recording>.wav     (16 kHz mono, concatenated)
    data/raw/negative/esc50/<orig_name>.wav        (44.1 kHz, as published)

Start small, size up later by raising --per-type / --per-category.

Usage:
    python -m src.data.download --config configs/default.yaml
    python -m src.data.download --per-type 60 --per-category 12   # bigger

Licences (verify before any non-portfolio use):
    ESC-50               CC BY-NC 3.0  (non-commercial)
    DroneAudioDataset    MIT (per repo) - confirm in the upstream LICENSE
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import load_config

# --- remote sources -------------------------------------------------------
DRONE_REPO = "saraalemadi/DroneAudioDataset"
DRONE_TREE = f"https://api.github.com/repos/{DRONE_REPO}/git/trees/master?recursive=1"
DRONE_RAW = f"https://raw.githubusercontent.com/{DRONE_REPO}/master/"
DRONE_SUBDIR = "Binary_Drone_Audio/yes_drone/"
DRONE_TYPES = ("bebop", "membo")
# leaf like  <recording>-<type>_<segidx>_.wav  ->  recording id + segment index
SEG_RE = re.compile(r"^(?P<rec>.+-(?:%s))_(?P<idx>\d+)_?\.wav$" % "|".join(DRONE_TYPES))

ESC50_META = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
ESC50_RAW = "https://github.com/karoldvl/ESC-50/raw/master/audio/"
# realistic outdoor negatives: ambient + traffic + birds + aerial confusers + human
ESC50_CATEGORIES = (
    "wind", "rain", "chirping_birds", "crickets", "engine",
    "car_horn", "train", "airplane", "helicopter", "footsteps",
)

UA = {"User-Agent": "acoustic-drone-detection/0.1"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                raise
            print(f"  retry ({attempt + 1}) {url}: {e}", file=sys.stderr)
    raise RuntimeError("unreachable")


# --- drone positives ------------------------------------------------------
def fetch_drone(raw_dir: Path, per_type: int, rng: np.random.Generator) -> int:
    import json

    print("[drone] listing DroneAudioDataset tree ...")
    tree = json.loads(_get(DRONE_TREE))["tree"]
    leaves = [
        n["path"].split("/")[-1]
        for n in tree
        if n["path"].startswith(DRONE_SUBDIR) and n["path"].endswith(".wav")
    ]

    # recording id -> list of (segment index, filename)
    recordings: dict[str, list[tuple[int, str]]] = {}
    for fn in leaves:
        m = SEG_RE.match(fn)
        if m:
            recordings.setdefault(m["rec"], []).append((int(m["idx"]), fn))

    out_dir = raw_dir / "drone" / "saraalemadi"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = 0
    for dtype in DRONE_TYPES:
        recs = sorted(r for r in recordings if r.endswith(f"-{dtype}"))
        rng.shuffle(recs)
        chosen = recs[:per_type]
        print(f"[drone] {dtype}: {len(chosen)}/{len(recs)} recordings")
        for rec in chosen:
            segs = sorted(recordings[rec])  # by segment index
            chunks, sr = [], None
            for _idx, fn in segs:
                data, sr = sf.read(io.BytesIO(_get(DRONE_RAW + DRONE_SUBDIR + fn)),
                                   dtype="float32")
                chunks.append(data)
            clip = np.concatenate(chunks)
            sf.write(out_dir / f"{rec}.wav", clip, sr, subtype="PCM_16")
            n_clips += 1
    print(f"[drone] wrote {n_clips} concatenated recordings -> {out_dir}")
    return n_clips


# --- negatives ------------------------------------------------------------
def fetch_negatives(raw_dir: Path, per_category: int,
                    rng: np.random.Generator) -> int:
    print("[neg] fetching ESC-50 metadata ...")
    meta = _get(ESC50_META).decode("utf-8").splitlines()
    header = meta[0].split(",")
    fi, ci = header.index("filename"), header.index("category")

    by_cat: dict[str, list[str]] = {}
    for line in meta[1:]:
        if not line:
            continue
        cols = line.split(",")
        by_cat.setdefault(cols[ci], []).append(cols[fi])

    out_dir = raw_dir / "negative" / "esc50"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = 0
    for cat in ESC50_CATEGORIES:
        files = sorted(by_cat.get(cat, []))
        if not files:
            print(f"[neg] WARNING: category '{cat}' not found in ESC-50")
            continue
        rng.shuffle(files)
        chosen = files[:per_category]
        for fn in chosen:
            (out_dir / fn).write_bytes(_get(ESC50_RAW + fn))
            n_clips += 1
        print(f"[neg] {cat}: {len(chosen)} clips")
    print(f"[neg] wrote {n_clips} ESC-50 clips -> {out_dir}")
    return n_clips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--per-type", type=int, default=30,
                    help="drone recordings per drone type (bebop, membo)")
    ap.add_argument("--per-category", type=int, default=6,
                    help="ESC-50 clips per negative category")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed for the random subset")
    ap.add_argument("--skip-drone", action="store_true")
    ap.add_argument("--skip-negatives", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    raw_dir = Path(cfg.paths["raw_dir"])

    n_d = n_n = 0
    if not args.skip_drone:
        n_d = fetch_drone(raw_dir, args.per_type, np.random.default_rng(seed))
    if not args.skip_negatives:
        n_n = fetch_negatives(raw_dir, args.per_category,
                              np.random.default_rng(seed + 1))
    print(f"\n[download] done: {n_d} drone + {n_n} negative clips under {raw_dir}")
    print("Next: python -m src.data.prepare --config", args.config)


if __name__ == "__main__":
    main()
