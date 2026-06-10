# acoustic-drone-detection
This repository attempts to classify drones by acoustic data
# Acoustic Drone Detection

Detect the presence of a drone (UAV) from short windows of audio. Binary,
window-level classification: **drone present** vs **not present**.

This repo currently covers **Phase 0 (setup & scoping)** and **Phase 1 (data
pipeline)**. Modelling (Phases 2+) builds on the clean, leak-free dataset this
pipeline produces.

## Task definition (locked)

| Decision        | Value      | Why |
|-----------------|------------|-----|
| Window length   | 1.0 s      | Long enough for rotor harmonics, short enough for low latency |
| Sample rate     | 16 kHz     | Drone acoustic energy sits below ~8 kHz; keeps models small |
| Channels        | mono       | Single-mic detection (multichannel is a later stretch goal) |
| Label           | binary     | `drone` (1) vs `negative` (0), per window |
| Headline metric | Recall @ fixed false-alarm rate, plus ROC-AUC, **evaluated under noise** |

Accuracy is deliberately *not* the headline metric: classes are imbalanced and
real-world performance is about catching drones in noise without false alarms.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place raw audio under `data/raw/` using this convention:

```
data/raw/<label>/<dataset>/<clip>.wav
        ^^^^^^^   ^^^^^^^^^
        drone     e.g. dregon, sesa
        negative  e.g. esc50, urbansound8k
```

`<label>` must be exactly `drone` or `negative`. Any audio format readable by
`soundfile`/`librosa` works (wav, flac, ogg, mp3). Then:

```bash
python -m src.data.prepare    --config configs/default.yaml   # resample -> data/interim
python -m src.data.manifest   --config configs/default.yaml   # window   -> data/manifests/windows.csv
python -m src.data.split      --config configs/default.yaml   # grouped split -> train/val/test.csv
python -m src.data.eda        --config configs/default.yaml   # stats + spectrogram pngs
```

Load a leak-free batch in training code:

```python
from src.data.dataset import AcousticWindowDataset
from src.features.spectrogram import LogMelSpectrogram

ds = AcousticWindowDataset("data/manifests/train.csv",
                           feature=LogMelSpectrogram.from_config("configs/default.yaml"))
x, y = ds[0]          # x: (1, n_mels, time), y: 0/1
```

## Data sources (verify each licence before use)

- **Drone positives:** DREGON (multi-rotor, multichannel), SESA, Kaggle drone-audio sets.
- **Negatives:** ESC-50, UrbanSound8K (wind, traffic, birds, voices, machinery).

Realistic negatives matter as much as positives — a detector that has only seen
silence as "no drone" will fire on every lawnmower.

## The two pitfalls this pipeline guards against

1. **Data leakage.** Windows are split **by source clip**, never individually, so
   no recording appears in both train and test. (See `src/data/split.py`.)
2. **Unrealistic negatives.** The convention encourages mixing real environmental
   sound into the negative class, not just silence.

## Layout

```
configs/default.yaml   all tunable parameters
data/raw/              original downloads (never edited)
data/interim/          resampled 16 kHz mono clips
data/manifests/        windows.csv, train/val/test.csv
src/data/              prepare, manifest, split, dataset, eda
src/features/          log-mel spectrogram transform
tests/                 pipeline smoke test
```