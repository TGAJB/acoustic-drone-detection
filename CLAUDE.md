# CLAUDE.md — Acoustic Drone Detection

Context for Claude Code working in this repository. Read this before making
changes, and keep it current as the project progresses.

## What this project is

A machine-learning system that detects whether a **drone (UAV) is present** from
short windows of audio. Portfolio project for someone strong in Python and
learning C++, aimed at UAV / defence ML roles. End goal: a robust, honestly
evaluated detector that can eventually run real-time on edge hardware (Raspberry
Pi / Jetson), with a C++ inference path as a stretch goal.

The task is **binary, window-level classification**: each 1-second window of
audio is labelled `drone` (1) or `negative` (0 = anything that isn't a drone).

## Locked task definition — do not change without good reason

| Decision        | Value   | Rationale |
|-----------------|---------|-----------|
| Window length   | 1.0 s   | Captures rotor harmonics; low detection latency |
| Sample rate     | 16 kHz  | Drone energy < ~8 kHz (Nyquist); keeps models small |
| Channels        | mono    | Single-mic deployment; multichannel is a later stretch |
| Label           | binary  | `drone`=1 vs `negative`=0, per window |
| Headline metric | Recall @ fixed false-alarm rate + ROC-AUC, **evaluated under noise** |

**Do NOT use raw accuracy as the success metric** — classes are imbalanced and it
is misleading. Always evaluate in noise, never on clean audio only.

## Phased roadmap

- **Phase 0 — Setup & scoping. DONE.** Repo skeleton, config, locked definition.
- **Phase 1 — Data pipeline.** resample → window → grouped split → dataset.
- **Phase 2 — Classical baseline.** MFCC/log-mel → SVM / random forest; build the
  metrics harness and a number to beat.
- **Phase 3 — CNN on log-mel spectrograms.** Augmentation, especially mixing drone
  audio with noise at controlled SNRs.
- **Phase 4 — Robustness & transfer learning.** Fine-tune a pretrained audio model
  (PANNs / AST / YAMNet); SNR-robustness study + false-alarm-per-hour.
- **Phase 5 — Real-time streaming inference** on a rolling audio buffer.
- **Phase 6 — Portfolio polish** (README, results, demo video).
- **Phase 7 (optional) — Hardware + C++.** Raspberry Pi / Jetson; ONNX/TFLite + C++
  inference; multichannel direction-of-arrival stretch goal.

## Repository layout

```
configs/default.yaml   all tunable params (sample rate, window, paths, split)
data/raw/              original downloads: data/raw/<label>/<dataset>/*.wav
data/interim/          resampled 16 kHz mono clips (generated; gitignored)
data/manifests/        windows.csv + train/val/test.csv (generated; gitignored)
src/config.py          loads configs/default.yaml
src/data/prepare.py    Phase 1.1 — resample raw -> interim
src/data/manifest.py   Phase 1.2 — window clips -> windows.csv
src/data/split.py      Phase 1.3 — grouped train/val/test split + leakage check
src/data/dataset.py    PyTorch Dataset over the manifest
src/data/eda.py        Phase 1.4 — dataset stats + example spectrograms
src/features/spectrogram.py  log-mel feature transform (torchaudio)
src/models/baseline.py Phase 2 baseline (stub)
src/models/cnn.py      Phase 3 CNN (stub)
src/train/train.py     Phase 3 training loop (stub)
src/eval/metrics.py    ROC-AUC, recall@FAR, SNR-sweep (Phase 2/4)
tests/make_synthetic.py  synthetic audio for smoke testing
tests/test_pipeline.py   end-to-end pipeline smoke test
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Framework is **PyTorch** (torch / torchaudio).

## Standard workflow

Place raw audio under `data/raw/<label>/<dataset>/` where `<label>` is exactly
`drone` or `negative`, then:

```bash
python tests/make_synthetic.py                          # optional: fake data to test wiring
python -m src.data.prepare  --config configs/default.yaml
python -m src.data.manifest --config configs/default.yaml
python -m src.data.split    --config configs/default.yaml
python -m src.data.eda      --config configs/default.yaml
pytest -q                                               # or: python tests/test_pipeline.py
```

## Conventions Claude Code must follow

- **Config-driven.** Never hard-code sample rate, window length, paths, or split
  sizes — read them from `configs/default.yaml` via `src.config.load_config`.
- **Never edit `data/raw/`.** It is the immutable source of truth; everything else
  is regenerated from it.
- **Split by recording, never by window.** Use `source_clip_id` as the group, and
  preserve the no-leakage assertion in `src/data/split.py`.
- **Realistic negatives.** The negative class must include real environmental
  sound (wind, traffic, birds, voices), not just silence.
- **Evaluate in noise.** Any evaluation added later must support an SNR sweep.
- **Reproducibility.** Respect `seed` from the config wherever randomness occurs.
- **The manifest is the single source of truth** for which windows exist and which
  split they belong to.

## Data sources (verify licences before use)

- Drone positives: DREGON, SESA, Kaggle drone-audio datasets.
- Negatives: ESC-50, UrbanSound8K.
