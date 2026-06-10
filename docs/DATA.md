# Data sources & provenance

The raw audio is **not** committed (`data/raw/` is gitignored). Regenerate the
starter dataset at any time with:

```bash
python -m src.data.download --config configs/default.yaml      # small default
python -m src.data.download --per-type 60 --per-category 12     # size up
```

Selection is deterministic for a given `seed` (from `configs/default.yaml`).

## Drone positives — `saraalemadi/DroneAudioDataset`

- Source: https://github.com/saraalemadi/DroneAudioDataset (`Binary_Drone_Audio/yes_drone`)
- Drones: Parrot **Bebop** and **Membo**.
- Published as 1-second segments named `<recording>-<type>_<NNN>_.wav`, where
  `<NNN>` is the segment index within one recording.
- **Grouping:** `src/data/download.py` concatenates all segments of a recording
  back into a single clip (`data/raw/drone/saraalemadi/<recording>.wav`). This
  makes `source_clip_id` correspond to one real recording, so the grouped
  train/val/test split cannot leak adjacent windows across splits.
- Licence: MIT per the upstream repo — **verify the LICENSE before non-portfolio
  use.**

## Negatives — `karoldvl/ESC-50`

- Source: https://github.com/karoldvl/ESC-50 (5-second environmental clips).
- Categories used (realistic outdoor / urban + hard buzzing confusers):
  `wind, rain, chirping_birds, crickets, engine, car_horn, train, airplane,
  helicopter, footsteps`.
- Each clip is already one recording → one `source_clip_id`.
- Licence: **CC BY-NC 3.0 (non-commercial).** Fine for a research/portfolio
  project; not for commercial use.

## Current starter snapshot

| Split | Drone rec. | Neg. rec. | Drone win. | Neg. win. |
|-------|-----------:|----------:|-----------:|----------:|
| train | —          | —         | 391        | 420       |
| val   | —          | —         | 86         | 90        |
| test  | —          | —         | 86         | 90        |

(60 drone recordings + 60 ESC-50 clips → 1163 windows. Leakage check passes.)

## Known gaps to fill when sizing up

- **Speech / voices:** ESC-50 has no clean speech. Add a speech source (e.g.
  a subset of LibriSpeech or UrbanSound8K street recordings) so the negative
  class covers human voice.
- **Drone diversity:** only two quadcopter models so far. DREGON / SESA and other
  Kaggle drone sets would broaden rotor types and recording conditions.
- **Real field SNR:** positives here are fairly clean; Phase 3 augmentation must
  mix drone audio with these negatives at controlled SNRs.
