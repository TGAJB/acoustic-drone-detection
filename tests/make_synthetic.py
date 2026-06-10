"""Generate tiny synthetic audio so the pipeline can be smoke-tested without
downloading any real datasets.

- "drone" clips: a fundamental + harmonics (a crude rotor buzz) + light noise.
- "negative" clips: filtered noise standing in for environment sound.

Writes several clips per class to data/raw/<label>/synthetic/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050  # deliberately != target 16k so resampling is exercised


def drone_clip(seconds, f0, rng):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15], start=1):
        sig += amp * np.sin(2 * np.pi * f0 * k * t)
    sig += 0.05 * rng.standard_normal(len(t))
    return 0.3 * sig / np.max(np.abs(sig))


def negative_clip(seconds, rng):
    n = int(SR * seconds)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(20) / 20, mode="same")
    return 0.3 * noise / np.max(np.abs(noise))


def main():
    rng = np.random.default_rng(0)
    root = Path("data/raw")
    n_clips = 6
    for i in range(n_clips):
        d = drone_clip(float(rng.uniform(2.5, 4.0)),
                       float(rng.uniform(90, 160)), rng)
        p = root / "drone" / "synthetic" / f"drone_{i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, d, SR)

        ncl = negative_clip(float(rng.uniform(2.5, 4.0)), rng)
        p = root / "negative" / "synthetic" / f"neg_{i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, ncl, SR)
    print(f"[synthetic] wrote {n_clips} drone + {n_clips} negative clips to {root}")


if __name__ == "__main__":
    main()
