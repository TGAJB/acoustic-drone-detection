"""Phase 3 - small CNN over log-mel spectrograms (PLACEHOLDER).

Plan: a compact convolutional classifier that takes the (1, n_mels, time)
spectrogram from src.features.spectrogram and outputs a drone/no-drone logit.
Train with src.train.train, augment with SpecAugment + noise mixing at controlled
SNRs (the most important augmentation for robustness).
"""
from __future__ import annotations

import torch.nn as nn


class SmallAudioCNN(nn.Module):
    """Minimal runnable skeleton - tune channels/depth in Phase 3."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
