"""Phase 3 - training loop (PLACEHOLDER).

Plan: wire the dataset, feature transform, model, and metrics into a standard
PyTorch loop driven by configs/default.yaml (`train` section).

Sketch:
    cfg   = load_config()
    feat  = LogMelSpectrogram.from_config()
    train = DataLoader(AcousticWindowDataset("data/manifests/train.csv", feature=feat),
                       batch_size=cfg.train["batch_size"], shuffle=True)
    val   = DataLoader(AcousticWindowDataset("data/manifests/val.csv", feature=feat))
    model = SmallAudioCNN()
    # weighted loss for class imbalance:
    #   loss_fn = nn.CrossEntropyLoss(weight=train.dataset.class_weights())
    # each epoch: train, then eval on val with src.eval.metrics
    # save the best checkpoint by recall_at_far on the val set.
"""
from __future__ import annotations


def main():
    raise NotImplementedError(
        "Phase 3 training loop not implemented yet. See module docstring.")


if __name__ == "__main__":
    main()
