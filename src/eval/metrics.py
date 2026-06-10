"""Evaluation metrics aligned with the locked success criteria.

The headline metric is **recall at a fixed false-alarm rate**, plus ROC-AUC,
and everything should ultimately be reported **under noise** (see snr_sweep,
wired up in Phase 4). These helpers are framework-agnostic: pass NumPy arrays of
ground-truth labels and predicted scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve. 0.5 = random, 1.0 = perfect."""
    return float(roc_auc_score(y_true, y_score))


def recall_at_far(y_true, y_score, target_far: float = 0.01):
    """Recall (true-positive rate) achievable while holding the false-alarm
    rate at or below `target_far`. Returns (recall, threshold, actual_far).

    This is the metric that matters for a real detector: 'how many drones can I
    catch while keeping false alarms acceptably rare'.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ok = fpr <= target_far
    if not ok.any():
        return 0.0, float("inf"), float(fpr.min())
    idx = np.where(ok)[0][-1]  # highest TPR among points within the FAR budget
    return float(tpr[idx]), float(thresholds[idx]), float(fpr[idx])


def confusion_at_threshold(y_true, y_score, threshold: float = 0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": precision, "recall": recall, "confusion_matrix": cm}


def snr_sweep(*args, **kwargs):
    """Phase 4: evaluate the model across a range of signal-to-noise ratios and
    return detection rate vs SNR. Implement once a trained model exists - mix
    drone audio with noise at each SNR, score, and record recall_at_far.
    """
    raise NotImplementedError("SNR robustness sweep arrives in Phase 4.")
