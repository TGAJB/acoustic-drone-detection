"""Phase 2 - classical baseline (PLACEHOLDER).

Plan: summarise each window with MFCC statistics (mean/std over time) and train a
simple scikit-learn classifier (SVM or RandomForest). This gives a fast,
unglamorous number to beat before reaching for deep learning.

Sketch:
    from sklearn.ensemble import RandomForestClassifier
    X_train, y_train = features_from_manifest("data/manifests/train.csv")
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced")
    clf.fit(X_train, y_train)
    # then score on test.csv and report src.eval.metrics.roc_auc / recall_at_far
"""
from __future__ import annotations


def main():
    raise NotImplementedError(
        "Phase 2 baseline not implemented yet. See module docstring for the plan.")


if __name__ == "__main__":
    main()
