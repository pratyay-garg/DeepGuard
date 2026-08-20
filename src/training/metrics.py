from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def accuracy(targets, predictions):
    return float(accuracy_score(targets, predictions))


def precision(targets, predictions, zero_division=0):
    return float(precision_score(targets, predictions, zero_division=zero_division))


def recall(targets, predictions, zero_division=0):
    return float(recall_score(targets, predictions, zero_division=zero_division))


def f1(targets, predictions, zero_division=0):
    return float(f1_score(targets, predictions, zero_division=zero_division))


def roc_auc(targets, probabilities):
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def compute_binary_metrics(targets, probabilities, predictions):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)

    return {
        "accuracy": accuracy(targets, predictions),
        "precision": precision(targets, predictions),
        "recall": recall(targets, predictions),
        "f1": f1(targets, predictions),
        "roc_auc": roc_auc(targets, probabilities),
    }
