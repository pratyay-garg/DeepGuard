from __future__ import annotations

from typing import Dict, Iterable, List

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


def compute_video_level_metrics(sample_rows):
    """Aggregate frame predictions into video-level metrics.

    The aggregation uses mean fake probability per video. If a row has no
    video_id, it falls back to sample_id, which keeps the function usable for
    generic prepared datasets.
    """
    if not sample_rows:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "samples": [],
        }

    grouped: Dict[str, List[dict]] = {}
    for row in sample_rows:
        group_id = row.get("video_id") or row.get("sample_id") or row.get("face_path")
        grouped.setdefault(str(group_id), []).append(row)

    video_rows = []
    for group_id, rows in grouped.items():
        true_labels = np.asarray([int(r.get("true_label", 0)) for r in rows], dtype=np.int64)
        fake_probabilities = np.asarray([float(r.get("fake_probability", 0.0)) for r in rows], dtype=np.float32)
        mean_probability = float(np.mean(fake_probabilities))
        true_label = int(true_labels[0]) if len(true_labels) else 0
        predicted_label = int(mean_probability >= 0.5)

        video_rows.append(
            {
                "video_id": group_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "fake_probability": mean_probability,
                "num_frames": len(rows),
            }
        )

    targets = np.asarray([row["true_label"] for row in video_rows], dtype=np.int64)
    probabilities = np.asarray([row["fake_probability"] for row in video_rows], dtype=np.float32)
    predictions = np.asarray([row["predicted_label"] for row in video_rows], dtype=np.int64)

    metrics = compute_binary_metrics(targets, probabilities, predictions)
    metrics["samples"] = video_rows
    return metrics
