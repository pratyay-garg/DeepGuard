from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from src.training.metrics import compute_binary_metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate a binary classifier and return aggregated metrics plus sample-level predictions."""
    model.eval()

    all_targets = []
    all_probabilities = []
    all_predictions = []
    sample_rows = []

    for images, labels, metadata in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        preds = (probabilities >= 0.5).to(torch.int64)

        all_targets.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())

        for idx, sample in enumerate(metadata):
            sample_rows.append(
                {
                    "video_id": sample.get("video_id"),
                    "frame_index": sample.get("frame_index"),
                    "timestamp": sample.get("timestamp"),
                    "true_label": int(labels[idx].item()),
                    "predicted_label": int(preds[idx].item()),
                    "fake_probability": float(probabilities[idx].item()),
                }
            )

    targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    probabilities = np.concatenate(all_probabilities) if all_probabilities else np.array([], dtype=np.float32)
    predictions = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)

    metrics = compute_binary_metrics(targets, probabilities, predictions)
    metrics["samples"] = sample_rows
    return metrics
