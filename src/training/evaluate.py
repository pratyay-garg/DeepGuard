from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from src.training.metrics import compute_binary_metrics, compute_video_level_metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate a binary classifier and return frame-level and video-level metrics."""
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

        # Metadata may be a dict of lists (default collate) or a list of dicts.
        batch_meta: list
        if isinstance(metadata, dict):
            # Convert {'video_id': [..], 'frame_index': [..], ...} -> list of dicts
            keys = list(metadata.keys())
            length = len(next(iter(metadata.values()))) if keys else 0
            batch_meta = [ {k: metadata[k][i] for k in keys} for i in range(length) ]
        else:
            batch_meta = list(metadata)

        for idx, sample in enumerate(batch_meta):
            # sample may already be a dict; ensure `.get` exists
            if isinstance(sample, dict):
                vid = sample.get("video_id")
                sample_id = sample.get("sample_id")
                fi = sample.get("frame_index")
                ts = sample.get("timestamp")
            else:
                # Fallback if sample is a tuple/list: try index-based access
                try:
                    vid, fi, ts = sample
                    sample_id = None
                except Exception:
                    vid, fi, ts, sample_id = None, None, None, None

            sample_rows.append(
                {
                    "video_id": vid,
                    "sample_id": sample_id,
                    "frame_index": fi,
                    "timestamp": ts,
                    "true_label": int(labels[idx].item()),
                    "predicted_label": int(preds[idx].item()),
                    "fake_probability": float(probabilities[idx].item()),
                }
            )

    targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    probabilities = np.concatenate(all_probabilities) if all_probabilities else np.array([], dtype=np.float32)
    predictions = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)

    metrics = compute_binary_metrics(targets, probabilities, predictions)
    video_metrics = compute_video_level_metrics(sample_rows)

    return {
        "frame_metrics": metrics,
        "video_metrics": video_metrics,
        "samples": sample_rows,
    }
