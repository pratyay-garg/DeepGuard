from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric, config, metrics=None):
    """Persist a training checkpoint with model/optimizer state and metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "metrics": metrics or {},
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path, model, optimizer=None, device=None):
    """Load a checkpoint into model and optional optimizer."""
    checkpoint = torch.load(path, map_location=device if device is not None else "cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
