from __future__ import annotations

import torch


def get_device():
    """Return the available torch device, preferring CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
