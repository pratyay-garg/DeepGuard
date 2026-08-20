from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from PIL import Image


@dataclass
class Sample:
    video_id: str
    frame_index: int
    timestamp: float
    face_path: str
    label: int


class FFPPDataset(torch.utils.data.Dataset):
    """Simple face-level FF++ dataset wrapper for a binary classification baseline."""

    def __init__(self, manifest_path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.samples = self._load_manifest()

    def _load_manifest(self):
        df = pd.read_csv(self.manifest_path)
        samples = []
        for row in df.to_dict(orient="records"):
            samples.append(
                Sample(
                    video_id=str(row.get("video_id", "")),
                    frame_index=int(row.get("frame_index", 0)),
                    timestamp=float(row.get("timestamp", 0.0)),
                    face_path=str(row.get("face_path", "")),
                    label=int(row.get("label", 0)),
                )
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = Path(sample.face_path)
        if not image_path.is_absolute():
            image_path = self.manifest_path.parent / image_path

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        metadata = {
            "video_id": sample.video_id,
            "frame_index": sample.frame_index,
            "timestamp": sample.timestamp,
            "face_path": str(image_path),
        }
        return image, sample.label, metadata
