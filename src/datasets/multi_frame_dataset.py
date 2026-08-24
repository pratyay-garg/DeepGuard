from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
from PIL import Image


@dataclass
class PreparedSequence:
    """A single prepared temporal sequence of frames."""

    sample_id: str
    image_paths: List[str]
    label: int
    metadata: Dict[str, Any]


class MultiFrameDataset(torch.utils.data.Dataset):
    """Dataset for temporal sequence training using consecutive frames.

    This dataset loads multiple consecutive frames from manifest files to support
    temporal models like TSM or Temporal CNN. Each sample is a sequence of frames.

    Args:
        manifest_path: Path to CSV manifest file
        transform: Optional transform to apply to each frame
        image_path_column: Column name in manifest for image paths
        label_column: Column name in manifest for labels
        id_column: Column name for sample/video IDs
        metadata_columns: Additional metadata columns to preserve
        root_dir: Optional root directory for resolving relative paths
        sequence_length: Number of consecutive frames to load per sample
        stride: Stride between sequences (default: sequence_length for non-overlapping)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform=None,
        image_path_column: str = "face_path",
        label_column: str = "label",
        id_column: str = "video_id",
        metadata_columns: Optional[Iterable[str]] = None,
        root_dir: str | Path | None = None,
        sequence_length: int = 32,
        stride: int = 32,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.id_column = id_column
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.sequence_length = sequence_length
        self.stride = stride

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {sequence_length}")

        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self.samples = self._load_manifest(metadata_columns=metadata_columns)

    def _load_manifest(self, metadata_columns: Optional[Iterable[str]] = None):
        df = pd.read_csv(self.manifest_path)

        if self.image_path_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain image path column '{self.image_path_column}'."
            )
        if self.label_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain label column '{self.label_column}'."
            )

        # Group by video_id to extract sequences
        if self.id_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain id column '{self.id_column}'."
            )

        grouped = df.groupby(self.id_column)

        samples = []
        for video_id, group in grouped:
            # Sort by frame_index if available, otherwise by index
            if "frame_index" in group.columns:
                group = group.sort_values("frame_index")
            else:
                group = group.sort_index()

            # Extract unique frames
            unique_frames = group[self.image_path_column].unique()

            # Create sequences
            num_sequences = len(unique_frames) // self.stride
            if num_sequences == 0:
                continue

            # Limit to first num_sequences
            unique_frames = unique_frames[: num_sequences * self.stride]

            for i in range(num_sequences):
                frame_indices = list(range(i * self.stride, i * self.stride + self.sequence_length))

                if len(frame_indices) < self.sequence_length:
                    continue

                frame_paths = [unique_frames[idx] for idx in frame_indices]

                # Extract metadata from first frame
                first_row = group.iloc[frame_indices[0]]
                metadata = {
                    column: first_row.get(column)
                    for column in (
                        "video_id",
                        "frame_index",
                        "timestamp",
                        "sample_id",
                        "original_file",
                    )
                    if column in df.columns
                }

                samples.append(
                    PreparedSequence(
                        sample_id=f"{video_id}_seq_{i}",
                        image_paths=frame_paths,
                        label=int(first_row.get(self.label_column, 0)),
                        metadata=metadata,
                    )
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)

        if candidate.is_absolute() and candidate.exists():
            return candidate

        search_roots = []
        if self.root_dir is not None:
            search_roots.append(self.root_dir)
        search_roots.extend(
            [
                self.manifest_path.parent,
                self.manifest_path.parent.parent,
                Path.cwd(),
            ]
        )

        for root in search_roots:
            if root is None:
                continue
            resolved = root / candidate
            if resolved.exists():
                return resolved

        return candidate

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load all frames
        frames = []
        for image_path in sample.image_paths:
            path = self._resolve_image_path(image_path)
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        # Stack into tensor [T, C, H, W]
        frames_tensor = torch.stack(frames)

        # Create metadata dict
        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata["num_frames"] = len(sample.image_paths)
        metadata["image_paths"] = sample.image_paths

        return frames_tensor, sample.label, metadata
