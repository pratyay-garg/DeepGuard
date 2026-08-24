from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import torch
from PIL import Image


@dataclass
class PreparedSample:
    """A single prepared sample row from a manifest file."""

    sample_id: str
    image_path: str
    label: int
    metadata: Dict[str, Any]


class PreparedDataset(torch.utils.data.Dataset):
    """Generic dataset for precomputed image crops described by a CSV manifest.

    The manifest must include at least one image path column and one label column.
    By default the loader expects the roadmap's columns:
    - image path: ``face_path``
    - label: ``label``

    Additional columns are returned in the metadata dictionary so the same class
    can be reused for any prepared dataset without model-specific code.
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
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.id_column = id_column
        self.root_dir = Path(root_dir) if root_dir is not None else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

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

        if metadata_columns is None:
            metadata_columns = [
                column
                for column in ("video_id", "frame_index", "timestamp", "sample_id")
                if column in df.columns
            ]
        else:
            metadata_columns = [column for column in metadata_columns if column in df.columns]

        samples = []
        for row in df.to_dict(orient="records"):
            sample_id = str(row.get(self.id_column, row.get("sample_id", "")))
            if not sample_id:
                sample_id = str(row.get(self.image_path_column, ""))

            metadata = {
                column: row.get(column)
                for column in metadata_columns
            }
            metadata[self.image_path_column] = row.get(self.image_path_column)

            samples.append(
                PreparedSample(
                    sample_id=sample_id,
                    image_path=str(row.get(self.image_path_column, "")),
                    label=int(row.get(self.label_column, 0)),
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

        # Fall back to the raw path so PIL raises a clear error if nothing matched.
        return candidate

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample.image_path)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata[self.image_path_column] = str(image_path)

        return image, sample.label, metadata

