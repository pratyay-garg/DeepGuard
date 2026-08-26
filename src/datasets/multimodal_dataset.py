from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
import torchaudio
from PIL import Image


@dataclass
class MultimodalSample:
    """A single multimodal sample with video frame and audio."""

    sample_id: str
    image_path: str
    audio_path: str
    label: int
    metadata: Dict[str, Any]


class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset for multimodal training using video frames and audio.

    This dataset loads a single frame and its corresponding audio from manifest files
    to support multimodal fusion models. Each sample contains:
    - A video frame (image)
    - An audio waveform/spectrogram
    - Label and metadata

    Args:
        manifest_path: Path to CSV manifest file
        transform: Optional transform to apply to image
        audio_transform: Optional transform to apply to audio
        image_path_column: Column name in manifest for image paths
        audio_path_column: Column name in manifest for audio paths
        label_column: Column name in manifest for labels
        id_column: Column name for sample/video IDs
        metadata_columns: Additional metadata columns to preserve
        root_dir: Optional root directory for resolving relative paths
        audio_config: Audio processing configuration
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform=None,
        audio_transform=None,
        image_path_column: str = "face_path",
        audio_path_column: str = "audio_path",
        label_column: str = "label",
        id_column: str = "video_id",
        metadata_columns: Optional[Iterable[str]] = None,
        root_dir: str | Path | None = None,
        audio_config: Optional[Dict[str, Any]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.audio_transform = audio_transform
        self.image_path_column = image_path_column
        self.audio_path_column = audio_path_column
        self.label_column = label_column
        self.id_column = id_column
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.audio_config = audio_config or {}

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.samples = self._load_manifest(metadata_columns=metadata_columns)

    def _load_manifest(self, metadata_columns: Optional[Iterable[str]] = None):
        df = pd.read_csv(self.manifest_path)

        if self.image_path_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain image path column '{self.image_path_column}'."
            )
        if self.audio_path_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain audio path column '{self.audio_path_column}'."
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
            metadata[self.audio_path_column] = row.get(self.audio_path_column)

            samples.append(
                MultimodalSample(
                    sample_id=sample_id,
                    image_path=str(row.get(self.image_path_column, "")),
                    audio_path=str(row.get(self.audio_path_column, "")),
                    label=int(row.get(self.label_column, 0)),
                    metadata=metadata,
                )
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)

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

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and process audio file."""
        path = self._resolve_path(audio_path)

        if not path.exists():
            # Return silence if audio not found
            sample_rate = self.audio_config.get("sample_rate", 16000)
            duration = self.audio_config.get("duration", 4.0)
            return torch.zeros(1, int(sample_rate * duration))

        waveform, sample_rate = torchaudio.load(str(path))

        # Resample if needed
        target_sr = self.audio_config.get("sample_rate", 16000)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim or pad to target duration
        target_duration = self.audio_config.get("duration", 4.0)
        target_samples = int(target_sr * target_duration)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel spectrogram from waveform."""
        sample_rate = self.audio_config.get("sample_rate", 16000)
        n_fft = self.audio_config.get("n_fft", 400)
        hop_length = self.audio_config.get("hop_length", 160)
        n_mels = self.audio_config.get("n_mels", 80)
        f_min = self.audio_config.get("f_min", 0.0)
        f_max = self.audio_config.get("f_max", None)
        power = self.audio_config.get("power", 2.0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

        mel_spec = mel_transform(waveform)

        # Log compression
        mel_spec = torch.log(mel_spec + 1e-6)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        return mel_spec

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load image
        image_path = self._resolve_path(sample.image_path)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Load and process audio
        waveform = self._load_audio(sample.audio_path)
        mel_spec = self._compute_mel_spectrogram(waveform)

        # Apply audio transform if provided
        if self.audio_transform is not None:
            mel_spec = self.audio_transform(mel_spec)

        # Create metadata dict
        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata[self.image_path_column] = str(image_path)
        metadata[self.audio_path_column] = sample.audio_path

        return image, mel_spec, sample.label, metadata


@dataclass
class MultimodalSequence:
    """A multimodal temporal sequence of frames with audio."""

    sample_id: str
    image_paths: List[str]
    audio_path: str
    label: int
    metadata: Dict[str, Any]


class MultimodalSequenceDataset(torch.utils.data.Dataset):
    """Dataset for multimodal temporal sequence training.

    This dataset loads multiple consecutive frames and their corresponding audio
    to support temporal multimodal fusion models.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform=None,
        audio_transform=None,
        image_path_column: str = "face_path",
        audio_path_column: str = "audio_path",
        label_column: str = "label",
        id_column: str = "video_id",
        metadata_columns: Optional[Iterable[str]] = None,
        root_dir: str | Path | None = None,
        sequence_length: int = 32,
        stride: int = 32,
        audio_config: Optional[Dict[str, Any]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.audio_transform = audio_transform
        self.image_path_column = image_path_column
        self.audio_path_column = audio_path_column
        self.label_column = label_column
        self.id_column = id_column
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.sequence_length = sequence_length
        self.stride = stride
        self.audio_config = audio_config or {}

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
        if self.audio_path_column not in df.columns:
            raise KeyError(
                f"Manifest {self.manifest_path} must contain audio path column '{self.audio_path_column}'."
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

            # Create sequences using sliding window
            seq_len = self.sequence_length
            stride = self.stride
            
            if len(unique_frames) < seq_len:
                continue
                
            num_sequences = (len(unique_frames) - seq_len) // stride + 1

            for i in range(num_sequences):
                start_idx = i * stride
                frame_indices = list(range(start_idx, start_idx + seq_len))

                frame_paths = [unique_frames[idx] for idx in frame_indices]

                # Get audio path from first frame (same for all frames in video)
                first_row = group.iloc[frame_indices[0]]
                audio_path = first_row.get(self.audio_path_column, "")

                # Extract metadata from first frame
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
                    MultimodalSequence(
                        sample_id=f"{video_id}_seq_{i}",
                        image_paths=frame_paths,
                        audio_path=audio_path,
                        label=int(first_row.get(self.label_column, 0)),
                        metadata=metadata,
                    )
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)

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

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and process audio file."""
        path = self._resolve_path(audio_path)

        if not path.exists():
            # Return silence if audio not found
            sample_rate = self.audio_config.get("sample_rate", 16000)
            duration = self.audio_config.get("duration", 4.0)
            return torch.zeros(1, int(sample_rate * duration))

        waveform, sample_rate = torchaudio.load(str(path))

        # Resample if needed
        target_sr = self.audio_config.get("sample_rate", 16000)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim or pad to target duration
        target_duration = self.audio_config.get("duration", 4.0)
        target_samples = int(target_sr * target_duration)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel spectrogram from waveform."""
        sample_rate = self.audio_config.get("sample_rate", 16000)
        n_fft = self.audio_config.get("n_fft", 400)
        hop_length = self.audio_config.get("hop_length", 160)
        n_mels = self.audio_config.get("n_mels", 80)
        f_min = self.audio_config.get("f_min", 0.0)
        f_max = self.audio_config.get("f_max", None)
        power = self.audio_config.get("power", 2.0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

        mel_spec = mel_transform(waveform)

        # Log compression
        mel_spec = torch.log(mel_spec + 1e-6)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        return mel_spec

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load all frames
        frames = []
        for image_path in sample.image_paths:
            path = self._resolve_path(image_path)
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        # Stack into tensor [T, C, H, W]
        frames_tensor = torch.stack(frames)

        # Load and process audio
        waveform = self._load_audio(sample.audio_path)
        mel_spec = self._compute_mel_spectrogram(waveform)

        # Apply audio transform if provided
        if self.audio_transform is not None:
            mel_spec = self.audio_transform(mel_spec)

        # Create metadata dict
        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata["num_frames"] = len(sample.image_paths)
        metadata["image_paths"] = sample.image_paths
        metadata["audio_path"] = sample.audio_path

        return frames_tensor, mel_spec, sample.label, metadata