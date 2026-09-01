import random
import torch
from pathlib import Path
from PIL import Image

from .multimodal_dataset import MultimodalSequenceDataset

class AVSyncDataset(MultimodalSequenceDataset):
    """
    Dataset for Audio-Video Synchronization (Phase 11).
    Inherits from MultimodalSequenceDataset to leverage existing audio/video loading.
    Generates positive pairs (sync) and negative pairs (unsync) dynamically.
    """
    def __init__(
        self,
        manifest_path,
        transform=None,
        audio_transform=None,
        image_path_column="face_path",
        audio_path_column="audio_path",
        label_column="label",
        id_column="video_id",
        metadata_columns=None,
        root_dir=None,
        sequence_length=32,
        stride=32,
        audio_config=None,
        negative_ratio=0.5,
    ):
        super().__init__(
            manifest_path=manifest_path,
            transform=transform,
            audio_transform=audio_transform,
            image_path_column=image_path_column,
            audio_path_column=audio_path_column,
            label_column=label_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            root_dir=root_dir,
            sequence_length=sequence_length,
            stride=stride,
            audio_config=audio_config,
        )
        self.negative_ratio = negative_ratio
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        is_negative = random.random() < self.negative_ratio
        target_audio_path = sample.audio_path
        
        if is_negative:
            # Easy negative: cross-video swap
            random_idx = random.randint(0, len(self.samples) - 1)
            for _ in range(10):
                if self.samples[random_idx].metadata.get("video_id") != sample.metadata.get("video_id"):
                    break
                random_idx = random.randint(0, len(self.samples) - 1)
            target_audio_path = self.samples[random_idx].audio_path
            
        # Load all frames
        frames = []
        for image_path in sample.image_paths:
            path = self._resolve_path(image_path)
            with Image.open(path) as img:
                image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        frames_tensor = torch.stack(frames)

        # Load and process audio from target_audio_path
        waveform = self._load_audio(target_audio_path)
        mel_spec = self._compute_mel_spectrogram(waveform)

        if self.audio_transform is not None:
            mel_spec = self.audio_transform(mel_spec)

        # Create metadata dict
        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata["num_frames"] = len(sample.image_paths)
        metadata["image_paths"] = sample.image_paths
        metadata["audio_path"] = target_audio_path
        metadata["is_sync"] = not is_negative

        # Label is 1 for sync, 0 for unsync
        sync_label = 0 if is_negative else 1

        return frames_tensor, mel_spec, sync_label, metadata
