from __future__ import annotations

import io
import torch
import torchaudio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from datasets import load_from_disk
from torch.utils.data import Dataset

from src.preprocessing.audio import compute_mel_spectrogram


class AudioDataset(Dataset):
    """
    Dataset for audio deepfake detection.
    
    Loads audio files (or waveforms) and computes log-Mel spectrograms on the fly.
    
    Expected dataset structure (from Hugging Face datasets library):
        - Each example has:
            - "audio": dict with "path" (str) and "bytes" (bytes for FLAC)
            - "label": int (0 for bonafide, 1 for spoof)
            - Optional: "notes": str (JSON metadata)
    
    Args:
        dataset_path: Path to the saved dataset (from prepare_audio_dataset.py)
        sample_rate: Target sample rate for audio (default: 16000)
        n_fft: FFT window size for spectrogram (default: 400)
        hop_length: Hop length for spectrogram (default: 160)
        n_mels: Number of Mel bands (default: 80)
        f_min: Minimum frequency for Mel scale (default: 0.0)
        f_max: Maximum frequency for Mel scale (default: sample_rate/2)
        power: Exponent for magnitude spectrogram (default: 2.0)
        transform: Optional transform to apply to the spectrogram
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        max_frames: int = 200,
        transform=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.power = power
        self.max_frames = max_frames
        self.transform = transform
        
        # Load the dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        self.dataset = load_from_disk(str(self.dataset_path))
        
        # Verify dataset has required fields
        if "audio" not in self.dataset.column_names:
            raise ValueError(
                "Dataset must contain an 'audio' field with waveform and sampling_rate. "
                "Check the dataset structure."
            )
        if "label" not in self.dataset.column_names:
            raise ValueError("Dataset must contain a 'label' field.")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a single item from the dataset.
        
        Returns:
            Tuple of (spectrogram, label, metadata)
            spectrogram: Tensor of shape [1, n_mels, time_frames]
            label: Integer label (0 or 1)
            metadata: Dictionary with additional information
        """
        example = self.dataset[index]
        
        # Extract audio - ASVspoof format has "audio" with "path" and "bytes" (FLAC)
        audio_info = example["audio"]
        
        # Load audio from bytes (FLAC format) or path
        if "bytes" in audio_info:
            # Load from bytes using torchaudio
            waveform, orig_sr = torchaudio.load(io.BytesIO(audio_info["bytes"]))
        elif "path" in audio_info:
            # Load from file path
            waveform, orig_sr = torchaudio.load(audio_info["path"])
        elif "array" in audio_info:
            # Standard HF format with array and sampling_rate
            waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
            orig_sr = audio_info["sampling_rate"]
        else:
            raise ValueError(f"Unknown audio format: {list(audio_info.keys())}")
        
        # Ensure waveform is mono and correct shape [1, num_samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            # Convert stereo to mono by averaging
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )
        
        # Compute log-Mel spectrogram
        spectrogram = compute_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=self.power,
        )  # Shape: [1, n_mels, time_frames]
        
        # Pad or truncate to fixed length
        time_frames = spectrogram.shape[-1]
        if time_frames < self.max_frames:
            # Pad with zeros
            pad_amount = self.max_frames - time_frames
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))
        elif time_frames > self.max_frames:
            # Truncate
            spectrogram = spectrogram[:, :, :self.max_frames]
        
        # Apply optional transform
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        # Prepare metadata
        metadata = {
            "index": index,
            "original_sampling_rate": orig_sr,
            "waveform_length": waveform.shape[1],
        }
        # Add any additional fields from the example
        for key in example.keys():
            if key not in ["audio", "label"]:
                metadata[key] = example[key]
        
        label = int(example["label"])
        
        return spectrogram, label, metadata