from __future__ import annotations

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Union, Tuple, Optional


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)
        
    Returns:
        Tuple of (waveform, sample_rate)
        waveform: Tensor of shape [1, num_samples] (mono)
        sample_rate: Actual sample rate of loaded audio
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
        
    return waveform, sr


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio waveform from original to target sample rate.
    
    Args:
        waveform: Input waveform tensor of shape [1, num_samples]
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled waveform tensor of shape [1, num_samples_resampled]
    """
    if orig_sr == target_sr:
        return waveform
    
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    power: float = 2.0,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Compute log-Mel spectrogram from audio waveform.
    
    Implements the audio mathematics described in the DeepGuard documentation:
    1. STFT: X(m,k) = sum_{n=0}^{N-1} x[n+m*H]*w[n]*e^(-j*2*pi*k*n/N)
    2. Power spectrum: P(m,k) = |X(m,k)|^2
    3. Mel filterbank: S_Mel = M * P
    4. Log amplitude: S_log = log(S_Mel + ε)
    
    Args:
        waveform: Input waveform tensor of shape [1, num_samples]
        sample_rate: Sample rate of audio (default: 16000 Hz)
        n_fft: FFT window length (default: 400 samples = 25ms at 16kHz)
        hop_length: Hop size (default: 160 samples = 10ms at 16kHz)
        n_mels: Number of Mel frequency bins (default: 80)
        f_min: Minimum frequency for Mel scale (default: 0 Hz)
        f_max: Maximum frequency for Mel scale (default: sample_rate/2)
        power: Exponent for magnitude spectrogram (default: 2.0 for power)
        normalized: Whether to normalize the Mel spectrogram
        
    Returns:
        Log-Mel spectrogram tensor of shape [1, n_mels, time_frames]
    """
    if f_max is None:
        f_max = sample_rate / 2.0
    
    # Create Mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
    )
    
    # Compute Mel spectrogram
    mel_spec = mel_spectrogram(waveform)  # Shape: [1, n_mels, time_frames]
    
    # Apply log compression with epsilon for numerical stability
    log_mel_spec = torch.log(mel_spec + 1e-6)
    
    return log_mel_spec


def compute_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Compute linear spectrogram (magnitude or power) from audio waveform.
    
    Args:
        waveform: Input waveform tensor of shape [1, num_samples]
        sample_rate: Sample rate of audio (default: 16000 Hz)
        n_fft: FFT window size (default: 400 samples = 25ms at 16kHz)
        hop_length: Hop length (default: 160 samples = 10ms at 16kHz)
        power: Exponent for magnitude spectrogram (default: 2.0 for power)
        
    Returns:
        Spectrogram tensor of shape [1, freq_bins, time_frames]
    """
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
    )
    
    spec = spectrogram(waveform)  # Shape: [1, freq_bins, time_frames]
    return spec