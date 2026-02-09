"""
Utility functions for voice cloning pipeline.

This module provides shared utilities for:
- Audio processing (mel-spectrogram, augmentation)
- File I/O
- Visualization
- Misc helpers
"""

import os
import random
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Compute mel-spectrogram from audio waveform.
    
    Args:
        audio: Audio waveform [T].
        sample_rate: Audio sample rate.
        n_fft: FFT size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        n_mels: Number of mel filterbanks.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        
    Returns:
        Mel-spectrogram [n_mels, T_frames].
    """
    # Compute STFT
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    
    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def load_audio(
    path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load and preprocess audio file.
    
    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        normalize: Whether to normalize audio.
        
    Returns:
        Audio waveform [T].
    """
    # Load audio
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    
    # Normalize
    if normalize and len(audio) > 0:
        audio = audio / (np.abs(audio).max() + 1e-8)
    
    return audio


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 24000,
) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio waveform [T].
        path: Output path.
        sample_rate: Audio sample rate.
    """
    import soundfile as sf
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save audio
    sf.write(path, audio, sample_rate)


def add_noise(
    audio: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Add Gaussian noise to audio with specified SNR.
    
    Args:
        audio: Audio waveform [T].
        snr_db: Signal-to-noise ratio in dB.
        
    Returns:
        Noisy audio [T].
    """
    if len(audio) == 0:
        return audio
    
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    
    return audio + noise


def time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Time-stretch audio.
    
    Args:
        audio: Audio waveform [T].
        rate: Stretch factor (< 1.0 = slower, > 1.0 = faster).
        
    Returns:
        Time-stretched audio [T'].
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float,
) -> np.ndarray:
    """
    Pitch-shift audio.
    
    Args:
        audio: Audio waveform [T].
        sample_rate: Audio sample rate.
        n_steps: Number of semitones to shift.
        
    Returns:
        Pitch-shifted audio [T].
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


def spec_augment(
    mel_spec: np.ndarray,
    time_mask_param: int = 20,
    freq_mask_param: int = 10,
) -> np.ndarray:
    """
    Apply SpecAugment to mel-spectrogram.
    
    Args:
        mel_spec: Mel-spectrogram [n_mels, T_frames].
        time_mask_param: Maximum time mask size.
        freq_mask_param: Maximum frequency mask size.
        
    Returns:
        Augmented mel-spectrogram [n_mels, T_frames].
    """
    mel_spec = mel_spec.copy()
    n_mels, n_frames = mel_spec.shape
    
    # Time masking
    if time_mask_param > 0 and n_frames > time_mask_param:
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, n_frames - t)
        mel_spec[:, t0:t0 + t] = 0
    
    # Frequency masking
    if freq_mask_param > 0 and n_mels > freq_mask_param:
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, n_mels - f)
        mel_spec[f0:f0 + f, :] = 0
    
    return mel_spec


def pad_or_trim_mel(
    mel: np.ndarray,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Pad or trim mel-spectrogram to target length.
    
    Args:
        mel: Mel-spectrogram [n_mels, T].
        target_length: Target time length. If None, returns as-is.
        
    Returns:
        Padded/trimmed mel-spectrogram [n_mels, target_length].
    """
    if target_length is None:
        return mel
    
    n_mels, current_length = mel.shape
    
    if current_length > target_length:
        # Trim
        return mel[:, :target_length]
    elif current_length < target_length:
        # Pad
        pad_width = target_length - current_length
        return np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        return mel


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
