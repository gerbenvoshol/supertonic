"""
PyTorch Dataset for voice cloning encoder training.

This module provides:
- VoiceCloneDataset: PyTorch Dataset that loads synthetic data from .npz files
- On-the-fly mel-spectrogram extraction
- Data augmentation (noise, time stretch, pitch shift, volume, SpecAugment)
- Collate function for variable-length sequences
- Helper function to create train/val/test dataloaders
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TrainingConfig
from utils import (
    compute_mel_spectrogram,
    add_noise,
    time_stretch,
    pitch_shift,
    spec_augment,
)

logger = logging.getLogger(__name__)


class VoiceCloneDataset(Dataset):
    """
    PyTorch Dataset for voice cloning encoder training.
    
    Loads synthetic training data from .npz files and performs:
    - On-the-fly mel-spectrogram extraction
    - Data augmentation (training split only)
    - Proper tensor formatting
    
    Attributes:
        data_dir: Directory containing train/val/test splits
        split: Data split ('train', 'val', or 'test')
        config: Training configuration
        file_paths: List of paths to .npz files
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize VoiceCloneDataset.
        
        Args:
            data_dir: Directory containing train/val/test subdirectories
            split: Data split ('train', 'val', or 'test')
            config: Training configuration. If None, uses default config.
            
        Raises:
            ValueError: If split is invalid or data directory doesn't exist
            FileNotFoundError: If split directory is empty
        """
        super().__init__()
        
        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
        
        self.split = split
        self.config = config or TrainingConfig()
        self.data_dir = Path(data_dir)
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Build path to split directory
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory does not exist: {self.split_dir}")
        
        # Load all .npz files in the split directory
        self.file_paths = sorted(list(self.split_dir.glob("*.npz")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {self.split_dir}")
        
        # Augmentation only for training
        self.augment = (split == "train")
        
        logger.info(f"Loaded {len(self.file_paths)} samples from {split} split")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with keys:
                - mel: Mel-spectrogram [n_mels, T]
                - style_ttl: Style TTL vector [style_ttl_size]
                - style_dp: Style DP vector [style_dp_size]
                - text: Text string (encoded as tensor)
                - lang: Language string (encoded as tensor)
                
        Raises:
            RuntimeError: If data loading or processing fails
        """
        file_path = self.file_paths[idx]
        
        try:
            # Load .npz file
            data = np.load(file_path, allow_pickle=True)
            
            # Extract data
            audio = data["audio"]
            style_ttl = data["style_ttl"]
            style_dp = data["style_dp"]
            text = str(data["text"])
            lang = str(data["lang"])
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError(f"Empty audio in {file_path}")
            
            # Apply data augmentation (training only)
            if self.augment:
                audio = self._apply_augmentation(audio)
            
            # Compute mel-spectrogram
            mel = compute_mel_spectrogram(
                audio=audio,
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
            )
            
            # Apply SpecAugment (training only, after mel computation)
            if self.augment and random.random() < self.config.augment_prob:
                mel = spec_augment(
                    mel,
                    time_mask_param=self.config.spec_augment_time_mask,
                    freq_mask_param=self.config.spec_augment_freq_mask,
                )
            
            # Convert to tensors
            mel = torch.from_numpy(mel).float()  # [n_mels, T]
            
            # Flatten style vectors
            style_ttl = torch.from_numpy(style_ttl.flatten()).float()
            style_dp = torch.from_numpy(style_dp.flatten()).float()
            
            return {
                "mel": mel,
                "style_ttl": style_ttl,
                "style_dp": style_dp,
                "text": text,
                "lang": lang,
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {file_path}: {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}") from e
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        All augmentations are applied probabilistically with probability
        specified by config.augment_prob (default 50%).
        
        Augmentations:
            1. Random noise injection (SNR 20-40 dB)
            2. Random time stretching (0.9-1.1x)
            3. Random pitch shifting (±1 semitone)
            4. Random volume scaling (0.8-1.2x)
        
        Args:
            audio: Audio waveform [T]
            
        Returns:
            Augmented audio [T']
        """
        # 1. Noise injection
        if random.random() < self.config.augment_prob:
            snr_db = random.uniform(
                self.config.augment_noise_snr_min,
                self.config.augment_noise_snr_max,
            )
            audio = add_noise(audio, snr_db)
        
        # 2. Time stretching
        if random.random() < self.config.augment_prob:
            rate = random.uniform(
                self.config.augment_time_stretch_min,
                self.config.augment_time_stretch_max,
            )
            audio = time_stretch(audio, rate)
        
        # 3. Pitch shifting
        if random.random() < self.config.augment_prob:
            n_steps = random.uniform(
                -self.config.augment_pitch_shift_semitones,
                self.config.augment_pitch_shift_semitones,
            )
            audio = pitch_shift(audio, self.config.sample_rate, n_steps)
        
        # 4. Volume scaling
        if random.random() < self.config.augment_prob:
            scale = random.uniform(
                self.config.augment_volume_min,
                self.config.augment_volume_max,
            )
            audio = audio * scale
        
        return audio


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length mel-spectrograms.
    
    Pads mel-spectrograms to the maximum length in the batch.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with keys:
            - mel: Padded mel-spectrograms [B, n_mels, T_max]
            - mel_lengths: Original lengths before padding [B]
            - style_ttl: Style TTL vectors [B, style_ttl_size]
            - style_dp: Style DP vectors [B, style_dp_size]
            - text: List of text strings [B]
            - lang: List of language strings [B]
    """
    # Extract components
    mels = [item["mel"] for item in batch]
    style_ttls = [item["style_ttl"] for item in batch]
    style_dps = [item["style_dp"] for item in batch]
    texts = [item["text"] for item in batch]
    langs = [item["lang"] for item in batch]
    
    # Get mel lengths
    mel_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    
    # Find max length
    max_len = mel_lengths.max().item()
    
    # Pad mels to max length
    n_mels = mels[0].shape[0]
    padded_mels = torch.zeros(len(batch), n_mels, max_len, dtype=torch.float32)
    
    for i, mel in enumerate(mels):
        padded_mels[i, :, :mel.shape[1]] = mel
    
    # Stack style vectors
    style_ttls = torch.stack(style_ttls, dim=0)
    style_dps = torch.stack(style_dps, dim=0)
    
    return {
        "mel": padded_mels,
        "mel_lengths": mel_lengths,
        "style_ttl": style_ttls,
        "style_dp": style_dps,
        "text": texts,
        "lang": langs,
    }


def get_dataloaders(
    data_dir: str,
    config: Optional[TrainingConfig] = None,
    train_shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for voice cloning.
    
    Args:
        data_dir: Directory containing train/val/test subdirectories
        config: Training configuration. If None, uses default config.
        train_shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Raises:
        ValueError: If data_dir is invalid or splits are missing
    """
    if config is None:
        config = TrainingConfig()
    
    # Create datasets
    try:
        train_dataset = VoiceCloneDataset(
            data_dir=data_dir,
            split="train",
            config=config,
        )
        logger.info(f"Created training dataset with {len(train_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create training dataset: {e}")
        raise
    
    try:
        val_dataset = VoiceCloneDataset(
            data_dir=data_dir,
            split="val",
            config=config,
        )
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create validation dataset: {e}")
        raise
    
    try:
        test_dataset = VoiceCloneDataset(
            data_dir=data_dir,
            split="test",
            config=config,
        )
        logger.info(f"Created test dataset with {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create test dataset: {e}")
        raise
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    logger.info(
        f"Created dataloaders - Train: {len(train_loader)} batches, "
        f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader


def validate_dataset(data_dir: str, config: Optional[TrainingConfig] = None) -> Dict[str, int]:
    """
    Validate dataset integrity and return statistics.
    
    Checks:
        - All splits exist and contain data
        - Files are readable and not corrupted
        - Data shapes are consistent
        
    Args:
        data_dir: Directory containing train/val/test subdirectories
        config: Training configuration. If None, uses default config.
        
    Returns:
        Dictionary with statistics:
            - train_samples: Number of training samples
            - val_samples: Number of validation samples
            - test_samples: Number of test samples
            - corrupted_files: Number of corrupted files
            
    Raises:
        ValueError: If validation fails critically
    """
    if config is None:
        config = TrainingConfig()
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    stats = {
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "corrupted_files": 0,
    }
    
    # Check each split
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            logger.warning(f"Split directory does not exist: {split_dir}")
            continue
        
        npz_files = list(split_dir.glob("*.npz"))
        split_key = f"{split}_samples"
        stats[split_key] = len(npz_files)
        
        # Check a few files for corruption
        sample_size = min(10, len(npz_files))
        for file_path in random.sample(npz_files, sample_size):
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # Check required keys
                required_keys = ["audio", "style_ttl", "style_dp", "text", "lang"]
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"Missing key '{key}' in {file_path}")
                
                # Check shapes
                audio = data["audio"]
                style_ttl = data["style_ttl"]
                style_dp = data["style_dp"]
                
                if len(audio) == 0:
                    raise ValueError(f"Empty audio in {file_path}")
                
                if style_ttl.size != config.style_ttl_size:
                    logger.warning(
                        f"style_ttl size mismatch in {file_path}: "
                        f"expected {config.style_ttl_size}, got {style_ttl.size}"
                    )
                
                if style_dp.size != config.style_dp_size:
                    logger.warning(
                        f"style_dp size mismatch in {file_path}: "
                        f"expected {config.style_dp_size}, got {style_dp.size}"
                    )
                    
            except Exception as e:
                logger.error(f"Corrupted file {file_path}: {e}")
                stats["corrupted_files"] += 1
    
    # Log statistics
    logger.info(f"Dataset validation complete:")
    logger.info(f"  Train samples: {stats['train_samples']}")
    logger.info(f"  Val samples: {stats['val_samples']}")
    logger.info(f"  Test samples: {stats['test_samples']}")
    logger.info(f"  Corrupted files: {stats['corrupted_files']}")
    
    return stats


if __name__ == "__main__":
    """Test the dataset implementation."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Test VoiceCloneDataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate dataset without creating dataloaders",
    )
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    
    # Validate dataset
    logger.info("Validating dataset...")
    try:
        stats = validate_dataset(args.data_dir, config)
        logger.info("Dataset validation successful!")
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        exit(1)
    
    if args.validate_only:
        exit(0)
    
    # Test dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=args.data_dir,
            config=config,
        )
        logger.info("Dataloaders created successfully!")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        exit(1)
    
    # Test loading a batch from each split
    logger.info("\nTesting batch loading...")
    
    for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        logger.info(f"\n{name} split:")
        try:
            batch = next(iter(loader))
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  mel shape: {batch['mel'].shape}")
            logger.info(f"  mel_lengths shape: {batch['mel_lengths'].shape}")
            logger.info(f"  style_ttl shape: {batch['style_ttl'].shape}")
            logger.info(f"  style_dp shape: {batch['style_dp'].shape}")
            logger.info(f"  text samples: {batch['text'][:2]}")
            logger.info(f"  lang samples: {batch['lang'][:2]}")
        except Exception as e:
            logger.error(f"  Failed to load batch: {e}")
            exit(1)
    
    logger.info("\n✓ All tests passed!")
