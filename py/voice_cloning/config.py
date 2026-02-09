"""
Configuration module for voice cloning knowledge distillation pipeline.

This module defines all hyperparameters, paths, and settings for:
- Synthetic data generation
- Audio processing
- Model architecture
- Training loop
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrainingConfig:
    """Configuration for voice cloning training pipeline."""
    
    # ========================================
    # Data Generation
    # ========================================
    num_samples: int = 10000
    """Number of synthetic training samples to generate."""
    
    num_val_samples: int = 1000
    """Number of validation samples."""
    
    num_test_samples: int = 500
    """Number of test samples."""
    
    texts: List[str] = field(default_factory=lambda: [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "This morning, I took a walk in the park.",
        "Technology is advancing rapidly every day.",
        "The weather is beautiful outside.",
        "Machine learning is transforming the world.",
        "Speech synthesis has come a long way.",
        "Natural language processing is fascinating.",
        "Deep learning models are incredibly powerful.",
        "Voice cloning technology is advancing.",
    ])
    """Diverse text corpus for synthetic data generation."""
    
    languages: List[str] = field(default_factory=lambda: ["en"])
    """Languages to generate (en, ko, es, pt, fr)."""
    
    # ========================================
    # Audio Processing
    # ========================================
    sample_rate: int = 24000
    """Audio sample rate (must match Supertonic TTS)."""
    
    n_fft: int = 1024
    """FFT size for mel-spectrogram."""
    
    hop_length: int = 256
    """Hop length for STFT."""
    
    win_length: int = 1024
    """Window length for STFT."""
    
    n_mels: int = 80
    """Number of mel filterbanks."""
    
    fmin: float = 0.0
    """Minimum frequency for mel-spectrogram."""
    
    fmax: float = 8000.0
    """Maximum frequency for mel-spectrogram."""
    
    # ========================================
    # Model Architecture
    # ========================================
    encoder_type: str = "cnn"
    """Encoder architecture type: 'cnn' or 'transformer'."""
    
    hidden_dim: int = 512
    """Hidden dimension for encoder."""
    
    num_layers: int = 6
    """Number of layers (transformer blocks or conv blocks)."""
    
    num_heads: int = 8
    """Number of attention heads (for transformer)."""
    
    dropout: float = 0.1
    """Dropout rate."""
    
    # ========================================
    # Style Vector Dimensions
    # ========================================
    style_ttl_shape: Tuple[int, int, int] = (1, 50, 256)
    """Shape of style_ttl vector (must match Supertonic TTS)."""
    
    style_dp_shape: Tuple[int, int, int] = (1, 8, 16)
    """Shape of style_dp vector (must match Supertonic TTS)."""
    
    @property
    def style_ttl_size(self) -> int:
        """Total size of style_ttl vector."""
        return self.style_ttl_shape[0] * self.style_ttl_shape[1] * self.style_ttl_shape[2]
    
    @property
    def style_dp_size(self) -> int:
        """Total size of style_dp vector."""
        return self.style_dp_shape[0] * self.style_dp_shape[1] * self.style_dp_shape[2]
    
    # ========================================
    # Training
    # ========================================
    batch_size: int = 32
    """Batch size for training."""
    
    learning_rate: float = 1e-4
    """Initial learning rate."""
    
    epochs: int = 100
    """Number of training epochs."""
    
    weight_decay: float = 1e-5
    """Weight decay for optimizer."""
    
    warmup_epochs: int = 5
    """Number of warmup epochs for learning rate scheduling."""
    
    gradient_clip: float = 1.0
    """Gradient clipping threshold."""
    
    use_amp: bool = True
    """Use automatic mixed precision (fp16) training."""
    
    # Loss weights
    lambda_ttl: float = 1.0
    """Weight for style_ttl loss."""
    
    lambda_dp: float = 1.0
    """Weight for style_dp loss."""
    
    lambda_cosine: float = 0.1
    """Weight for cosine similarity loss."""
    
    # Early stopping
    early_stopping_patience: int = 10
    """Number of epochs without improvement before early stopping."""
    
    # ========================================
    # Data Augmentation
    # ========================================
    augment_noise_snr_min: float = 20.0
    """Minimum SNR for noise augmentation (dB)."""
    
    augment_noise_snr_max: float = 40.0
    """Maximum SNR for noise augmentation (dB)."""
    
    augment_time_stretch_min: float = 0.9
    """Minimum time stretch factor."""
    
    augment_time_stretch_max: float = 1.1
    """Maximum time stretch factor."""
    
    augment_pitch_shift_semitones: float = 1.0
    """Maximum pitch shift in semitones (±)."""
    
    augment_volume_min: float = 0.8
    """Minimum volume scaling factor."""
    
    augment_volume_max: float = 1.2
    """Maximum volume scaling factor."""
    
    spec_augment_time_mask: int = 20
    """Maximum time mask size for SpecAugment."""
    
    spec_augment_freq_mask: int = 10
    """Maximum frequency mask size for SpecAugment."""
    
    augment_prob: float = 0.5
    """Probability of applying augmentation."""
    
    # ========================================
    # Paths
    # ========================================
    onnx_dir: str = "../../assets/onnx"
    """Path to ONNX models directory."""
    
    voice_styles_dir: str = "../../assets/voice_styles"
    """Path to voice styles directory."""
    
    data_dir: str = "./training_data"
    """Directory for generated training data."""
    
    checkpoint_dir: str = "./checkpoints"
    """Directory for model checkpoints."""
    
    output_dir: str = "./output"
    """Directory for outputs (exported models, etc.)."""
    
    # ========================================
    # Data Loading
    # ========================================
    num_workers: int = 4
    """Number of workers for data loading."""
    
    pin_memory: bool = True
    """Pin memory for faster GPU transfer."""
    
    # ========================================
    # Logging
    # ========================================
    log_interval: int = 10
    """Log training metrics every N batches."""
    
    save_interval: int = 5
    """Save checkpoint every N epochs."""
    
    use_wandb: bool = False
    """Use Weights & Biases for logging."""
    
    use_tensorboard: bool = True
    """Use TensorBoard for logging."""
    
    # ========================================
    # Reproducibility
    # ========================================
    seed: int = 42
    """Random seed for reproducibility."""
    
    # ========================================
    # TTS Generation Parameters
    # ========================================
    tts_total_step: int = 5
    """Number of denoising steps for TTS generation."""
    
    tts_speed: float = 1.05
    """Speed factor for TTS generation."""
    
    # ========================================
    # Style Sampling Strategy
    # ========================================
    style_sample_mode: str = "mixed"
    """Style sampling mode: 'random', 'perturb', 'interpolate', or 'mixed'."""
    
    style_perturb_std: float = 0.1
    """Standard deviation for style perturbation."""
    
    style_interp_alpha_min: float = 0.0
    """Minimum interpolation factor for style mixing."""
    
    style_interp_alpha_max: float = 1.0
    """Maximum interpolation factor for style mixing."""


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def load_config_from_yaml(path: str) -> TrainingConfig:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file.
        
    Returns:
        TrainingConfig instance.
    """
    import yaml
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def save_config_to_yaml(config: TrainingConfig, path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: TrainingConfig instance.
        path: Path to save YAML configuration file.
    """
    import yaml
    from dataclasses import asdict
    with open(path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
