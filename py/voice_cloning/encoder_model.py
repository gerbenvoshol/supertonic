"""
Audio encoder architectures for voice cloning.

This module implements two encoder architectures:
1. CNNEncoder: Convolutional encoder with attention pooling (default)
2. TransformerEncoder: Transformer-based encoder with attention pooling

Both encoders take mel-spectrograms as input and produce style embeddings
for voice cloning (style_ttl and style_dp vectors).
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import TrainingConfig


class AttentionPooling(nn.Module):
    """
    Attention pooling layer that learns to weight important time frames.
    
    This module computes attention weights over the temporal dimension
    and produces a weighted sum of the input features.
    
    Args:
        hidden_dim: Dimension of input features.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Input tensor of shape [B, T, C]
            mask: Optional boolean mask of shape [B, T] (True for valid positions)
            
        Returns:
            Pooled tensor of shape [B, C]
        """
        # x must be [B, T, C]
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]
        
        # Apply attention weights
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, C]
        
        return pooled


class CNNEncoder(nn.Module):
    """
    Convolutional encoder with attention pooling.
    
    This encoder uses strided convolutions for downsampling and attention pooling
    for temporal aggregation. It's efficient and works well for voice cloning tasks.
    
    Architecture:
        - Conv1d(80, 128, k=7, s=2) + BatchNorm1d + GELU
        - Conv1d(128, 256, k=5, s=2) + BatchNorm1d + GELU
        - Conv1d(256, 512, k=3, s=2) + BatchNorm1d + GELU
        - Conv1d(512, 512, k=3, s=2) + BatchNorm1d + GELU
        - Conv1d(512, 512, k=3, s=1) + BatchNorm1d + GELU (no stride)
        - Attention pooling
        - FC(512, 512) + GELU + Dropout
        - Two output heads (style_ttl and style_dp)
    
    Args:
        config: Training configuration.
        use_checkpoint: Use gradient checkpointing for memory efficiency.
    """
    
    def __init__(self, config: TrainingConfig, use_checkpoint: bool = False):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(80, 128, kernel_size=7, stride=2),
            self._make_conv_block(128, 256, kernel_size=5, stride=2),
            self._make_conv_block(256, 512, kernel_size=3, stride=2),
            self._make_conv_block(512, 512, kernel_size=3, stride=2),
            self._make_conv_block(512, 512, kernel_size=3, stride=1),
        ])
        
        # Attention pooling
        self.attention_pool = AttentionPooling(512)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Output heads
        self.style_ttl_head = nn.Linear(512, config.style_ttl_size)
        self.style_dp_head = nn.Linear(512, config.style_dp_size)
        
        # Initialize weights
        self._init_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                        kernel_size: int, stride: int) -> nn.Module:
        """Create a convolutional block with BatchNorm and GELU."""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _apply_conv_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutional blocks."""
        for block in self.conv_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
    
    def forward(self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mel: Mel-spectrogram tensor of shape [B, 80, T]
            mask: Optional boolean mask of shape [B, T] (True for valid positions)
            
        Returns:
            Tuple of (style_ttl, style_dp) tensors:
                - style_ttl: [B, 1, 50, 256]
                - style_dp: [B, 1, 8, 16]
        """
        batch_size = mel.size(0)
        
        # Apply convolutional blocks
        x = self._apply_conv_blocks(mel)  # [B, 512, T']
        
        # Adjust mask for downsampling
        if mask is not None:
            # Calculate downsampling factor (2^4 = 16 from 4 strided convs)
            downsample_factor = 16
            mask_len = (mask.sum(dim=1) / downsample_factor).long()
            new_mask = torch.zeros(batch_size, x.size(2), dtype=torch.bool, device=x.device)
            for i, length in enumerate(mask_len):
                new_mask[i, :length] = True
            mask = new_mask
        
        # Transpose from [B, C, T'] to [B, T', C] for attention pooling
        x = x.transpose(1, 2)  # [B, T', 512]
        
        # Attention pooling
        x = self.attention_pool(x, mask)  # [B, 512]
        
        # Projection
        x = self.projection(x)  # [B, 512]
        
        # Output heads
        style_ttl = self.style_ttl_head(x)  # [B, 12800]
        style_dp = self.style_dp_head(x)    # [B, 128]
        
        # Reshape to target dimensions
        style_ttl = style_ttl.view(batch_size, *self.config.style_ttl_shape)  # [B, 1, 50, 256]
        style_dp = style_dp.view(batch_size, *self.config.style_dp_shape)     # [B, 1, 8, 16]
        
        return style_ttl, style_dp
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer.
    
    Args:
        d_model: Dimension of the model.
        max_len: Maximum sequence length.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [B, T, C]
            
        Returns:
            Tensor with positional encoding added [B, T, C]
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder with attention pooling.
    
    This encoder uses transformer blocks for sequence modeling and attention pooling
    for temporal aggregation. It's more powerful but computationally intensive.
    
    Architecture:
        - Conv stem: Conv1d(80, 512, k=7, s=4) + LayerNorm
        - Positional encoding (learned)
        - 6x Transformer Encoder Blocks (d_model=512, nhead=8, dim_feedforward=2048)
        - Attention pooling
        - Two output heads (style_ttl and style_dp)
    
    Args:
        config: Training configuration.
        use_checkpoint: Use gradient checkpointing for memory efficiency.
    """
    
    def __init__(self, config: TrainingConfig, use_checkpoint: bool = False):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        d_model = config.hidden_dim
        
        # Convolutional stem for initial feature extraction
        self.conv_stem = nn.Sequential(
            nn.Conv1d(config.n_mels, d_model, kernel_size=7, stride=4, padding=3),
            nn.GELU()
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=d_model * 4,  # 2048 for d_model=512
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Output heads
        self.style_ttl_head = nn.Linear(d_model, config.style_ttl_size)
        self.style_dp_head = nn.Linear(d_model, config.style_dp_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mel: Mel-spectrogram tensor of shape [B, 80, T]
            mask: Optional boolean mask of shape [B, T] (True for valid positions)
            
        Returns:
            Tuple of (style_ttl, style_dp) tensors:
                - style_ttl: [B, 1, 50, 256]
                - style_dp: [B, 1, 8, 16]
        """
        batch_size = mel.size(0)
        
        # Conv stem
        x = self.conv_stem(mel)  # [B, d_model, T']
        x = x.transpose(1, 2)    # [B, T', d_model]
        x = self.layer_norm(x)
        
        # Adjust mask for downsampling (stride=4)
        if mask is not None:
            downsample_factor = 4
            mask_len = (mask.sum(dim=1) / downsample_factor).long()
            new_mask = torch.zeros(batch_size, x.size(1), dtype=torch.bool, device=x.device)
            for i, length in enumerate(mask_len):
                new_mask[i, :length] = True
            # Transformer expects inverse mask (True for positions to ignore)
            src_key_padding_mask = ~new_mask
        else:
            src_key_padding_mask = None
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        if self.use_checkpoint and self.training:
            x = checkpoint(self.transformer_encoder, x, src_key_padding_mask, use_reentrant=False)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Attention pooling (use original mask convention: True for valid)
        if mask is not None:
            pool_mask = new_mask
        else:
            pool_mask = None
        x = self.attention_pool(x, pool_mask)  # [B, d_model]
        
        # Output heads
        style_ttl = self.style_ttl_head(x)  # [B, 12800]
        style_dp = self.style_dp_head(x)    # [B, 128]
        
        # Reshape to target dimensions
        style_ttl = style_ttl.view(batch_size, *self.config.style_ttl_shape)  # [B, 1, 50, 256]
        style_dp = style_dp.view(batch_size, *self.config.style_dp_shape)     # [B, 1, 8, 16]
        
        return style_ttl, style_dp
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def create_encoder(config: TrainingConfig, use_checkpoint: bool = False) -> nn.Module:
    """
    Factory function to create an encoder based on configuration.
    
    Args:
        config: Training configuration.
        use_checkpoint: Use gradient checkpointing for memory efficiency.
        
    Returns:
        Encoder model (CNNEncoder or TransformerEncoder).
        
    Raises:
        ValueError: If encoder_type is not recognized.
    """
    if config.encoder_type.lower() == "cnn":
        encoder = CNNEncoder(config, use_checkpoint=use_checkpoint)
    elif config.encoder_type.lower() == "transformer":
        encoder = TransformerEncoder(config, use_checkpoint=use_checkpoint)
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}. "
                        f"Choose 'cnn' or 'transformer'.")
    
    num_params = encoder.get_num_params()
    print(f"Created {config.encoder_type.upper()} encoder with {num_params:,} parameters")
    
    return encoder


def save_encoder(model: nn.Module, path: str, config: Optional[TrainingConfig] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                epoch: Optional[int] = None,
                metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Save encoder checkpoint.
    
    Args:
        model: Encoder model to save.
        path: Path to save checkpoint.
        config: Optional training configuration.
        optimizer: Optional optimizer state.
        epoch: Optional current epoch number.
        metrics: Optional training metrics.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
    }
    
    if config is not None:
        from dataclasses import asdict
        checkpoint['config'] = asdict(config)
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_encoder(checkpoint_path: str, config: Optional[TrainingConfig] = None,
                device: str = 'cpu', use_checkpoint: bool = False) -> nn.Module:
    """
    Load encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Optional training configuration. If not provided, will use config from checkpoint.
        device: Device to load model on ('cpu' or 'cuda').
        use_checkpoint: Use gradient checkpointing for memory efficiency.
        
    Returns:
        Loaded encoder model.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If checkpoint is invalid or incompatible.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint if not provided
    if config is None:
        if 'config' not in checkpoint:
            raise ValueError("No config provided and checkpoint doesn't contain config")
        config = TrainingConfig(**checkpoint['config'])
    
    # Create model
    model = create_encoder(config, use_checkpoint=use_checkpoint)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise ValueError(f"Failed to load model state dict: {e}")
    
    model.to(device)
    model.eval()
    
    print(f"Loaded {checkpoint['model_type']} from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    return model


def load_checkpoint_full(checkpoint_path: str, config: Optional[TrainingConfig] = None,
                        device: str = 'cpu', use_checkpoint: bool = False
                        ) -> Tuple[nn.Module, Optional[Dict], Optional[int], Optional[Dict]]:
    """
    Load encoder checkpoint with full metadata (optimizer, epoch, metrics).
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Optional training configuration.
        device: Device to load model on.
        use_checkpoint: Use gradient checkpointing for memory efficiency.
        
    Returns:
        Tuple of (model, optimizer_state_dict, epoch, metrics).
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if config is None:
        if 'config' not in checkpoint:
            raise ValueError("No config provided and checkpoint doesn't contain config")
        config = TrainingConfig(**checkpoint['config'])
    
    # Create and load model
    model = create_encoder(config, use_checkpoint=use_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    epoch = checkpoint.get('epoch', None)
    metrics = checkpoint.get('metrics', None)
    
    return model, optimizer_state, epoch, metrics
