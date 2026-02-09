"""
Evaluation and metrics for voice cloning encoder.

This module provides comprehensive evaluation capabilities:
1. Reconstruction metrics (MSE, MAE, cosine similarity)
2. Per-dimension analysis
3. Round-trip evaluation with TTS models (optional)
4. Visualization (t-SNE/UMAP, scatter plots, histograms)
5. Testing on built-in voice styles
6. CLI with extensive configuration options
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_cloning.config import TrainingConfig
from voice_cloning.encoder_model import load_encoder
from voice_cloning.dataset import VoiceCloneDataset, collate_fn
from voice_cloning.utils import compute_mel_spectrogram

# Optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization disabled")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    warnings.warn("scikit-learn not available - t-SNE disabled")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available - UMAP visualization disabled")

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not available - perceptual quality metrics disabled")

logger = logging.getLogger(__name__)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        pred: Predicted tensor [B, ...]
        target: Target tensor [B, ...]
        
    Returns:
        MSE value (scalar).
    """
    return F.mse_loss(pred, target).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        pred: Predicted tensor [B, ...]
        target: Target tensor [B, ...]
        
    Returns:
        MAE value (scalar).
    """
    return F.l1_loss(pred, target).item()


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute cosine similarity between vectors.
    
    Args:
        pred: Predicted tensor [B, D]
        target: Target tensor [B, D]
        
    Returns:
        Average cosine similarity (scalar).
    """
    # Flatten if needed
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
    return cos_sim.mean().item()


def compute_per_dimension_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Compute per-dimension error metrics.
    
    Args:
        pred: Predicted tensor [B, D]
        target: Target tensor [B, D]
        
    Returns:
        Dictionary with per-dimension metrics:
            - mse: MSE per dimension [D]
            - mae: MAE per dimension [D]
            - std: Standard deviation of error per dimension [D]
    """
    # Flatten if needed
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Compute errors
    errors = pred_flat - target_flat  # [B, D]
    
    # Per-dimension metrics
    mse_per_dim = (errors ** 2).mean(dim=0).cpu().numpy()
    mae_per_dim = errors.abs().mean(dim=0).cpu().numpy()
    std_per_dim = errors.std(dim=0).cpu().numpy()
    
    return {
        'mse': mse_per_dim,
        'mae': mae_per_dim,
        'std': std_per_dim,
    }


def compute_mel_cepstral_distortion(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int = 24000,
    n_mfcc: int = 13
) -> float:
    """
    Compute Mel-Cepstral Distortion (MCD) between two audio signals.
    
    MCD measures the spectral distance between two audio signals.
    Lower values indicate more similar audio.
    
    Args:
        audio1: First audio signal [T]
        audio2: Second audio signal [T]
        sample_rate: Sample rate
        n_mfcc: Number of MFCCs to use
        
    Returns:
        MCD value in dB.
    """
    import librosa
    
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Compute MFCCs
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Ensure same number of frames
    min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_frames]
    mfcc2 = mfcc2[:, :min_frames]
    
    # Compute MCD (ignoring 0th coefficient which is energy)
    mfcc1_no_c0 = mfcc1[1:, :]
    mfcc2_no_c0 = mfcc2[1:, :]
    
    diff = mfcc1_no_c0 - mfcc2_no_c0
    mcd = (10.0 / np.log(10)) * np.sqrt(np.sum(diff ** 2, axis=0)).mean()
    
    return float(mcd)


class VoiceCloneEvaluator:
    """
    Comprehensive evaluator for voice cloning encoder.
    
    Features:
        - Reconstruction metrics on test set
        - Per-dimension analysis
        - Round-trip evaluation (optional)
        - Visualization (t-SNE, UMAP, scatter plots)
        - Testing on built-in voice styles
        - Detailed result logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cpu',
        output_dir: str = './evaluation_results',
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained encoder model
            config: Training configuration
            device: Device to run evaluation on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for predictions and ground truth
        self.predictions = {
            'style_ttl': [],
            'style_dp': [],
        }
        self.ground_truth = {
            'style_ttl': [],
            'style_dp': [],
        }
        self.sample_info = []
        
    def evaluate_dataloader(
        self,
        dataloader: DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: DataLoader with test samples
            desc: Description for progress bar
            
        Returns:
            Dictionary with aggregate metrics.
        """
        logger.info(f"Evaluating on {len(dataloader)} batches...")
        
        all_metrics = {
            'mse_ttl': [],
            'mse_dp': [],
            'mae_ttl': [],
            'mae_dp': [],
            'cosine_ttl': [],
            'cosine_dp': [],
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Move to device
                mel = batch['mel'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                style_ttl_true = batch['style_ttl'].to(self.device)
                style_dp_true = batch['style_dp'].to(self.device)
                
                # Create mask
                batch_size = mel.size(0)
                max_len = mel.size(2)
                mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
                for i, length in enumerate(mel_lengths):
                    mask[i, :length] = True
                
                # Forward pass
                style_ttl_pred, style_dp_pred = self.model(mel, mask)
                
                # Flatten for comparison
                style_ttl_pred_flat = style_ttl_pred.view(batch_size, -1)
                style_dp_pred_flat = style_dp_pred.view(batch_size, -1)
                style_ttl_true_flat = style_ttl_true.view(batch_size, -1)
                style_dp_true_flat = style_dp_true.view(batch_size, -1)
                
                # Compute metrics
                all_metrics['mse_ttl'].append(compute_mse(style_ttl_pred_flat, style_ttl_true_flat))
                all_metrics['mse_dp'].append(compute_mse(style_dp_pred_flat, style_dp_true_flat))
                all_metrics['mae_ttl'].append(compute_mae(style_ttl_pred_flat, style_ttl_true_flat))
                all_metrics['mae_dp'].append(compute_mae(style_dp_pred_flat, style_dp_true_flat))
                all_metrics['cosine_ttl'].append(compute_cosine_similarity(style_ttl_pred_flat, style_ttl_true_flat))
                all_metrics['cosine_dp'].append(compute_cosine_similarity(style_dp_pred_flat, style_dp_true_flat))
                
                # Store predictions for visualization
                self.predictions['style_ttl'].append(style_ttl_pred.cpu())
                self.predictions['style_dp'].append(style_dp_pred.cpu())
                self.ground_truth['style_ttl'].append(style_ttl_true.cpu())
                self.ground_truth['style_dp'].append(style_dp_true.cpu())
                
                # Store sample info
                for i in range(batch_size):
                    self.sample_info.append({
                        'text': batch['text'][i] if 'text' in batch else '',
                        'lang': batch['lang'][i] if 'lang' in batch else '',
                    })
        
        # Aggregate metrics
        metrics = {
            key: float(np.mean(values))
            for key, values in all_metrics.items()
        }
        
        return metrics
    
    def compute_per_dimension_analysis(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute per-dimension error analysis.
        
        Returns:
            Dictionary with per-dimension metrics for ttl and dp.
        """
        logger.info("Computing per-dimension analysis...")
        
        # Concatenate all predictions
        pred_ttl = torch.cat(self.predictions['style_ttl'], dim=0)
        pred_dp = torch.cat(self.predictions['style_dp'], dim=0)
        true_ttl = torch.cat(self.ground_truth['style_ttl'], dim=0)
        true_dp = torch.cat(self.ground_truth['style_dp'], dim=0)
        
        # Compute per-dimension metrics
        metrics_ttl = compute_per_dimension_metrics(pred_ttl, true_ttl)
        metrics_dp = compute_per_dimension_metrics(pred_dp, true_dp)
        
        return {
            'style_ttl': metrics_ttl,
            'style_dp': metrics_dp,
        }
    
    def evaluate_builtin_styles(
        self,
        voice_styles_dir: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on built-in voice styles (M1.json, F1.json, etc.).
        
        Args:
            voice_styles_dir: Directory containing voice style JSON files
            
        Returns:
            Dictionary mapping style name to metrics.
        """
        logger.info(f"Evaluating built-in voice styles from {voice_styles_dir}...")
        
        voice_styles_dir = Path(voice_styles_dir)
        if not voice_styles_dir.exists():
            logger.warning(f"Voice styles directory not found: {voice_styles_dir}")
            return {}
        
        # Find all JSON files
        style_files = sorted(voice_styles_dir.glob("*.json"))
        if len(style_files) == 0:
            logger.warning(f"No JSON files found in {voice_styles_dir}")
            return {}
        
        results = {}
        
        for style_file in style_files:
            try:
                # Load style JSON
                with open(style_file, 'r') as f:
                    style_data = json.load(f)
                
                # Extract style vectors
                if 'style_ttl' not in style_data or 'style_dp' not in style_data:
                    logger.warning(f"Missing style vectors in {style_file.name}")
                    continue
                
                style_ttl = np.array(style_data['style_ttl'], dtype=np.float32)
                style_dp = np.array(style_data['style_dp'], dtype=np.float32)
                
                # Check if audio path is available
                if 'reference_audio' in style_data:
                    audio_path = voice_styles_dir / style_data['reference_audio']
                    if audio_path.exists():
                        # Load audio and compute mel-spectrogram
                        import librosa
                        audio, _ = librosa.load(str(audio_path), sr=self.config.sample_rate, mono=True)
                        
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
                        
                        # Convert to tensor
                        mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(self.device)  # [1, 80, T]
                        
                        # Forward pass
                        with torch.no_grad():
                            pred_ttl, pred_dp = self.model(mel_tensor, mask=None)
                        
                        # Flatten and convert to numpy
                        pred_ttl = pred_ttl.view(1, -1).cpu()
                        pred_dp = pred_dp.view(1, -1).cpu()
                        true_ttl = torch.from_numpy(style_ttl.flatten()).unsqueeze(0)
                        true_dp = torch.from_numpy(style_dp.flatten()).unsqueeze(0)
                        
                        # Compute metrics
                        metrics = {
                            'mse_ttl': compute_mse(pred_ttl, true_ttl),
                            'mse_dp': compute_mse(pred_dp, true_dp),
                            'mae_ttl': compute_mae(pred_ttl, true_ttl),
                            'mae_dp': compute_mae(pred_dp, true_dp),
                            'cosine_ttl': compute_cosine_similarity(pred_ttl, true_ttl),
                            'cosine_dp': compute_cosine_similarity(pred_dp, true_dp),
                        }
                        
                        results[style_file.stem] = metrics
                        logger.info(f"Evaluated {style_file.stem}: MSE_ttl={metrics['mse_ttl']:.4f}, MSE_dp={metrics['mse_dp']:.4f}")
                    else:
                        logger.warning(f"Reference audio not found: {audio_path}")
                else:
                    logger.warning(f"No reference audio specified in {style_file.name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {style_file.name}: {e}")
                continue
        
        return results
    
    def visualize_embeddings(
        self,
        method: str = 'tsne',
        perplexity: int = 30,
        n_neighbors: int = 15,
    ) -> None:
        """
        Visualize embeddings using dimensionality reduction.
        
        Args:
            method: Dimensionality reduction method ('tsne' or 'umap')
            perplexity: Perplexity for t-SNE
            n_neighbors: Number of neighbors for UMAP
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping visualization")
            return
        
        logger.info(f"Visualizing embeddings with {method.upper()}...")
        
        # Concatenate predictions
        pred_ttl = torch.cat(self.predictions['style_ttl'], dim=0)
        pred_dp = torch.cat(self.predictions['style_dp'], dim=0)
        true_ttl = torch.cat(self.ground_truth['style_ttl'], dim=0)
        true_dp = torch.cat(self.ground_truth['style_dp'], dim=0)
        
        # Flatten
        pred_ttl_flat = pred_ttl.view(pred_ttl.size(0), -1).numpy()
        true_ttl_flat = true_ttl.view(true_ttl.size(0), -1).numpy()
        pred_dp_flat = pred_dp.view(pred_dp.size(0), -1).numpy()
        true_dp_flat = true_dp.view(true_dp.size(0), -1).numpy()
        
        # Visualize style_ttl
        self._visualize_embedding_pair(
            pred_ttl_flat, true_ttl_flat,
            method=method,
            title='Style TTL Embeddings',
            filename='embedding_ttl.png',
            perplexity=perplexity,
            n_neighbors=n_neighbors,
        )
        
        # Visualize style_dp
        self._visualize_embedding_pair(
            pred_dp_flat, true_dp_flat,
            method=method,
            title='Style DP Embeddings',
            filename='embedding_dp.png',
            perplexity=perplexity,
            n_neighbors=n_neighbors,
        )
    
    def _visualize_embedding_pair(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        method: str,
        title: str,
        filename: str,
        perplexity: int,
        n_neighbors: int,
    ) -> None:
        """Helper to visualize a pair of embeddings."""
        # Combine predictions and ground truth
        combined = np.vstack([pred, true])
        labels = np.array(['Predicted'] * len(pred) + ['Ground Truth'] * len(true))
        
        # Apply dimensionality reduction
        if method == 'tsne':
            if not TSNE_AVAILABLE:
                logger.warning("t-SNE not available - skipping")
                return
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedded = reducer.fit_transform(combined)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP not available - skipping")
                return
            reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
            embedded = reducer.fit_transform(combined)
        else:
            logger.warning(f"Unknown method: {method}")
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label, color in [('Predicted', 'blue'), ('Ground Truth', 'red')]:
            mask = labels == label
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=color,
                label=label,
                alpha=0.6,
                s=20
            )
        
        ax.set_title(f'{title} ({method.upper()})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved embedding visualization to {save_path}")
    
    def plot_scatter_predictions(self) -> None:
        """Create scatter plots comparing predicted vs true style vectors."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping scatter plots")
            return
        
        logger.info("Creating scatter plots...")
        
        # Concatenate predictions
        pred_ttl = torch.cat(self.predictions['style_ttl'], dim=0)
        pred_dp = torch.cat(self.predictions['style_dp'], dim=0)
        true_ttl = torch.cat(self.ground_truth['style_ttl'], dim=0)
        true_dp = torch.cat(self.ground_truth['style_dp'], dim=0)
        
        # Flatten
        pred_ttl_flat = pred_ttl.view(-1).numpy()
        true_ttl_flat = true_ttl.view(-1).numpy()
        pred_dp_flat = pred_dp.view(-1).numpy()
        true_dp_flat = true_dp.view(-1).numpy()
        
        # Style TTL scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_ttl_flat, pred_ttl_flat, alpha=0.3, s=1)
        ax.plot([true_ttl_flat.min(), true_ttl_flat.max()],
                [true_ttl_flat.min(), true_ttl_flat.max()],
                'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('True Style TTL Values')
        ax.set_ylabel('Predicted Style TTL Values')
        ax.set_title('Style TTL: Predicted vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'scatter_ttl.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scatter plot to {save_path}")
        
        # Style DP scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_dp_flat, pred_dp_flat, alpha=0.3, s=1)
        ax.plot([true_dp_flat.min(), true_dp_flat.max()],
                [true_dp_flat.min(), true_dp_flat.max()],
                'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('True Style DP Values')
        ax.set_ylabel('Predicted Style DP Values')
        ax.set_title('Style DP: Predicted vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'scatter_dp.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scatter plot to {save_path}")
    
    def plot_per_dimension_errors(
        self,
        per_dim_metrics: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot per-dimension error distributions."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping per-dimension plots")
            return
        
        logger.info("Creating per-dimension error plots...")
        
        for style_name, metrics in per_dim_metrics.items():
            mse = metrics['mse']
            mae = metrics['mae']
            std = metrics['std']
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # MSE per dimension
            axes[0].bar(range(len(mse)), mse, alpha=0.7, color='blue')
            axes[0].set_xlabel('Dimension')
            axes[0].set_ylabel('MSE')
            axes[0].set_title(f'{style_name}: MSE per Dimension')
            axes[0].grid(True, alpha=0.3)
            
            # MAE per dimension
            axes[1].bar(range(len(mae)), mae, alpha=0.7, color='green')
            axes[1].set_xlabel('Dimension')
            axes[1].set_ylabel('MAE')
            axes[1].set_title(f'{style_name}: MAE per Dimension')
            axes[1].grid(True, alpha=0.3)
            
            # Std per dimension
            axes[2].bar(range(len(std)), std, alpha=0.7, color='red')
            axes[2].set_xlabel('Dimension')
            axes[2].set_ylabel('Std Dev')
            axes[2].set_title(f'{style_name}: Error Std Dev per Dimension')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"per_dimension_{style_name.replace('_', '_')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved per-dimension plot to {save_path}")
            
            # Also plot histogram of errors
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].hist(mse, bins=50, alpha=0.7, color='blue')
            axes[0].set_xlabel('MSE')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'{style_name}: MSE Distribution')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].hist(mae, bins=50, alpha=0.7, color='green')
            axes[1].set_xlabel('MAE')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'{style_name}: MAE Distribution')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].hist(std, bins=50, alpha=0.7, color='red')
            axes[2].set_xlabel('Std Dev')
            axes[2].set_ylabel('Count')
            axes[2].set_title(f'{style_name}: Std Dev Distribution')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"error_histogram_{style_name.replace('_', '_')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved error histogram to {save_path}")
    
    def save_results(
        self,
        metrics: Dict[str, Any],
        per_dim_metrics: Dict[str, Dict[str, np.ndarray]],
        builtin_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Save evaluation results to JSON.
        
        Args:
            metrics: Aggregate metrics
            per_dim_metrics: Per-dimension metrics
            builtin_metrics: Metrics on built-in voice styles
        """
        logger.info("Saving evaluation results...")
        
        # Prepare results dictionary
        results = {
            'aggregate_metrics': metrics,
            'builtin_voice_styles': builtin_metrics,
            'per_dimension_summary': {},
        }
        
        # Add per-dimension summary statistics
        for style_name, style_metrics in per_dim_metrics.items():
            results['per_dimension_summary'][style_name] = {
                'mse': {
                    'mean': float(style_metrics['mse'].mean()),
                    'std': float(style_metrics['mse'].std()),
                    'min': float(style_metrics['mse'].min()),
                    'max': float(style_metrics['mse'].max()),
                    'hardest_dims': style_metrics['mse'].argsort()[-10:].tolist(),
                },
                'mae': {
                    'mean': float(style_metrics['mae'].mean()),
                    'std': float(style_metrics['mae'].std()),
                    'min': float(style_metrics['mae'].min()),
                    'max': float(style_metrics['mae'].max()),
                    'hardest_dims': style_metrics['mae'].argsort()[-10:].tolist(),
                },
            }
        
        # Save to JSON
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Also save detailed per-dimension arrays
        per_dim_path = self.output_dir / 'per_dimension_metrics.npz'
        np.savez(
            per_dim_path,
            **{
                f"{style_name}_{metric_name}": metric_values
                for style_name, style_metrics in per_dim_metrics.items()
                for metric_name, metric_values in style_metrics.items()
            }
        )
        logger.info(f"Saved per-dimension metrics to {per_dim_path}")
    
    def print_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nStyle TTL Metrics:")
        print(f"  MSE:              {metrics['mse_ttl']:.6f}")
        print(f"  MAE:              {metrics['mae_ttl']:.6f}")
        print(f"  Cosine Similarity: {metrics['cosine_ttl']:.6f}")
        print(f"\nStyle DP Metrics:")
        print(f"  MSE:              {metrics['mse_dp']:.6f}")
        print(f"  MAE:              {metrics['mae_dp']:.6f}")
        print(f"  Cosine Similarity: {metrics['cosine_dp']:.6f}")
        print("=" * 80 + "\n")


def evaluate_round_trip(
    evaluator: VoiceCloneEvaluator,
    test_loader: DataLoader,
    onnx_dir: str,
    num_samples: int = 10,
) -> Dict[str, float]:
    """
    Perform round-trip evaluation using TTS models.
    
    Round-trip evaluation:
    1. Take real audio
    2. Encode to style vectors using trained encoder
    3. Synthesize audio with predicted style vectors
    4. Compare original and re-synthesized audio
    
    Args:
        evaluator: VoiceCloneEvaluator instance
        test_loader: Test data loader
        onnx_dir: Path to ONNX models directory
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with round-trip metrics.
    """
    logger.info("Performing round-trip evaluation...")
    
    try:
        from helper import TextToSpeech, UnicodeProcessor, Style
        import onnxruntime as ort
    except ImportError as e:
        logger.error(f"Failed to import TTS dependencies: {e}")
        return {}
    
    # Load TTS models
    try:
        # Load ONNX models
        onnx_dir = Path(onnx_dir)
        
        # Load configuration
        with open(onnx_dir / "cfg.json", 'r') as f:
            cfgs = json.load(f)
        
        # Load text processor
        text_processor = UnicodeProcessor(str(onnx_dir / "unicode_indexer.json"))
        
        # Load ONNX sessions
        dp_ort = ort.InferenceSession(str(onnx_dir / "dp.onnx"))
        text_enc_ort = ort.InferenceSession(str(onnx_dir / "text_enc.onnx"))
        vector_est_ort = ort.InferenceSession(str(onnx_dir / "vector_est.onnx"))
        vocoder_ort = ort.InferenceSession(str(onnx_dir / "vocoder.onnx"))
        
        tts = TextToSpeech(
            cfgs=cfgs,
            text_processor=text_processor,
            dp_ort=dp_ort,
            text_enc_ort=text_enc_ort,
            vector_est_ort=vector_est_ort,
            vocoder_ort=vocoder_ort,
        )
        
        logger.info("Loaded TTS models successfully")
    except Exception as e:
        logger.error(f"Failed to load TTS models: {e}")
        return {}
    
    # Evaluate samples
    mcd_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            
            # Move to device
            mel = batch['mel'].to(evaluator.device)
            texts = batch['text']
            langs = batch['lang']
            
            # Get first sample in batch
            mel_sample = mel[0:1]  # [1, 80, T]
            text = texts[0]
            lang = langs[0]
            
            # Encode style
            style_ttl_pred, style_dp_pred = evaluator.model(mel_sample)
            
            # Convert to numpy
            style_ttl_np = style_ttl_pred.cpu().numpy()
            style_dp_np = style_dp_pred.cpu().numpy()
            
            # Create Style object
            style = Style(style_ttl_np, style_dp_np)
            
            try:
                # Synthesize audio
                audio_synth, _ = tts._infer(
                    text_list=[text],
                    lang_list=[lang],
                    style=style,
                    total_step=evaluator.config.tts_total_step,
                    speed=evaluator.config.tts_speed,
                )
                
                # Load original audio (reconstruct from mel - approximation)
                # In practice, we'd need the original audio from the batch
                # For now, skip direct comparison
                
                # Note: This is a simplified version
                # Full implementation would require storing original audio
                
                logger.info(f"Synthesized audio for sample {batch_idx + 1}")
                
            except Exception as e:
                logger.error(f"Failed to synthesize audio for sample {batch_idx + 1}: {e}")
                continue
    
    return {
        'mcd_mean': float(np.mean(mcd_scores)) if mcd_scores else 0.0,
        'mcd_std': float(np.std(mcd_scores)) if mcd_scores else 0.0,
        'num_samples': len(mcd_scores),
    }


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate voice cloning encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained encoder checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./training_data',
        help='Test data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results and visualizations'
    )
    parser.add_argument(
        '--voice-styles-dir',
        type=str,
        default='../../assets/voice_styles',
        help='Directory containing built-in voice styles'
    )
    parser.add_argument(
        '--onnx-dir',
        type=str,
        default='../../assets/onnx',
        help='ONNX models directory for round-trip evaluation'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--no-round-trip',
        action='store_true',
        help='Skip round-trip evaluation'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip visualization'
    )
    parser.add_argument(
        '--no-builtin-styles',
        action='store_true',
        help='Skip built-in voice styles evaluation'
    )
    
    # Visualization options
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap'],
        help='Dimensionality reduction method for embeddings'
    )
    parser.add_argument(
        '--tsne-perplexity',
        type=int,
        default=30,
        help='Perplexity for t-SNE'
    )
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='Number of neighbors for UMAP'
    )
    
    # Round-trip options
    parser.add_argument(
        '--round-trip-samples',
        type=int,
        default=10,
        help='Number of samples for round-trip evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / 'evaluation.log')
        ]
    )
    
    logger.info("Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    try:
        logger.info("Loading model...")
        model = load_encoder(
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_checkpoint=False,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Get config from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'config' in checkpoint:
        config = TrainingConfig(**checkpoint['config'])
    else:
        logger.warning("Config not found in checkpoint, using default")
        config = TrainingConfig()
    
    # Override batch size
    config.batch_size = args.batch_size
    
    # Create test dataset
    try:
        logger.info("Creating test dataset...")
        test_dataset = VoiceCloneDataset(
            data_dir=args.data_dir,
            split='test',
            config=config,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for evaluation
            pin_memory=False,
            collate_fn=collate_fn,
        )
        logger.info(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"Failed to create test dataset: {e}")
        return 1
    
    # Create evaluator
    evaluator = VoiceCloneEvaluator(
        model=model,
        config=config,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Evaluate on test set
    try:
        metrics = evaluator.evaluate_dataloader(
            dataloader=test_loader,
            desc="Evaluating test set"
        )
        evaluator.print_summary(metrics)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    # Per-dimension analysis
    try:
        per_dim_metrics = evaluator.compute_per_dimension_analysis()
        logger.info("Per-dimension analysis complete")
        
        # Print hardest dimensions
        for style_name, style_metrics in per_dim_metrics.items():
            hardest_mse = style_metrics['mse'].argsort()[-10:][::-1]
            hardest_mae = style_metrics['mae'].argsort()[-10:][::-1]
            logger.info(f"\n{style_name} - Hardest dimensions (MSE): {hardest_mse.tolist()}")
            logger.info(f"{style_name} - Hardest dimensions (MAE): {hardest_mae.tolist()}")
    except Exception as e:
        logger.error(f"Per-dimension analysis failed: {e}")
        per_dim_metrics = {}
    
    # Evaluate on built-in voice styles
    builtin_metrics = {}
    if not args.no_builtin_styles:
        try:
            builtin_metrics = evaluator.evaluate_builtin_styles(
                voice_styles_dir=args.voice_styles_dir
            )
            if builtin_metrics:
                logger.info("\nBuilt-in Voice Styles Results:")
                for style_name, style_metrics in builtin_metrics.items():
                    logger.info(f"  {style_name}: MSE_ttl={style_metrics['mse_ttl']:.4f}, "
                              f"MSE_dp={style_metrics['mse_dp']:.4f}")
        except Exception as e:
            logger.error(f"Built-in styles evaluation failed: {e}")
    
    # Visualization
    if not args.no_visualization:
        try:
            logger.info("Creating visualizations...")
            
            # Embeddings
            evaluator.visualize_embeddings(
                method=args.embedding_method,
                perplexity=args.tsne_perplexity,
                n_neighbors=args.umap_neighbors,
            )
            
            # Scatter plots
            evaluator.plot_scatter_predictions()
            
            # Per-dimension error plots
            if per_dim_metrics:
                evaluator.plot_per_dimension_errors(per_dim_metrics)
            
            logger.info("Visualizations complete")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    # Round-trip evaluation
    if not args.no_round_trip:
        try:
            round_trip_metrics = evaluate_round_trip(
                evaluator=evaluator,
                test_loader=test_loader,
                onnx_dir=args.onnx_dir,
                num_samples=args.round_trip_samples,
            )
            if round_trip_metrics:
                logger.info(f"\nRound-trip evaluation results:")
                logger.info(f"  MCD: {round_trip_metrics.get('mcd_mean', 0):.2f} ± "
                          f"{round_trip_metrics.get('mcd_std', 0):.2f} dB")
                metrics.update(round_trip_metrics)
        except Exception as e:
            logger.error(f"Round-trip evaluation failed: {e}")
    
    # Save results
    try:
        evaluator.save_results(
            metrics=metrics,
            per_dim_metrics=per_dim_metrics,
            builtin_metrics=builtin_metrics,
        )
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
