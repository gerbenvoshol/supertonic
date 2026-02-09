"""
Training script for voice cloning encoder.

This module implements the complete training loop for the voice cloning encoder,
including:
- Multi-loss training (MSE + cosine similarity)
- Mixed precision training (fp16)
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- TensorBoard logging
- Progress tracking
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainingConfig, load_config_from_yaml, save_config_to_yaml
from dataset import get_dataloaders
from encoder_model import create_encoder, save_encoder
from utils import set_seed, get_device, AverageMeter, count_parameters, format_time

logger = logging.getLogger(__name__)


class MultiLoss(nn.Module):
    """
    Multi-loss function for voice cloning encoder training.
    
    Combines:
    1. MSE loss on style_ttl vectors
    2. MSE loss on style_dp vectors
    3. Cosine similarity loss for better direction matching
    
    Args:
        lambda_ttl: Weight for style_ttl MSE loss.
        lambda_dp: Weight for style_dp MSE loss.
        lambda_cosine: Weight for cosine similarity loss.
    """
    
    def __init__(self, lambda_ttl: float = 1.0, lambda_dp: float = 1.0, 
                 lambda_cosine: float = 0.1):
        super().__init__()
        self.lambda_ttl = lambda_ttl
        self.lambda_dp = lambda_dp
        self.lambda_cosine = lambda_cosine
        
    def forward(
        self,
        pred_ttl: torch.Tensor,
        target_ttl: torch.Tensor,
        pred_dp: torch.Tensor,
        target_dp: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-loss.
        
        Args:
            pred_ttl: Predicted style_ttl [B, ...]
            target_ttl: Target style_ttl [B, ...]
            pred_dp: Predicted style_dp [B, ...]
            target_dp: Target style_dp [B, ...]
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        # Flatten for loss computation
        pred_ttl_flat = pred_ttl.flatten(1)
        target_ttl_flat = target_ttl.flatten(1)
        pred_dp_flat = pred_dp.flatten(1)
        target_dp_flat = target_dp.flatten(1)
        
        # MSE losses
        loss_ttl = F.mse_loss(pred_ttl_flat, target_ttl_flat)
        loss_dp = F.mse_loss(pred_dp_flat, target_dp_flat)
        
        # Cosine similarity loss (1 - cosine_similarity)
        # Combine both vectors for cosine similarity
        pred_combined = torch.cat([pred_ttl_flat, pred_dp_flat], dim=1)
        target_combined = torch.cat([target_ttl_flat, target_dp_flat], dim=1)
        
        # Normalize
        pred_norm = F.normalize(pred_combined, p=2, dim=1)
        target_norm = F.normalize(target_combined, p=2, dim=1)
        
        # Compute cosine similarity (between -1 and 1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
        
        # Loss is 1 - cosine_similarity (minimize to maximize similarity)
        loss_cosine = 1.0 - cosine_sim
        
        # Weighted combination
        total_loss = (
            self.lambda_ttl * loss_ttl +
            self.lambda_dp * loss_dp +
            self.lambda_cosine * loss_cosine
        )
        
        # Return loss dict for logging
        loss_dict = {
            "loss_ttl": loss_ttl.item(),
            "loss_dp": loss_dp.item(),
            "loss_cosine": loss_cosine.item(),
            "cosine_sim": cosine_sim.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, loss_dict


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that increases linearly during warmup,
    then decreases following a cosine curve.
    
    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (0.5 = half cosine).
        last_epoch: The index of the last epoch (-1 for first epoch).
        
    Returns:
        Learning rate scheduler.
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Encoder model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (updated per step).
        scaler: Gradient scaler for mixed precision.
        device: Device to train on.
        epoch: Current epoch number.
        config: Training configuration.
        writer: TensorBoard writer.
        
    Returns:
        Dictionary of average training metrics.
    """
    model.train()
    
    # Meters for tracking
    loss_meter = AverageMeter()
    loss_ttl_meter = AverageMeter()
    loss_dp_meter = AverageMeter()
    loss_cosine_meter = AverageMeter()
    cosine_sim_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        mel = batch["mel"].to(device)
        style_ttl_target = batch["style_ttl"].to(device)
        style_dp_target = batch["style_dp"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        
        # Create mask from lengths
        batch_size, _, max_len = mel.shape
        mask = torch.arange(max_len, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.use_amp and scaler is not None:
            with autocast():
                # Forward pass
                style_ttl_pred, style_dp_pred = model(mel, mask)
                
                # Reshape targets to match predictions
                style_ttl_target = style_ttl_target.view(style_ttl_pred.shape)
                style_dp_target = style_dp_target.view(style_dp_pred.shape)
                
                # Compute loss
                loss, loss_dict = criterion(
                    style_ttl_pred, style_ttl_target,
                    style_dp_pred, style_dp_target
                )
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass without mixed precision
            style_ttl_pred, style_dp_pred = model(mel, mask)
            
            # Reshape targets
            style_ttl_target = style_ttl_target.view(style_ttl_pred.shape)
            style_dp_target = style_dp_target.view(style_dp_pred.shape)
            
            # Compute loss
            loss, loss_dict = criterion(
                style_ttl_pred, style_ttl_target,
                style_dp_pred, style_dp_target
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Optimizer step
            optimizer.step()
        
        # Update learning rate scheduler (per step)
        if scheduler is not None:
            scheduler.step()
        
        # Update meters
        batch_size = mel.size(0)
        loss_meter.update(loss_dict["total_loss"], batch_size)
        loss_ttl_meter.update(loss_dict["loss_ttl"], batch_size)
        loss_dp_meter.update(loss_dict["loss_dp"], batch_size)
        loss_cosine_meter.update(loss_dict["loss_cosine"], batch_size)
        cosine_sim_meter.update(loss_dict["cosine_sim"], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            "cosine": f"{cosine_sim_meter.avg:.4f}",
        })
        
        # TensorBoard logging
        if writer is not None and batch_idx % config.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", loss_dict["total_loss"], global_step)
            writer.add_scalar("train/loss_ttl", loss_dict["loss_ttl"], global_step)
            writer.add_scalar("train/loss_dp", loss_dict["loss_dp"], global_step)
            writer.add_scalar("train/loss_cosine", loss_dict["loss_cosine"], global_step)
            writer.add_scalar("train/cosine_sim", loss_dict["cosine_sim"], global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
    
    return {
        "loss": loss_meter.avg,
        "loss_ttl": loss_ttl_meter.avg,
        "loss_dp": loss_dp_meter.avg,
        "loss_cosine": loss_cosine_meter.avg,
        "cosine_sim": cosine_sim_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiLoss,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Encoder model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to validate on.
        epoch: Current epoch number.
        config: Training configuration.
        
    Returns:
        Dictionary of average validation metrics.
    """
    model.eval()
    
    # Meters for tracking
    loss_meter = AverageMeter()
    loss_ttl_meter = AverageMeter()
    loss_dp_meter = AverageMeter()
    loss_cosine_meter = AverageMeter()
    cosine_sim_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs} [Val]")
    
    for batch in pbar:
        # Move to device
        mel = batch["mel"].to(device)
        style_ttl_target = batch["style_ttl"].to(device)
        style_dp_target = batch["style_dp"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        
        # Create mask from lengths
        batch_size, _, max_len = mel.shape
        mask = torch.arange(max_len, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        
        # Forward pass
        style_ttl_pred, style_dp_pred = model(mel, mask)
        
        # Reshape targets
        style_ttl_target = style_ttl_target.view(style_ttl_pred.shape)
        style_dp_target = style_dp_target.view(style_dp_pred.shape)
        
        # Compute loss
        loss, loss_dict = criterion(
            style_ttl_pred, style_ttl_target,
            style_dp_pred, style_dp_target
        )
        
        # Update meters
        batch_size = mel.size(0)
        loss_meter.update(loss_dict["total_loss"], batch_size)
        loss_ttl_meter.update(loss_dict["loss_ttl"], batch_size)
        loss_dp_meter.update(loss_dict["loss_dp"], batch_size)
        loss_cosine_meter.update(loss_dict["loss_cosine"], batch_size)
        cosine_sim_meter.update(loss_dict["cosine_sim"], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "cosine": f"{cosine_sim_meter.avg:.4f}",
        })
    
    return {
        "loss": loss_meter.avg,
        "loss_ttl": loss_ttl_meter.avg,
        "loss_dp": loss_dp_meter.avg,
        "loss_cosine": loss_cosine_meter.avg,
        "cosine_sim": cosine_sim_meter.avg,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    epoch: int,
    best_val_loss: float,
    train_history: list,
    config: TrainingConfig,
    checkpoint_path: str,
    is_best: bool = False,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Encoder model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        train_history: Training history.
        config: Training configuration.
        checkpoint_path: Path to save checkpoint.
        is_best: Whether this is the best model so far.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "config": config.__dict__,
        "model_type": model.__class__.__name__,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # Create directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best model if applicable
    if is_best:
        best_path = Path(checkpoint_path).parent / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, list]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Encoder model to load weights into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        scaler: Optional scaler to load state into.
        device: Device to load tensors to.
        
    Returns:
        Tuple of (start_epoch, best_val_loss, train_history).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Load scaler
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    train_history = checkpoint.get("train_history", [])
    
    logger.info(f"Resumed from epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss, train_history


def train(
    config: TrainingConfig,
    resume_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Main training function.
    
    Args:
        config: Training configuration.
        resume_checkpoint: Optional path to checkpoint to resume from.
        device: Device to train on. If None, auto-detected.
    """
    # Setup device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    set_seed(config.seed)
    logger.info(f"Set random seed to {config.seed}")
    
    # Create output directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = checkpoint_dir / "config.yaml"
    save_config_to_yaml(config, str(config_path))
    logger.info(f"Saved config to {config_path}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        config=config,
        train_shuffle=True,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating encoder model...")
    model = create_encoder(config, use_checkpoint=False)
    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = MultiLoss(
        lambda_ttl=config.lambda_ttl,
        lambda_dp=config.lambda_dp,
        lambda_cosine=config.lambda_cosine,
    )
    logger.info(f"Loss weights - TTL: {config.lambda_ttl}, DP: {config.lambda_dp}, "
                f"Cosine: {config.lambda_cosine}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    
    # Create learning rate scheduler
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(f"Scheduler: Cosine with warmup ({config.warmup_epochs} epochs)")
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config.use_amp and device.type == "cuda" else None
    if config.use_amp:
        logger.info(f"Using mixed precision training (AMP): {device.type == 'cuda'}")
    
    # TensorBoard writer
    writer = None
    if config.use_tensorboard:
        log_dir = checkpoint_dir / "logs"
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logs: {log_dir}")
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_loss = float("inf")
    train_history = []
    epochs_without_improvement = 0
    
    if resume_checkpoint is not None:
        start_epoch, best_val_loss, train_history = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        epochs_without_improvement = 0
        for record in reversed(train_history):
            if record["val_loss"] > best_val_loss:
                epochs_without_improvement += 1
            else:
                break
    
    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training from epoch {start_epoch} to {config.epochs}")
    logger.info(f"{'='*60}\n")
    
    train_start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
            writer=writer,
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
        )
        
        # Compute epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(f"\nEpoch {epoch}/{config.epochs} - {format_time(epoch_time)}")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f} "
                   f"(TTL: {train_metrics['loss_ttl']:.4f}, "
                   f"DP: {train_metrics['loss_dp']:.4f}, "
                   f"Cosine: {train_metrics['loss_cosine']:.4f})")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f} "
                   f"(TTL: {val_metrics['loss_ttl']:.4f}, "
                   f"DP: {val_metrics['loss_dp']:.4f}, "
                   f"Cosine: {val_metrics['loss_cosine']:.4f})")
        logger.info(f"  Train Cosine Sim: {train_metrics['cosine_sim']:.4f}, "
                   f"Val Cosine Sim: {val_metrics['cosine_sim']:.4f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/loss_ttl", val_metrics["loss_ttl"], epoch)
            writer.add_scalar("val/loss_dp", val_metrics["loss_dp"], epoch)
            writer.add_scalar("val/loss_cosine", val_metrics["loss_cosine"], epoch)
            writer.add_scalar("val/cosine_sim", val_metrics["cosine_sim"], epoch)
            writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
        
        # Update training history
        train_history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_cosine_sim": train_metrics["cosine_sim"],
            "val_cosine_sim": val_metrics["cosine_sim"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
        })
        
        # Check for improvement
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            logger.info(f"  ✓ New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoint
        if epoch % config.save_interval == 0 or is_best:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_history=train_history,
                config=config,
                checkpoint_path=str(checkpoint_path),
                is_best=is_best,
            )
        
        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Training complete (epoch retains its value after loop)
    total_time = time.time() - train_start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Total time: {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}\n")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_val_loss=best_val_loss,
        train_history=train_history,
        config=config,
        checkpoint_path=str(final_checkpoint_path),
        is_best=False,
    )
    
    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    logger.info("Training finished successfully!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train voice cloning encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)",
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to training data directory",
    )
    
    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints",
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    
    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load config
    if args.config is not None:
        logger.info(f"Loading config from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        logger.info("Using default config")
        config = TrainingConfig()
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Validate config
    if not Path(config.data_dir).exists():
        logger.error(f"Data directory does not exist: {config.data_dir}")
        return
    
    # Print configuration
    logger.info("\n" + "="*60)
    logger.info("Training Configuration")
    logger.info("="*60)
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Checkpoint directory: {config.checkpoint_dir}")
    logger.info(f"Encoder type: {config.encoder_type}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Warmup epochs: {config.warmup_epochs}")
    logger.info(f"Early stopping patience: {config.early_stopping_patience}")
    logger.info(f"Gradient clip: {config.gradient_clip}")
    logger.info(f"Mixed precision: {config.use_amp}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.seed}")
    logger.info("="*60 + "\n")
    
    # Start training
    try:
        train(
            config=config,
            resume_checkpoint=args.resume,
            device=device,
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
