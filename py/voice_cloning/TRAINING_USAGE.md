# Voice Cloning Encoder Training Guide

This guide explains how to use the training script (`train.py`) for the voice cloning encoder.

## Overview

The training script implements a complete training pipeline for the voice cloning encoder with:
- Multi-loss training (MSE + cosine similarity)
- Mixed precision training (fp16)
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- TensorBoard logging

## Quick Start

### Basic Training

```bash
python train.py \
  --data-dir ./training_data \
  --checkpoint-dir ./checkpoints \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4
```

### Training with Config File

```bash
python train.py --config config.yaml
```

### Resume Training

```bash
python train.py --resume ./checkpoints/checkpoint_epoch_50.pt
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | None | Path to YAML config file |
| `--data-dir` | str | None | Path to training data directory |
| `--checkpoint-dir` | str | None | Directory for saving checkpoints |
| `--epochs` | int | None | Number of training epochs |
| `--batch-size` | int | None | Training batch size |
| `--lr` | float | None | Learning rate |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--device` | str | auto | Device: auto/cuda/mps/cpu |

**Note:** Command line arguments override config file settings.

## Training Features

### Multi-Loss Training

The training uses a weighted combination of three losses:

1. **MSE Loss (style_ttl)**: Mean squared error for style_ttl vectors
2. **MSE Loss (style_dp)**: Mean squared error for style_dp vectors  
3. **Cosine Similarity Loss**: Encourages better direction matching

Configurable via:
- `config.lambda_ttl` (default: 1.0)
- `config.lambda_dp` (default: 1.0)
- `config.lambda_cosine` (default: 0.1)

### Mixed Precision Training

Automatic mixed precision (fp16) is enabled by default on CUDA devices:
- Faster training
- Lower memory usage
- Minimal accuracy impact

Controlled by `config.use_amp` (default: True)

### Learning Rate Scheduling

Cosine annealing with warmup:
- Linear warmup for first N epochs
- Cosine decay for remaining epochs
- Helps stabilize training and improve convergence

Configurable via:
- `config.learning_rate` (default: 1e-4)
- `config.warmup_epochs` (default: 5)

### Early Stopping

Training stops if validation loss doesn't improve for N epochs:
- Prevents overfitting
- Saves time and resources
- Configurable via `config.early_stopping_patience` (default: 10)

### Model Checkpointing

Automatic checkpoint saving:
- **Best model**: Saved whenever validation loss improves
- **Periodic**: Saved every N epochs (configurable)
- **Final**: Saved at end of training

Checkpoints contain:
- Model weights
- Optimizer state
- Scheduler state
- Scaler state (for mixed precision)
- Training history
- Configuration

### TensorBoard Logging

Tracks metrics in real-time:
- Training/validation losses
- Loss components (TTL, DP, cosine)
- Learning rate
- Cosine similarity

To view:
```bash
tensorboard --logdir ./checkpoints/logs
```

Enable/disable: `config.use_tensorboard` (default: True)

## Data Requirements

Training data should be organized as:
```
training_data/
├── train/
│   ├── sample_0000.npz
│   ├── sample_0001.npz
│   └── ...
├── val/
│   ├── sample_0000.npz
│   └── ...
└── test/
    ├── sample_0000.npz
    └── ...
```

Each `.npz` file contains:
- `audio`: Audio waveform [T]
- `style_ttl`: Target style_ttl vector
- `style_dp`: Target style_dp vector
- `text`: Text string
- `lang`: Language string

## Output Structure

After training, the checkpoint directory contains:

```
checkpoints/
├── config.yaml              # Training configuration
├── best_model.pt            # Best model checkpoint
├── final_checkpoint.pt      # Final checkpoint
├── checkpoint_epoch_5.pt    # Periodic checkpoints
├── checkpoint_epoch_10.pt
├── ...
├── training_history.json    # Training metrics
└── logs/                    # TensorBoard logs
    └── events.out.tfevents.*
```

## Example Configurations

### Fast Development Training

```yaml
# fast_train.yaml
batch_size: 64
epochs: 20
learning_rate: 1e-3
warmup_epochs: 2
early_stopping_patience: 5
save_interval: 2
```

```bash
python train.py --config fast_train.yaml --data-dir ./training_data
```

### Production Training

```yaml
# prod_train.yaml
batch_size: 32
epochs: 200
learning_rate: 5e-5
warmup_epochs: 10
early_stopping_patience: 20
save_interval: 10
gradient_clip: 1.0
use_amp: true
```

```bash
python train.py --config prod_train.yaml --data-dir ./training_data
```

## Monitoring Training

### Progress Bar

Training shows real-time progress with tqdm:
```
Epoch 10/100 [Train]: 100%|████████| 312/312 [02:15<00:00, 2.31it/s, loss=0.0234, lr=0.000095, cosine=0.9523]
Epoch 10/100 [Val]:   100%|████████| 32/32 [00:12<00:00, 2.53it/s, loss=0.0198, cosine=0.9612]
```

### Console Logging

Detailed epoch summaries:
```
Epoch 10/100 - 2m 27s
  Train Loss: 0.0234 (TTL: 0.0180, DP: 0.0042, Cosine: 0.0012)
  Val Loss: 0.0198 (TTL: 0.0152, DP: 0.0036, Cosine: 0.0010)
  Train Cosine Sim: 0.9523, Val Cosine Sim: 0.9612
  Learning Rate: 0.000095
  ✓ New best validation loss: 0.0198
```

### TensorBoard

Real-time metrics visualization:
```bash
tensorboard --logdir ./checkpoints/logs
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

### Out of Memory

1. Reduce batch size: `--batch-size 16`
2. Use gradient checkpointing (in encoder_model.py)
3. Reduce number of workers: edit `config.num_workers`

### Training Too Slow

1. Increase batch size: `--batch-size 64`
2. Reduce data augmentation
3. Use faster device (CUDA > MPS > CPU)

### Poor Convergence

1. Adjust learning rate: `--lr 5e-5`
2. Increase warmup: edit `config.warmup_epochs`
3. Adjust loss weights: edit `config.lambda_*`
4. Check data quality and preprocessing

### Resume Training Fails

Ensure the checkpoint file matches the model architecture and configuration.

## Advanced Usage

### Custom Loss Weights

Edit `config.yaml`:
```yaml
lambda_ttl: 2.0      # Emphasize style_ttl
lambda_dp: 1.0       # Standard weight for style_dp
lambda_cosine: 0.05  # Reduce cosine influence
```

### Custom Augmentation

Edit `config.yaml`:
```yaml
augment_prob: 0.7                # Apply augmentation more often
augment_noise_snr_min: 15.0      # More aggressive noise
spec_augment_time_mask: 30       # Larger time masking
```

### Multi-GPU Training

Currently single-GPU. For multi-GPU, modify train.py to use:
```python
model = torch.nn.DataParallel(model)
```

## Performance Tips

1. **Use CUDA**: 10-50x faster than CPU
2. **Enable AMP**: 2-3x speedup on modern GPUs
3. **Optimize batch size**: Find max that fits in memory
4. **Pin memory**: Already enabled by default
5. **Increase workers**: Edit `config.num_workers` based on CPU cores

## Citation

If you use this training pipeline in your research, please cite the Supertonic project.
