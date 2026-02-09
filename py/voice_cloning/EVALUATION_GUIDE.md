# Voice Cloning Encoder Evaluation Guide

## Overview

The `evaluate.py` script provides comprehensive evaluation capabilities for the voice cloning encoder model. It computes reconstruction metrics, performs per-dimension analysis, generates visualizations, and optionally performs round-trip evaluation using TTS models.

## Features

### 1. Reconstruction Metrics
- **MSE (Mean Squared Error)**: Measures average squared error for style_ttl and style_dp
- **MAE (Mean Absolute Error)**: Measures average absolute error
- **Cosine Similarity**: Measures angular similarity between predicted and true vectors
- Separate metrics for both style_ttl and style_dp vectors

### 2. Per-Dimension Analysis
- Computes MSE, MAE, and standard deviation for each dimension
- Identifies the hardest-to-predict dimensions
- Generates visualizations showing error distributions
- Saves detailed per-dimension metrics to .npz files

### 3. Round-Trip Evaluation (Optional)
- Encodes audio to style vectors using the trained encoder
- Synthesizes new audio using predicted style vectors
- Computes Mel-Cepstral Distortion (MCD) between original and synthesized audio
- Framework for PESQ and other perceptual quality metrics

### 4. Visualization
- **t-SNE/UMAP Embeddings**: 2D projection of predicted vs true style vectors
- **Scatter Plots**: Direct comparison of predicted vs true values
- **Error Histograms**: Distribution of per-dimension errors
- Separate visualizations for style_ttl and style_dp

### 5. Built-in Voice Styles Testing
- Tests encoder on pre-defined voice styles (M1.json, F1.json, etc.)
- Loads reference audio and compares with ground truth
- Generates comprehensive metrics for each voice style

### 6. Results Persistence
- Saves aggregate metrics to JSON
- Saves per-dimension arrays to .npz
- Comprehensive logging to file and console
- Per-sample tracking for detailed analysis

## Usage

### Basic Usage

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data-dir ./training_data \
  --output-dir ./evaluation_results
```

### Full Evaluation with All Features

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data-dir ./training_data \
  --output-dir ./evaluation_results \
  --voice-styles-dir ../../assets/voice_styles \
  --onnx-dir ../../assets/onnx \
  --batch-size 32 \
  --device cuda \
  --embedding-method umap
```

### Skip Optional Components

```bash
# Skip round-trip evaluation
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data-dir ./training_data \
  --output-dir ./evaluation_results \
  --no-round-trip

# Skip visualization
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data-dir ./training_data \
  --output-dir ./evaluation_results \
  --no-visualization

# Skip built-in voice styles
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data-dir ./training_data \
  --output-dir ./evaluation_results \
  --no-builtin-styles
```

## Command-Line Arguments

### Required Arguments
- `--checkpoint`: Path to trained encoder checkpoint (.pt file)

### Data Directories
- `--data-dir`: Directory containing test data (default: `./training_data`)
- `--output-dir`: Directory for results and visualizations (default: `./evaluation_results`)
- `--voice-styles-dir`: Directory with built-in voice styles (default: `../../assets/voice_styles`)
- `--onnx-dir`: Directory with ONNX models for round-trip evaluation (default: `../../assets/onnx`)

### Evaluation Options
- `--batch-size`: Batch size for evaluation (default: 32)
- `--device`: Device to run on - 'cpu' or 'cuda' (default: auto-detect)
- `--no-round-trip`: Skip round-trip evaluation (flag)
- `--no-visualization`: Skip visualization generation (flag)
- `--no-builtin-styles`: Skip built-in voice styles evaluation (flag)

### Visualization Options
- `--embedding-method`: Method for dimensionality reduction - 'tsne' or 'umap' (default: 'tsne')
- `--tsne-perplexity`: Perplexity parameter for t-SNE (default: 30)
- `--umap-neighbors`: Number of neighbors for UMAP (default: 15)

### Round-Trip Options
- `--round-trip-samples`: Number of samples for round-trip evaluation (default: 10)

## Output Files

The evaluation script generates the following outputs in the `--output-dir`:

### JSON Results
- `evaluation_results.json`: Aggregate metrics and summary statistics
  - Reconstruction metrics (MSE, MAE, cosine similarity)
  - Built-in voice styles results
  - Per-dimension summary statistics

### NumPy Arrays
- `per_dimension_metrics.npz`: Detailed per-dimension metrics
  - style_ttl_mse, style_ttl_mae, style_ttl_std
  - style_dp_mse, style_dp_mae, style_dp_std

### Visualizations
- `embedding_ttl.png`: t-SNE/UMAP visualization of style_ttl
- `embedding_dp.png`: t-SNE/UMAP visualization of style_dp
- `scatter_ttl.png`: Scatter plot of predicted vs true style_ttl
- `scatter_dp.png`: Scatter plot of predicted vs true style_dp
- `per_dimension_style_ttl.png`: Bar plots of per-dimension errors for style_ttl
- `per_dimension_style_dp.png`: Bar plots of per-dimension errors for style_dp
- `error_histogram_style_ttl.png`: Histograms of error distributions for style_ttl
- `error_histogram_style_dp.png`: Histograms of error distributions for style_dp

### Logs
- `evaluation.log`: Detailed evaluation log with timestamps

## Dependencies

### Required Dependencies
```
torch
numpy
tqdm
librosa
```

### Optional Dependencies
```
matplotlib        # For visualization
scikit-learn     # For t-SNE
umap-learn       # For UMAP
pesq             # For perceptual quality metrics
onnxruntime      # For round-trip evaluation
```

The script gracefully handles missing optional dependencies and disables the corresponding features.

## Code Structure

### Main Components

1. **Metric Functions**
   - `compute_mse()`: Mean Squared Error
   - `compute_mae()`: Mean Absolute Error
   - `compute_cosine_similarity()`: Cosine similarity
   - `compute_per_dimension_metrics()`: Per-dimension analysis
   - `compute_mel_cepstral_distortion()`: MCD for audio comparison

2. **VoiceCloneEvaluator Class**
   - `evaluate_dataloader()`: Evaluate on test set
   - `compute_per_dimension_analysis()`: Per-dimension error analysis
   - `evaluate_builtin_styles()`: Test on built-in voice styles
   - `visualize_embeddings()`: Create t-SNE/UMAP visualizations
   - `plot_scatter_predictions()`: Create scatter plots
   - `plot_per_dimension_errors()`: Create error distribution plots
   - `save_results()`: Save all results to disk
   - `print_summary()`: Print evaluation summary

3. **Helper Functions**
   - `evaluate_round_trip()`: Round-trip evaluation with TTS
   - `main()`: CLI entry point with argparse

## Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

Style TTL Metrics:
  MSE:              0.003421
  MAE:              0.042156
  Cosine Similarity: 0.987654

Style DP Metrics:
  MSE:              0.001234
  MAE:              0.028945
  Cosine Similarity: 0.992341
================================================================================
```

## Performance Tips

1. **Use CUDA**: Enable GPU acceleration with `--device cuda` for faster evaluation
2. **Adjust Batch Size**: Use larger batch sizes (64, 128) for faster throughput
3. **Skip Expensive Operations**: Use `--no-round-trip` and `--no-visualization` for quick metrics
4. **Parallel Workers**: The script uses efficient PyTorch DataLoader for parallel processing

## Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**
   - The script will warn and continue without that feature
   - Install missing packages: `pip install matplotlib scikit-learn umap-learn`

2. **CUDA out of memory**
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

3. **No voice styles found**
   - Verify `--voice-styles-dir` path is correct
   - Check that JSON files contain 'style_ttl' and 'style_dp' fields

4. **Round-trip evaluation fails**
   - Ensure ONNX models are available in `--onnx-dir`
   - Check that all required ONNX files exist (dp.onnx, text_enc.onnx, etc.)

## Integration with Training

The evaluation script is designed to work seamlessly with the training pipeline:

1. Train your model with `train.py`
2. Checkpoints are saved to `--checkpoint-dir`
3. Run evaluation on the best checkpoint:
   ```bash
   python evaluate.py --checkpoint checkpoints/best_model.pt
   ```

## Citation and License

This evaluation script is part of the Supertonic voice cloning project. See the main README for license information.
