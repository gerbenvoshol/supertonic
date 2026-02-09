"""
ONNX export script for trained voice cloning encoder.

This module exports PyTorch encoder models to ONNX format with:
- Dynamic axes for variable-length audio
- Validation against PyTorch inference
- Model optimization
- Comprehensive metadata and statistics
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import checker, helper, shape_inference

from config import TrainingConfig
from encoder_model import load_encoder
from utils import set_seed


def export_encoder_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    config: TrainingConfig,
    opset_version: int = 14,
    device: str = "cpu",
    validate: bool = True,
) -> Dict:
    """
    Export PyTorch encoder to ONNX format.
    
    Args:
        model: PyTorch encoder model.
        output_path: Path to save ONNX model.
        config: Training configuration.
        opset_version: ONNX opset version.
        device: Device for model ('cpu', 'cuda', or 'mps').
        validate: Whether to validate exported model.
        
    Returns:
        Dictionary with export statistics and validation results.
    """
    print(f"\n{'='*60}")
    print("Starting ONNX Export")
    print(f"{'='*60}")
    
    # Set model to eval mode
    model.eval()
    model.to(device)
    
    # Create dummy input with dynamic time dimension
    batch_size = 1
    n_mels = config.n_mels
    time_steps = 200  # Dummy length for export
    
    dummy_input = torch.randn(batch_size, n_mels, time_steps, device=device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Device: {device}")
    print(f"Opset version: {opset_version}")
    
    # Get PyTorch output for validation
    with torch.no_grad():
        pytorch_style_ttl, pytorch_style_dp = model(dummy_input)
    
    print(f"Output shapes:")
    print(f"  style_ttl: {pytorch_style_ttl.shape}")
    print(f"  style_dp: {pytorch_style_dp.shape}")
    
    # Define dynamic axes (batch and time dimensions)
    dynamic_axes = {
        'mel_spectrogram': {0: 'batch', 2: 'time'},
        'style_ttl': {0: 'batch'},
        'style_dp': {0: 'batch'},
    }
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    export_start = time.time()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['style_ttl', 'style_dp'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    
    export_time = time.time() - export_start
    print(f"Export completed in {export_time:.2f}s")
    
    # Get model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Verify ONNX model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Run shape inference
    onnx_model = shape_inference.infer_shapes(onnx_model)
    
    # Get model statistics
    num_nodes = len(onnx_model.graph.node)
    num_initializers = len(onnx_model.graph.initializer)
    
    # Count parameters
    total_params = 0
    for initializer in onnx_model.graph.initializer:
        dims = initializer.dims
        params = np.prod(dims) if dims else 0
        total_params += params
    
    print(f"\nModel statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Initializers: {num_initializers}")
    print(f"  Parameters: {total_params:,}")
    
    # Validation results
    validation_results = {}
    
    if validate:
        print(f"\n{'='*60}")
        print("Validating ONNX Model")
        print(f"{'='*60}")
        validation_results = validate_onnx_model(
            output_path,
            dummy_input.cpu().numpy(),
            pytorch_style_ttl.cpu().numpy(),
            pytorch_style_dp.cpu().numpy(),
            config,
        )
    
    # Prepare export metadata
    export_metadata = {
        'export_time': export_time,
        'model_size_mb': model_size_mb,
        'num_nodes': num_nodes,
        'num_initializers': num_initializers,
        'total_parameters': int(total_params),
        'opset_version': opset_version,
        'validation': validation_results,
    }
    
    return export_metadata


def validate_onnx_model(
    onnx_path: str,
    input_array: np.ndarray,
    pytorch_ttl: np.ndarray,
    pytorch_dp: np.ndarray,
    config: TrainingConfig,
    num_runs: int = 10,
) -> Dict:
    """
    Validate ONNX model against PyTorch outputs.
    
    Args:
        onnx_path: Path to ONNX model.
        input_array: Input mel-spectrogram array.
        pytorch_ttl: PyTorch style_ttl output.
        pytorch_dp: PyTorch style_dp output.
        config: Training configuration.
        num_runs: Number of inference runs for timing.
        
    Returns:
        Dictionary with validation results.
    """
    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    print(f"\nCreating ONNX Runtime session...")
    print(f"Providers: {providers}")
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"Input: {input_name}")
    print(f"Outputs: {output_names}")
    
    # Run inference
    print(f"\nRunning ONNX inference...")
    ort_inputs = {input_name: input_array}
    ort_outputs = session.run(output_names, ort_inputs)
    
    onnx_ttl = ort_outputs[0]
    onnx_dp = ort_outputs[1]
    
    print(f"ONNX output shapes:")
    print(f"  style_ttl: {onnx_ttl.shape}")
    print(f"  style_dp: {onnx_dp.shape}")
    
    # Compare outputs
    print(f"\nComparing outputs with PyTorch...")
    
    # Check shapes
    shapes_match = (
        onnx_ttl.shape == pytorch_ttl.shape and
        onnx_dp.shape == pytorch_dp.shape
    )
    print(f"  Shapes match: {shapes_match}")
    
    # Numerical comparison
    rtol, atol = 1e-3, 1e-3
    
    ttl_close = np.allclose(onnx_ttl, pytorch_ttl, rtol=rtol, atol=atol)
    dp_close = np.allclose(onnx_dp, pytorch_dp, rtol=rtol, atol=atol)
    
    # Compute errors
    ttl_max_diff = np.max(np.abs(onnx_ttl - pytorch_ttl))
    ttl_mean_diff = np.mean(np.abs(onnx_ttl - pytorch_ttl))
    
    dp_max_diff = np.max(np.abs(onnx_dp - pytorch_dp))
    dp_mean_diff = np.mean(np.abs(onnx_dp - pytorch_dp))
    
    print(f"\n  style_ttl:")
    print(f"    Numerically close: {ttl_close}")
    print(f"    Max abs diff: {ttl_max_diff:.6f}")
    print(f"    Mean abs diff: {ttl_mean_diff:.6f}")
    
    print(f"\n  style_dp:")
    print(f"    Numerically close: {dp_close}")
    print(f"    Max abs diff: {dp_max_diff:.6f}")
    print(f"    Mean abs diff: {dp_mean_diff:.6f}")
    
    # Validate with different input sizes
    print(f"\nValidating with variable input lengths...")
    
    test_lengths = [100, 200, 300, 500]
    variable_length_results = []
    
    for length in test_lengths:
        test_input = np.random.randn(1, config.n_mels, length).astype(np.float32)
        try:
            outputs = session.run(output_names, {input_name: test_input})
            success = (
                outputs[0].shape == (1, *config.style_ttl_shape) and
                outputs[1].shape == (1, *config.style_dp_shape)
            )
            variable_length_results.append({
                'length': length,
                'success': success,
                'output_shapes': [out.shape for out in outputs]
            })
            status = "✓" if success else "✗"
            print(f"  {status} Length {length}: shapes {[out.shape for out in outputs]}")
        except Exception as e:
            variable_length_results.append({
                'length': length,
                'success': False,
                'error': str(e)
            })
            print(f"  ✗ Length {length}: {e}")
    
    # Measure inference time
    print(f"\nMeasuring inference time ({num_runs} runs)...")
    
    inference_times = []
    for _ in range(num_runs):
        start = time.time()
        session.run(output_names, ort_inputs)
        inference_times.append(time.time() - start)
    
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"  Average: {avg_time*1000:.2f} ms")
    print(f"  Std dev: {std_time*1000:.2f} ms")
    print(f"  Min: {min_time*1000:.2f} ms")
    print(f"  Max: {max_time*1000:.2f} ms")
    
    # Check memory usage (approximate from model size)
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    # Overall validation status
    validation_passed = all([
        shapes_match,
        ttl_close,
        dp_close,
        all(r['success'] for r in variable_length_results),
    ])
    
    if validation_passed:
        print(f"\n{'='*60}")
        print("✓ VALIDATION PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("✗ VALIDATION FAILED")
        print(f"{'='*60}")
    
    return {
        'passed': validation_passed,
        'shapes_match': shapes_match,
        'style_ttl': {
            'numerically_close': bool(ttl_close),
            'max_abs_diff': float(ttl_max_diff),
            'mean_abs_diff': float(ttl_mean_diff),
        },
        'style_dp': {
            'numerically_close': bool(dp_close),
            'max_abs_diff': float(dp_max_diff),
            'mean_abs_diff': float(dp_mean_diff),
        },
        'variable_length_tests': variable_length_results,
        'inference_time': {
            'avg_ms': float(avg_time * 1000),
            'std_ms': float(std_time * 1000),
            'min_ms': float(min_time * 1000),
            'max_ms': float(max_time * 1000),
        },
        'memory_mb': float(memory_mb),
        'rtol': rtol,
        'atol': atol,
    }


def optimize_onnx_model(
    input_path: str,
    output_path: str,
    optimization_level: str = "all",
) -> Dict:
    """
    Optimize ONNX model using ONNX Runtime optimizations.
    
    Args:
        input_path: Path to input ONNX model.
        output_path: Path to save optimized model.
        optimization_level: Optimization level ('basic', 'extended', or 'all').
        
    Returns:
        Dictionary with optimization results.
    """
    print(f"\n{'='*60}")
    print("Optimizing ONNX Model")
    print(f"{'='*60}")
    
    # Map optimization level
    opt_level_map = {
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    
    opt_level = opt_level_map.get(optimization_level.lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    
    print(f"Optimization level: {optimization_level}")
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level
    sess_options.optimized_model_filepath = output_path
    
    # Create session (this will save optimized model)
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    print(f"Running optimization...")
    optimize_start = time.time()
    
    session = ort.InferenceSession(input_path, sess_options, providers=providers)
    
    optimize_time = time.time() - optimize_start
    
    # Get model sizes
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    optimized_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    size_reduction = ((original_size_mb - optimized_size_mb) / original_size_mb) * 100
    
    print(f"\nOptimization completed in {optimize_time:.2f}s")
    print(f"Original size: {original_size_mb:.2f} MB")
    print(f"Optimized size: {optimized_size_mb:.2f} MB")
    print(f"Size reduction: {size_reduction:.2f}%")
    
    # Load and verify optimized model
    print(f"\nVerifying optimized model...")
    optimized_model = onnx.load(output_path)
    checker.check_model(optimized_model)
    print("✓ Optimized model is valid")
    
    num_nodes_optimized = len(optimized_model.graph.node)
    print(f"Nodes in optimized model: {num_nodes_optimized}")
    
    return {
        'optimization_time': optimize_time,
        'original_size_mb': original_size_mb,
        'optimized_size_mb': optimized_size_mb,
        'size_reduction_percent': size_reduction,
        'num_nodes': num_nodes_optimized,
    }


def save_export_metadata(
    metadata: Dict,
    output_path: str,
    checkpoint_path: str,
    config: TrainingConfig,
    opset_version: int,
) -> None:
    """
    Save export metadata to JSON file.
    
    Args:
        metadata: Export and validation metadata.
        output_path: ONNX model output path.
        checkpoint_path: Source checkpoint path.
        config: Training configuration.
        opset_version: ONNX opset version.
    """
    from dataclasses import asdict
    
    full_metadata = {
        'export_info': {
            'export_datetime': datetime.now().isoformat(),
            'source_checkpoint': str(checkpoint_path),
            'output_path': str(output_path),
            'onnx_version': onnx.__version__,
            'opset_version': opset_version,
        },
        'model_architecture': {
            'encoder_type': config.encoder_type,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'dropout': config.dropout,
        },
        'input_output_spec': {
            'input': {
                'name': 'mel_spectrogram',
                'shape': ['B', config.n_mels, 'T'],
                'dtype': 'float32',
                'description': 'Mel-spectrogram with dynamic time dimension',
            },
            'outputs': {
                'style_ttl': {
                    'name': 'style_ttl',
                    'shape': ['B', *config.style_ttl_shape],
                    'dtype': 'float32',
                },
                'style_dp': {
                    'name': 'style_dp',
                    'shape': ['B', *config.style_dp_shape],
                    'dtype': 'float32',
                },
            },
        },
        'audio_config': {
            'sample_rate': config.sample_rate,
            'n_fft': config.n_fft,
            'hop_length': config.hop_length,
            'win_length': config.win_length,
            'n_mels': config.n_mels,
            'fmin': config.fmin,
            'fmax': config.fmax,
        },
        'export_statistics': metadata,
    }
    
    # Save metadata
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    print(f"\nSaved metadata to {metadata_path}")


def main():
    """Main function for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export trained voice cloning encoder to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained encoder checkpoint (.pt file)',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output ONNX file path',
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (minimum 14)',
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Create optimized version of the model',
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate exported model against PyTorch',
    )
    
    parser.add_argument(
        '--no-validate',
        dest='validate',
        action='store_false',
        help='Skip model validation',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for model export',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    
    args = parser.parse_args()
    
    # Validate opset version
    if args.opset < 14:
        print(f"Warning: Opset version {args.opset} < 14 may have compatibility issues.")
        print("Recommended: opset >= 14")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print("ONNX Export Configuration")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Opset: {args.opset}")
    print(f"Device: {args.device}")
    print(f"Optimize: {args.optimize}")
    print(f"Validate: {args.validate}")
    
    # Load model
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    
    model = load_encoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_checkpoint=False,  # Disable gradient checkpointing for export
    )
    
    # Get config from model (already loaded by load_encoder)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = TrainingConfig(**checkpoint['config'])
    
    # Export to ONNX
    export_metadata = export_encoder_to_onnx(
        model=model,
        output_path=args.output,
        config=config,
        opset_version=args.opset,
        device=args.device,
        validate=args.validate,
    )
    
    # Optimize if requested
    if args.optimize:
        optimized_path = Path(args.output).with_stem(
            Path(args.output).stem + '_optimized'
        )
        
        optimize_metadata = optimize_onnx_model(
            input_path=args.output,
            output_path=str(optimized_path),
            optimization_level='all',
        )
        
        export_metadata['optimization'] = optimize_metadata
    
    # Save metadata
    save_export_metadata(
        metadata=export_metadata,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        config=config,
        opset_version=args.opset,
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"✓ Model exported to: {args.output}")
    print(f"  Size: {export_metadata['model_size_mb']:.2f} MB")
    print(f"  Parameters: {export_metadata['total_parameters']:,}")
    
    if args.validate and export_metadata['validation'].get('passed'):
        print(f"✓ Validation: PASSED")
        inference_time = export_metadata['validation']['inference_time']['avg_ms']
        print(f"  Inference time: {inference_time:.2f} ms")
    elif args.validate:
        print(f"✗ Validation: FAILED")
    
    if args.optimize:
        optimized_path = Path(args.output).with_stem(
            Path(args.output).stem + '_optimized'
        )
        print(f"✓ Optimized model: {optimized_path}")
        opt_size = export_metadata['optimization']['optimized_size_mb']
        reduction = export_metadata['optimization']['size_reduction_percent']
        print(f"  Size: {opt_size:.2f} MB ({reduction:.1f}% reduction)")
    
    print(f"\n{'='*60}")
    print("Export completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
