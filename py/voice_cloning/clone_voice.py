"""
End-to-end voice cloning script for Supertonic TTS.

This script provides a complete pipeline for voice cloning:
1. Load and preprocess input audio (resample, convert to mono, normalize)
2. Extract mel-spectrogram features
3. Run encoder inference (PyTorch or ONNX mode)
4. Generate Supertonic-compatible voice style JSON
5. Optionally test the cloned voice with TTS

Example usage:
    # Clone voice using PyTorch model
    python clone_voice.py --input speaker.wav --output cloned_voice.json \\
        --encoder checkpoints/encoder.pth --mode pytorch --name "My Voice"
    
    # Clone voice using ONNX model (production mode)
    python clone_voice.py --input speaker.wav --output cloned_voice.json \\
        --encoder output/encoder.onnx --mode onnx --device cpu
    
    # Clone voice and run TTS test
    python clone_voice.py --input speaker.wav --output cloned_voice.json \\
        --encoder encoder.onnx --test --test-text "Hello world" \\
        --test-output test_output.wav --onnx-dir ../../assets/onnx
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_cloning.config import TrainingConfig, get_default_config

# Conditional import for PyTorch (required for PyTorch mode)
try:
    import torch
    from voice_cloning.encoder_model import create_encoder, load_encoder
    from voice_cloning.utils import get_device
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

# Conditional import for TTS testing
try:
    from helper import Style, load_text_to_speech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# Import audio utility functions directly to avoid torch dependency
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
    # Create directory if needed
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    # Save audio
    sf.write(path, audio, sample_rate)


def validate_audio_file(audio_path: str) -> None:
    """
    Validate that the input audio file exists and is readable.
    
    Args:
        audio_path: Path to audio file.
        
    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If file format is not supported.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported audio format: {ext}. "
                        f"Supported formats: {', '.join(valid_extensions)}")


def validate_encoder_model(encoder_path: str, mode: str) -> str:
    """
    Validate encoder model path and determine inference mode.
    
    Args:
        encoder_path: Path to encoder model file.
        mode: Inference mode ('auto', 'pytorch', or 'onnx').
        
    Returns:
        Detected/validated mode ('pytorch' or 'onnx').
        
    Raises:
        FileNotFoundError: If encoder model doesn't exist.
        ValueError: If mode is invalid or incompatible with file type.
    """
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
    
    ext = os.path.splitext(encoder_path)[1].lower()
    
    # Auto-detect mode from file extension
    if mode == 'auto':
        if ext == '.onnx':
            mode = 'onnx'
        elif ext in ['.pth', '.pt']:
            mode = 'pytorch'
        else:
            raise ValueError(f"Cannot auto-detect mode for file extension: {ext}. "
                           "Please specify --mode explicitly.")
    
    # Validate mode matches file extension
    if mode == 'onnx' and ext != '.onnx':
        raise ValueError(f"Mode is 'onnx' but file extension is {ext}. "
                        "Expected .onnx file.")
    elif mode == 'pytorch' and ext not in ['.pth', '.pt']:
        raise ValueError(f"Mode is 'pytorch' but file extension is {ext}. "
                        "Expected .pth or .pt file.")
    
    return mode


def load_and_preprocess_audio(
    audio_path: str,
    config: TrainingConfig,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to input audio file.
        config: Training configuration.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (audio waveform, duration in seconds).
    """
    if verbose:
        print(f"Loading audio from: {audio_path}")
    
    start_time = time.time()
    
    # Load audio (automatically resamples to target sample rate, converts to mono, normalizes)
    audio = load_audio(audio_path, sample_rate=config.sample_rate, normalize=True)
    
    duration = len(audio) / config.sample_rate
    load_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Loaded audio: duration={duration:.2f}s, "
              f"samples={len(audio)}, sr={config.sample_rate}Hz ({load_time:.2f}s)")
    
    # Warn about audio length
    if duration < 3.0:
        warnings.warn(f"Audio is very short ({duration:.2f}s). "
                     "Voice cloning may not work well with <3s of audio. "
                     "Consider using longer audio samples (5-30s recommended).")
    elif duration > 30.0:
        warnings.warn(f"Audio is quite long ({duration:.2f}s). "
                     "Consider using shorter clips (5-30s recommended) for better quality.")
    
    return audio, duration


def extract_mel_spectrogram(
    audio: np.ndarray,
    config: TrainingConfig,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract mel-spectrogram from audio waveform.
    
    Args:
        audio: Audio waveform [T].
        config: Training configuration.
        verbose: Whether to print progress messages.
        
    Returns:
        Mel-spectrogram [n_mels, T_frames].
    """
    if verbose:
        print("Extracting mel-spectrogram...")
    
    start_time = time.time()
    
    mel = compute_mel_spectrogram(
        audio=audio,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    
    extract_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Extracted mel-spectrogram: shape={mel.shape}, "
              f"frames={mel.shape[1]} ({extract_time:.3f}s)")
    
    return mel


def run_pytorch_inference(
    mel: np.ndarray,
    encoder_path: str,
    config: TrainingConfig,
    device: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run encoder inference using PyTorch model.
    
    Args:
        mel: Mel-spectrogram [n_mels, T_frames].
        encoder_path: Path to PyTorch encoder checkpoint.
        config: Training configuration.
        device: Device to run inference on ('cpu', 'cuda', or 'mps').
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (style_ttl, style_dp) numpy arrays.
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available. Install torch to use PyTorch mode.")
    
    if verbose:
        print(f"Running PyTorch inference on device: {device}")
    
    # Load model
    start_time = time.time()
    model = load_encoder(encoder_path, config=config, device=device)
    model.eval()
    load_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Loaded model ({load_time:.2f}s)")
    
    # Prepare input
    mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, n_mels, T]
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        style_ttl, style_dp = model(mel_tensor)
    inference_time = time.time() - start_time
    
    # Convert to numpy
    style_ttl_np = style_ttl.cpu().numpy()
    style_dp_np = style_dp.cpu().numpy()
    
    if verbose:
        print(f"  ✓ Inference completed: style_ttl={style_ttl_np.shape}, "
              f"style_dp={style_dp_np.shape} ({inference_time:.3f}s)")
    
    return style_ttl_np, style_dp_np


def run_onnx_inference(
    mel: np.ndarray,
    encoder_path: str,
    config: TrainingConfig,
    device: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run encoder inference using ONNX model.
    
    Args:
        mel: Mel-spectrogram [n_mels, T_frames].
        encoder_path: Path to ONNX encoder model.
        config: Training configuration.
        device: Device to run inference on ('cpu', 'cuda', or 'mps').
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (style_ttl, style_dp) numpy arrays.
    """
    import onnxruntime as ort
    
    if verbose:
        print(f"Running ONNX inference on device: {device}")
    
    # Set providers based on device
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device == 'mps':
        # ONNX Runtime doesn't support MPS, fallback to CPU
        warnings.warn("ONNX Runtime doesn't support MPS. Using CPU instead.")
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    # Load ONNX model
    start_time = time.time()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(encoder_path, sess_options=session_options, 
                                   providers=providers)
    load_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Loaded ONNX model ({load_time:.2f}s)")
    
    # Prepare input
    mel_input = mel[np.newaxis, :, :].astype(np.float32)  # [1, n_mels, T]
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Run inference
    start_time = time.time()
    outputs = session.run(output_names, {input_name: mel_input})
    inference_time = time.time() - start_time
    
    style_ttl_np = outputs[0]
    style_dp_np = outputs[1]
    
    if verbose:
        print(f"  ✓ Inference completed: style_ttl={style_ttl_np.shape}, "
              f"style_dp={style_dp_np.shape} ({inference_time:.3f}s)")
    
    return style_ttl_np, style_dp_np


def format_style_json(
    style_ttl: np.ndarray,
    style_dp: np.ndarray,
    name: str,
    description: str,
    audio_path: str,
    verbose: bool = True
) -> Dict:
    """
    Format style vectors into Supertonic-compatible JSON structure.
    
    Args:
        style_ttl: Style TTL vector [1, 1, 50, 256].
        style_dp: Style DP vector [1, 1, 8, 16].
        name: Name for the voice style.
        description: Description for the voice style.
        audio_path: Path to original audio file (for metadata).
        verbose: Whether to print progress messages.
        
    Returns:
        Dictionary with voice style data.
    """
    if verbose:
        print("Formatting output JSON...")
    
    # Remove batch dimension and flatten to list
    style_ttl_data = style_ttl[0].flatten().tolist()  # [1, 50, 256] -> [12800]
    style_dp_data = style_dp[0].flatten().tolist()    # [1, 8, 16] -> [128]
    
    # Get dimensions (batch=1, then actual dimensions)
    # Format: [batch_size, height, width] to match helper.py load_voice_style
    ttl_dims = [1, style_ttl.shape[2], style_ttl.shape[3]]  # [1, 50, 256]
    dp_dims = [1, style_dp.shape[2], style_dp.shape[3]]    # [1, 8, 16]
    
    # Create JSON structure matching Supertonic format
    voice_style = {
        "name": name,
        "description": description,
        "metadata": {
            "source_audio": os.path.basename(audio_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "style_ttl": {
            "dims": ttl_dims,
            "data": [style_ttl_data]  # List of flattened arrays
        },
        "style_dp": {
            "dims": dp_dims,
            "data": [style_dp_data]  # List of flattened arrays
        }
    }
    
    if verbose:
        print(f"  ✓ Formatted JSON: ttl_size={len(style_ttl_data)}, "
              f"dp_size={len(style_dp_data)}")
    
    return voice_style


def save_voice_style(voice_style: Dict, output_path: str, verbose: bool = True) -> None:
    """
    Save voice style to JSON file.
    
    Args:
        voice_style: Voice style dictionary.
        output_path: Path to output JSON file.
        verbose: Whether to print progress messages.
    """
    if verbose:
        print(f"Saving voice style to: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(voice_style, f, indent=2)
    
    if verbose:
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"  ✓ Saved voice style ({file_size:.1f} KB)")


def run_tts_test(
    voice_style_path: str,
    test_text: str,
    test_output: str,
    onnx_dir: str,
    total_step: int = 5,
    speed: float = 1.05,
    language: str = "en",
    verbose: bool = True
) -> None:
    """
    Test the cloned voice by generating speech.
    
    Args:
        voice_style_path: Path to voice style JSON file.
        test_text: Text to synthesize.
        test_output: Path to save output audio.
        onnx_dir: Path to TTS ONNX models directory.
        total_step: Number of denoising steps.
        speed: Speed factor for TTS.
        language: Language code.
        verbose: Whether to print progress messages.
    """
    if not TTS_AVAILABLE:
        raise RuntimeError("TTS testing requires helper module which is not available.")
    
    if verbose:
        print("\n" + "="*60)
        print("Running TTS test...")
        print("="*60)
    
    # Load TTS engine
    if verbose:
        print(f"Loading TTS models from: {onnx_dir}")
    start_time = time.time()
    tts = load_text_to_speech(onnx_dir, use_gpu=False)
    load_time = time.time() - start_time
    if verbose:
        print(f"  ✓ Loaded TTS models ({load_time:.2f}s)")
    
    # Load voice style
    if verbose:
        print(f"Loading voice style from: {voice_style_path}")
    style = load_voice_style([voice_style_path], verbose=False)
    
    # Generate speech
    if verbose:
        print(f"Generating speech: '{test_text}'")
    start_time = time.time()
    wav, duration = tts(test_text, language, style, total_step, speed)
    gen_time = time.time() - start_time
    
    # Save audio
    audio_np = wav[0]  # Remove batch dimension
    save_audio(audio_np, test_output, sample_rate=tts.sample_rate)
    
    if verbose:
        print(f"  ✓ Generated speech: duration={duration[0]:.2f}s, "
              f"generation_time={gen_time:.2f}s")
        print(f"  ✓ Saved test audio to: {test_output}")
        print("="*60)


def load_voice_style(voice_style_paths: list, verbose: bool = False) -> "Style":
    """
    Load voice style from JSON file(s) for TTS testing.
    
    Args:
        voice_style_paths: List of paths to voice style JSON files.
        verbose: Whether to print progress messages.
        
    Returns:
        Style object for TTS.
    """
    bsz = len(voice_style_paths)
    
    # Read first file to get dimensions
    with open(voice_style_paths[0], "r") as f:
        first_style = json.load(f)
    ttl_dims = first_style["style_ttl"]["dims"]
    dp_dims = first_style["style_dp"]["dims"]
    
    # Pre-allocate arrays with full batch size
    ttl_style = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp_style = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)
    
    # Fill in the data
    for i, voice_style_path in enumerate(voice_style_paths):
        with open(voice_style_path, "r") as f:
            voice_style = json.load(f)
        
        ttl_data = np.array(
            voice_style["style_ttl"]["data"], dtype=np.float32
        ).flatten()
        ttl_style[i] = ttl_data.reshape(ttl_dims[1], ttl_dims[2])
        
        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        dp_style[i] = dp_data.reshape(dp_dims[1], dp_dims[2])
    
    if verbose:
        print(f"Loaded {bsz} voice styles")
    
    return Style(ttl_style, dp_style)


def print_statistics(
    style_ttl: np.ndarray,
    style_dp: np.ndarray,
    verbose: bool = True
) -> None:
    """
    Print statistics about the generated style vectors.
    
    Args:
        style_ttl: Style TTL vector.
        style_dp: Style DP vector.
        verbose: Whether to print statistics.
    """
    if not verbose:
        return
    
    print("\nStyle Vector Statistics:")
    print("-" * 40)
    print(f"Style TTL:")
    print(f"  Shape: {style_ttl.shape}")
    print(f"  Mean: {style_ttl.mean():.4f}")
    print(f"  Std: {style_ttl.std():.4f}")
    print(f"  Min: {style_ttl.min():.4f}")
    print(f"  Max: {style_ttl.max():.4f}")
    print(f"\nStyle DP:")
    print(f"  Shape: {style_dp.shape}")
    print(f"  Mean: {style_dp.mean():.4f}")
    print(f"  Std: {style_dp.std():.4f}")
    print(f"  Min: {style_dp.min():.4f}")
    print(f"  Max: {style_dp.max():.4f}")
    print("-" * 40)


def main():
    """Main entry point for voice cloning script."""
    parser = argparse.ArgumentParser(
        description="End-to-end voice cloning for Supertonic TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone voice using PyTorch model
  python clone_voice.py --input speaker.wav --output cloned_voice.json \\
      --encoder checkpoints/encoder.pth --mode pytorch --name "My Voice"
  
  # Clone voice using ONNX model (production mode)
  python clone_voice.py --input speaker.wav --output cloned_voice.json \\
      --encoder output/encoder.onnx --mode onnx --device cpu
  
  # Clone voice and run TTS test
  python clone_voice.py --input speaker.wav --output cloned_voice.json \\
      --encoder encoder.onnx --test --test-text "Hello world" \\
      --test-output test_output.wav --onnx-dir ../../assets/onnx
        """
    )
    
    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input audio file (WAV, MP3, FLAC, etc.)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output JSON file')
    
    # Encoder arguments
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to encoder model (.pth for PyTorch, .onnx for ONNX)')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'pytorch', 'onnx'],
                       help='Inference mode (auto: detect from file extension)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for inference (auto: detect best available)')
    
    # Voice style metadata
    parser.add_argument('--name', type=str, default='cloned_voice',
                       help='Name for the voice style')
    parser.add_argument('--description', type=str, default=None,
                       help='Description for the voice style (auto-generated if not provided)')
    
    # TTS testing arguments
    parser.add_argument('--test', action='store_true',
                       help='Run TTS test after cloning')
    parser.add_argument('--test-text', type=str, 
                       default='Hello, this is a test of the cloned voice.',
                       help='Text to use for TTS test')
    parser.add_argument('--test-output', type=str, default='test_output.wav',
                       help='Where to save test audio')
    parser.add_argument('--onnx-dir', type=str, default='../../assets/onnx',
                       help='Path to TTS ONNX models for testing')
    parser.add_argument('--test-language', type=str, default='en',
                       choices=['en', 'ko', 'es', 'pt', 'fr'],
                       help='Language for TTS test')
    parser.add_argument('--test-speed', type=float, default=1.05,
                       help='Speed factor for TTS test')
    parser.add_argument('--test-steps', type=int, default=5,
                       help='Number of denoising steps for TTS test')
    
    # Other arguments
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("\n" + "="*60)
        print("Supertonic Voice Cloning")
        print("="*60)
    
    try:
        # Step 1: Validate inputs
        if verbose:
            print("\n[Step 1/6] Validating inputs...")
        validate_audio_file(args.input)
        mode = validate_encoder_model(args.encoder, args.mode)
        
        # Determine device
        if args.device == 'auto':
            if mode == 'pytorch':
                if not PYTORCH_AVAILABLE:
                    raise RuntimeError("PyTorch is not available. Cannot auto-detect device.")
                device_torch = get_device()
                device = str(device_torch).split(':')[0]  # Remove device index if present
            else:
                # For ONNX, default to CPU
                device = 'cpu'
        else:
            device = args.device
        
        if verbose:
            print(f"  ✓ Mode: {mode}")
            print(f"  ✓ Device: {device}")
        
        # Step 2: Load configuration
        if verbose:
            print("\n[Step 2/6] Loading configuration...")
        config = get_default_config()
        if verbose:
            print(f"  ✓ Loaded config: sample_rate={config.sample_rate}Hz, "
                  f"n_mels={config.n_mels}")
        
        # Step 3: Load and preprocess audio
        if verbose:
            print("\n[Step 3/6] Loading and preprocessing audio...")
        audio, duration = load_and_preprocess_audio(args.input, config, verbose)
        
        # Step 4: Extract mel-spectrogram
        if verbose:
            print("\n[Step 4/6] Extracting features...")
        mel = extract_mel_spectrogram(audio, config, verbose)
        
        # Step 5: Run encoder inference
        if verbose:
            print(f"\n[Step 5/6] Running {mode.upper()} inference...")
        
        if mode == 'pytorch':
            style_ttl, style_dp = run_pytorch_inference(
                mel, args.encoder, config, device, verbose
            )
        elif mode == 'onnx':
            style_ttl, style_dp = run_onnx_inference(
                mel, args.encoder, config, device, verbose
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Print statistics
        print_statistics(style_ttl, style_dp, verbose)
        
        # Step 6: Generate and save output
        if verbose:
            print("\n[Step 6/6] Generating output...")
        
        # Auto-generate description if not provided
        if args.description is None:
            description = f"Voice cloned from {os.path.basename(args.input)} ({duration:.1f}s)"
        else:
            description = args.description
        
        voice_style = format_style_json(
            style_ttl, style_dp, args.name, description, args.input, verbose
        )
        save_voice_style(voice_style, args.output, verbose)
        
        # Success message
        if verbose:
            print("\n" + "="*60)
            print("✓ Voice cloning completed successfully!")
            print(f"  Output saved to: {args.output}")
            print("="*60)
        
        # Step 7: Optional TTS test
        if args.test:
            if not TTS_AVAILABLE:
                print("\n⚠ Warning: TTS testing is not available (helper module missing)")
            else:
                run_tts_test(
                    voice_style_path=args.output,
                    test_text=args.test_text,
                    test_output=args.test_output,
                    onnx_dir=args.onnx_dir,
                    total_step=args.test_steps,
                    speed=args.test_speed,
                    language=args.test_language,
                    verbose=verbose
                )
        
        return 0
        
    except Exception as e:
        if verbose:
            print("\n" + "="*60)
            print("✗ Error occurred:")
            print(f"  {type(e).__name__}: {e}")
            print("="*60)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
