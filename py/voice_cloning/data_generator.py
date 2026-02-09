"""
Synthetic training data generation for voice cloning pipeline.

This module generates synthetic training data by:
1. Loading ONNX models (text_encoder, duration_predictor, vector_estimator, vocoder)
2. Sampling style vectors using various strategies (random, perturb, interpolate, mixed)
3. Running full TTS pipeline to synthesize audio
4. Saving (audio, style_ttl, style_dp) tuples for training

The generated data is split into train/val/test sets and saved as numpy .npz files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Add parent directory to path to import helper
sys.path.append(str(Path(__file__).parent.parent))

from helper import (
    Style,
    TextToSpeech,
    load_text_to_speech,
    load_voice_style,
)


class StyleSampler:
    """
    Generates style vectors using various sampling strategies.
    
    Strategies:
        - random: Generate completely random style vectors
        - perturb: Start from existing styles and add Gaussian noise
        - interpolate: Mix two existing styles at random ratios
        - mixed: Randomly select from all strategies
    """
    
    def __init__(
        self,
        reference_styles: Optional[List[Style]] = None,
        style_ttl_shape: Tuple[int, int, int] = (1, 50, 256),
        style_dp_shape: Tuple[int, int, int] = (1, 8, 16),
        perturb_std: float = 0.1,
        interp_alpha_min: float = 0.0,
        interp_alpha_max: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize style sampler.
        
        Args:
            reference_styles: List of reference Style objects from existing voice styles
            style_ttl_shape: Shape of style_ttl vector (batch, seq_len, hidden_dim)
            style_dp_shape: Shape of style_dp vector (batch, seq_len, hidden_dim)
            perturb_std: Standard deviation for Gaussian noise in perturb strategy
            interp_alpha_min: Minimum interpolation factor
            interp_alpha_max: Maximum interpolation factor
            seed: Random seed for reproducibility
        """
        self.reference_styles = reference_styles or []
        self.style_ttl_shape = style_ttl_shape
        self.style_dp_shape = style_dp_shape
        self.perturb_std = perturb_std
        self.interp_alpha_min = interp_alpha_min
        self.interp_alpha_max = interp_alpha_max
        
        if seed is not None:
            np.random.seed(seed)
    
    def sample_random(self) -> Style:
        """Generate completely random style vectors."""
        # Sample from standard normal distribution
        style_ttl = np.random.randn(*self.style_ttl_shape).astype(np.float32)
        style_dp = np.random.randn(*self.style_dp_shape).astype(np.float32)
        return Style(style_ttl, style_dp)
    
    def sample_perturb(self) -> Style:
        """
        Start from an existing style and add Gaussian noise.
        
        Returns:
            Style: Perturbed style vector
            
        Raises:
            ValueError: If no reference styles are available
        """
        if not self.reference_styles:
            raise ValueError("No reference styles available for perturbation")
        
        # Randomly select a reference style
        ref_style = np.random.choice(self.reference_styles)
        
        # Add Gaussian noise
        noise_ttl = np.random.randn(*self.style_ttl_shape).astype(np.float32) * self.perturb_std
        noise_dp = np.random.randn(*self.style_dp_shape).astype(np.float32) * self.perturb_std
        
        style_ttl = ref_style.ttl + noise_ttl
        style_dp = ref_style.dp + noise_dp
        
        return Style(style_ttl, style_dp)
    
    def sample_interpolate(self) -> Style:
        """
        Mix two existing styles at a random ratio.
        
        Returns:
            Style: Interpolated style vector
            
        Raises:
            ValueError: If fewer than 2 reference styles are available
        """
        if len(self.reference_styles) < 2:
            raise ValueError("Need at least 2 reference styles for interpolation")
        
        # Randomly select two different styles
        style1, style2 = np.random.choice(self.reference_styles, size=2, replace=False)
        
        # Random interpolation factor
        alpha = np.random.uniform(self.interp_alpha_min, self.interp_alpha_max)
        
        # Linear interpolation
        style_ttl = alpha * style1.ttl + (1 - alpha) * style2.ttl
        style_dp = alpha * style1.dp + (1 - alpha) * style2.dp
        
        return Style(style_ttl, style_dp)
    
    def sample_mixed(self) -> Style:
        """
        Randomly select from all available strategies.
        
        Returns:
            Style: Sampled style vector
        """
        # Determine available strategies
        strategies = ["random"]
        if self.reference_styles:
            strategies.append("perturb")
        if len(self.reference_styles) >= 2:
            strategies.append("interpolate")
        
        # Randomly select a strategy
        strategy = np.random.choice(strategies)
        
        if strategy == "random":
            return self.sample_random()
        elif strategy == "perturb":
            return self.sample_perturb()
        else:  # interpolate
            return self.sample_interpolate()
    
    def sample(self, mode: str = "mixed") -> Style:
        """
        Sample a style vector using the specified strategy.
        
        Args:
            mode: Sampling strategy ('random', 'perturb', 'interpolate', or 'mixed')
            
        Returns:
            Style: Sampled style vector
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode == "random":
            return self.sample_random()
        elif mode == "perturb":
            return self.sample_perturb()
        elif mode == "interpolate":
            return self.sample_interpolate()
        elif mode == "mixed":
            return self.sample_mixed()
        else:
            raise ValueError(f"Invalid sampling mode: {mode}")


class SyntheticDataGenerator:
    """
    Generates synthetic training data for voice cloning.
    
    For each sample:
    1. Sample random style vectors (style_ttl, style_dp)
    2. Pick random text from corpus
    3. Run full TTS pipeline to synthesize audio
    4. Save (audio, style_ttl, style_dp) as .npz file
    """
    
    def __init__(
        self,
        tts: TextToSpeech,
        style_sampler: StyleSampler,
        texts: List[str],
        languages: List[str],
        output_dir: str,
        tts_total_step: int = 5,
        tts_speed: float = 1.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            tts: TextToSpeech instance for synthesis
            style_sampler: StyleSampler for generating style vectors
            texts: Diverse text corpus for generation
            languages: List of languages to use
            output_dir: Directory to save generated data
            tts_total_step: Number of denoising steps for TTS
            tts_speed: Speed factor for TTS generation
            seed: Random seed for reproducibility
        """
        self.tts = tts
        self.style_sampler = style_sampler
        self.texts = texts
        self.languages = languages
        self.output_dir = Path(output_dir)
        self.tts_total_step = tts_total_step
        self.tts_speed = tts_speed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_sample(
        self,
        sample_id: int,
        split: str,
        style_sample_mode: str = "mixed",
    ) -> Dict[str, Any]:
        """
        Generate a single training sample.
        
        Args:
            sample_id: Unique identifier for the sample
            split: Data split ('train', 'val', or 'test')
            style_sample_mode: Style sampling strategy
            
        Returns:
            Dict containing generation metadata
        """
        # Sample random text and language
        text = np.random.choice(self.texts)
        lang = np.random.choice(self.languages)
        
        # Sample style vectors
        style = self.style_sampler.sample(mode=style_sample_mode)
        
        # Synthesize audio
        try:
            wav, duration = self.tts(
                text=text,
                lang=lang,
                style=style,
                total_step=self.tts_total_step,
                speed=self.tts_speed,
            )
            
            # Validate output
            if duration is None or len(duration) == 0:
                raise ValueError("TTS returned empty duration")
            
            # Trim audio to actual duration
            duration_samples = int(self.tts.sample_rate * duration[0].item())
            audio = wav[0, :duration_samples]
            
            # Save as .npz file
            if split == "train":
                save_dir = self.train_dir
            elif split == "val":
                save_dir = self.val_dir
            else:
                save_dir = self.test_dir
            
            save_path = save_dir / f"sample_{sample_id:06d}.npz"
            np.savez_compressed(
                save_path,
                audio=audio,
                style_ttl=style.ttl,
                style_dp=style.dp,
                text=text,
                lang=lang,
                duration=duration[0].item(),
                sample_rate=self.tts.sample_rate,
            )
            
            return {
                "sample_id": sample_id,
                "split": split,
                "text": text,
                "lang": lang,
                "duration": duration[0].item(),
                "audio_length": len(audio),
                "save_path": str(save_path),
            }
            
        except Exception as e:
            print(f"\nError generating sample {sample_id}: {e}")
            print(f"  Text: '{text[:50]}...' (lang={lang})")
            return None
    
    def generate_dataset(
        self,
        num_train: int,
        num_val: int,
        num_test: int,
        style_sample_mode: str = "mixed",
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete dataset with train/val/test splits.
        
        Args:
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples
            style_sample_mode: Style sampling strategy
            resume: Whether to resume from existing progress
            
        Returns:
            Dict containing generation statistics
        """
        total_samples = num_train + num_val + num_test
        
        # Check for existing samples if resuming
        existing_train = len(list(self.train_dir.glob("*.npz"))) if resume else 0
        existing_val = len(list(self.val_dir.glob("*.npz"))) if resume else 0
        existing_test = len(list(self.test_dir.glob("*.npz"))) if resume else 0
        
        print(f"\n{'='*60}")
        print(f"Generating Synthetic Dataset")
        print(f"{'='*60}")
        print(f"Train samples: {num_train} (existing: {existing_train})")
        print(f"Val samples: {num_val} (existing: {existing_val})")
        print(f"Test samples: {num_test} (existing: {existing_test})")
        print(f"Total samples: {total_samples}")
        print(f"Style sampling mode: {style_sample_mode}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Determine which samples to generate
        splits = []
        if existing_train < num_train:
            splits.extend([("train", i) for i in range(existing_train, num_train)])
        if existing_val < num_val:
            splits.extend([("val", i) for i in range(existing_val, num_val)])
        if existing_test < num_test:
            splits.extend([("test", i) for i in range(existing_test, num_test)])
        
        if not splits:
            print("All samples already exist. Nothing to generate.")
            return self._get_generation_stats()
        
        # Generate samples with progress bar
        successful = 0
        failed = 0
        start_time = time.time()
        
        with tqdm(total=len(splits), desc="Generating samples") as pbar:
            for split, idx in splits:
                result = self.generate_sample(idx, split, style_sample_mode)
                
                if result is not None:
                    successful += 1
                else:
                    failed += 1
                
                pbar.update(1)
                
                # Update progress bar with stats
                elapsed = time.time() - start_time
                samples_per_sec = successful / elapsed if elapsed > 0 and successful > 0 else 0
                remaining = len(splits) - (successful + failed)
                eta = remaining / samples_per_sec if samples_per_sec > 0 else 0
                
                pbar.set_postfix({
                    "success": successful,
                    "failed": failed,
                    "rate": f"{samples_per_sec:.2f} samples/s",
                    "ETA": f"{eta/60:.1f}m",
                })
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"{'='*60}")
        print(f"Successfully generated: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Average time per sample: {elapsed_time/successful:.2f} seconds")
        print(f"{'='*60}\n")
        
        return self._get_generation_stats()
    
    def _get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated dataset."""
        train_samples = list(self.train_dir.glob("*.npz"))
        val_samples = list(self.val_dir.glob("*.npz"))
        test_samples = list(self.test_dir.glob("*.npz"))
        
        return {
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(test_samples),
            "total": len(train_samples) + len(val_samples) + len(test_samples),
            "train_dir": str(self.train_dir),
            "val_dir": str(self.val_dir),
            "test_dir": str(self.test_dir),
        }


def load_reference_styles(voice_styles_dir: str) -> List[Style]:
    """
    Load all reference voice styles from directory.
    
    Args:
        voice_styles_dir: Path to voice styles directory
        
    Returns:
        List of Style objects
    """
    voice_styles_path = Path(voice_styles_dir)
    
    if not voice_styles_path.exists():
        print(f"Warning: Voice styles directory not found: {voice_styles_dir}")
        print("Proceeding with random sampling only.")
        return []
    
    style_files = sorted(voice_styles_path.glob("*.json"))
    
    if not style_files:
        print(f"Warning: No .json files found in {voice_styles_dir}")
        print("Proceeding with random sampling only.")
        return []
    
    print(f"Loading {len(style_files)} reference voice styles...")
    
    styles = []
    for style_file in style_files:
        try:
            style = load_voice_style([str(style_file)])
            styles.append(style)
        except Exception as e:
            print(f"Warning: Failed to load {style_file.name}: {e}")
    
    print(f"Successfully loaded {len(styles)} reference styles")
    return styles


def save_metadata(
    output_dir: str,
    args: argparse.Namespace,
    stats: Dict[str, Any],
) -> None:
    """
    Save generation metadata to JSON file.
    
    Args:
        output_dir: Output directory
        args: Command line arguments
        stats: Generation statistics
    """
    metadata = {
        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "num_samples": args.num_samples,
            "num_val_samples": args.num_val_samples,
            "num_test_samples": args.num_test_samples,
            "style_sample_mode": args.style_sample_mode,
            "tts_total_step": args.tts_total_step,
            "tts_speed": args.tts_speed,
            "seed": args.seed,
            "onnx_dir": args.onnx_dir,
            "voice_styles_dir": args.voice_styles_dir,
            "perturb_std": args.perturb_std,
            "interp_alpha_min": args.interp_alpha_min,
            "interp_alpha_max": args.interp_alpha_max,
        },
        "texts": args.texts,
        "languages": args.languages,
        "statistics": stats,
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for voice cloning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=1000,
        help="Number of validation samples to generate",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=500,
        help="Number of test samples to generate",
    )
    
    # Path parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_data",
        help="Directory to save generated data",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="../../assets/onnx",
        help="Path to ONNX models directory",
    )
    parser.add_argument(
        "--voice-styles-dir",
        type=str,
        default="../../assets/voice_styles",
        help="Path to voice styles directory",
    )
    
    # Style sampling parameters
    parser.add_argument(
        "--style-sample-mode",
        type=str,
        default="mixed",
        choices=["random", "perturb", "interpolate", "mixed"],
        help="Style sampling strategy",
    )
    parser.add_argument(
        "--perturb-std",
        type=float,
        default=0.1,
        help="Standard deviation for style perturbation",
    )
    parser.add_argument(
        "--interp-alpha-min",
        type=float,
        default=0.0,
        help="Minimum interpolation factor",
    )
    parser.add_argument(
        "--interp-alpha-max",
        type=float,
        default=1.0,
        help="Maximum interpolation factor",
    )
    
    # TTS parameters
    parser.add_argument(
        "--tts-total-step",
        type=int,
        default=5,
        help="Number of denoising steps for TTS generation",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.05,
        help="Speed factor for TTS generation",
    )
    
    # Text corpus
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=[
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
            "Artificial intelligence is changing our lives.",
            "The sun sets beautifully over the ocean.",
            "Music brings joy to people around the world.",
            "Communication is key to understanding each other.",
            "Innovation drives progress in society.",
        ],
        help="Text corpus for generation (space-separated)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        help="Languages to use for generation (en, ko, es, pt, fr)",
    )
    
    # Other parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing progress (regenerate all)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for inference (default: CPU)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for data generation."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("Synthetic Training Data Generator for Voice Cloning")
    print("="*60 + "\n")
    
    # Load TTS models
    print("Loading TTS models...")
    try:
        tts = load_text_to_speech(args.onnx_dir, use_gpu=args.use_gpu)
        print("✓ TTS models loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load TTS models: {e}")
        print(f"Please ensure ONNX models are available at: {args.onnx_dir}")
        sys.exit(1)
    
    # Load reference voice styles
    reference_styles = load_reference_styles(args.voice_styles_dir)
    
    # Get style vector shapes from config or first reference style
    if reference_styles:
        style_ttl_shape = reference_styles[0].ttl.shape
        style_dp_shape = reference_styles[0].dp.shape
    else:
        # Default shapes if no reference styles available
        style_ttl_shape = (1, 50, 256)
        style_dp_shape = (1, 8, 16)
    
    print(f"Style vector shapes: ttl={style_ttl_shape}, dp={style_dp_shape}\n")
    
    # Create style sampler
    style_sampler = StyleSampler(
        reference_styles=reference_styles,
        style_ttl_shape=style_ttl_shape,
        style_dp_shape=style_dp_shape,
        perturb_std=args.perturb_std,
        interp_alpha_min=args.interp_alpha_min,
        interp_alpha_max=args.interp_alpha_max,
        seed=args.seed,
    )
    
    # Create data generator
    generator = SyntheticDataGenerator(
        tts=tts,
        style_sampler=style_sampler,
        texts=args.texts,
        languages=args.languages,
        output_dir=args.output_dir,
        tts_total_step=args.tts_total_step,
        tts_speed=args.tts_speed,
        seed=args.seed,
    )
    
    # Generate dataset
    stats = generator.generate_dataset(
        num_train=args.num_samples,
        num_val=args.num_val_samples,
        num_test=args.num_test_samples,
        style_sample_mode=args.style_sample_mode,
        resume=not args.no_resume,
    )
    
    # Save metadata
    save_metadata(args.output_dir, args, stats)
    
    print("\n✓ Data generation complete!")
    print(f"Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
