# Supertonic TTS - Pure C Implementation

This directory contains a pure C implementation of the Supertonic Text-to-Speech system, converted from the C++ version.

## Features

- **Pure C (C11)**: No C++ dependencies
- **ONNX Runtime C API**: Uses the native C API for model inference
- **cJSON**: For JSON parsing instead of nlohmann/json
- **Cross-platform**: Works on macOS and Linux
- **Memory-managed**: Proper allocation/deallocation with cleanup functions
- **Same functionality**: Maintains all features from the C++ version
- **Voice Builder**: Create custom voice styles from WAV files

## Files

- `supertonic.h` - Main header file with function declarations and structures
- `supertonic.c` - Core implementation of TTS system
- `wav_utils.h/c` - WAV file reading and writing utilities
- `example_onnx.c` - Example program demonstrating usage
- `audiobook_generator.c` - Tool for converting text files to audiobooks
- `voice_builder.c` - Tool for creating custom voice styles from WAV files
- `Makefile` - Build system for compiling the project

## Dependencies

### Required Libraries

1. **ONNX Runtime** - For running ONNX models
   ```bash
   # macOS (Homebrew)
   brew install onnxruntime
   
   # Linux - download from https://github.com/microsoft/onnxruntime/releases
   # Extract and install to /usr/local or specify ONNXRUNTIME_ROOT
   ```

**Note:** cJSON is now bundled in `vendor/cjson/` and no longer requires separate installation!

## Downloading Resources (Models and Voice Styles)

Before building and running the examples, you need to download the ONNX models and voice styles.

### Automatic Download (Recommended)

Use the provided script to automatically download all necessary resources:

```bash
./resource.sh
```

This script will:
- Check if Git LFS is installed (required for large model files)
- Download ONNX models and voice styles from Hugging Face
- Verify that all required files are present
- Handle updates if resources already exist

### Manual Download

Alternatively, you can manually download the resources:

```bash
cd ..
git clone https://huggingface.co/Supertone/supertonic-2 assets
```

> **Note:** The Hugging Face repository uses Git LFS. Please ensure Git LFS is installed and initialized before cloning:
> - macOS: `brew install git-lfs && git lfs install`
> - Generic: see `https://git-lfs.com` for installers

## Building

### Default Build

```bash
make
```

This will compile all source files and create three executables: `example_onnx`, `audiobook_generator`, and `voice_builder`.

### Custom Library Paths

If your libraries are installed in non-standard locations (e.g., downloaded ONNX Runtime package), you have three options:

#### Option 1: Create Makefile.local (Recommended)

This method saves your settings so you don't have to specify them every time:

```bash
# Create a Makefile.local with your custom paths
cat > Makefile.local << EOF
ONNXRUNTIME_ROOT = /home/user/Downloads/onnxruntime-linux-x64-1.23.2
CJSON_ROOT = /usr/local
EOF

# Now just run make normally
make
```

The `Makefile.local` file is ignored by git, so your personal settings won't be committed.

#### Option 2: Command-line Arguments

Specify paths each time you build:

```bash
make ONNXRUNTIME_ROOT=/home/user/Downloads/onnxruntime-linux-x64-1.23.2 CJSON_ROOT=/usr/local
```

#### Option 3: Environment Variables

Set environment variables in your shell:

```bash
export ONNXRUNTIME_ROOT=/home/user/Downloads/onnxruntime-linux-x64-1.23.2
export CJSON_ROOT=/usr/local
make
```

### Verifying Your Configuration

Check if your library paths are correctly configured:

```bash
make show-config
```

This will display your current settings and verify that the library directories exist.

### Common ONNX Runtime Locations

- **System install**: `/usr/local` (default)
- **Homebrew (macOS)**: `/opt/homebrew` or `/usr/local`
- **Downloaded package**: `/home/user/Downloads/onnxruntime-linux-x64-gpu-1.23.2`
- **Custom install**: `/opt/onnxruntime`

### Clean

```bash
make clean
```

## Usage

### Quick Start

The simplest way to get started is to run the example with default settings:

```bash
./example_onnx
```

Or with custom text:

```bash
./example_onnx --text "Hello, world! This is my first text-to-speech test."
```

### Getting Help

Show all available options and examples:

```bash
./example_onnx --help
```

List available voice styles:

```bash
./example_onnx --list-voices
```

### Basic Examples

**Simple text-to-speech:**
```bash
./example_onnx --text "Welcome to Supertonic!"
```

**Use a female voice:**
```bash
./example_onnx --text "Hello!" --voice-style ../assets/voice_styles/F1.json
```

**Change language and speed:**
```bash
./example_onnx --text "Hola mundo" --lang es --speed 1.2
```

**Save to a specific file:**
```bash
./example_onnx --text "Testing output" --output my_speech.wav
```

**Multiple iterations for testing:**
```bash
./example_onnx --text "Test" --n-test 5
```

### Advanced Usage

**Multiple texts with different voices:**
```bash
./example_onnx \
    --text "Hello from voice 1|Hello from voice 2" \
    --voice-style "../assets/voice_styles/M1.json,../assets/voice_styles/F1.json"
```

**Full control over all parameters:**
```bash
./example_onnx \
    --onnx-dir ../assets/onnx \
    --voice-style "../assets/voice_styles/M1.json" \
    --text "This is a complete example." \
    --lang "en" \
    --total-step 5 \
    --speed 1.05 \
    --n-test 1 \
    --output my_audio.wav
```

### Command-line Options

- `--help, -h` - Show help message with all options and examples
- `--list-voices` - List all available voice styles in the assets directory
- `--text <text>` - Text to synthesize (use `|` to separate multiple texts)
- `--voice-style <path>` - Path to voice style JSON (use `,` to separate multiple)
- `--lang <code>` - Language code: `en`, `ko`, `es`, `pt`, `fr` (use `,` for multiple)
- `--speed <float>` - Speech speed factor (default: 1.05)
- `--total-step <n>` - Number of denoising steps (default: 5)
- `--n-test <n>` - Number of test iterations (default: 1)
- `--onnx-dir <path>` - Path to ONNX models (default: ../assets/onnx)
- `--save-dir <path>` - Output directory (default: results)
- `--output <file>` - Output WAV file (overrides --save-dir for single file)
- `--batch` - Use batch processing mode

### Available Languages

- `en` - English
- `ko` - Korean
- `es` - Spanish
- `pt` - Portuguese
- `fr` - French

### Available Voice Styles

The default installation includes several voice styles in `../assets/voice_styles/`:
- `M1.json` - Male voice 1
- `F1.json` - Female voice 1
- Additional voices may be available depending on your download

Use `./example_onnx --list-voices` to see all available voices on your system.

### Batch Mode Example

```bash
./example_onnx \
    --voice-style "../assets/voice_styles/M1.json,../assets/voice_styles/F1.json" \
    --text "Hello world|Bonjour le monde" \
    --lang "en,fr" \
    --batch
```

## Audiobook Generator

The `audiobook_generator` program converts text files into complete audiobooks with intelligent pause insertion and chapter handling.

### Features

- **Automatic Pause Insertion**: Natural pauses at punctuation marks (can be disabled)
- **Paragraph Detection**: Longer pauses for paragraph breaks
- **Custom Pause Directives**: `[PAUSE:ms]` for precise timing control
- **Sentence-by-Sentence Processing**: Memory-efficient handling of large texts
- **Real-time Progress**: Visual progress bar with statistics
- **Flexible Control**: Enable or disable automatic pauses

### Quick Start

```bash
# Basic usage with automatic pauses
./audiobook_generator --input story.txt

# Show help and all options
./audiobook_generator --help
```

### Command-line Options

- `--input <file>` - **Required**: Input text file to convert
- `--output <file>` - Output WAV file (default: audiobook.wav)
- `--voice <file>` - Voice style JSON (default: ../assets/voice_styles/M1.json)
- `--onnx-dir <dir>` - ONNX model directory (default: ../assets/onnx)
- `--lang <code>` - Language: en, ko, es, pt, fr (default: en)
- `--speed <float>` - Speech speed multiplier (default: 1.05, range: 0.5-2.0)
- `--steps <int>` - Inference steps for quality (default: 5, more = higher quality)
- `--no-auto-pause` - Disable automatic pause insertion
- `--help, -h` - Show comprehensive help message

### Automatic Pause Behavior

When automatic pauses are **enabled** (default):
- **Period/Exclamation/Question** (`.!?`): 500ms pause
- **Comma** (`,`): 250ms pause
- **Semicolon/Colon** (`;:`): 350ms pause
- **Paragraph breaks** (empty line): 800ms pause
- **Custom directives** (`[PAUSE:ms]`): Always work

When automatic pauses are **disabled** (`--no-auto-pause`):
- No automatic pauses at punctuation
- No automatic paragraph break pauses
- Custom `[PAUSE:ms]` directives still work
- Gives complete manual control

### When to Use `--no-auto-pause`

Use this option when:
- Text has non-standard punctuation usage
- You want complete control using only `[PAUSE:ms]` directives
- Text already has natural pacing
- Creating fast-paced narration without breaks
- Punctuation is used for formatting rather than pauses

### Usage Examples

**Basic audiobook with automatic pauses:**
```bash
./audiobook_generator --input story.txt --output story.wav
```

**Use female voice with faster speech:**
```bash
./audiobook_generator --input book.txt \
    --voice ../assets/voice_styles/F1.json \
    --speed 1.2
```

**Manual pause control only:**
```bash
./audiobook_generator --input script.txt --no-auto-pause
```

**Spanish audiobook with high quality:**
```bash
./audiobook_generator --input libro.txt \
    --lang es \
    --speed 1.0 \
    --steps 7 \
    --output libro.wav
```

### Text File Format

For best results, format your text file as follows:

**Paragraph Structure:**
```
This is the first paragraph. It contains multiple sentences.
Another sentence in the same paragraph.

This is a new paragraph after an empty line.
The generator will add an 800ms pause for the paragraph break.
```

**Custom Pauses:**
```
This sentence ends with a dramatic pause. [PAUSE:1500]
After 1.5 seconds, the story continues.

You can add pauses anywhere. [PAUSE:500] Even mid-sentence.
```

**Chapter Breaks:**
```
This is the end of chapter one.

[PAUSE:2000]

Chapter Two

The story continues after a 2-second pause.
```

### Tips for Best Results

1. **Clean Text**: Remove excessive formatting, headers, and page numbers
2. **One Paragraph Per Section**: Use natural paragraph breaks
3. **Double Line Breaks**: Use empty lines between logical sections
4. **Custom Pauses**: Add `[PAUSE:ms]` for dramatic effect or chapter breaks
5. **Voice Selection**: Try different voices (M1, F1) for different characters
6. **Speed Adjustment**: 1.0 = normal, 1.2 = faster, 0.9 = slower
7. **Quality Steps**: Use 5-7 steps for good quality, higher for critical productions

### Output Format

- **Format**: WAV (uncompressed audio)
- **Sample Rate**: 24,000 Hz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM

### Statistics Output

After generation, you'll see statistics including:
- Sentences processed
- Audio duration (seconds and minutes)
- Processing time
- Real-time factor (speed of generation)
- Pause mode (automatic or manual)
- Sample rate and file size

Example output:
```
=== Statistics ===
Sentences processed: 42 / 42
Audio duration: 125.34 seconds (2.09 minutes)
Processing time: 45.67 seconds
Real-time factor: 0.364x
Pause mode: Automatic + custom directives
Sample rate: 24000 Hz
Total samples: 3008160
File size: 5.73 MB
```

## Voice Builder

⚠️ **WARNING: EXPERIMENTAL TOOL WITH SIGNIFICANT LIMITATIONS**

The `voice_builder` tool is an **experimental demonstration** that uses simplified audio feature extraction. **It will NOT produce high-quality voice styles** and generated audio will likely be **distorted or of poor quality**.

### Why It Produces Poor Quality

The voice_builder uses basic audio statistics (energy, zero-crossing rate, windowed features) rather than deep learning embeddings. Real voice styles require:
- Trained neural network encoder models
- Proper voice embedding extraction
- Large datasets for training

The current implementation fills most of the 12,800+ features with pseudo-random values, which the TTS model interprets as noise, resulting in distorted speech.

### ⭐ Recommended Alternative

**For production-quality voice styles**, please use the official Voice Builder service:
- **URL**: https://supertonic.supertone.ai/voice_builder
- Provides high-quality voice embeddings
- Uses trained encoder models
- Produces clear, natural-sounding TTS output

### Features (Experimental)

- **WAV Input**: Supports 16-bit PCM, 8-bit PCM, and 32-bit float WAV files
- **Audio Analysis**: Extracts voice characteristics using spectral features
- **Style Generation**: Creates text-to-latent (50×256=12800) and duration predictor (8×16=128) style vectors
- **JSON Output**: Generates voice style files compatible with all Supertonic implementations
- **Cross-platform**: Works on macOS and Linux

### Quick Start

```bash
# Generate voice style from a WAV file
./voice_builder --input my_voice.wav

# Specify custom output filename
./voice_builder --input my_voice.wav --output my_custom_voice.json

# Show help
./voice_builder --help
```

### Command-line Options

- `--input <file>` - **Required**: Input WAV file containing voice audio
- `--output <file>` - Output JSON file (default: voice_style.json)
- `--help, -h` - Show help message with examples and requirements

### Audio Requirements

For best results, use audio that meets these requirements:

- **Format**: WAV (PCM 16-bit, 8-bit, or 32-bit float)
- **Sample Rate**: 16-24 kHz recommended (other rates supported)
- **Channels**: Mono or stereo
- **Duration**: At least 3-5 seconds of clear voice audio
- **Quality**: Clean recording with minimal background noise
- **Content**: Natural speech with varied intonation

### Usage Examples

**Basic voice style generation:**
```bash
./voice_builder --input recordings/my_voice.wav
```

**With custom output path:**
```bash
./voice_builder --input audio.wav --output ../assets/voice_styles/custom.json
```

**Using the generated voice style:**
```bash
# After generating the voice style, use it with example_onnx
./example_onnx --voice-style voice_style.json --text "Hello, this is my custom voice!"

# Or with audiobook generator
./audiobook_generator --input book.txt --voice voice_style.json --output audiobook.wav
```

### Output Format

The generated JSON file contains two main components:

1. **style_ttl**: Text-to-latent style features (50×256 = 12,800 elements)
   - Shape: [1, 50, 256]
   - Controls the voice characteristics during speech generation
   - Captures prosody, pitch patterns, and speaking style

2. **style_dp**: Duration predictor style features (8×16 = 128 elements)
   - Shape: [1, 8, 16]
   - Controls timing and rhythm of speech
   - Influences speaking rate and pause patterns

Example structure:
```json
{
  "style_ttl": {
    "dims": [1, 50, 256],
    "data": [0.012, -0.149, -0.140, ...]
  },
  "style_dp": {
    "dims": [1, 8, 16],
    "data": [0.074, -0.588, -0.037, ...]
  }
}
```

### Tips for Best Results

1. **Recording Quality**: Use a good microphone in a quiet environment
2. **Audio Content**: Record 5-10 seconds of natural, expressive speech
3. **Multiple Takes**: Try generating styles from different recordings and compare
4. **Testing**: Test generated styles with various texts to evaluate quality
5. **Sample Rate**: Match your TTS model's preferred sample rate (24 kHz for Supertonic)

### Technical Details

The voice builder performs the following steps:

1. **WAV Parsing**: Reads and validates the input WAV file
2. **Audio Analysis**: Extracts statistical features (energy, zero-crossing rate, spectral characteristics)
3. **Feature Engineering**: Generates windowed statistics across the audio
4. **Normalization**: Applies zero-mean, unit-variance normalization
5. **JSON Generation**: Formats features into the required JSON structure

### Limitations

**CRITICAL:** This tool is for **educational/experimental purposes only**:

- ❌ **Poor Audio Quality**: Generated voice styles produce distorted or unnatural speech
- ❌ **Not Production Ready**: Uses simplified statistics, not deep learning embeddings
- ❌ **Random Features**: ~99% of features are pseudo-random values, not real voice characteristics
- ❌ **Cannot Capture Voice Identity**: Basic audio stats cannot represent prosody, pitch patterns, or speaking style

**Technical Explanation (based on ONNX model analysis):**

The TTS system uses two types of style embeddings extracted by trained neural networks:

1. **style_ttl** (50×256 = 12,800 values)
   - Used in cross-attention layers (SpeechPromptedAttention)
   - Conditions text encoder on voice timbre, tone, prosody
   - **Should be**: Output from speech encoder network
     - 6 ConvNeXt blocks
     - 4 self-attention layers  
     - 2 cross-attention layers
   - **Currently**: Random values from audio statistics

2. **style_dp** (8×16 = 128 values)  
   - Concatenated with text features (64+128=192)
   - Predicts phoneme durations via MLP
   - Conditions on speaking rate and rhythm
   - **Should be**: Output from duration encoder network
     - 2 attention layers
     - 6 ConvNeXt blocks
   - **Currently**: Random values from audio statistics

**Why This Fails:**

Real encoders use:
- Mel-spectrogram extraction (FFT + mel filterbank)
- ConvNeXt blocks (depthwise conv, layer norm, GELU activation)
- Multi-head attention with relative positional embeddings
- Learned representations from thousands of voice samples

This C tool:
- Extracts ~20 basic statistics (energy, ZCR, windowed means)
- Generates ~12,780 pseudo-random values
- Cannot capture voice identity or prosodic patterns

**Proper Implementation Would Need:**
- ONNX Runtime integration (~500 KB library)
- Pre-trained encoder models (~50 MB)
- Mel-spectrogram computation code (~200 lines)
- Neural network inference infrastructure (~800+ lines)
- Total: ~1000+ lines of additional C code

### Production Alternative

For actual voice cloning and high-quality TTS:
- Use the official **[Voice Builder](https://supertonic.supertone.ai/voice_builder)** service
- Or use pre-trained voice styles from the assets directory
- The official service uses trained encoder models to extract proper voice embeddings

### Alternative Improvement Approaches (Q&A)

**Q: Could we use the ONNX models to improve voice_builder?**

**Option 1: Use an Audio Encoder Model**
- ❌ **Not Possible**: The ONNX assets include only synthesis models (vocoder, text_encoder, duration_predictor, vector_estimator)
- ❌ **Missing**: Audio encoder model that converts audio → style vectors
- ❌ **Why**: Encoder models require training and are not publicly available

**Option 2: Optimization-Based Approach (Simulated Annealing)**
- ⚠️ **Theoretically Possible but Impractical**

**Concept**: Start with an existing good voice.json, iteratively adjust to match target audio

**Algorithm**:
```
1. Load existing voice.json (e.g., M5.json) as starting point
2. Loop until convergence:
   a. Synthesize test phrase ("Hello, world!") with current style
   b. Extract features from synthesized + target audio (MFCC, spectral)
   c. Calculate distance/similarity metric
   d. Perturb style vectors (simulated annealing)
   e. Accept/reject based on temperature schedule
3. Output optimized voice.json
```

**Practical Challenges**:
- ⏱️ **Time**: Each iteration = 1-2 seconds TTS synthesis
  - Need 1000+ iterations → 30+ minutes minimum
- 🎯 **Search Space**: 12,928 dimensions (style_ttl: 12,800 + style_dp: 128)
  - High-dimensional optimization is very difficult
- 📉 **Local Minima**: Likely to converge to poor solutions
  - Simulated annealing helps but no guarantees
- 🎨 **Complexity**: Requires:
  - Feature extraction (MFCC, spectral analysis)
  - Distance metrics (DTW, cosine similarity)
  - Optimization algorithm (annealing schedule)
  - Audio comparison logic
  - ~500+ lines of additional code
- ❓ **Quality**: No guarantee of good results
  - May match features but sound unnatural
  - Voice identity is complex, not just spectral matching

**Verdict**: Theoretically feasible but:
- Too slow (30+ minutes per voice)
- Uncertain quality (may not improve over random baseline)
- High implementation complexity
- Better to just use official Voice Builder service (seconds, guaranteed quality)

**Why Official Service is Superior**:
```
Official Encoder:          Optimization Approach:
audio → [encoder] → style  audio → [iterative TTS] → style
  ~2 seconds                 ~30+ minutes
  Guaranteed quality         Uncertain quality
  Trained on data            Blind search
```

### Integration with Supertonic

Voice styles generated by this tool are compatible with:
- C implementation (this directory)
- Python implementation (`py/`)
- Node.js implementation (`nodejs/`)
- All other Supertonic TTS implementations

Simply place the generated JSON file in your voice_styles directory or reference it directly in your TTS calls.

## API Reference

### Core Structures

```c
// Configuration
typedef struct Config {
    AEConfig ae;
    TTLConfig ttl;
} Config;

// Unicode Processor
typedef struct UnicodeProcessor UnicodeProcessor;

// Voice Style
typedef struct Style Style;

// Text-to-Speech Engine
typedef struct TextToSpeech TextToSpeech;

// Synthesis Result
typedef struct SynthesisResult {
    float* wav;
    size_t wav_size;
    float* duration;
    size_t duration_count;
} SynthesisResult;
```

### Key Functions

#### Loading

```c
// Load configuration
Config* load_cfgs(const char* onnx_dir);

// Load text processor
UnicodeProcessor* load_text_processor(const char* onnx_dir);

// Load voice style
Style* load_voice_style(const char** paths, int count, int verbose);

// Load complete TTS system
TextToSpeech* load_text_to_speech(OrtEnv* env, const char* onnx_dir, int use_gpu);
```

#### Synthesis

```c
// Single-speaker synthesis with chunking
SynthesisResult* tts_call(
    TextToSpeech* tts,
    OrtMemoryInfo* memory_info,
    const char* text,
    const char* lang,
    const Style* style,
    int total_step,
    float speed,
    float silence_duration
);

// Batch synthesis
SynthesisResult* tts_batch(
    TextToSpeech* tts,
    OrtMemoryInfo* memory_info,
    const char** text_list,
    const char** lang_list,
    int batch_size,
    const Style* style,
    int total_step,
    float speed
);
```

#### Cleanup

```c
void unicode_processor_free(UnicodeProcessor* processor);
void style_free(Style* style);
void tts_free(TextToSpeech* tts);
void synthesis_result_free(SynthesisResult* result);
```

### WAV Utilities

```c
// Write WAV file
int write_wav_file(
    const char* filename,
    const float* audio_data,
    size_t audio_size,
    int sample_rate
);
```

## Example Code

```c
#include "supertonic.h"
#include "wav_utils.h"

int main() {
    // Initialize ONNX Runtime
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TTS", &env);
    
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    // Load TTS system
    TextToSpeech* tts = load_text_to_speech(env, "../assets/onnx", 0);
    
    // Load voice style
    const char* voice_paths[] = {"../assets/voice_styles/M1.json"};
    Style* style = load_voice_style(voice_paths, 1, 1);
    
    // Synthesize speech
    SynthesisResult* result = tts_call(
        tts, memory_info,
        "Hello, world!",
        "en",
        style,
        5,      // total_step
        1.05f,  // speed
        0.3f    // silence_duration
    );
    
    // Save to file
    write_wav_file("output.wav", result->wav, 
                   (size_t)(tts->sample_rate * result->duration[0]),
                   tts->sample_rate);
    
    // Cleanup
    synthesis_result_free(result);
    style_free(style);
    tts_free(tts);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseEnv(env);
    
    return 0;
}
```

## Differences from C++ Version

1. **Memory Management**: Manual memory allocation with `malloc/free` instead of RAII
2. **Error Handling**: Return codes and NULL checks instead of exceptions
3. **API Style**: C-style functions instead of classes and methods
4. **JSON Parsing**: cJSON library instead of nlohmann/json
5. **ONNX Runtime**: C API instead of C++ API
6. **String Handling**: Manual string manipulation instead of std::string

## Performance

The C implementation should have similar performance to the C++ version as both use:
- ONNX Runtime for inference
- Same model architecture
- Similar memory layout

## Limitations

- GPU support not yet implemented (CPU only)
- Limited Unicode normalization compared to full ICU library
- Simplified text chunking algorithm

## Can We Train an Encoder Model?

### The Question

"The official Voice Builder is expensive. Can we distill or retrain an encoder using the ONNX models?"

### Short Answer

✅ **YES**, it's technically feasible through **knowledge distillation**, but requires:
- Python ML framework (PyTorch recommended)
- GPU resources for training
- Hours to days of training time  
- ~850 lines of training code
- ML/audio processing expertise

This is a significant engineering project beyond the scope of this C implementation, but we provide a complete roadmap below.

### Knowledge Distillation Approach

**Core Idea:** Use the existing TTS models (which convert text+style → audio) to generate synthetic training data for an encoder model (which converts audio → style).

**Training Pipeline:**

```
Step 1: Generate Synthetic Training Data
-------------------------------------------
for i in range(10000):
    style_ttl = random_vector([1, 50, 256])
    style_dp = random_vector([1, 8, 16])
    
    audio = tts_synthesize(
        text="The quick brown fox jumps over the lazy dog",
        style_ttl=style_ttl,
        style_dp=style_dp
    )
    
    dataset.add(audio, style_ttl, style_dp)

Step 2: Train Encoder Model
----------------------------
encoder = EncoderModel()  # CNN or Transformer
optimizer = Adam(encoder.parameters())

for epoch in range(100):
    for audio, true_style_ttl, true_style_dp in dataloader:
        mel = extract_mel_spectrogram(audio)
        pred_ttl, pred_dp = encoder(mel)
        
        loss = mse_loss(pred_ttl, true_style_ttl) + \
               mse_loss(pred_dp, true_style_dp)
        
        loss.backward()
        optimizer.step()

Step 3: Export and Integrate
-----------------------------
torch.onnx.export(encoder, "audio_encoder.onnx")
# Integrate with C implementation using ONNX Runtime
```

### Implementation Roadmap

**Phase 1: Data Generation (~200 lines Python)**
```python
# data_generator.py
import torch
import onnxruntime as ort
import numpy as np

def generate_training_data(num_samples=10000):
    # Load TTS models
    tts = load_tts_models()
    
    # Generate diverse style vectors
    styles = generate_random_styles(num_samples)
    
    # Synthesize audio
    dataset = []
    for style_ttl, style_dp in styles:
        audio = tts.synthesize(
            text=random_text(),
            style_ttl=style_ttl,
            style_dp=style_dp
        )
        dataset.append((audio, style_ttl, style_dp))
    
    return dataset
```

**Phase 2: Encoder Architecture (~300 lines PyTorch)**
```python
# encoder_model.py
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Mel-spectrogram extractor
        self.mel_extractor = MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        # CNN backbone (or use Transformer)
        self.backbone = nn.Sequential(
            ConvBlock(80, 256, kernel=3),
            ConvBlock(256, 512, kernel=3),
            ConvBlock(512, 512, kernel=3),
        )
        
        # Style heads
        self.style_ttl_head = nn.Linear(512, 50*256)
        self.style_dp_head = nn.Linear(512, 8*16)
    
    def forward(self, audio):
        # Extract mel-spectrogram
        mel = self.mel_extractor(audio)  # [B, 80, T]
        
        # CNN encoding
        features = self.backbone(mel)  # [B, 512, T']
        
        # Global pooling
        pooled = features.mean(dim=2)  # [B, 512]
        
        # Predict styles
        style_ttl = self.style_ttl_head(pooled).view(-1, 1, 50, 256)
        style_dp = self.style_dp_head(pooled).view(-1, 1, 8, 16)
        
        return style_ttl, style_dp
```

**Phase 3: Training Loop (~200 lines)**
```python
# train.py
def train_encoder(encoder, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        encoder.train()
        train_loss = 0
        for audio, true_ttl, true_dp in train_loader:
            pred_ttl, pred_dp = encoder(audio)
            
            loss = F.mse_loss(pred_ttl, true_ttl) + \
                   F.mse_loss(pred_dp, true_dp)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = validate(encoder, val_loader)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), 'best_encoder.pth')
        
        scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

**Phase 4: Evaluation (~100 lines)**
```python
# evaluate.py
def evaluate_encoder(encoder, test_audio_files):
    """Test encoder on real voice recordings"""
    for audio_file in test_audio_files:
        # Encode real audio
        audio = load_audio(audio_file)
        style_ttl, style_dp = encoder(audio)
        
        # Synthesize with predicted style
        synth_audio = tts.synthesize(
            text="Hello, how are you?",
            style_ttl=style_ttl,
            style_dp=style_dp
        )
        
        # Save and evaluate
        save_audio(f"output_{audio_file}", synth_audio)
        
        # Metrics: perceptual similarity, speaker verification score
        similarity = compute_similarity(audio, synth_audio)
        print(f"{audio_file}: similarity={similarity:.3f}")
```

**Phase 5: ONNX Export (~50 lines)**
```python
# export_onnx.py
def export_to_onnx(encoder, output_path="audio_encoder.onnx"):
    encoder.eval()
    
    # Dummy input
    dummy_audio = torch.randn(1, 24000 * 5)  # 5 seconds
    
    # Export
    torch.onnx.export(
        encoder,
        dummy_audio,
        output_path,
        input_names=['audio'],
        output_names=['style_ttl', 'style_dp'],
        dynamic_axes={
            'audio': {1: 'audio_length'},
        },
        opset_version=14
    )
    
    print(f"Encoder exported to {output_path}")
```

**Phase 6: C Integration (~100 lines)**
```c
// encoder_inference.c
OrtSession* load_encoder(const char* model_path) {
    // Load ONNX encoder model
    // Similar to existing TTS model loading
}

void extract_voice_style(const char* wav_path, Style* style) {
    // 1. Load audio
    float* audio = load_wav_file(wav_path);
    
    // 2. Run encoder inference
    OrtValue* input = create_tensor(audio);
    OrtValue* outputs[2];
    
    OrtRun(encoder_session, input, outputs);
    
    // 3. Extract style vectors
    copy_tensor_data(outputs[0], style->ttl);
    copy_tensor_data(outputs[1], style->dp);
}
```

### Requirements

**Software:**
- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- librosa (audio processing)
- onnx, onnxruntime
- numpy, scipy

**Hardware:**
- GPU: NVIDIA with CUDA (4GB+ VRAM recommended)
- RAM: 16GB+ for dataset generation
- Storage: ~10GB for training data

**Time:**
- Data generation: 3-6 hours (10K samples)
- Training: 6-24 hours (depends on model size, GPU)
- Evaluation/tuning: Variable

**Expertise:**
- Machine learning (PyTorch, training loops)
- Audio processing (mel-spectrograms, feature extraction)
- Model optimization (hyperparameter tuning)

### Challenges and Solutions

**Challenge 1: Training Data Diversity**
- **Problem:** Random styles may not cover real voice variations
- **Solution:** 
  - Use stratified sampling (vary energy, pitch range, etc.)
  - Generate styles from existing voice.json files with perturbations
  - Include diverse text content (different phonemes, lengths)

**Challenge 2: Quality Validation**
- **Problem:** How to measure if encoder works well?
- **Solution:**
  - Reconstruction error on test set
  - Perceptual metrics (MCD - Mel Cepstral Distortion, STOI - Short-Time Objective Intelligibility)
  - Human evaluation on real voices
  - Speaker verification scores

**Challenge 3: Overfitting**
- **Problem:** Model memorizes training data
- **Solution:**
  - Data augmentation (noise, reverb, pitch shift)
  - Regularization (dropout, weight decay)
  - Early stopping on validation set
  - Large diverse dataset

**Challenge 4: Generalization to Real Voices**
- **Problem:** Trained on synthetic, tested on real
- **Solution:**
  - Fine-tune on small set of real voice pairs (if available)
  - Use domain adaptation techniques
  - Test extensively on diverse real recordings

**Challenge 5: Model Architecture**
- **Problem:** What architecture works best?
- **Solution:**
  - Start with proven models (Wav2Vec2, HuBERT architectures)
  - Experiment with CNN vs Transformer
  - Try different pooling strategies (mean, attention-based)
  - Ablation studies

### Suggested Model Architectures

**Option 1: CNN-based Encoder (Simpler, Faster)**
```
Input: Raw audio [B, T] or Mel-spectrogram [B, 80, T']
↓
Conv1D Blocks (3-5 layers, increasing channels)
↓
Global Average Pooling or Attention Pooling
↓
Fully Connected Layers
↓
Output: style_ttl [B, 1, 50, 256], style_dp [B, 1, 8, 16]
```

**Option 2: Transformer-based Encoder (More Powerful)**
```
Input: Mel-spectrogram [B, 80, T']
↓
Convolutional Stem (feature extraction)
↓
Transformer Encoder Blocks (6-12 layers)
↓
Attention Pooling (attend to important frames)
↓
Output Projection
↓
Output: style_ttl [B, 1, 50, 256], style_dp [B, 1, 8, 16]
```

**Option 3: Pre-trained Transfer Learning (Fastest)**
```
Use Wav2Vec2 or HuBERT pre-trained encoder
↓
Freeze most layers, fine-tune top layers
↓
Add projection heads for style outputs
↓
Train only projection heads on synthetic data
```

### Expected Results

**Best Case:**
- Encoder captures basic voice characteristics
- Generated voices sound similar to target (70-80% match)
- Works reasonably on diverse speakers
- Usable alternative to commercial service

**Realistic Case:**
- Encoder learns general voice features
- Quality varies by speaker
- Better than random baseline
- Requires per-speaker fine-tuning for best results

**Worst Case:**
- Encoder fails to generalize
- Output similar to random styles
- Needs more training data or different approach

### Cost-Benefit Analysis

**Training Encoder Once:**
- Time: 1-2 weeks (development + training)
- Cost: GPU hours ($50-200 on cloud)
- Expertise: ML/audio background needed
- Output: Reusable encoder model

**Using Official Service:**
- Time: Instant per voice
- Cost: $X per voice (ongoing)
- Expertise: None needed
- Output: High-quality styles

**Breakeven Point:**
If you need to clone N voices:
- Official: N × $price_per_voice
- Trained encoder: $fixed_training_cost + minimal_per_voice

Training becomes cost-effective if N is large enough.

### Community Contribution

**Call for Contributions:**

If someone in the community trains a high-quality encoder:
1. Share the trained ONNX model
2. Document training process and dataset size
3. Provide evaluation metrics
4. Create PR with C integration code

This would create an open-source alternative to the commercial Voice Builder service!

### Practical Recommendations

1. **Start Small**: Generate 1,000 samples first, train a small model, validate the approach
2. **Iterate**: Gradually increase dataset size and model complexity
3. **Measure**: Track metrics at each step to ensure improvement
4. **Share**: Contribute trained models back to the community
5. **Alternative**: Consider optimization-based approaches (documented in earlier section) as interim solution

### Conclusion

**Training an encoder is feasible but requires significant effort:**
- ✅ Technically sound approach (knowledge distillation)
- ✅ Complete implementation roadmap provided
- ⚠️ Weeks of development time
- ⚠️ Uncertain quality until tested
- ⚠️ Requires ML/audio expertise

**For most users:** Official Voice Builder is still the practical choice

**For advanced developers:** This documentation provides everything needed to implement an open-source alternative

**For the community:** Trained models can be shared to benefit everyone!

## Troubleshooting

### Library Not Found

If you get errors about missing libraries:

```bash
# macOS
export DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

### cJSON Not Found

Make sure cJSON is properly installed:

```bash
# Check if cJSON is installed
pkg-config --cflags --libs libcjson

# If not found, install or specify CJSON_ROOT
make CJSON_ROOT=/path/to/cjson
```

### ONNX Runtime Version

This implementation is tested with ONNX Runtime 1.16+. If you have issues, try:

```bash
# Check ONNX Runtime version
pkg-config --modversion onnxruntime
```

## License

Same as the parent Supertonic project.
