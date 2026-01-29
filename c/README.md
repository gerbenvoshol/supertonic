# Supertonic TTS - Pure C Implementation

This directory contains a pure C implementation of the Supertonic Text-to-Speech system, converted from the C++ version.

## Features

- **Pure C (C11)**: No C++ dependencies
- **ONNX Runtime C API**: Uses the native C API for model inference
- **cJSON**: For JSON parsing instead of nlohmann/json
- **Cross-platform**: Works on macOS and Linux
- **Memory-managed**: Proper allocation/deallocation with cleanup functions
- **Same functionality**: Maintains all features from the C++ version

## Files

- `supertonic.h` - Main header file with function declarations and structures
- `supertonic.c` - Core implementation of TTS system
- `wav_utils.h/c` - WAV file writing utilities
- `example_onnx.c` - Example program demonstrating usage
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

This will compile all source files and create both executables: `example_onnx` and `audiobook_generator`.

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
