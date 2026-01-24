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

2. **cJSON** - For JSON parsing
   ```bash
   # macOS (Homebrew)
   brew install cjson
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install libcjson-dev
   
   # Or build from source: https://github.com/DaveGamble/cJSON
   ```

## Building

### Default Build

```bash
make
```

This will compile all source files and create the `example_onnx` executable.

### Custom Library Paths

If your libraries are installed in non-standard locations:

```bash
make ONNXRUNTIME_ROOT=/path/to/onnxruntime CJSON_ROOT=/path/to/cjson
```

### Clean

```bash
make clean
```

## Usage

### Basic Usage

```bash
./example_onnx
```

This uses default settings:
- ONNX models from `../assets/onnx`
- Voice style from `../assets/voice_styles/M1.json`
- Default text in English
- 4 test iterations
- Output saved to `results/` directory

### Advanced Usage

```bash
./example_onnx \
    --onnx-dir ../assets/onnx \
    --voice-style "../assets/voice_styles/M1.json" \
    --text "Hello, this is a test." \
    --lang "en" \
    --total-step 5 \
    --speed 1.05 \
    --n-test 1 \
    --save-dir output
```

### Command-line Options

- `--onnx-dir <path>` - Path to ONNX models directory
- `--voice-style <path1,path2,...>` - Comma-separated voice style JSON files
- `--text <text1|text2|...>` - Pipe-separated text inputs
- `--lang <lang1,lang2,...>` - Comma-separated language codes (en, ko, es, pt, fr)
- `--total-step <n>` - Number of denoising steps (default: 5)
- `--speed <float>` - Speech speed factor (default: 1.05)
- `--n-test <n>` - Number of test iterations (default: 4)
- `--save-dir <path>` - Output directory (default: results)
- `--batch` - Use batch mode

### Batch Mode Example

```bash
./example_onnx \
    --voice-style "../assets/voice_styles/M1.json,../assets/voice_styles/F1.json" \
    --text "Hello world|Bonjour le monde" \
    --lang "en,fr" \
    --batch
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
