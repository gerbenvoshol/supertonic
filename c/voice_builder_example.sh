#!/bin/bash
# Example script demonstrating voice builder workflow

set -e

echo "============================================"
echo "Voice Builder Example Workflow"
echo "============================================"
echo ""

# Check if voice_builder exists
if [ ! -f "./voice_builder" ]; then
    echo "Error: voice_builder not found. Please run 'make voice_builder' first."
    exit 1
fi

echo "Step 1: Generate a test WAV file"
echo "---------------------------------"

# Create simple test WAV generator
cat > /tmp/generate_test_wav.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    const char* filename = "example_voice.wav";
    int sample_rate = 24000;
    float duration = 5.0;
    int num_samples = (int)(sample_rate * duration);
    
    float* audio = (float*)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        float t = (float)i / sample_rate;
        float sample = 0.3f * sinf(2.0f * M_PI * 220.0f * t);
        sample += 0.2f * sinf(2.0f * M_PI * 330.0f * t);
        sample += 0.15f * sinf(2.0f * M_PI * 440.0f * t);
        float envelope = 0.5f + 0.5f * sinf(2.0f * M_PI * 2.0f * t);
        audio[i] = sample * envelope * 0.8f;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        free(audio);
        return 1;
    }
    
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = num_samples * bits_per_sample / 8;
    
    fwrite("RIFF", 1, 4, file);
    int32_t chunk_size = 36 + data_size;
    fwrite(&chunk_size, 4, 1, file);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    int32_t fmt_chunk_size = 16;
    fwrite(&fmt_chunk_size, 4, 1, file);
    int16_t audio_format = 1;
    fwrite(&audio_format, 2, 1, file);
    int16_t num_channels_16 = num_channels;
    fwrite(&num_channels_16, 2, 1, file);
    fwrite(&sample_rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file);
    int16_t block_align_16 = block_align;
    fwrite(&block_align_16, 2, 1, file);
    int16_t bits_per_sample_16 = bits_per_sample;
    fwrite(&bits_per_sample_16, 2, 1, file);
    fwrite("data", 1, 4, file);
    fwrite(&data_size, 4, 1, file);
    
    for (int i = 0; i < num_samples; i++) {
        float sample = audio[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t int_sample = (int16_t)(sample * 32767);
        fwrite(&int_sample, 2, 1, file);
    }
    
    fclose(file);
    free(audio);
    
    printf("Generated: %s (%.1f seconds at %d Hz)\n", filename, duration, sample_rate);
    return 0;
}
EOF

gcc -std=c11 -O2 /tmp/generate_test_wav.c -o /tmp/generate_test_wav -lm
/tmp/generate_test_wav
echo ""

echo "Step 2: Generate voice style from WAV"
echo "--------------------------------------"
./voice_builder --input example_voice.wav --output example_voice_style.json
echo ""

echo "Step 3: View generated JSON structure"
echo "--------------------------------------"
echo "First 20 lines of generated JSON:"
head -20 example_voice_style.json
echo "..."
echo ""

echo "============================================"
echo "Complete! You now have:"
echo "  - example_voice.wav (input audio)"
echo "  - example_voice_style.json (voice style)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Record your own voice as a WAV file"
echo "  2. Run: ./voice_builder --input your_voice.wav"
echo "  3. Use the generated JSON with example_onnx:"
echo "     ./example_onnx --voice-style voice_style.json --text \"Hello!\""
echo ""
