/*
 * Simple test WAV file generator
 * Generates a test WAV file with a simple sine wave
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_RATE 24000
#define DURATION 3.0
#define FREQUENCY 440.0

int main() {
    const char* filename = "test_voice.wav";
    int sample_rate = SAMPLE_RATE;
    float duration = DURATION;
    int num_samples = (int)(sample_rate * duration);
    
    /* Generate sine wave with varying amplitude (simulates voice) */
    float* audio = (float*)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        float t = (float)i / sample_rate;
        /* Mix multiple frequencies to simulate voice characteristics */
        float sample = 0.3f * sinf(2.0f * M_PI * FREQUENCY * t);
        sample += 0.2f * sinf(2.0f * M_PI * FREQUENCY * 1.5f * t);
        sample += 0.1f * sinf(2.0f * M_PI * FREQUENCY * 2.0f * t);
        /* Add amplitude modulation */
        float envelope = 0.5f + 0.5f * sinf(2.0f * M_PI * 3.0f * t);
        audio[i] = sample * envelope;
    }
    
    /* Write WAV file */
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(audio);
        return 1;
    }
    
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = num_samples * bits_per_sample / 8;
    
    /* RIFF header */
    fwrite("RIFF", 1, 4, file);
    int32_t chunk_size = 36 + data_size;
    fwrite(&chunk_size, 4, 1, file);
    fwrite("WAVE", 1, 4, file);
    
    /* fmt chunk */
    fwrite("fmt ", 1, 4, file);
    int32_t fmt_chunk_size = 16;
    fwrite(&fmt_chunk_size, 4, 1, file);
    int16_t audio_format = 1; /* PCM */
    fwrite(&audio_format, 2, 1, file);
    int16_t num_channels_16 = num_channels;
    fwrite(&num_channels_16, 2, 1, file);
    fwrite(&sample_rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file);
    int16_t block_align_16 = block_align;
    fwrite(&block_align_16, 2, 1, file);
    int16_t bits_per_sample_16 = bits_per_sample;
    fwrite(&bits_per_sample_16, 2, 1, file);
    
    /* data chunk */
    fwrite("data", 1, 4, file);
    fwrite(&data_size, 4, 1, file);
    
    /* Write audio data */
    for (int i = 0; i < num_samples; i++) {
        float sample = audio[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t int_sample = (int16_t)(sample * 32767);
        fwrite(&int_sample, 2, 1, file);
    }
    
    fclose(file);
    free(audio);
    
    printf("Generated test WAV file: %s\n", filename);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %.2f seconds\n", duration);
    printf("  Samples: %d\n", num_samples);
    
    return 0;
}
