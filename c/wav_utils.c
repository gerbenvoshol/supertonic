#include "wav_utils.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int write_wav_file(const char* filename, const float* audio_data, size_t audio_size, int sample_rate) {
    if (!filename) {
        fprintf(stderr, "Error: filename is NULL\n");
        return -1;
    }
    if (!audio_data) {
        fprintf(stderr, "Error: audio_data is NULL\n");
        return -1;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }
    
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = audio_size * bits_per_sample / 8;
    
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
    for (size_t i = 0; i < audio_size; i++) {
        float sample = audio_data[i];
        /* Clamp to [-1.0, 1.0] */
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t int_sample = (int16_t)(sample * 32767);
        fwrite(&int_sample, 2, 1, file);
    }
    
    fclose(file);
    return 0;
}
