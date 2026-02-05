#include "wav_utils.h"
#include <stdio.h>
#include <stdlib.h>
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

WavData* read_wav_file(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: filename is NULL\n");
        return NULL;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return NULL;
    }
    
    /* Read RIFF header */
    char riff[4];
    if (fread(riff, 1, 4, file) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "Error: Not a valid WAV file (missing RIFF header)\n");
        fclose(file);
        return NULL;
    }
    
    int32_t chunk_size;
    fread(&chunk_size, 4, 1, file);
    
    char wave[4];
    if (fread(wave, 1, 4, file) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: Not a valid WAV file (missing WAVE header)\n");
        fclose(file);
        return NULL;
    }
    
    /* Find fmt chunk */
    char fmt[4];
    int32_t fmt_size = 0;
    int16_t audio_format = 0;
    int16_t num_channels = 0;
    int32_t sample_rate = 0;
    int32_t byte_rate = 0;
    int16_t block_align = 0;
    int16_t bits_per_sample = 0;
    
    while (1) {
        if (fread(fmt, 1, 4, file) != 4) {
            fprintf(stderr, "Error: Could not find fmt chunk\n");
            fclose(file);
            return NULL;
        }
        
        fread(&fmt_size, 4, 1, file);
        
        if (memcmp(fmt, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, file);
            fread(&num_channels, 2, 1, file);
            fread(&sample_rate, 4, 1, file);
            fread(&byte_rate, 4, 1, file);
            fread(&block_align, 2, 1, file);
            fread(&bits_per_sample, 2, 1, file);
            
            /* Skip any extra format bytes */
            if (fmt_size > 16) {
                fseek(file, fmt_size - 16, SEEK_CUR);
            }
            break;
        } else {
            /* Skip this chunk */
            fseek(file, fmt_size, SEEK_CUR);
        }
    }
    
    /* Find data chunk */
    char data[4];
    int32_t data_size = 0;
    
    while (1) {
        if (fread(data, 1, 4, file) != 4) {
            fprintf(stderr, "Error: Could not find data chunk\n");
            fclose(file);
            return NULL;
        }
        
        fread(&data_size, 4, 1, file);
        
        if (memcmp(data, "data", 4) == 0) {
            break;
        } else {
            /* Skip this chunk */
            fseek(file, data_size, SEEK_CUR);
        }
    }
    
    /* Allocate result structure */
    WavData* wav_data = (WavData*)malloc(sizeof(WavData));
    if (!wav_data) {
        fprintf(stderr, "Error: Failed to allocate WavData structure\n");
        fclose(file);
        return NULL;
    }
    
    wav_data->sample_rate = sample_rate;
    wav_data->num_channels = num_channels;
    
    /* Calculate number of samples */
    size_t num_samples = data_size / (bits_per_sample / 8);
    wav_data->audio_size = num_samples;
    
    /* Allocate audio buffer */
    wav_data->audio_data = (float*)malloc(num_samples * sizeof(float));
    if (!wav_data->audio_data) {
        fprintf(stderr, "Error: Failed to allocate audio buffer\n");
        free(wav_data);
        fclose(file);
        return NULL;
    }
    
    /* Read and convert audio data based on format */
    if (bits_per_sample == 16 && audio_format == 1) {
        /* 16-bit PCM */
        for (size_t i = 0; i < num_samples; i++) {
            int16_t sample;
            if (fread(&sample, 2, 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read audio sample\n");
                free(wav_data->audio_data);
                free(wav_data);
                fclose(file);
                return NULL;
            }
            wav_data->audio_data[i] = sample / 32768.0f;
        }
    } else if (bits_per_sample == 8 && audio_format == 1) {
        /* 8-bit PCM */
        for (size_t i = 0; i < num_samples; i++) {
            uint8_t sample;
            if (fread(&sample, 1, 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read audio sample\n");
                free(wav_data->audio_data);
                free(wav_data);
                fclose(file);
                return NULL;
            }
            wav_data->audio_data[i] = (sample - 128) / 128.0f;
        }
    } else if (bits_per_sample == 32 && audio_format == 3) {
        /* 32-bit float */
        if (fread(wav_data->audio_data, sizeof(float), num_samples, file) != num_samples) {
            fprintf(stderr, "Error: Failed to read audio data\n");
            free(wav_data->audio_data);
            free(wav_data);
            fclose(file);
            return NULL;
        }
    } else {
        fprintf(stderr, "Error: Unsupported audio format (format=%d, bits=%d)\n", 
                audio_format, bits_per_sample);
        free(wav_data->audio_data);
        free(wav_data);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    
    printf("Loaded WAV file: %s\n", filename);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Channels: %d\n", num_channels);
    printf("  Bits per sample: %d\n", bits_per_sample);
    printf("  Audio samples: %zu\n", num_samples);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate / num_channels);
    
    return wav_data;
}

void wav_data_free(WavData* data) {
    if (data) {
        if (data->audio_data) {
            free(data->audio_data);
        }
        free(data);
    }
}
