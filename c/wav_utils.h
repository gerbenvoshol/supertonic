#ifndef WAV_UTILS_H
#define WAV_UTILS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* audio_data;
    size_t audio_size;
    int sample_rate;
    int num_channels;
} WavData;

int write_wav_file(const char* filename, const float* audio_data, size_t audio_size, int sample_rate);
WavData* read_wav_file(const char* filename);
void wav_data_free(WavData* data);

#ifdef __cplusplus
}
#endif

#endif /* WAV_UTILS_H */
