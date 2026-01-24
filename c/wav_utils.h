#ifndef WAV_UTILS_H
#define WAV_UTILS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int write_wav_file(const char* filename, const float* audio_data, size_t audio_size, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* WAV_UTILS_H */
