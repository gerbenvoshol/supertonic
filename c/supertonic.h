#ifndef SUPERTONIC_H
#define SUPERTONIC_H

#include <stdint.h>
#include <stddef.h>
#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Available languages */
extern const char* AVAILABLE_LANGS[];
extern const int AVAILABLE_LANGS_COUNT;

/* Configuration structures */
typedef struct {
    int sample_rate;
    int base_chunk_size;
} AEConfig;

typedef struct {
    int chunk_compress_factor;
    int latent_dim;
} TTLConfig;

typedef struct {
    AEConfig ae;
    TTLConfig ttl;
} Config;

/* Unicode Processor */
typedef struct {
    int64_t* indexer;
    size_t indexer_size;
} UnicodeProcessor;

UnicodeProcessor* unicode_processor_create(const char* unicode_indexer_json_path);
void unicode_processor_free(UnicodeProcessor* processor);

int unicode_processor_call(
    UnicodeProcessor* processor,
    const char** text_list,
    const char** lang_list,
    int batch_size,
    int64_t*** text_ids_out,
    int* text_ids_rows,
    int* text_ids_cols,
    float**** text_mask_out,
    int* text_mask_batch,
    int* text_mask_channels,
    int* text_mask_len
);

/* Style structure */
typedef struct {
    float* ttl_data;
    int64_t* ttl_shape;
    size_t ttl_shape_len;
    size_t ttl_data_size;
    
    float* dp_data;
    int64_t* dp_shape;
    size_t dp_shape_len;
    size_t dp_data_size;
} Style;

Style* style_create(
    const float* ttl_data, size_t ttl_data_size, const int64_t* ttl_shape, size_t ttl_shape_len,
    const float* dp_data, size_t dp_data_size, const int64_t* dp_shape, size_t dp_shape_len
);
void style_free(Style* style);

/* Text-to-Speech structure */
typedef struct {
    Config cfgs;
    UnicodeProcessor* text_processor;
    OrtSession* dp_session;
    OrtSession* text_enc_session;
    OrtSession* vector_est_session;
    OrtSession* vocoder_session;
    int sample_rate;
    int base_chunk_size;
    int chunk_compress_factor;
    int ldim;
} TextToSpeech;

typedef struct {
    float* wav;
    size_t wav_size;
    float* duration;
    size_t duration_count;
} SynthesisResult;

TextToSpeech* tts_create(
    const Config* cfgs,
    UnicodeProcessor* text_processor,
    OrtSession* dp_session,
    OrtSession* text_enc_session,
    OrtSession* vector_est_session,
    OrtSession* vocoder_session
);

void tts_free(TextToSpeech* tts);

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

void synthesis_result_free(SynthesisResult* result);

/* Utility functions */
float*** length_to_mask(const int64_t* lengths, int count, int max_len);
float*** get_latent_mask(const int64_t* wav_lengths, int count, int base_chunk_size, int chunk_compress_factor);
void free_3d_float_array(float*** array, int dim1, int dim2);
void free_2d_int64_array(int64_t** array, int rows);

/* ONNX model loading */
typedef struct {
    OrtSession* dp;
    OrtSession* text_enc;
    OrtSession* vector_est;
    OrtSession* vocoder;
} OnnxModels;

OrtSession* load_onnx(OrtEnv* env, const char* onnx_path, const OrtSessionOptions* opts);
OnnxModels* load_onnx_all(OrtEnv* env, const char* onnx_dir, const OrtSessionOptions* opts);
void onnx_models_free(OnnxModels* models);

/* Configuration and processor loading */
Config* load_cfgs(const char* onnx_dir);
UnicodeProcessor* load_text_processor(const char* onnx_dir);

/* Voice style loading */
Style* load_voice_style(const char** voice_style_paths, int count, int verbose);

/* TextToSpeech loading */
TextToSpeech* load_text_to_speech(OrtEnv* env, const char* onnx_dir, int use_gpu);

/* Tensor creation utilities */
OrtValue* array_to_tensor_3d(
    OrtMemoryInfo* memory_info,
    float*** array,
    int64_t dim0, int64_t dim1, int64_t dim2
);

OrtValue* int_array_to_tensor_2d(
    OrtMemoryInfo* memory_info,
    int64_t** array,
    int64_t dim0, int64_t dim1
);

/* String utilities */
char* sanitize_filename(const char* text, int max_len);
char** chunk_text(const char* text, int max_len, int* chunk_count);
void free_string_array(char** array, int count);

#ifdef __cplusplus
}
#endif

#endif /* SUPERTONIC_H */
