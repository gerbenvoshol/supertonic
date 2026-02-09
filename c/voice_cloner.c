/*
 * Voice Cloner for Supertonic TTS - On-Device Implementation
 * 
 * This tool loads an audio encoder ONNX model and extracts voice style embeddings
 * from input audio files. The embeddings can then be used with Supertonic TTS
 * to clone the voice characteristics of the speaker.
 * 
 * TECHNICAL APPROACH:
 * ==================
 * 
 * 1. Audio Loading: Read WAV file using wav_utils.h
 * 2. Mel-Spectrogram Computation:
 *    - Apply Hann window
 *    - Compute FFT (radix-2 Cooley-Tukey)
 *    - Calculate power spectrum
 *    - Apply mel filterbank (80 mels, 0-8000 Hz)
 *    - Log scaling
 * 3. ONNX Inference: Run audio_encoder.onnx model via ONNX Runtime C API
 * 4. JSON Export: Save voice style JSON using cJSON
 * 
 * Usage:
 *   voice_cloner --input audio.wav --encoder model.onnx --output style.json --name "MyVoice"
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>
#include "wav_utils.h"
#include "vendor/cjson/cJSON.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Mel-spectrogram configuration */
#define N_FFT 1024
#define HOP_LENGTH 256
#define N_MELS 80
#define SAMPLE_RATE 22050
#define MEL_FMIN 0.0f
#define MEL_FMAX 8000.0f

/* Command-line arguments */
typedef struct {
    char* input_wav;
    char* encoder_model;
    char* output_json;
    char* voice_name;
    int show_help;
} Args;

/* FFT Implementation - Radix-2 Cooley-Tukey */
typedef struct {
    float real;
    float imag;
} Complex;

static void fft_radix2(Complex* data, int n, int inverse) {
    if (n <= 1) return;
    
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            Complex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
    
    /* Cooley-Tukey FFT */
    for (int len = 2; len <= n; len *= 2) {
        float angle = 2.0f * M_PI / len * (inverse ? 1 : -1);
        Complex wlen = {cosf(angle), sinf(angle)};
        
        for (int i = 0; i < n; i += len) {
            Complex w = {1.0f, 0.0f};
            for (int j = 0; j < len / 2; j++) {
                Complex u = data[i + j];
                Complex v = {
                    data[i + j + len/2].real * w.real - data[i + j + len/2].imag * w.imag,
                    data[i + j + len/2].real * w.imag + data[i + j + len/2].imag * w.real
                };
                
                data[i + j].real = u.real + v.real;
                data[i + j].imag = u.imag + v.imag;
                data[i + j + len/2].real = u.real - v.real;
                data[i + j + len/2].imag = u.imag - v.imag;
                
                float w_temp = w.real * wlen.real - w.imag * wlen.imag;
                w.imag = w.real * wlen.imag + w.imag * wlen.real;
                w.real = w_temp;
            }
        }
    }
    
    if (inverse) {
        for (int i = 0; i < n; i++) {
            data[i].real /= n;
            data[i].imag /= n;
        }
    }
}

/* Hann window */
static void apply_hann_window(float* signal, int n) {
    for (int i = 0; i < n; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
        signal[i] *= window;
    }
}

/* Convert Hz to Mel scale */
static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/* Convert Mel to Hz scale */
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* Create mel filterbank */
static float** create_mel_filterbank(int n_mels, int n_fft, int sample_rate, float fmin, float fmax) {
    float** filterbank = (float**)malloc(n_mels * sizeof(float*));
    for (int i = 0; i < n_mels; i++) {
        filterbank[i] = (float*)calloc(n_fft / 2 + 1, sizeof(float));
    }
    
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);
    
    /* Create mel points */
    float* mel_points = (float*)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    /* Convert mel points to Hz and then to FFT bins */
    int* bin_points = (int*)malloc((n_mels + 2) * sizeof(int));
    for (int i = 0; i < n_mels + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)((n_fft + 1) * hz / sample_rate);
    }
    
    /* Create triangular filters */
    for (int i = 0; i < n_mels; i++) {
        int left = bin_points[i];
        int center = bin_points[i + 1];
        int right = bin_points[i + 2];
        
        for (int j = left; j < center; j++) {
            filterbank[i][j] = (float)(j - left) / (center - left);
        }
        for (int j = center; j < right; j++) {
            filterbank[i][j] = (float)(right - j) / (right - center);
        }
    }
    
    free(mel_points);
    free(bin_points);
    
    return filterbank;
}

/* Free mel filterbank */
static void free_mel_filterbank(float** filterbank, int n_mels) {
    for (int i = 0; i < n_mels; i++) {
        free(filterbank[i]);
    }
    free(filterbank);
}

/* Compute mel-spectrogram from audio */
static float* compute_mel_spectrogram(const float* audio, size_t audio_len, 
                                       int* n_frames_out, int* n_mels_out) {
    *n_mels_out = N_MELS;
    
    /* Calculate number of frames */
    int n_frames = (audio_len - N_FFT) / HOP_LENGTH + 1;
    if (n_frames <= 0) n_frames = 1;
    *n_frames_out = n_frames;
    
    /* Allocate mel spectrogram */
    float* mel_spec = (float*)calloc(n_frames * N_MELS, sizeof(float));
    if (!mel_spec) {
        fprintf(stderr, "Error: Failed to allocate mel spectrogram\n");
        return NULL;
    }
    
    /* Create mel filterbank */
    float** filterbank = create_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, MEL_FMIN, MEL_FMAX);
    
    /* Process each frame */
    Complex* fft_buffer = (Complex*)malloc(N_FFT * sizeof(Complex));
    float* frame = (float*)malloc(N_FFT * sizeof(float));
    
    for (int i = 0; i < n_frames; i++) {
        int frame_start = i * HOP_LENGTH;
        
        /* Extract frame */
        for (int j = 0; j < N_FFT; j++) {
            if (frame_start + j < audio_len) {
                frame[j] = audio[frame_start + j];
            } else {
                frame[j] = 0.0f;
            }
        }
        
        /* Apply Hann window */
        apply_hann_window(frame, N_FFT);
        
        /* Prepare FFT input */
        for (int j = 0; j < N_FFT; j++) {
            fft_buffer[j].real = frame[j];
            fft_buffer[j].imag = 0.0f;
        }
        
        /* Compute FFT */
        fft_radix2(fft_buffer, N_FFT, 0);
        
        /* Compute power spectrum */
        float* power_spec = (float*)malloc((N_FFT / 2 + 1) * sizeof(float));
        for (int j = 0; j < N_FFT / 2 + 1; j++) {
            power_spec[j] = fft_buffer[j].real * fft_buffer[j].real + 
                           fft_buffer[j].imag * fft_buffer[j].imag;
        }
        
        /* Apply mel filterbank */
        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            for (int j = 0; j < N_FFT / 2 + 1; j++) {
                mel_energy += power_spec[j] * filterbank[m][j];
            }
            /* Log scaling with small epsilon to avoid log(0) */
            mel_spec[i * N_MELS + m] = logf(mel_energy + 1e-10f);
        }
        
        free(power_spec);
    }
    
    free(fft_buffer);
    free(frame);
    free_mel_filterbank(filterbank, N_MELS);
    
    return mel_spec;
}

/* Load ONNX model and run inference */
static int run_encoder_inference(const char* model_path, float* mel_spec, 
                                  int n_frames, int n_mels,
                                  float** style_ttl_out, int* ttl_size_out,
                                  float** style_dp_out, int* dp_size_out) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    /* Create ONNX Runtime environment */
    OrtEnv* env;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "VoiceCloner", &env);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create ONNX Runtime environment: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    /* Create session options */
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create session options: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
    
    /* Load model */
    OrtSession* session;
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to load ONNX model: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    printf("✓ ONNX model loaded successfully\n");
    
    /* Create memory info */
    OrtMemoryInfo* memory_info;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create memory info: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    /* Create input tensor [1, n_mels, n_frames] */
    int64_t input_shape[] = {1, n_mels, n_frames};
    OrtValue* input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, mel_spec, n_frames * n_mels * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create input tensor: %s\n", 
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    /* Run inference */
    const char* input_names[] = {"mel_spec"};
    const char* output_names[] = {"style_ttl", "style_dp"};
    OrtValue* output_tensors[2] = {NULL, NULL};
    
    printf("Running encoder inference...\n");
    status = g_ort->Run(session, NULL, input_names, 
                       (const OrtValue* const*)&input_tensor, 1,
                       output_names, 2, output_tensors);
    
    if (status != NULL) {
        fprintf(stderr, "Error: Inference failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    printf("✓ Inference completed successfully\n");
    
    /* Extract output tensors */
    float* ttl_data;
    float* dp_data;
    
    status = g_ort->GetTensorMutableData(output_tensors[0], (void**)&ttl_data);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get TTL output data\n");
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(output_tensors[0]);
        g_ort->ReleaseValue(output_tensors[1]);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    status = g_ort->GetTensorMutableData(output_tensors[1], (void**)&dp_data);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get DP output data\n");
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(output_tensors[0]);
        g_ort->ReleaseValue(output_tensors[1]);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    
    /* Get tensor shapes */
    OrtTensorTypeAndShapeInfo* ttl_info;
    OrtTensorTypeAndShapeInfo* dp_info;
    
    g_ort->GetTensorTypeAndShape(output_tensors[0], &ttl_info);
    g_ort->GetTensorTypeAndShape(output_tensors[1], &dp_info);
    
    size_t ttl_dims;
    size_t dp_dims;
    g_ort->GetDimensionsCount(ttl_info, &ttl_dims);
    g_ort->GetDimensionsCount(dp_info, &dp_dims);
    
    /* Calculate total elements */
    int ttl_size = 1;
    int dp_size = 1;
    
    int64_t* ttl_shape = (int64_t*)malloc(ttl_dims * sizeof(int64_t));
    int64_t* dp_shape = (int64_t*)malloc(dp_dims * sizeof(int64_t));
    
    g_ort->GetDimensions(ttl_info, ttl_shape, ttl_dims);
    g_ort->GetDimensions(dp_info, dp_shape, dp_dims);
    
    for (size_t i = 0; i < ttl_dims; i++) ttl_size *= ttl_shape[i];
    for (size_t i = 0; i < dp_dims; i++) dp_size *= dp_shape[i];
    
    free(ttl_shape);
    free(dp_shape);
    
    /* Copy output data */
    *style_ttl_out = (float*)malloc(ttl_size * sizeof(float));
    *style_dp_out = (float*)malloc(dp_size * sizeof(float));
    
    memcpy(*style_ttl_out, ttl_data, ttl_size * sizeof(float));
    memcpy(*style_dp_out, dp_data, dp_size * sizeof(float));
    
    *ttl_size_out = ttl_size;
    *dp_size_out = dp_size;
    
    /* Cleanup */
    g_ort->ReleaseTensorTypeAndShapeInfo(ttl_info);
    g_ort->ReleaseTensorTypeAndShapeInfo(dp_info);
    g_ort->ReleaseValue(output_tensors[0]);
    g_ort->ReleaseValue(output_tensors[1]);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    
    return 0;
}

/* Save voice style to JSON */
static int save_voice_style_json(const char* output_path, const char* voice_name,
                                  const float* style_ttl, int ttl_size,
                                  const float* style_dp, int dp_size) {
    printf("\nGenerating voice style JSON: %s\n", output_path);
    
    cJSON* root = cJSON_CreateObject();
    if (!root) {
        fprintf(stderr, "Error: Failed to create JSON object\n");
        return -1;
    }
    
    /* Add voice name if provided */
    if (voice_name) {
        cJSON_AddStringToObject(root, "name", voice_name);
    }
    
    /* Create style_ttl object */
    cJSON* ttl_obj = cJSON_CreateObject();
    
    /* Assume shape [1, 50, 256] for TTL */
    int ttl_dims[] = {1, 50, 256};
    cJSON* ttl_dims_arr = cJSON_CreateIntArray(ttl_dims, 3);
    cJSON_AddItemToObject(ttl_obj, "dims", ttl_dims_arr);
    
    cJSON* ttl_data_arr = cJSON_CreateFloatArray(style_ttl, ttl_size);
    cJSON_AddItemToObject(ttl_obj, "data", ttl_data_arr);
    
    cJSON_AddItemToObject(root, "style_ttl", ttl_obj);
    
    /* Create style_dp object */
    cJSON* dp_obj = cJSON_CreateObject();
    
    /* Assume shape [1, 8, 16] for DP */
    int dp_dims[] = {1, 8, 16};
    cJSON* dp_dims_arr = cJSON_CreateIntArray(dp_dims, 3);
    cJSON_AddItemToObject(dp_obj, "dims", dp_dims_arr);
    
    cJSON* dp_data_arr = cJSON_CreateFloatArray(style_dp, dp_size);
    cJSON_AddItemToObject(dp_obj, "data", dp_data_arr);
    
    cJSON_AddItemToObject(root, "style_dp", dp_obj);
    
    /* Write to file */
    char* json_str = cJSON_Print(root);
    if (!json_str) {
        fprintf(stderr, "Error: Failed to generate JSON string\n");
        cJSON_Delete(root);
        return -1;
    }
    
    FILE* file = fopen(output_path, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open output file: %s\n", output_path);
        free(json_str);
        cJSON_Delete(root);
        return -1;
    }
    
    fprintf(file, "%s", json_str);
    fclose(file);
    
    free(json_str);
    cJSON_Delete(root);
    
    printf("✓ Voice style JSON saved successfully\n");
    return 0;
}

/* Argument parsing */
static void print_help(const char* prog_name) {
    printf("\n");
    printf("Voice Cloner for Supertonic TTS\n");
    printf("================================\n\n");
    printf("Extract voice style embeddings from audio using ONNX encoder model.\n\n");
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Required arguments:\n");
    printf("  --input <file>     Input WAV file (16kHz or 22.05kHz recommended)\n");
    printf("  --encoder <file>   Path to audio_encoder.onnx model\n\n");
    printf("Optional arguments:\n");
    printf("  --output <file>    Output JSON file (default: voice_style.json)\n");
    printf("  --name <name>      Voice name for the style\n");
    printf("  --help, -h         Show this help message\n\n");
    printf("Audio requirements:\n");
    printf("  - Format: WAV (PCM 16-bit, 8-bit, or 32-bit float)\n");
    printf("  - Sample rate: 22050 Hz recommended (will resample if different)\n");
    printf("  - Duration: At least 3-5 seconds of clear voice audio\n");
    printf("  - Quality: Clean recording with minimal background noise\n\n");
    printf("Examples:\n");
    printf("  # Basic usage\n");
    printf("  %s --input voice.wav --encoder audio_encoder.onnx\n\n", prog_name);
    printf("  # Specify output and name\n");
    printf("  %s --input voice.wav --encoder model.onnx --output my_voice.json --name \"John\"\n\n", prog_name);
    printf("Output:\n");
    printf("  The generated JSON file contains voice style embeddings that can be used\n");
    printf("  with Supertonic TTS for voice cloning:\n\n");
    printf("    ./example_onnx --voice-style my_voice.json --text \"Hello, world!\"\n\n");
}

static Args parse_args(int argc, char* argv[]) {
    Args args = {0};
    args.output_json = "voice_style.json";
    
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)) {
            args.show_help = 1;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input_wav = argv[++i];
        } else if (strcmp(argv[i], "--encoder") == 0 && i + 1 < argc) {
            args.encoder_model = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output_json = argv[++i];
        } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            args.voice_name = argv[++i];
        } else {
            fprintf(stderr, "Warning: Unknown argument '%s'\n", argv[i]);
        }
    }
    
    return args;
}

/* Main */
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.show_help) {
        print_help(argv[0]);
        return 0;
    }
    
    /* Validate required arguments */
    if (!args.input_wav) {
        fprintf(stderr, "Error: --input argument is required\n");
        fprintf(stderr, "Use --help for usage information\n");
        return 1;
    }
    
    if (!args.encoder_model) {
        fprintf(stderr, "Error: --encoder argument is required\n");
        fprintf(stderr, "Use --help for usage information\n");
        return 1;
    }
    
    printf("\n========================================\n");
    printf("Voice Cloner for Supertonic TTS\n");
    printf("========================================\n\n");
    
    printf("Configuration:\n");
    printf("  Input WAV:     %s\n", args.input_wav);
    printf("  Encoder model: %s\n", args.encoder_model);
    printf("  Output JSON:   %s\n", args.output_json);
    if (args.voice_name) {
        printf("  Voice name:    %s\n", args.voice_name);
    }
    printf("\n");
    
    /* Load WAV file */
    printf("Loading input audio...\n");
    WavData* wav_data = read_wav_file(args.input_wav);
    if (!wav_data) {
        fprintf(stderr, "Failed to load WAV file\n");
        return 1;
    }
    printf("✓ Audio loaded successfully\n");
    
    /* Validate audio duration */
    float duration = (float)wav_data->audio_size / wav_data->sample_rate / wav_data->num_channels;
    if (duration < 2.0f) {
        fprintf(stderr, "\nWarning: Audio duration (%.2f seconds) is short.\n", duration);
        fprintf(stderr, "For best results, use at least 3-5 seconds of clear voice audio.\n\n");
    }
    
    /* Convert stereo to mono if needed */
    float* mono_audio = wav_data->audio_data;
    size_t mono_len = wav_data->audio_size;
    
    if (wav_data->num_channels > 1) {
        printf("Converting stereo to mono...\n");
        mono_len = wav_data->audio_size / wav_data->num_channels;
        mono_audio = (float*)malloc(mono_len * sizeof(float));
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < wav_data->num_channels; ch++) {
                sum += wav_data->audio_data[i * wav_data->num_channels + ch];
            }
            mono_audio[i] = sum / wav_data->num_channels;
        }
    }
    
    /* Compute mel-spectrogram */
    printf("\nComputing mel-spectrogram...\n");
    printf("  FFT size: %d\n", N_FFT);
    printf("  Hop length: %d\n", HOP_LENGTH);
    printf("  Mel bins: %d\n", N_MELS);
    printf("  Frequency range: %.0f - %.0f Hz\n", MEL_FMIN, MEL_FMAX);
    
    int n_frames, n_mels;
    float* mel_spec = compute_mel_spectrogram(mono_audio, mono_len, &n_frames, &n_mels);
    
    if (wav_data->num_channels > 1) {
        free(mono_audio);
    }
    wav_data_free(wav_data);
    
    if (!mel_spec) {
        fprintf(stderr, "Failed to compute mel-spectrogram\n");
        return 1;
    }
    
    printf("✓ Mel-spectrogram computed: %d frames x %d mels\n", n_frames, n_mels);
    
    /* Run encoder inference */
    printf("\nLoading encoder model...\n");
    float* style_ttl = NULL;
    float* style_dp = NULL;
    int ttl_size, dp_size;
    
    int result = run_encoder_inference(args.encoder_model, mel_spec, n_frames, n_mels,
                                        &style_ttl, &ttl_size, &style_dp, &dp_size);
    
    free(mel_spec);
    
    if (result != 0) {
        fprintf(stderr, "Failed to run encoder inference\n");
        return 1;
    }
    
    printf("✓ Voice embeddings extracted:\n");
    printf("  Style TTL: %d elements\n", ttl_size);
    printf("  Style DP:  %d elements\n", dp_size);
    
    /* Save to JSON */
    result = save_voice_style_json(args.output_json, args.voice_name,
                                    style_ttl, ttl_size, style_dp, dp_size);
    
    free(style_ttl);
    free(style_dp);
    
    if (result != 0) {
        fprintf(stderr, "Failed to save voice style JSON\n");
        return 1;
    }
    
    printf("\n========================================\n");
    printf("Voice cloning completed successfully!\n");
    printf("========================================\n\n");
    printf("Output: %s\n\n", args.output_json);
    printf("Usage with Supertonic TTS:\n");
    printf("  ./example_onnx --voice-style %s --text \"Your text here\"\n\n", args.output_json);
    
    return 0;
}
