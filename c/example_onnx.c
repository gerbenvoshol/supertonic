#define _POSIX_C_SOURCE 200809L
#include "supertonic.h"
#include "wav_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    char* onnx_dir;
    int total_step;
    float speed;
    int n_test;
    char** voice_style;
    int voice_style_count;
    char** text;
    int text_count;
    char** lang;
    int lang_count;
    char* save_dir;
    int batch;
} Args;

static void free_args(Args* args) {
    free(args->onnx_dir);
    for (int i = 0; i < args->voice_style_count; i++) {
        free(args->voice_style[i]);
    }
    free(args->voice_style);
    for (int i = 0; i < args->text_count; i++) {
        free(args->text[i]);
    }
    free(args->text);
    for (int i = 0; i < args->lang_count; i++) {
        free(args->lang[i]);
    }
    free(args->lang);
    free(args->save_dir);
}

static char** split_string(const char* str, char delim, int* count) {
    int n = 1;
    for (const char* p = str; *p; p++) {
        if (*p == delim) n++;
    }
    
    char** result = malloc(n * sizeof(char*));
    int idx = 0;
    const char* start = str;
    const char* end;
    
    while ((end = strchr(start, delim)) != NULL) {
        size_t len = end - start;
        result[idx] = malloc(len + 1);
        memcpy(result[idx], start, len);
        result[idx][len] = 0;
        idx++;
        start = end + 1;
    }
    
    result[idx] = strdup(start);
    *count = n;
    return result;
}

static Args parse_args(int argc, char* argv[]) {
    Args args = {0};
    args.onnx_dir = strdup("../assets/onnx");
    args.total_step = 5;
    args.speed = 1.05f;
    args.n_test = 4;
    args.voice_style = malloc(sizeof(char*));
    args.voice_style[0] = strdup("../assets/voice_styles/M1.json");
    args.voice_style_count = 1;
    args.text = malloc(sizeof(char*));
    args.text[0] = strdup("This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen.");
    args.text_count = 1;
    args.lang = malloc(sizeof(char*));
    args.lang[0] = strdup("en");
    args.lang_count = 1;
    args.save_dir = strdup("results");
    args.batch = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--onnx-dir") == 0 && i + 1 < argc) {
            free(args.onnx_dir);
            args.onnx_dir = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--total-step") == 0 && i + 1 < argc) {
            args.total_step = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--speed") == 0 && i + 1 < argc) {
            args.speed = atof(argv[++i]);
        } else if (strcmp(argv[i], "--n-test") == 0 && i + 1 < argc) {
            args.n_test = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--voice-style") == 0 && i + 1 < argc) {
            for (int j = 0; j < args.voice_style_count; j++) {
                free(args.voice_style[j]);
            }
            free(args.voice_style);
            args.voice_style = split_string(argv[++i], ',', &args.voice_style_count);
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            for (int j = 0; j < args.text_count; j++) {
                free(args.text[j]);
            }
            free(args.text);
            args.text = split_string(argv[++i], '|', &args.text_count);
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            for (int j = 0; j < args.lang_count; j++) {
                free(args.lang[j]);
            }
            free(args.lang);
            args.lang = split_string(argv[++i], ',', &args.lang_count);
        } else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) {
            free(args.save_dir);
            args.save_dir = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0) {
            args.batch = 1;
        }
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    printf("=== TTS Inference with ONNX Runtime (C) ===\n\n");
    
    srand(time(NULL));
    
    Args args = parse_args(argc, argv);
    
    if (args.voice_style_count != args.text_count) {
        fprintf(stderr, "Error: Number of voice styles (%d) must match number of texts (%d)\n",
                args.voice_style_count, args.text_count);
        free_args(&args);
        return 1;
    }
    
    int bsz = args.voice_style_count;
    
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TTS", &env);
    
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    printf("Loading Text-to-Speech model...\n");
    TextToSpeech* tts = load_text_to_speech(env, args.onnx_dir, 0);
    if (!tts) {
        fprintf(stderr, "Failed to load TTS\n");
        free_args(&args);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    printf("\n");
    
    printf("Loading voice style...\n");
    Style* style = load_voice_style((const char**)args.voice_style, args.voice_style_count, 1);
    if (!style) {
        fprintf(stderr, "Failed to load voice style\n");
        tts_free(tts);
        free_args(&args);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    
    mkdir(args.save_dir, 0755);
    
    for (int n = 0; n < args.n_test; n++) {
        printf("\n[%d/%d] Starting synthesis...\n", n + 1, args.n_test);
        
        clock_t start = clock();
        SynthesisResult* result;
        
        if (args.batch) {
            result = tts_batch(tts, memory_info, (const char**)args.text, (const char**)args.lang,
                              bsz, style, args.total_step, args.speed);
        } else {
            result = tts_call(tts, memory_info, args.text[0], args.lang[0], style,
                            args.total_step, args.speed, 0.3f);
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Generating speech from text...\n");
        printf("  -> Generating speech from text completed in %.2f sec\n", elapsed);
        
        if (!result) {
            fprintf(stderr, "Synthesis failed\n");
            continue;
        }
        
        int sample_rate = tts->sample_rate;
        int wav_shape_1 = result->wav_size / bsz;
        
        for (int b = 0; b < bsz; b++) {
            char* fname_base = sanitize_filename(args.text[b], 20);
            char fname[512];
            snprintf(fname, sizeof(fname), "%s_%d.wav", fname_base, n + 1);
            free(fname_base);
            
            int wav_len = (int)(sample_rate * result->duration[b]);
            
            char output_path[1024];
            snprintf(output_path, sizeof(output_path), "%s/%s", args.save_dir, fname);
            
            write_wav_file(output_path, result->wav + b * wav_shape_1, wav_len, sample_rate);
            printf("Saved: %s\n", output_path);
        }
        
        synthesis_result_free(result);
        clear_tensor_buffers();
    }
    
    printf("\n=== Synthesis completed successfully! ===\n");
    
    style_free(style);
    tts_free(tts);
    free_args(&args);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseEnv(env);
    
    return 0;
}
