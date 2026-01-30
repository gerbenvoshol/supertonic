#define _POSIX_C_SOURCE 200809L
#include "supertonic.h"
#include "wav_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <dirent.h>

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
    char* output_file;
    int batch;
    int show_help;
    int list_voices;
} Args;

static void print_help(const char* program_name) {
    printf("=== Supertonic TTS - Example ONNX Runtime ===\n\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --help, -h              Show this help message\n");
    printf("  --list-voices           List available voice styles\n");
    printf("  --text <text>           Text to synthesize (use | to separate multiple texts)\n");
    printf("  --voice-style <path>    Path to voice style JSON (use , to separate multiple)\n");
    printf("  --lang <code>           Language code: en, ko, es, pt, fr (use , for multiple)\n");
    printf("  --speed <float>         Speech speed factor (default: 1.05)\n");
    printf("  --total-step <n>        Number of denoising steps (default: 5)\n");
    printf("  --n-test <n>            Number of test iterations (default: 1)\n");
    printf("  --onnx-dir <path>       Path to ONNX models (default: ../assets/onnx)\n");
    printf("  --save-dir <path>       Output directory (default: results)\n");
    printf("  --output <file>         Output WAV file (overrides --save-dir for single file)\n");
    printf("  --batch                 Use batch processing mode\n");
    printf("\n");
    printf("Examples:\n");
    printf("  # Simple usage with custom text\n");
    printf("  %s --text \"Hello, world!\"\n\n", program_name);
    printf("  # Use a different voice\n");
    printf("  %s --text \"Hello!\" --voice-style ../assets/voice_styles/F1.json\n\n", program_name);
    printf("  # Multiple texts with different voices\n");
    printf("  %s --text \"Hello|Goodbye\" --voice-style \"../assets/voice_styles/M1.json,../assets/voice_styles/F1.json\"\n\n", program_name);
    printf("  # Change language and speed\n");
    printf("  %s --text \"Hola mundo\" --lang es --speed 1.2\n\n", program_name);
    printf("  # Save to specific file\n");
    printf("  %s --text \"Test\" --output my_audio.wav\n\n", program_name);
    printf("\n");
    printf("Available Languages:\n");
    printf("  en - English\n");
    printf("  ko - Korean\n");
    printf("  es - Spanish\n");
    printf("  pt - Portuguese\n");
    printf("  fr - French\n");
    printf("\n");
    printf("Available Voice Styles (in ../assets/voice_styles/):\n");
    printf("  M1.json - Male voice 1\n");
    printf("  F1.json - Female voice 1\n");
    printf("  (Use --list-voices to see all available voices)\n");
    printf("\n");
}

static void list_available_voices(const char* voice_dir) {
    printf("=== Available Voice Styles ===\n\n");
    printf("Scanning directory: %s\n\n", voice_dir);
    
    DIR* dir = opendir(voice_dir);
    if (!dir) {
        fprintf(stderr, "Error: Could not open voice styles directory: %s\n", voice_dir);
        fprintf(stderr, "Please ensure the assets have been downloaded using ./resource.sh\n");
        return;
    }
    
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            const char* ext = strrchr(entry->d_name, '.');
            if (ext && strcmp(ext, ".json") == 0) {
                count++;
                printf("  %d. %s\n", count, entry->d_name);
                printf("     Path: %s/%s\n", voice_dir, entry->d_name);
            }
        }
    }
    closedir(dir);
    
    if (count == 0) {
        printf("  No voice style files found.\n");
        printf("  Please run ./resource.sh to download voice styles.\n");
    }
    printf("\n");
    printf("Usage: --voice-style %s/<filename>\n", voice_dir);
    printf("\n");
}

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
    if (args->output_file) {
        free(args->output_file);
    }
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
    args.n_test = 1;  // Changed default from 4 to 1 for single test
    args.voice_style = malloc(sizeof(char*));
    args.voice_style[0] = strdup("../assets/voice_styles/M1.json");
    args.voice_style_count = 1;
    args.text = malloc(sizeof(char*));
    args.text[0] = strdup("Hello! This is a test of the Supertonic text to speech system.");
    args.text_count = 1;
    args.lang = malloc(sizeof(char*));
    args.lang[0] = strdup("en");
    args.lang_count = 1;
    args.save_dir = strdup("results");
    args.output_file = NULL;
    args.batch = 0;
    args.show_help = 0;
    args.list_voices = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            args.show_help = 1;
        } else if (strcmp(argv[i], "--list-voices") == 0) {
            args.list_voices = 1;
        } else if (strcmp(argv[i], "--onnx-dir") == 0 && i + 1 < argc) {
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
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            if (args.output_file) free(args.output_file);
            args.output_file = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0) {
            args.batch = 1;
        } else {
            fprintf(stderr, "Warning: Unknown option '%s' (use --help for usage)\n", argv[i]);
        }
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    // Handle --help
    if (args.show_help) {
        print_help(argv[0]);
        free_args(&args);
        return 0;
    }
    
    // Handle --list-voices
    if (args.list_voices) {
        // Try to find voice styles directory
        const char* voice_dirs[] = {
            "../assets/voice_styles",
            "assets/voice_styles",
            "./voice_styles"
        };
        
        int found = 0;
        for (int i = 0; i < 3; i++) {
            DIR* dir = opendir(voice_dirs[i]);
            if (dir) {
                closedir(dir);
                list_available_voices(voice_dirs[i]);
                found = 1;
                break;
            }
        }
        
        if (!found) {
            fprintf(stderr, "Error: Could not find voice styles directory.\n");
            fprintf(stderr, "Please run ./resource.sh to download voice styles.\n");
        }
        
        free_args(&args);
        return 0;
    }
    
    printf("=== Supertonic TTS - ONNX Runtime Inference ===\n\n");
    
    srand(time(NULL));
    
    // Validate arguments
    if (args.voice_style_count != args.text_count) {
        fprintf(stderr, "Error: Number of voice styles (%d) must match number of texts (%d)\n",
                args.voice_style_count, args.text_count);
        fprintf(stderr, "Use --help for usage information.\n");
        free_args(&args);
        return 1;
    }
    
    // Show configuration
    printf("Configuration:\n");
    printf("  ONNX Models:   %s\n", args.onnx_dir);
    printf("  Voice Style:   %s\n", args.voice_style[0]);
    if (args.voice_style_count > 1) {
        for (int i = 1; i < args.voice_style_count; i++) {
            printf("                 %s\n", args.voice_style[i]);
        }
    }
    printf("  Language:      %s\n", args.lang[0]);
    printf("  Speed:         %.2f\n", args.speed);
    printf("  Steps:         %d\n", args.total_step);
    printf("  Iterations:    %d\n", args.n_test);
    if (args.output_file) {
        printf("  Output:        %s\n", args.output_file);
    } else {
        printf("  Output Dir:    %s\n", args.save_dir);
    }
    printf("\n");
    
    printf("Text to synthesize:\n");
    for (int i = 0; i < args.text_count; i++) {
        printf("  [%d] \"%s\"\n", i + 1, args.text[i]);
    }
    printf("\n");
    
    int bsz = args.voice_style_count;
    
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TTS", &env);
    
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    printf("Loading Text-to-Speech model...\n");
    TextToSpeech* tts = load_text_to_speech(env, args.onnx_dir, 0);
    if (!tts) {
        fprintf(stderr, "Failed to load TTS model from: %s\n", args.onnx_dir);
        fprintf(stderr, "Please run ./resource.sh to download the models.\n");
        free_args(&args);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    printf("✓ Model loaded successfully\n\n");
    
    printf("Loading voice style...\n");
    Style* style = load_voice_style((const char**)args.voice_style, args.voice_style_count, 1);
    if (!style) {
        fprintf(stderr, "Failed to load voice style: %s\n", args.voice_style[0]);
        fprintf(stderr, "Please check the file exists and run ./resource.sh if needed.\n");
        tts_free(tts);
        free_args(&args);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    printf("✓ Voice style loaded successfully\n\n");
    
    // Create output directory if needed
    if (!args.output_file) {
        mkdir(args.save_dir, 0755);
    }
    
    for (int n = 0; n < args.n_test; n++) {
        if (args.n_test > 1) {
            printf("[%d/%d] Starting synthesis...\n", n + 1, args.n_test);
        } else {
            printf("Starting synthesis...\n");
        }
        
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
        
        if (!result) {
            fprintf(stderr, "Synthesis failed\n");
            continue;
        }
        
        printf("✓ Synthesis completed in %.2f seconds\n", elapsed);
        
        int sample_rate = tts->sample_rate;
        int wav_shape_1 = result->wav_size / bsz;
        
        for (int b = 0; b < bsz; b++) {
            char output_path[1024];
            
            if (args.output_file) {
                // Use specified output file
                snprintf(output_path, sizeof(output_path), "%s", args.output_file);
            } else {
                // Generate filename from text
                char* fname_base = sanitize_filename(args.text[b], 20);
                char fname[512];
                if (args.n_test > 1) {
                    snprintf(fname, sizeof(fname), "%s_%d.wav", fname_base, n + 1);
                } else {
                    snprintf(fname, sizeof(fname), "%s.wav", fname_base);
                }
                free(fname_base);
                snprintf(output_path, sizeof(output_path), "%s/%s", args.save_dir, fname);
            }
            
            int wav_len = (int)(sample_rate * result->duration[b]);
            write_wav_file(output_path, result->wav + b * wav_shape_1, wav_len, sample_rate);
            
            double audio_duration = result->duration[b];
            double rtf = elapsed / audio_duration;
            
            printf("✓ Saved: %s\n", output_path);
            printf("  Audio duration: %.2f seconds\n", audio_duration);
            printf("  Real-time factor: %.2fx\n", rtf);
        }
        
        synthesis_result_free(result);
        clear_tensor_buffers();
        
        if (args.n_test > 1) {
            printf("\n");
        }
    }
    
    printf("\n=== Synthesis completed successfully! ===\n");
    
    style_free(style);
    tts_free(tts);
    free_args(&args);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseEnv(env);
    
    return 0;
}
