/*
 * Audiobook Generator for Supertonic TTS
 * 
 * Reads text files and generates complete audiobooks with:
 * - Automatic pauses at punctuation marks
 * - Paragraph breaks
 * - Custom pause directives [PAUSE:ms]
 * - Sentence-by-sentence processing
 * - Real-time progress display
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "supertonic.h"
#include "wav_utils.h"

#define MAX_LINE_LENGTH 4096
#define MAX_SENTENCE_LENGTH 1024
#define DEFAULT_SAMPLE_RATE 24000

/* Pause durations in seconds */
#define PAUSE_PERIOD 0.5f       /* Period, exclamation, question mark */
#define PAUSE_COMMA 0.25f       /* Comma */
#define PAUSE_SEMICOLON 0.35f   /* Semicolon, colon */
#define PAUSE_PARAGRAPH 0.8f    /* Empty line / paragraph break */

/* Command line arguments */
typedef struct {
    char* input_file;
    char* output_file;
    char* voice_style;
    char* onnx_dir;
    char* lang;
    float speed;
    int steps;
} AudiobookArgs;

/* Statistics tracking */
typedef struct {
    int total_sentences;
    int processed_sentences;
    double total_audio_duration;
    double total_processing_time;
    size_t total_samples;
} AudiobookStats;

/* Sentence structure */
typedef struct {
    char text[MAX_SENTENCE_LENGTH];
    float pause_after;  /* Pause duration in seconds */
} Sentence;

/* Dynamic array of sentences */
typedef struct {
    Sentence* sentences;
    size_t count;
    size_t capacity;
} SentenceArray;

/* Initialize sentence array */
static SentenceArray* sentence_array_create(void) {
    SentenceArray* arr = (SentenceArray*)malloc(sizeof(SentenceArray));
    if (!arr) return NULL;
    
    arr->capacity = 100;
    arr->count = 0;
    arr->sentences = (Sentence*)malloc(arr->capacity * sizeof(Sentence));
    if (!arr->sentences) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

/* Add sentence to array */
static int sentence_array_add(SentenceArray* arr, const char* text, float pause) {
    if (!arr || !text) return -1;
    
    if (arr->count >= arr->capacity) {
        size_t new_capacity = arr->capacity * 2;
        Sentence* new_sentences = (Sentence*)realloc(arr->sentences, 
                                                      new_capacity * sizeof(Sentence));
        if (!new_sentences) return -1;
        arr->sentences = new_sentences;
        arr->capacity = new_capacity;
    }
    
    strncpy(arr->sentences[arr->count].text, text, MAX_SENTENCE_LENGTH - 1);
    arr->sentences[arr->count].text[MAX_SENTENCE_LENGTH - 1] = '\0';
    arr->sentences[arr->count].pause_after = pause;
    arr->count++;
    
    return 0;
}

/* Free sentence array */
static void sentence_array_free(SentenceArray* arr) {
    if (arr) {
        free(arr->sentences);
        free(arr);
    }
}

/* Check if character is sentence ending punctuation */
static int is_sentence_end(char c) {
    return (c == '.' || c == '!' || c == '?');
}

/* Get pause duration for punctuation */
static float get_pause_for_punctuation(char c) {
    if (c == '.' || c == '!' || c == '?') {
        return PAUSE_PERIOD;
    } else if (c == ',') {
        return PAUSE_COMMA;
    } else if (c == ';' || c == ':') {
        return PAUSE_SEMICOLON;
    }
    return 0.0f;
}

/* Trim whitespace from string */
static char* trim_whitespace(char* str) {
    char* end;
    
    /* Trim leading space */
    while (isspace((unsigned char)*str)) str++;
    
    if (*str == 0) return str;
    
    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    
    end[1] = '\0';
    return str;
}

/* Parse custom pause directive [PAUSE:ms] */
static int parse_pause_directive(const char* text, float* pause_duration) {
    const char* start = strstr(text, "[PAUSE:");
    if (!start) return 0;
    
    const char* end = strchr(start, ']');
    if (!end) return 0;
    
    int ms;
    if (sscanf(start, "[PAUSE:%d]", &ms) == 1) {
        *pause_duration = ms / 1000.0f;  /* Convert ms to seconds */
        return (int)(end - text + 1);  /* Return length of directive */
    }
    
    return 0;
}

/* Remove custom pause directives from text */
static void remove_pause_directives(char* text) {
    char* src = text;
    char* dst = text;
    
    while (*src) {
        if (strncmp(src, "[PAUSE:", 7) == 0) {
            /* Find end of directive */
            char* end = strchr(src, ']');
            if (end) {
                src = end + 1;
                continue;
            }
        }
        *dst++ = *src++;
    }
    *dst = '\0';
}

/* Parse text into sentences with pause information */
static SentenceArray* parse_text_to_sentences(const char* text) {
    SentenceArray* sentences = sentence_array_create();
    if (!sentences) return NULL;
    
    char buffer[MAX_SENTENCE_LENGTH];
    int buffer_pos = 0;
    const char* p = text;
    int last_was_newline = 0;
    int consecutive_newlines = 0;
    
    while (*p) {
        /* Check for custom pause directive */
        float custom_pause = 0.0f;
        int directive_len = parse_pause_directive(p, &custom_pause);
        if (directive_len > 0) {
            /* Add current buffer as sentence with custom pause */
            if (buffer_pos > 0) {
                buffer[buffer_pos] = '\0';
                char* trimmed = trim_whitespace(buffer);
                if (strlen(trimmed) > 0) {
                    sentence_array_add(sentences, trimmed, custom_pause);
                }
                buffer_pos = 0;
            }
            p += directive_len;
            continue;
        }
        
        /* Track consecutive newlines for paragraph breaks */
        if (*p == '\n') {
            if (last_was_newline) {
                consecutive_newlines++;
                if (consecutive_newlines >= 1 && buffer_pos > 0) {
                    /* Paragraph break - add sentence with paragraph pause */
                    buffer[buffer_pos] = '\0';
                    char* trimmed = trim_whitespace(buffer);
                    if (strlen(trimmed) > 0) {
                        sentence_array_add(sentences, trimmed, PAUSE_PARAGRAPH);
                    }
                    buffer_pos = 0;
                }
            }
            last_was_newline = 1;
            p++;
            continue;
        } else {
            consecutive_newlines = 0;
            last_was_newline = 0;
        }
        
        /* Add character to buffer */
        if (buffer_pos < MAX_SENTENCE_LENGTH - 1) {
            buffer[buffer_pos++] = *p;
        }
        
        /* Check for sentence-ending punctuation */
        if (is_sentence_end(*p)) {
            /* Look ahead to see if there's more content */
            const char* next = p + 1;
            while (*next && isspace((unsigned char)*next)) next++;
            
            if (*next && !is_sentence_end(*next)) {
                /* End of sentence, add it */
                buffer[buffer_pos] = '\0';
                char* trimmed = trim_whitespace(buffer);
                if (strlen(trimmed) > 0) {
                    float pause = get_pause_for_punctuation(*p);
                    sentence_array_add(sentences, trimmed, pause);
                }
                buffer_pos = 0;
            }
        } else if ((*p == ',' || *p == ';' || *p == ':') && buffer_pos > 50) {
            /* Optional: break long sentences at commas/semicolons */
            /* For now, we'll keep them together but could split if needed */
        }
        
        p++;
    }
    
    /* Add remaining buffer */
    if (buffer_pos > 0) {
        buffer[buffer_pos] = '\0';
        char* trimmed = trim_whitespace(buffer);
        if (strlen(trimmed) > 0) {
            sentence_array_add(sentences, trimmed, 0.0f);
        }
    }
    
    return sentences;
}

/* Read entire text file */
static char* read_text_file(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Allocate buffer */
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }
    
    /* Read file */
    size_t read = fread(buffer, 1, size, f);
    buffer[read] = '\0';
    fclose(f);
    
    return buffer;
}

/* Generate silence samples */
static float* generate_silence(float duration_sec, int sample_rate, size_t* out_samples) {
    size_t num_samples = (size_t)(duration_sec * sample_rate);
    *out_samples = num_samples;
    
    float* silence = (float*)calloc(num_samples, sizeof(float));
    return silence;
}

/* Append audio samples to buffer */
static int append_audio(float** dest, size_t* dest_size, size_t* dest_capacity,
                       const float* src, size_t src_size) {
    if (!dest || !dest_size || !dest_capacity || !src) return -1;
    
    size_t new_size = *dest_size + src_size;
    
    if (new_size > *dest_capacity) {
        size_t new_capacity = new_size * 2;
        float* new_buffer = (float*)realloc(*dest, new_capacity * sizeof(float));
        if (!new_buffer) return -1;
        *dest = new_buffer;
        *dest_capacity = new_capacity;
    }
    
    memcpy(*dest + *dest_size, src, src_size * sizeof(float));
    *dest_size = new_size;
    
    return 0;
}

/* Print progress bar */
static void print_progress(int current, int total, double elapsed_time) {
    int bar_width = 50;
    float progress = (float)current / total;
    int pos = (int)(bar_width * progress);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%) - %.2fs", 
           current, total, progress * 100, elapsed_time);
    fflush(stdout);
}

/* Parse command line arguments */
static AudiobookArgs parse_args(int argc, char* argv[]) {
    AudiobookArgs args = {
        .input_file = NULL,
        .output_file = "audiobook.wav",
        .voice_style = "../assets/voice_styles/M1.json",
        .onnx_dir = "../assets/onnx",
        .lang = "en",
        .speed = 1.05f,
        .steps = 5
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input_file = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
            args.voice_style = argv[++i];
        } else if (strcmp(argv[i], "--onnx-dir") == 0 && i + 1 < argc) {
            args.onnx_dir = argv[++i];
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            args.lang = argv[++i];
        } else if (strcmp(argv[i], "--speed") == 0 && i + 1 < argc) {
            args.speed = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args.steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Audiobook Generator for Supertonic TTS\n\n");
            printf("Usage: %s --input <file> [OPTIONS]\n\n", argv[0]);
            printf("Required:\n");
            printf("  --input <file>      Input text file\n\n");
            printf("Options:\n");
            printf("  --output <file>     Output WAV file (default: audiobook.wav)\n");
            printf("  --voice <file>      Voice style JSON (default: ../assets/voice_styles/M1.json)\n");
            printf("  --onnx-dir <dir>    ONNX model directory (default: ../assets/onnx)\n");
            printf("  --lang <code>       Language: en, ko, es, pt, fr (default: en)\n");
            printf("  --speed <float>     Speech speed (default: 1.05)\n");
            printf("  --steps <int>       Inference steps (default: 5)\n");
            printf("  --help, -h          Show this help message\n\n");
            printf("Features:\n");
            printf("  - Automatic pauses: period/!/? (500ms), comma (250ms), ;/: (350ms)\n");
            printf("  - Paragraph breaks: 800ms pause\n");
            printf("  - Custom pauses: [PAUSE:1000] for 1000ms pause\n");
            printf("  - Real-time progress display\n\n");
            exit(0);
        }
    }
    
    return args;
}

/* Main audiobook generation function */
int main(int argc, char* argv[]) {
    printf("=== Supertonic Audiobook Generator ===\n\n");
    
    /* Parse arguments */
    AudiobookArgs args = parse_args(argc, argv);
    
    if (!args.input_file) {
        fprintf(stderr, "Error: --input is required\n");
        fprintf(stderr, "Use --help for usage information\n");
        return 1;
    }
    
    /* Read input file */
    printf("Reading text file: %s\n", args.input_file);
    char* text_content = read_text_file(args.input_file);
    if (!text_content) {
        return 1;
    }
    
    /* Parse text into sentences */
    printf("Parsing text into sentences...\n");
    SentenceArray* sentences = parse_text_to_sentences(text_content);
    free(text_content);
    
    if (!sentences || sentences->count == 0) {
        fprintf(stderr, "Error: No sentences found in text\n");
        sentence_array_free(sentences);
        return 1;
    }
    
    printf("Found %zu sentences\n\n", sentences->count);
    
    /* Initialize TTS */
    printf("Loading TTS models from: %s\n", args.onnx_dir);
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        fprintf(stderr, "Error: Failed to get ONNX Runtime API\n");
        sentence_array_free(sentences);
        return 1;
    }
    
    OrtEnv* env = NULL;
    if (ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "audiobook", &env) != NULL) {
        fprintf(stderr, "Error: Failed to create ONNX Runtime environment\n");
        sentence_array_free(sentences);
        return 1;
    }
    
    TextToSpeech* tts = text_to_speech_create(env, ort_api, args.onnx_dir, 0);
    if (!tts) {
        fprintf(stderr, "Error: Failed to load TTS models\n");
        ort_api->ReleaseEnv(env);
        sentence_array_free(sentences);
        return 1;
    }
    
    /* Load voice style */
    printf("Loading voice style: %s\n", args.voice_style);
    Style* style = style_load_from_file(args.voice_style);
    if (!style) {
        fprintf(stderr, "Error: Failed to load voice style\n");
        text_to_speech_free(tts);
        ort_api->ReleaseEnv(env);
        sentence_array_free(sentences);
        return 1;
    }
    
    printf("\n=== Starting Audiobook Generation ===\n\n");
    printf("Configuration:\n");
    printf("  Language: %s\n", args.lang);
    printf("  Speed: %.2fx\n", args.speed);
    printf("  Steps: %d\n", args.steps);
    printf("  Output: %s\n\n", args.output_file);
    
    /* Allocate audio buffer */
    size_t audio_capacity = 1000000;  /* Start with 1M samples */
    size_t audio_size = 0;
    float* audio_buffer = (float*)malloc(audio_capacity * sizeof(float));
    if (!audio_buffer) {
        fprintf(stderr, "Error: Failed to allocate audio buffer\n");
        style_free(style);
        text_to_speech_free(tts);
        ort_api->ReleaseEnv(env);
        sentence_array_free(sentences);
        return 1;
    }
    
    /* Process each sentence */
    clock_t start_time = clock();
    AudiobookStats stats = {0};
    stats.total_sentences = (int)sentences->count;
    
    for (size_t i = 0; i < sentences->count; i++) {
        Sentence* sent = &sentences->sentences[i];
        
        /* Remove pause directives from text before synthesis */
        char clean_text[MAX_SENTENCE_LENGTH];
        strncpy(clean_text, sent->text, MAX_SENTENCE_LENGTH);
        remove_pause_directives(clean_text);
        
        /* Skip empty sentences */
        char* trimmed = trim_whitespace(clean_text);
        if (strlen(trimmed) == 0) {
            continue;
        }
        
        /* Synthesize speech */
        float* wav_data = NULL;
        size_t wav_samples = 0;
        float duration = 0.0f;
        
        int result = text_to_speech_synthesize(tts, trimmed, args.lang, style,
                                              args.steps, args.speed, 0.0f,
                                              &wav_data, &wav_samples, &duration);
        
        if (result != 0 || !wav_data) {
            fprintf(stderr, "\nWarning: Failed to synthesize sentence %zu\n", i + 1);
            continue;
        }
        
        /* Append synthesized audio */
        if (append_audio(&audio_buffer, &audio_size, &audio_capacity, 
                        wav_data, wav_samples) != 0) {
            fprintf(stderr, "\nError: Failed to append audio\n");
            free(wav_data);
            break;
        }
        
        free(wav_data);
        stats.total_audio_duration += duration;
        
        /* Add pause after sentence if specified */
        if (sent->pause_after > 0.0f) {
            size_t silence_samples = 0;
            float* silence = generate_silence(sent->pause_after, DEFAULT_SAMPLE_RATE, 
                                             &silence_samples);
            if (silence) {
                append_audio(&audio_buffer, &audio_size, &audio_capacity,
                           silence, silence_samples);
                free(silence);
                stats.total_audio_duration += sent->pause_after;
            }
        }
        
        /* Update progress */
        stats.processed_sentences = (int)(i + 1);
        double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        print_progress(stats.processed_sentences, stats.total_sentences, elapsed);
    }
    
    printf("\n\n=== Generation Complete ===\n\n");
    
    /* Calculate statistics */
    stats.total_processing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    stats.total_samples = audio_size;
    
    /* Write output file */
    printf("Writing output file: %s\n", args.output_file);
    if (write_wav_file(args.output_file, audio_buffer, audio_size, 
                      DEFAULT_SAMPLE_RATE, 16) != 0) {
        fprintf(stderr, "Error: Failed to write output file\n");
        free(audio_buffer);
        style_free(style);
        text_to_speech_free(tts);
        ort_api->ReleaseEnv(env);
        sentence_array_free(sentences);
        return 1;
    }
    
    /* Print statistics */
    printf("\n=== Statistics ===\n");
    printf("Sentences processed: %d / %d\n", 
           stats.processed_sentences, stats.total_sentences);
    printf("Audio duration: %.2f seconds (%.2f minutes)\n",
           stats.total_audio_duration, stats.total_audio_duration / 60.0);
    printf("Processing time: %.2f seconds\n", stats.total_processing_time);
    printf("Real-time factor: %.3fx\n", 
           stats.total_processing_time / stats.total_audio_duration);
    printf("Sample rate: %d Hz\n", DEFAULT_SAMPLE_RATE);
    printf("Total samples: %zu\n", stats.total_samples);
    printf("File size: %.2f MB\n", 
           (stats.total_samples * sizeof(int16_t)) / (1024.0 * 1024.0));
    
    printf("\n✓ Audiobook generation completed successfully!\n");
    
    /* Cleanup */
    free(audio_buffer);
    style_free(style);
    text_to_speech_free(tts);
    ort_api->ReleaseEnv(env);
    sentence_array_free(sentences);
    
    return 0;
}
