/*
 * Voice Mixer for Supertonic TTS
 * 
 * A practical tool for blending high-quality voice styles to create custom voice personalities.
 * Unlike voice_builder.c which generates pseudo-random styles, this tool interpolates between
 * existing voice styles that were created by the official encoder, ensuring high-quality output.
 * 
 * Voice styles consist of two vectors:
 * - style_ttl: Text-to-latent features [1, 50, 256] = 12,800 floats
 * - style_dp: Duration predictor features [1, 8, 16] = 128 floats
 * 
 * These vectors can be linearly interpolated to create new voices that blend characteristics
 * of the source voices.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include "vendor/cjson/cJSON.h"

#define MAX_VOICES 10
#define MAX_PATH 512
#define DEFAULT_TTL_DIM1 50
#define DEFAULT_TTL_DIM2 256
#define DEFAULT_DP_DIM1 8
#define DEFAULT_DP_DIM2 16

/* Voice style structure */
typedef struct {
    char filename[MAX_PATH];
    float* ttl_data;
    int ttl_dim1;
    int ttl_dim2;
    int ttl_size;
    float* dp_data;
    int dp_dim1;
    int dp_dim2;
    int dp_size;
} VoiceStyle;

/* Blend mode enumeration */
typedef enum {
    BLEND_LINEAR,
    BLEND_SLERP
} BlendMode;

/* Function prototypes */
VoiceStyle* load_voice_style(const char* path);
void free_voice_style(VoiceStyle* style);
int save_voice_style_json(const VoiceStyle* style, const char* output_path);
VoiceStyle* blend_voices_linear(VoiceStyle** voices, float* weights, int num_voices);
VoiceStyle* blend_voices_slerp(VoiceStyle** voices, float* weights, int num_voices);
void normalize_weights(float* weights, int count);
void list_voice_files(const char* voice_dir);
void print_usage(const char* program_name);

/* Load a voice style from JSON file */
VoiceStyle* load_voice_style(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path);
        return NULL;
    }
    
    /* Read file content */
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* json_data = (char*)malloc(file_size + 1);
    if (!json_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    size_t read_size = fread(json_data, 1, file_size, file);
    json_data[read_size] = '\0';
    fclose(file);
    
    /* Parse JSON */
    cJSON* root = cJSON_Parse(json_data);
    free(json_data);
    
    if (!root) {
        fprintf(stderr, "Error: Failed to parse JSON from %s\n", path);
        return NULL;
    }
    
    /* Allocate voice style structure */
    VoiceStyle* style = (VoiceStyle*)calloc(1, sizeof(VoiceStyle));
    if (!style) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        cJSON_Delete(root);
        return NULL;
    }
    
    strncpy(style->filename, path, MAX_PATH - 1);
    
    /* Extract style_ttl */
    cJSON* style_ttl = cJSON_GetObjectItem(root, "style_ttl");
    if (!style_ttl) {
        fprintf(stderr, "Error: Missing style_ttl in %s\n", path);
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    cJSON* ttl_dims = cJSON_GetObjectItem(style_ttl, "dims");
    cJSON* ttl_data = cJSON_GetObjectItem(style_ttl, "data");
    
    if (!ttl_dims || !ttl_data) {
        fprintf(stderr, "Error: Invalid style_ttl format in %s\n", path);
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    /* Get dimensions */
    if (cJSON_GetArraySize(ttl_dims) >= 3) {
        cJSON* dim1_item = cJSON_GetArrayItem(ttl_dims, 1);
        cJSON* dim2_item = cJSON_GetArrayItem(ttl_dims, 2);
        if (dim1_item && dim2_item) {
            style->ttl_dim1 = dim1_item->valueint;
            style->ttl_dim2 = dim2_item->valueint;
        } else {
            style->ttl_dim1 = DEFAULT_TTL_DIM1;
            style->ttl_dim2 = DEFAULT_TTL_DIM2;
        }
    } else {
        style->ttl_dim1 = DEFAULT_TTL_DIM1;
        style->ttl_dim2 = DEFAULT_TTL_DIM2;
    }
    style->ttl_size = style->ttl_dim1 * style->ttl_dim2;
    
    /* Allocate and load ttl data */
    style->ttl_data = (float*)malloc(style->ttl_size * sizeof(float));
    if (!style->ttl_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    int ttl_array_size = cJSON_GetArraySize(ttl_data);
    for (int i = 0; i < style->ttl_size && i < ttl_array_size; i++) {
        cJSON* item = cJSON_GetArrayItem(ttl_data, i);
        style->ttl_data[i] = (float)item->valuedouble;
    }
    
    /* Extract style_dp */
    cJSON* style_dp = cJSON_GetObjectItem(root, "style_dp");
    if (!style_dp) {
        fprintf(stderr, "Error: Missing style_dp in %s\n", path);
        free(style->ttl_data);
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    cJSON* dp_dims = cJSON_GetObjectItem(style_dp, "dims");
    cJSON* dp_data = cJSON_GetObjectItem(style_dp, "data");
    
    if (!dp_dims || !dp_data) {
        fprintf(stderr, "Error: Invalid style_dp format in %s\n", path);
        free(style->ttl_data);
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    /* Get dimensions */
    if (cJSON_GetArraySize(dp_dims) >= 3) {
        cJSON* dim1_item = cJSON_GetArrayItem(dp_dims, 1);
        cJSON* dim2_item = cJSON_GetArrayItem(dp_dims, 2);
        if (dim1_item && dim2_item) {
            style->dp_dim1 = dim1_item->valueint;
            style->dp_dim2 = dim2_item->valueint;
        } else {
            style->dp_dim1 = DEFAULT_DP_DIM1;
            style->dp_dim2 = DEFAULT_DP_DIM2;
        }
    } else {
        style->dp_dim1 = DEFAULT_DP_DIM1;
        style->dp_dim2 = DEFAULT_DP_DIM2;
    }
    style->dp_size = style->dp_dim1 * style->dp_dim2;
    
    /* Allocate and load dp data */
    style->dp_data = (float*)malloc(style->dp_size * sizeof(float));
    if (!style->dp_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(style->ttl_data);
        free(style);
        cJSON_Delete(root);
        return NULL;
    }
    
    int dp_array_size = cJSON_GetArraySize(dp_data);
    for (int i = 0; i < style->dp_size && i < dp_array_size; i++) {
        cJSON* item = cJSON_GetArrayItem(dp_data, i);
        style->dp_data[i] = (float)item->valuedouble;
    }
    
    cJSON_Delete(root);
    return style;
}

/* Free voice style memory */
void free_voice_style(VoiceStyle* style) {
    if (style) {
        if (style->ttl_data) free(style->ttl_data);
        if (style->dp_data) free(style->dp_data);
        free(style);
    }
}

/* Normalize weights to sum to 1.0 */
void normalize_weights(float* weights, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += weights[i];
    }
    
    if (sum > 0.0001f) {
        for (int i = 0; i < count; i++) {
            weights[i] /= sum;
        }
    } else {
        /* If all weights are zero, use equal weights */
        float equal_weight = 1.0f / count;
        for (int i = 0; i < count; i++) {
            weights[i] = equal_weight;
        }
    }
}

/* Linear interpolation (weighted average) */
VoiceStyle* blend_voices_linear(VoiceStyle** voices, float* weights, int num_voices) {
    if (num_voices == 0) {
        fprintf(stderr, "Error: No voices to blend\n");
        return NULL;
    }
    
    /* Allocate result */
    VoiceStyle* result = (VoiceStyle*)calloc(1, sizeof(VoiceStyle));
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    /* Use dimensions from first voice */
    result->ttl_dim1 = voices[0]->ttl_dim1;
    result->ttl_dim2 = voices[0]->ttl_dim2;
    result->ttl_size = voices[0]->ttl_size;
    result->dp_dim1 = voices[0]->dp_dim1;
    result->dp_dim2 = voices[0]->dp_dim2;
    result->dp_size = voices[0]->dp_size;
    
    /* Allocate arrays */
    result->ttl_data = (float*)calloc(result->ttl_size, sizeof(float));
    result->dp_data = (float*)calloc(result->dp_size, sizeof(float));
    
    if (!result->ttl_data || !result->dp_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_voice_style(result);
        return NULL;
    }
    
    /* Blend style_ttl */
    for (int i = 0; i < result->ttl_size; i++) {
        float value = 0.0f;
        for (int v = 0; v < num_voices; v++) {
            value += weights[v] * voices[v]->ttl_data[i];
        }
        result->ttl_data[i] = value;
    }
    
    /* Blend style_dp */
    for (int i = 0; i < result->dp_size; i++) {
        float value = 0.0f;
        for (int v = 0; v < num_voices; v++) {
            value += weights[v] * voices[v]->dp_data[i];
        }
        result->dp_data[i] = value;
    }
    
    snprintf(result->filename, MAX_PATH, "mixed_voice.json");
    return result;
}

/* Calculate dot product */
float dot_product(const float* a, const float* b, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

/* Calculate vector magnitude */
float vector_magnitude(const float* v, int size) {
    return sqrtf(dot_product(v, v, size));
}

/* Normalize vector to unit length */
void normalize_vector(float* v, int size) {
    float mag = vector_magnitude(v, size);
    if (mag > 1e-8f) {
        for (int i = 0; i < size; i++) {
            v[i] /= mag;
        }
    }
}

/* Spherical linear interpolation between two vectors */
void slerp_two_vectors(const float* v0, const float* v1, float t, float* result, int size) {
    /* Compute dot product (cosine of angle) */
    float dot = dot_product(v0, v1, size);
    
    /* Clamp dot product to valid range */
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    
    /* If vectors are very close, use linear interpolation */
    if (fabsf(dot) > 0.9995f) {
        for (int i = 0; i < size; i++) {
            result[i] = (1.0f - t) * v0[i] + t * v1[i];
        }
        return;
    }
    
    /* Calculate angle between vectors */
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    
    if (fabsf(sin_theta) < 1e-8f) {
        /* Fall back to linear interpolation */
        for (int i = 0; i < size; i++) {
            result[i] = (1.0f - t) * v0[i] + t * v1[i];
        }
        return;
    }
    
    /* Compute slerp coefficients */
    float c0 = sinf((1.0f - t) * theta) / sin_theta;
    float c1 = sinf(t * theta) / sin_theta;
    
    /* Interpolate */
    for (int i = 0; i < size; i++) {
        result[i] = c0 * v0[i] + c1 * v1[i];
    }
}

/* Spherical interpolation for multiple voices */
VoiceStyle* blend_voices_slerp(VoiceStyle** voices, float* weights, int num_voices) {
    if (num_voices == 0) {
        fprintf(stderr, "Error: No voices to blend\n");
        return NULL;
    }
    
    if (num_voices == 1) {
        /* Single voice, just copy it */
        return blend_voices_linear(voices, weights, num_voices);
    }
    
    if (num_voices == 2) {
        /* Two voices: direct slerp */
        VoiceStyle* result = (VoiceStyle*)calloc(1, sizeof(VoiceStyle));
        if (!result) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return NULL;
        }
        
        result->ttl_dim1 = voices[0]->ttl_dim1;
        result->ttl_dim2 = voices[0]->ttl_dim2;
        result->ttl_size = voices[0]->ttl_size;
        result->dp_dim1 = voices[0]->dp_dim1;
        result->dp_dim2 = voices[0]->dp_dim2;
        result->dp_size = voices[0]->dp_size;
        
        result->ttl_data = (float*)malloc(result->ttl_size * sizeof(float));
        result->dp_data = (float*)malloc(result->dp_size * sizeof(float));
        
        if (!result->ttl_data || !result->dp_data) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_voice_style(result);
            return NULL;
        }
        
        /* Slerp ttl data */
        float t = weights[1] / (weights[0] + weights[1]);
        slerp_two_vectors(voices[0]->ttl_data, voices[1]->ttl_data, t, result->ttl_data, result->ttl_size);
        
        /* Slerp dp data */
        slerp_two_vectors(voices[0]->dp_data, voices[1]->dp_data, t, result->dp_data, result->dp_size);
        
        snprintf(result->filename, MAX_PATH, "mixed_voice.json");
        return result;
    }
    
    /* For more than 2 voices, do sequential slerp */
    VoiceStyle* current = NULL;
    float accumulated_weight = 0.0f;
    
    for (int i = 0; i < num_voices; i++) {
        accumulated_weight += weights[i];
        
        if (current == NULL) {
            /* First voice: copy it */
            current = (VoiceStyle*)calloc(1, sizeof(VoiceStyle));
            if (!current) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                return NULL;
            }
            
            current->ttl_dim1 = voices[i]->ttl_dim1;
            current->ttl_dim2 = voices[i]->ttl_dim2;
            current->ttl_size = voices[i]->ttl_size;
            current->dp_dim1 = voices[i]->dp_dim1;
            current->dp_dim2 = voices[i]->dp_dim2;
            current->dp_size = voices[i]->dp_size;
            
            current->ttl_data = (float*)malloc(current->ttl_size * sizeof(float));
            current->dp_data = (float*)malloc(current->dp_size * sizeof(float));
            
            if (!current->ttl_data || !current->dp_data) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                free_voice_style(current);
                return NULL;
            }
            
            memcpy(current->ttl_data, voices[i]->ttl_data, current->ttl_size * sizeof(float));
            memcpy(current->dp_data, voices[i]->dp_data, current->dp_size * sizeof(float));
        } else {
            /* Slerp current with next voice */
            float t = weights[i] / accumulated_weight;
            
            float* new_ttl = (float*)malloc(current->ttl_size * sizeof(float));
            float* new_dp = (float*)malloc(current->dp_size * sizeof(float));
            
            if (!new_ttl || !new_dp) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                if (new_ttl) free(new_ttl);
                if (new_dp) free(new_dp);
                free_voice_style(current);
                return NULL;
            }
            
            slerp_two_vectors(current->ttl_data, voices[i]->ttl_data, t, new_ttl, current->ttl_size);
            slerp_two_vectors(current->dp_data, voices[i]->dp_data, t, new_dp, current->dp_size);
            
            free(current->ttl_data);
            free(current->dp_data);
            current->ttl_data = new_ttl;
            current->dp_data = new_dp;
        }
    }
    
    if (current) {
        snprintf(current->filename, MAX_PATH, "mixed_voice.json");
    }
    return current;
}

/* Save voice style to JSON file */
int save_voice_style_json(const VoiceStyle* style, const char* output_path) {
    if (!style || !output_path) {
        fprintf(stderr, "Error: Invalid arguments\n");
        return -1;
    }
    
    cJSON* root = cJSON_CreateObject();
    if (!root) {
        fprintf(stderr, "Error: Failed to create JSON object\n");
        return -1;
    }
    
    /* Create style_ttl */
    cJSON* style_ttl = cJSON_CreateObject();
    
    int ttl_dims_arr[3] = {1, style->ttl_dim1, style->ttl_dim2};
    cJSON* ttl_dims = cJSON_CreateIntArray(ttl_dims_arr, 3);
    cJSON_AddItemToObject(style_ttl, "dims", ttl_dims);
    
    cJSON* ttl_data_array = cJSON_CreateFloatArray(style->ttl_data, style->ttl_size);
    if (!ttl_data_array) {
        fprintf(stderr, "Error: Failed to create ttl_data array\n");
        cJSON_Delete(root);
        return -1;
    }
    cJSON_AddItemToObject(style_ttl, "data", ttl_data_array);
    
    cJSON_AddItemToObject(root, "style_ttl", style_ttl);
    
    /* Create style_dp */
    cJSON* style_dp = cJSON_CreateObject();
    
    int dp_dims_arr[3] = {1, style->dp_dim1, style->dp_dim2};
    cJSON* dp_dims = cJSON_CreateIntArray(dp_dims_arr, 3);
    cJSON_AddItemToObject(style_dp, "dims", dp_dims);
    
    cJSON* dp_data_array = cJSON_CreateFloatArray(style->dp_data, style->dp_size);
    if (!dp_data_array) {
        fprintf(stderr, "Error: Failed to create dp_data array\n");
        cJSON_Delete(root);
        return -1;
    }
    cJSON_AddItemToObject(style_dp, "data", dp_data_array);
    
    cJSON_AddItemToObject(root, "style_dp", style_dp);
    
    /* Convert to string and save */
    char* json_string = cJSON_Print(root);
    if (!json_string) {
        fprintf(stderr, "Error: Failed to generate JSON string\n");
        cJSON_Delete(root);
        return -1;
    }
    
    FILE* file = fopen(output_path, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot open output file: %s\n", output_path);
        free(json_string);
        cJSON_Delete(root);
        return -1;
    }
    
    fprintf(file, "%s", json_string);
    fclose(file);
    
    free(json_string);
    cJSON_Delete(root);
    
    return 0;
}

/* List available voice files in directory */
void list_voice_files(const char* voice_dir) {
    DIR* dir = opendir(voice_dir);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory: %s\n", voice_dir);
        return;
    }
    
    printf("Available voice styles in %s:\n", voice_dir);
    printf("─────────────────────────────────────\n");
    
    struct dirent* entry;
    int count = 0;
    while ((entry = readdir(dir)) != NULL) {
        /* Check if filename ends with .json */
        size_t len = strlen(entry->d_name);
        if (len > 5 && strcmp(entry->d_name + len - 5, ".json") == 0) {
            /* Skip hidden files */
            if (entry->d_name[0] != '.') {
                printf("  • %s\n", entry->d_name);
                count++;
            }
        }
    }
    
    if (count == 0) {
        printf("  (no .json files found)\n");
    } else {
        printf("─────────────────────────────────────\n");
        printf("Total: %d voice style(s)\n", count);
    }
    
    closedir(dir);
}

/* Print usage information */
void print_usage(const char* program_name) {
    printf("\n");
    printf("Voice Mixer for Supertonic TTS\n");
    printf("================================\n\n");
    printf("Blend multiple high-quality voice styles to create custom voice personalities.\n");
    printf("Unlike voice_builder, this tool interpolates between existing voice styles\n");
    printf("created by the official encoder, ensuring high-quality output.\n\n");
    
    printf("Usage: %s [options]\n\n", program_name);
    
    printf("Basic Options:\n");
    printf("  --voices <paths>        Comma-separated paths to voice style JSON files\n");
    printf("  --weights <values>      Comma-separated blend weights (default: equal)\n");
    printf("  --output <file>         Output JSON file (default: mixed_voice.json)\n");
    printf("  --mode <type>           Blend mode: 'linear' or 'slerp' (default: linear)\n");
    printf("  --help, -h              Show this help message\n\n");
    
    printf("Advanced Options:\n");
    printf("  --random                Generate random blend from available voices\n");
    printf("  --num-random <n>        Number of voices for random blend (default: 2)\n");
    printf("  --voice-dir <path>      Directory with voice styles (default: ../assets/voice_styles)\n");
    printf("  --list-voices           List available voices in voice-dir\n\n");
    
    printf("Blend Modes:\n");
    printf("  linear                  Standard weighted average (default)\n");
    printf("  slerp                   Spherical interpolation for smoother blends\n\n");
    
    printf("Examples:\n");
    printf("  # Basic: blend two voices equally\n");
    printf("  %s --voices ../assets/voice_styles/M1.json,../assets/voice_styles/F1.json \\\n", program_name);
    printf("    --output mixed.json\n\n");
    
    printf("  # Weighted blend: 70%% male, 30%% female\n");
    printf("  %s --voices ../assets/voice_styles/M1.json,../assets/voice_styles/F1.json \\\n", program_name);
    printf("    --weights 0.7,0.3 --output mixed.json\n\n");
    
    printf("  # Blend three voices\n");
    printf("  %s --voices M1.json,F1.json,M3.json \\\n", program_name);
    printf("    --weights 0.5,0.3,0.2 --output mixed.json\n\n");
    
    printf("  # Spherical interpolation mode\n");
    printf("  %s --voices M1.json,F1.json \\\n", program_name);
    printf("    --weights 0.7,0.3 --mode slerp --output mixed.json\n\n");
    
    printf("  # Random blend from available voices\n");
    printf("  %s --random --voice-dir ../assets/voice_styles \\\n", program_name);
    printf("    --output random_voice.json\n\n");
    
    printf("  # List available voices\n");
    printf("  %s --list-voices --voice-dir ../assets/voice_styles\n\n", program_name);
}

/* Main program */
int main(int argc, char* argv[]) {
    char* voice_paths[MAX_VOICES];
    float weights[MAX_VOICES];
    int num_voices = 0;
    char* output_path = "mixed_voice.json";
    BlendMode mode = BLEND_LINEAR;
    int random_mode = 0;
    int num_random = 2;
    char* voice_dir = "../assets/voice_styles";
    int list_voices_flag = 0;
    int weights_provided = 0;
    
    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--voices") == 0 && i + 1 < argc) {
            /* Parse comma-separated voice paths */
            char* paths = argv[++i];
            char* token = strtok(paths, ",");
            while (token != NULL && num_voices < MAX_VOICES) {
                voice_paths[num_voices++] = token;
                token = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            /* Parse comma-separated weights */
            char* weights_str = argv[++i];
            char* token = strtok(weights_str, ",");
            int w_count = 0;
            while (token != NULL && w_count < MAX_VOICES) {
                weights[w_count++] = atof(token);
                token = strtok(NULL, ",");
            }
            weights_provided = 1;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "linear") == 0) {
                mode = BLEND_LINEAR;
            } else if (strcmp(argv[i], "slerp") == 0) {
                mode = BLEND_SLERP;
            } else {
                fprintf(stderr, "Error: Unknown mode '%s'. Use 'linear' or 'slerp'.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--random") == 0) {
            random_mode = 1;
        } else if (strcmp(argv[i], "--num-random") == 0 && i + 1 < argc) {
            num_random = atoi(argv[++i]);
            if (num_random < 2) num_random = 2;
            if (num_random > MAX_VOICES) num_random = MAX_VOICES;
        } else if (strcmp(argv[i], "--voice-dir") == 0 && i + 1 < argc) {
            voice_dir = argv[++i];
        } else if (strcmp(argv[i], "--list-voices") == 0) {
            list_voices_flag = 1;
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage information\n");
            return 1;
        }
    }
    
    /* Handle list-voices flag */
    if (list_voices_flag) {
        list_voice_files(voice_dir);
        return 0;
    }
    
    /* Handle random mode */
    if (random_mode) {
        /* Scan directory for voice files */
        DIR* dir = opendir(voice_dir);
        if (!dir) {
            fprintf(stderr, "Error: Cannot open voice directory: %s\n", voice_dir);
            return 1;
        }
        
        char* available_voices[MAX_VOICES];
        int num_available = 0;
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL && num_available < MAX_VOICES) {
            /* Check if filename ends with .json */
            size_t len = strlen(entry->d_name);
            if (len > 5 && strcmp(entry->d_name + len - 5, ".json") == 0) {
                /* Skip hidden files */
                if (entry->d_name[0] != '.') {
                    char* full_path = (char*)malloc(MAX_PATH);
                    if (!full_path) {
                        fprintf(stderr, "Error: Memory allocation failed\n");
                        closedir(dir);
                        for (int j = 0; j < num_available; j++) {
                            free(available_voices[j]);
                        }
                        return 1;
                    }
                    snprintf(full_path, MAX_PATH, "%s/%s", voice_dir, entry->d_name);
                    available_voices[num_available++] = full_path;
                }
            }
        }
        closedir(dir);
        
        if (num_available < 2) {
            fprintf(stderr, "Error: Need at least 2 voice files for random blend\n");
            for (int i = 0; i < num_available; i++) {
                free(available_voices[i]);
            }
            return 1;
        }
        
        /* Select random voices */
        srand(time(NULL));
        if (num_random > num_available) {
            num_random = num_available;
        }
        
        /* Simple random selection without replacement */
        num_voices = num_random;
        int* selected = (int*)calloc(num_available, sizeof(int));
        
        for (int i = 0; i < num_random; i++) {
            int idx;
            do {
                idx = rand() % num_available;
            } while (selected[idx]);
            selected[idx] = 1;
            voice_paths[i] = available_voices[idx];
            
            /* Generate random weights */
            weights[i] = (float)rand() / RAND_MAX;
        }
        
        free(selected);
        
        /* Free unused paths */
        for (int i = 0; i < num_available; i++) {
            int used = 0;
            for (int j = 0; j < num_voices; j++) {
                if (available_voices[i] == voice_paths[j]) {
                    used = 1;
                    break;
                }
            }
            if (!used) {
                free(available_voices[i]);
            }
        }
        
        weights_provided = 1;
        
        printf("Random blend selected:\n");
        for (int i = 0; i < num_voices; i++) {
            printf("  • %s (weight: %.3f)\n", voice_paths[i], weights[i]);
        }
        printf("\n");
    }
    
    /* Validate required arguments */
    if (num_voices == 0) {
        fprintf(stderr, "Error: No voices specified. Use --voices or --random.\n");
        fprintf(stderr, "Use --help for usage information\n");
        return 1;
    }
    
    /* Set default equal weights if not provided */
    if (!weights_provided) {
        float equal_weight = 1.0f / num_voices;
        for (int i = 0; i < num_voices; i++) {
            weights[i] = equal_weight;
        }
    }
    
    /* Normalize weights */
    normalize_weights(weights, num_voices);
    
    printf("\n========================================\n");
    printf("Voice Mixer for Supertonic TTS\n");
    printf("========================================\n\n");
    
    printf("Blending %d voice style(s):\n", num_voices);
    for (int i = 0; i < num_voices; i++) {
        printf("  • %s (weight: %.3f)\n", voice_paths[i], weights[i]);
    }
    printf("\nBlend mode: %s\n", mode == BLEND_SLERP ? "slerp" : "linear");
    printf("Output file: %s\n\n", output_path);
    
    /* Load all voice styles */
    VoiceStyle* voices[MAX_VOICES];
    for (int i = 0; i < num_voices; i++) {
        printf("Loading voice %d/%d: %s...\n", i + 1, num_voices, voice_paths[i]);
        voices[i] = load_voice_style(voice_paths[i]);
        if (!voices[i]) {
            fprintf(stderr, "Error: Failed to load voice: %s\n", voice_paths[i]);
            /* Free already loaded voices */
            for (int j = 0; j < i; j++) {
                free_voice_style(voices[j]);
            }
            return 1;
        }
    }
    
    printf("\nBlending voices...\n");
    
    /* Blend voices */
    VoiceStyle* result = NULL;
    if (mode == BLEND_SLERP) {
        result = blend_voices_slerp(voices, weights, num_voices);
    } else {
        result = blend_voices_linear(voices, weights, num_voices);
    }
    
    if (!result) {
        fprintf(stderr, "Error: Failed to blend voices\n");
        for (int i = 0; i < num_voices; i++) {
            free_voice_style(voices[i]);
        }
        return 1;
    }
    
    /* Save result */
    printf("Saving blended voice to %s...\n", output_path);
    if (save_voice_style_json(result, output_path) != 0) {
        fprintf(stderr, "Error: Failed to save output file\n");
        free_voice_style(result);
        for (int i = 0; i < num_voices; i++) {
            free_voice_style(voices[i]);
        }
        return 1;
    }
    
    /* Cleanup */
    free_voice_style(result);
    for (int i = 0; i < num_voices; i++) {
        free_voice_style(voices[i]);
    }
    
    /* Cleanup random mode allocations */
    if (random_mode) {
        for (int i = 0; i < num_voices; i++) {
            free(voice_paths[i]);
        }
    }
    
    printf("\n========================================\n");
    printf("Voice blending completed successfully!\n");
    printf("========================================\n\n");
    printf("Output file: %s\n\n", output_path);
    printf("To test the blended voice:\n");
    printf("  ./example_onnx --voice-style %s --text \"Hello, world!\"\n\n", output_path);
    
    return 0;
}
