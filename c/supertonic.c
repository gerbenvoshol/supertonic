#define _POSIX_C_SOURCE 200809L
#include "supertonic.h"
#include "wav_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "vendor/cjson/cJSON.h"

/* Available languages */
const char* AVAILABLE_LANGS[] = {"en", "ko", "es", "pt", "fr"};
const int AVAILABLE_LANGS_COUNT = 5;

/* Hangul decomposition constants */
#define HANGUL_SBASE 0xAC00
#define HANGUL_LBASE 0x1100
#define HANGUL_VBASE 0x1161
#define HANGUL_TBASE 0x11A7
#define HANGUL_LCOUNT 19
#define HANGUL_VCOUNT 21
#define HANGUL_TCOUNT 28
#define HANGUL_NCOUNT (HANGUL_VCOUNT * HANGUL_TCOUNT)
#define HANGUL_SCOUNT (HANGUL_LCOUNT * HANGUL_NCOUNT)

/* Latin decomposition structure */
typedef struct {
    uint32_t composed;
    uint16_t decomposed[3];
    int decomposed_len;
} LatinDecomp;

static const LatinDecomp LATIN_DECOMPOSITIONS[] = {
    {0x00C1, {0x0041, 0x0301}, 2}, {0x00E1, {0x0061, 0x0301}, 2},
    {0x00C9, {0x0045, 0x0301}, 2}, {0x00E9, {0x0065, 0x0301}, 2},
    {0x00CD, {0x0049, 0x0301}, 2}, {0x00ED, {0x0069, 0x0301}, 2},
    {0x00D3, {0x004F, 0x0301}, 2}, {0x00F3, {0x006F, 0x0301}, 2},
    {0x00DA, {0x0055, 0x0301}, 2}, {0x00FA, {0x0075, 0x0301}, 2},
    {0x00C0, {0x0041, 0x0300}, 2}, {0x00E0, {0x0061, 0x0300}, 2},
    {0x00C8, {0x0045, 0x0300}, 2}, {0x00E8, {0x0065, 0x0300}, 2},
    {0x00CC, {0x0049, 0x0300}, 2}, {0x00EC, {0x0069, 0x0300}, 2},
    {0x00D2, {0x004F, 0x0300}, 2}, {0x00F2, {0x006F, 0x0300}, 2},
    {0x00D9, {0x0055, 0x0300}, 2}, {0x00F9, {0x0075, 0x0300}, 2},
    {0x00C2, {0x0041, 0x0302}, 2}, {0x00E2, {0x0061, 0x0302}, 2},
    {0x00CA, {0x0045, 0x0302}, 2}, {0x00EA, {0x0065, 0x0302}, 2},
    {0x00CE, {0x0049, 0x0302}, 2}, {0x00EE, {0x0069, 0x0302}, 2},
    {0x00D4, {0x004F, 0x0302}, 2}, {0x00F4, {0x006F, 0x0302}, 2},
    {0x00DB, {0x0055, 0x0302}, 2}, {0x00FB, {0x0075, 0x0302}, 2},
    {0x00C3, {0x0041, 0x0303}, 2}, {0x00E3, {0x0061, 0x0303}, 2},
    {0x00D1, {0x004E, 0x0303}, 2}, {0x00F1, {0x006E, 0x0303}, 2},
    {0x00D5, {0x004F, 0x0303}, 2}, {0x00F5, {0x006F, 0x0303}, 2},
    {0x00C4, {0x0041, 0x0308}, 2}, {0x00E4, {0x0061, 0x0308}, 2},
    {0x00CB, {0x0045, 0x0308}, 2}, {0x00EB, {0x0065, 0x0308}, 2},
    {0x00CF, {0x0049, 0x0308}, 2}, {0x00EF, {0x0069, 0x0308}, 2},
    {0x00D6, {0x004F, 0x0308}, 2}, {0x00F6, {0x006F, 0x0308}, 2},
    {0x00DC, {0x0055, 0x0308}, 2}, {0x00FC, {0x0075, 0x0308}, 2},
    {0x00C7, {0x0043, 0x0327}, 2}, {0x00E7, {0x0063, 0x0327}, 2},
    {0, {0}, 0}
};

static const int LATIN_DECOMPOSITIONS_COUNT = sizeof(LATIN_DECOMPOSITIONS) / sizeof(LatinDecomp) - 1;

/* Helper: trim whitespace */
static char* trim(const char* str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) {
        char* result = malloc(1);
        result[0] = 0;
        return result;
    }
    const char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    size_t len = end - str + 1;
    char* result = malloc(len + 1);
    memcpy(result, str, len);
    result[len] = 0;
    return result;
}

/* Helper: string replace */
static char* str_replace(const char* orig, const char* rep, const char* with) {
    char* result;
    char* ins = (char*)orig;
    char* tmp;
    int len_rep = strlen(rep);
    int len_with = strlen(with);
    int len_front;
    int count;
    
    if (!orig || !rep) return NULL;
    if (len_rep == 0) {
        return strdup(orig);
    }
    
    for (count = 0; (tmp = strstr(ins, rep)); ++count) {
        ins = tmp + len_rep;
    }
    
    tmp = result = malloc(strlen(orig) + (len_with - len_rep) * count + 1);
    if (!result) return NULL;
    
    while (count--) {
        ins = strstr(orig, rep);
        len_front = ins - orig;
        tmp = strncpy(tmp, orig, len_front) + len_front;
        tmp = strcpy(tmp, with) + len_with;
        orig += len_front + len_rep;
    }
    strcpy(tmp, orig);
    return result;
}

/* Helper: remove substring */
static char* str_remove(const char* orig, const char* sub) {
    return str_replace(orig, sub, "");
}

/* Helper: remove emoji (4-byte UTF-8) */
static char* remove_emoji(const char* text) {
    size_t len = strlen(text);
    char* result = malloc(len + 1);
    size_t j = 0;
    
    for (size_t i = 0; i < len; ) {
        unsigned char c = (unsigned char)text[i];
        if ((c == 0xF0) && (i + 3 < len) && 
            ((unsigned char)text[i+1] == 0x9F)) {
            i += 4;
        } else {
            result[j++] = text[i++];
        }
    }
    result[j] = 0;
    return result;
}

/* Decompose character */
static void decompose_character(uint32_t codepoint, uint16_t* output, int* output_len) {
    *output_len = 0;
    
    if (codepoint >= HANGUL_SBASE && codepoint < HANGUL_SBASE + HANGUL_SCOUNT) {
        uint32_t sIndex = codepoint - HANGUL_SBASE;
        uint32_t lIndex = sIndex / HANGUL_NCOUNT;
        uint32_t vIndex = (sIndex % HANGUL_NCOUNT) / HANGUL_TCOUNT;
        uint32_t tIndex = sIndex % HANGUL_TCOUNT;
        
        output[(*output_len)++] = (uint16_t)(HANGUL_LBASE + lIndex);
        output[(*output_len)++] = (uint16_t)(HANGUL_VBASE + vIndex);
        if (tIndex > 0) {
            output[(*output_len)++] = (uint16_t)(HANGUL_TBASE + tIndex);
        }
        return;
    }
    
    for (int i = 0; i < LATIN_DECOMPOSITIONS_COUNT; i++) {
        if (LATIN_DECOMPOSITIONS[i].composed == codepoint) {
            for (int j = 0; j < LATIN_DECOMPOSITIONS[i].decomposed_len; j++) {
                output[(*output_len)++] = LATIN_DECOMPOSITIONS[i].decomposed[j];
            }
            return;
        }
    }
    
    output[(*output_len)++] = (uint16_t)(codepoint & 0xFFFF);
}

/* UTF-8 to Unicode values */
static uint16_t* text_to_unicode_values(const char* text, size_t* count) {
    size_t len = strlen(text);
    uint16_t* temp = malloc(len * 3 * sizeof(uint16_t));
    size_t pos = 0;
    size_t i = 0;
    
    while (i < len) {
        uint32_t codepoint = 0;
        unsigned char c = (unsigned char)text[i];
        
        if ((c & 0x80) == 0) {
            codepoint = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            codepoint = (c & 0x1F) << 6;
            codepoint |= ((unsigned char)text[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            codepoint = (c & 0x0F) << 12;
            codepoint |= ((unsigned char)text[i + 1] & 0x3F) << 6;
            codepoint |= ((unsigned char)text[i + 2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            codepoint = (c & 0x07) << 18;
            codepoint |= ((unsigned char)text[i + 1] & 0x3F) << 12;
            codepoint |= ((unsigned char)text[i + 2] & 0x3F) << 6;
            codepoint |= ((unsigned char)text[i + 3] & 0x3F);
            i += 4;
        } else {
            i += 1;
            continue;
        }
        
        uint16_t decomposed[3];
        int decomp_len;
        decompose_character(codepoint, decomposed, &decomp_len);
        for (int j = 0; j < decomp_len; j++) {
            temp[pos++] = decomposed[j];
        }
    }
    
    uint16_t* result = malloc(pos * sizeof(uint16_t));
    memcpy(result, temp, pos * sizeof(uint16_t));
    free(temp);
    *count = pos;
    return result;
}

/* Preprocess text */
static char* preprocess_text(const char* text, const char* lang) {
    char* result = strdup(text);
    char* temp;
    
    /* Symbol replacements - use UTF-8 encoded strings */
    temp = str_replace(result, "\xE2\x80\x93", "-"); free(result); result = temp;  /* en dash */
    temp = str_replace(result, "\xE2\x80\x91", "-"); free(result); result = temp;  /* non-breaking hyphen */
    temp = str_replace(result, "\xE2\x80\x94", "-"); free(result); result = temp;  /* em dash */
    temp = str_replace(result, "_", " "); free(result); result = temp;
    temp = str_replace(result, "\xE2\x80\x9C", "\""); free(result); result = temp;  /* left double quote */
    temp = str_replace(result, "\xE2\x80\x9D", "\""); free(result); result = temp;  /* right double quote */
    temp = str_replace(result, "\xE2\x80\x98", "'"); free(result); result = temp;   /* left single quote */
    temp = str_replace(result, "\xE2\x80\x99", "'"); free(result); result = temp;   /* right single quote */
    temp = str_replace(result, "\xC2\xB4", "'"); free(result); result = temp;       /* acute accent */
    temp = str_replace(result, "`", "'"); free(result); result = temp;
    temp = str_replace(result, "[", " "); free(result); result = temp;
    temp = str_replace(result, "]", " "); free(result); result = temp;
    temp = str_replace(result, "|", " "); free(result); result = temp;
    temp = str_replace(result, "/", " "); free(result); result = temp;
    temp = str_replace(result, "#", " "); free(result); result = temp;
    
    temp = remove_emoji(result); free(result); result = temp;
    
    temp = str_remove(result, "\xE2\x99\xA5"); free(result); result = temp;  /* heart */
    temp = str_remove(result, "\xE2\x98\x86"); free(result); result = temp;  /* star */
    temp = str_remove(result, "\xE2\x99\xA1"); free(result); result = temp;  /* white heart */
    temp = str_remove(result, "\xC2\xA9"); free(result); result = temp;      /* copyright */
    temp = str_remove(result, "\\"); free(result); result = temp;
    
    temp = str_replace(result, "@", " at "); free(result); result = temp;
    temp = str_replace(result, "e.g.,", "for example, "); free(result); result = temp;
    temp = str_replace(result, "i.e.,", "that is, "); free(result); result = temp;
    
    temp = str_replace(result, " ,", ","); free(result); result = temp;
    temp = str_replace(result, " .", "."); free(result); result = temp;
    temp = str_replace(result, " !", "!"); free(result); result = temp;
    temp = str_replace(result, " ?", "?"); free(result); result = temp;
    temp = str_replace(result, " ;", ";"); free(result); result = temp;
    temp = str_replace(result, " :", ":"); free(result); result = temp;
    temp = str_replace(result, " '", "'"); free(result); result = temp;
    
    while (strstr(result, "\"\"")) {
        temp = str_replace(result, "\"\"", "\""); free(result); result = temp;
    }
    while (strstr(result, "''")) {
        temp = str_replace(result, "''", "'"); free(result); result = temp;
    }
    while (strstr(result, "  ")) {
        temp = str_replace(result, "  ", " "); free(result); result = temp;
    }
    
    temp = trim(result); free(result); result = temp;
    
    if (strlen(result) > 0) {
        char last = result[strlen(result) - 1];
        if (last != '.' && last != '!' && last != '?' && last != ';' && 
            last != ':' && last != ',' && last != '\'' && last != '"' &&
            last != ')' && last != ']' && last != '}' && last != '>') {
            temp = malloc(strlen(result) + 2);
            strcpy(temp, result);
            strcat(temp, ".");
            free(result);
            result = temp;
        }
    }
    
    int valid = 0;
    for (int i = 0; i < AVAILABLE_LANGS_COUNT; i++) {
        if (strcmp(lang, AVAILABLE_LANGS[i]) == 0) {
            valid = 1;
            break;
        }
    }
    if (!valid) {
        fprintf(stderr, "Invalid language: %s\n", lang);
        free(result);
        return NULL;
    }
    
    temp = malloc(strlen(result) + strlen(lang) * 2 + 10);
    sprintf(temp, "<%s>%s</%s>", lang, result, lang);
    free(result);
    result = temp;
    
    return result;
}

/* Unicode Processor implementation */
UnicodeProcessor* unicode_processor_create(const char* unicode_indexer_json_path) {
    if (!unicode_indexer_json_path) {
        fprintf(stderr, "Error: unicode_indexer_json_path is NULL\n");
        return NULL;
    }
    
    FILE* file = fopen(unicode_indexer_json_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open unicode indexer: %s\n", unicode_indexer_json_path);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* json_str = malloc(file_size + 1);
    if (!json_str) {
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(json_str, 1, file_size, file);
    fclose(file);
    json_str[file_size] = 0;
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Failed to read complete unicode indexer file\n");
        free(json_str);
        return NULL;
    }
    
    cJSON* json = cJSON_Parse(json_str);
    free(json_str);
    
    if (!json) {
        fprintf(stderr, "Failed to parse JSON\n");
        return NULL;
    }
    
    int array_size = cJSON_GetArraySize(json);
    
    UnicodeProcessor* processor = malloc(sizeof(UnicodeProcessor));
    if (!processor) {
        cJSON_Delete(json);
        return NULL;
    }
    processor->indexer = malloc(array_size * sizeof(int64_t));
    if (!processor->indexer) {
        free(processor);
        cJSON_Delete(json);
        return NULL;
    }
    processor->indexer_size = array_size;
    
    for (int i = 0; i < array_size; i++) {
        cJSON* item = cJSON_GetArrayItem(json, i);
        processor->indexer[i] = (int64_t)item->valueint;
    }
    
    cJSON_Delete(json);
    return processor;
}

void unicode_processor_free(UnicodeProcessor* processor) {
    if (processor) {
        free(processor->indexer);
        free(processor);
    }
}

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
) {
    if (!processor || !text_list || !lang_list || batch_size <= 0) {
        return -1;
    }
    
    char** processed_texts = malloc(batch_size * sizeof(char*));
    uint16_t** unicode_vals = malloc(batch_size * sizeof(uint16_t*));
    size_t* unicode_counts = malloc(batch_size * sizeof(size_t));
    
    if (!processed_texts || !unicode_vals || !unicode_counts) {
        free(processed_texts);
        free(unicode_vals);
        free(unicode_counts);
        return -1;
    }
    
    for (int i = 0; i < batch_size; i++) {
        processed_texts[i] = preprocess_text(text_list[i], lang_list[i]);
        if (!processed_texts[i]) {
            for (int j = 0; j < i; j++) {
                free(processed_texts[j]);
                free(unicode_vals[j]);
            }
            free(processed_texts);
            free(unicode_vals);
            free(unicode_counts);
            return -1;
        }
        unicode_vals[i] = text_to_unicode_values(processed_texts[i], &unicode_counts[i]);
        if (!unicode_vals[i]) {
            free(processed_texts[i]);
            for (int j = 0; j < i; j++) {
                free(processed_texts[j]);
                free(unicode_vals[j]);
            }
            free(processed_texts);
            free(unicode_vals);
            free(unicode_counts);
            return -1;
        }
    }
    
    int64_t max_len = 0;
    for (int i = 0; i < batch_size; i++) {
        if ((int64_t)unicode_counts[i] > max_len) {
            max_len = (int64_t)unicode_counts[i];
        }
    }
    
    int64_t** text_ids = malloc(batch_size * sizeof(int64_t*));
    if (!text_ids) {
        for (int i = 0; i < batch_size; i++) {
            free(processed_texts[i]);
            free(unicode_vals[i]);
        }
        free(processed_texts);
        free(unicode_vals);
        free(unicode_counts);
        return -1;
    }
    
    for (int i = 0; i < batch_size; i++) {
        text_ids[i] = calloc(max_len, sizeof(int64_t));
        if (!text_ids[i]) {
            for (int j = 0; j < i; j++) {
                free(text_ids[j]);
            }
            free(text_ids);
            for (int j = 0; j < batch_size; j++) {
                free(processed_texts[j]);
                free(unicode_vals[j]);
            }
            free(processed_texts);
            free(unicode_vals);
            free(unicode_counts);
            return -1;
        }
        for (size_t j = 0; j < unicode_counts[i]; j++) {
            if (unicode_vals[i][j] < processor->indexer_size) {
                text_ids[i][j] = processor->indexer[unicode_vals[i][j]];
            }
        }
    }
    
    // Convert size_t to int64_t for length_to_mask
    int64_t* lengths = malloc(batch_size * sizeof(int64_t));
    if (!lengths) {
        for (int i = 0; i < batch_size; i++) {
            free(text_ids[i]);
            free(processed_texts[i]);
            free(unicode_vals[i]);
        }
        free(text_ids);
        free(processed_texts);
        free(unicode_vals);
        free(unicode_counts);
        return -1;
    }
    for (int i = 0; i < batch_size; i++) {
        lengths[i] = (int64_t)unicode_counts[i];
    }
    
    float*** text_mask = length_to_mask(lengths, batch_size, max_len);
    free(lengths);
    
    if (!text_mask) {
        for (int i = 0; i < batch_size; i++) {
            free(text_ids[i]);
            free(processed_texts[i]);
            free(unicode_vals[i]);
        }
        free(text_ids);
        free(processed_texts);
        free(unicode_vals);
        free(unicode_counts);
        return -1;
    }
    
    for (int i = 0; i < batch_size; i++) {
        free(processed_texts[i]);
        free(unicode_vals[i]);
    }
    free(processed_texts);
    free(unicode_vals);
    free(unicode_counts);
    
    *text_ids_out = text_ids;
    *text_ids_rows = batch_size;
    *text_ids_cols = max_len;
    *text_mask_out = text_mask;
    *text_mask_batch = batch_size;
    *text_mask_channels = 1;
    *text_mask_len = max_len;
    
    return 0;
}

/* Style implementation */
Style* style_create(
    const float* ttl_data, size_t ttl_data_size, const int64_t* ttl_shape, size_t ttl_shape_len,
    const float* dp_data, size_t dp_data_size, const int64_t* dp_shape, size_t dp_shape_len
) {
    Style* style = malloc(sizeof(Style));
    
    style->ttl_data = malloc(ttl_data_size * sizeof(float));
    memcpy(style->ttl_data, ttl_data, ttl_data_size * sizeof(float));
    style->ttl_data_size = ttl_data_size;
    
    style->ttl_shape = malloc(ttl_shape_len * sizeof(int64_t));
    memcpy(style->ttl_shape, ttl_shape, ttl_shape_len * sizeof(int64_t));
    style->ttl_shape_len = ttl_shape_len;
    
    style->dp_data = malloc(dp_data_size * sizeof(float));
    memcpy(style->dp_data, dp_data, dp_data_size * sizeof(float));
    style->dp_data_size = dp_data_size;
    
    style->dp_shape = malloc(dp_shape_len * sizeof(int64_t));
    memcpy(style->dp_shape, dp_shape, dp_shape_len * sizeof(int64_t));
    style->dp_shape_len = dp_shape_len;
    
    return style;
}

void style_free(Style* style) {
    if (style) {
        free(style->ttl_data);
        free(style->ttl_shape);
        free(style->dp_data);
        free(style->dp_shape);
        free(style);
    }
}

/* Utility functions */
float*** length_to_mask(const int64_t* lengths, int count, int max_len) {
    if (max_len == -1) {
        max_len = 0;
        for (int i = 0; i < count; i++) {
            if (lengths[i] > max_len) max_len = lengths[i];
        }
    }
    
    float*** mask = malloc(count * sizeof(float**));
    for (int i = 0; i < count; i++) {
        mask[i] = malloc(1 * sizeof(float*));
        mask[i][0] = malloc(max_len * sizeof(float));
        for (int j = 0; j < max_len; j++) {
            mask[i][0][j] = (j < lengths[i]) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

float*** get_latent_mask(const int64_t* wav_lengths, int count, int base_chunk_size, int chunk_compress_factor) {
    int latent_size = base_chunk_size * chunk_compress_factor;
    int64_t* latent_lengths = malloc(count * sizeof(int64_t));
    for (int i = 0; i < count; i++) {
        latent_lengths[i] = (wav_lengths[i] + latent_size - 1) / latent_size;
    }
    float*** mask = length_to_mask(latent_lengths, count, -1);
    free(latent_lengths);
    return mask;
}

void free_3d_float_array(float*** array, int dim1, int dim2) {
    if (array) {
        for (int i = 0; i < dim1; i++) {
            if (array[i]) {
                for (int j = 0; j < dim2; j++) {
                    free(array[i][j]);
                }
                free(array[i]);
            }
        }
        free(array);
    }
}

void free_2d_int64_array(int64_t** array, int rows) {
    if (array) {
        for (int i = 0; i < rows; i++) {
            free(array[i]);
        }
        free(array);
    }
}

/* ONNX model loading */
OrtSession* load_onnx(OrtEnv* env, const char* onnx_path, const OrtSessionOptions* opts) {
    if (!env || !onnx_path || !opts) {
        fprintf(stderr, "Error: NULL parameter passed to load_onnx\n");
        return NULL;
    }
    
    OrtSession* session;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = g_ort->CreateSession(env, onnx_path, opts, &session);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error creating session: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }
    return session;
}

OnnxModels* load_onnx_all(OrtEnv* env, const char* onnx_dir, const OrtSessionOptions* opts) {
    if (!env || !onnx_dir || !opts) {
        fprintf(stderr, "Error: NULL parameter passed to load_onnx_all\n");
        return NULL;
    }
    
    OnnxModels* models = malloc(sizeof(OnnxModels));
    char path[1024];
    
    snprintf(path, sizeof(path), "%s/duration_predictor.onnx", onnx_dir);
    models->dp = load_onnx(env, path, opts);
    
    snprintf(path, sizeof(path), "%s/text_encoder.onnx", onnx_dir);
    models->text_enc = load_onnx(env, path, opts);
    
    snprintf(path, sizeof(path), "%s/vector_estimator.onnx", onnx_dir);
    models->vector_est = load_onnx(env, path, opts);
    
    snprintf(path, sizeof(path), "%s/vocoder.onnx", onnx_dir);
    models->vocoder = load_onnx(env, path, opts);
    
    if (!models->dp || !models->text_enc || !models->vector_est || !models->vocoder) {
        onnx_models_free(models);
        return NULL;
    }
    
    return models;
}

void onnx_models_free(OnnxModels* models) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (models) {
        if (models->dp) g_ort->ReleaseSession(models->dp);
        if (models->text_enc) g_ort->ReleaseSession(models->text_enc);
        if (models->vector_est) g_ort->ReleaseSession(models->vector_est);
        if (models->vocoder) g_ort->ReleaseSession(models->vocoder);
        free(models);
    }
}

/* Configuration loading */
Config* load_cfgs(const char* onnx_dir) {
    if (!onnx_dir) {
        fprintf(stderr, "Error: onnx_dir is NULL\n");
        return NULL;
    }
    
    char cfg_path[1024];
    snprintf(cfg_path, sizeof(cfg_path), "%s/tts.json", onnx_dir);
    
    FILE* file = fopen(cfg_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", cfg_path);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* json_str = malloc(file_size + 1);
    if (!json_str) {
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(json_str, 1, file_size, file);
    fclose(file);
    json_str[file_size] = 0;
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Failed to read complete config file\n");
        free(json_str);
        return NULL;
    }
    
    cJSON* json = cJSON_Parse(json_str);
    free(json_str);
    
    if (!json) {
        fprintf(stderr, "Failed to parse config JSON\n");
        return NULL;
    }
    
    Config* cfg = malloc(sizeof(Config));
    if (!cfg) {
        cJSON_Delete(json);
        return NULL;
    }
    
    cJSON* ae = cJSON_GetObjectItem(json, "ae");
    cfg->ae.sample_rate = cJSON_GetObjectItem(ae, "sample_rate")->valueint;
    cfg->ae.base_chunk_size = cJSON_GetObjectItem(ae, "base_chunk_size")->valueint;
    
    cJSON* ttl = cJSON_GetObjectItem(json, "ttl");
    cfg->ttl.chunk_compress_factor = cJSON_GetObjectItem(ttl, "chunk_compress_factor")->valueint;
    cfg->ttl.latent_dim = cJSON_GetObjectItem(ttl, "latent_dim")->valueint;
    
    cJSON_Delete(json);
    return cfg;
}

UnicodeProcessor* load_text_processor(const char* onnx_dir) {
    if (!onnx_dir) {
        fprintf(stderr, "Error: onnx_dir is NULL\n");
        return NULL;
    }
    
    char path[1024];
    snprintf(path, sizeof(path), "%s/unicode_indexer.json", onnx_dir);
    return unicode_processor_create(path);
}

/* Voice style loading */
Style* load_voice_style(const char** voice_style_paths, int count, int verbose) {
    if (!voice_style_paths || count <= 0) {
        fprintf(stderr, "Error: Invalid voice_style_paths or count\n");
        return NULL;
    }
    
    FILE* first_file = fopen(voice_style_paths[0], "r");
    if (!first_file) {
        fprintf(stderr, "Failed to open voice style file: %s\n", voice_style_paths[0]);
        return NULL;
    }
    
    fseek(first_file, 0, SEEK_END);
    long file_size = ftell(first_file);
    fseek(first_file, 0, SEEK_SET);
    
    char* json_str = malloc(file_size + 1);
    if (!json_str) {
        fclose(first_file);
        return NULL;
    }
    size_t bytes_read = fread(json_str, 1, file_size, first_file);
    fclose(first_file);
    json_str[file_size] = 0;
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Failed to read complete voice style file\n");
        free(json_str);
        return NULL;
    }
    
    cJSON* first_json = cJSON_Parse(json_str);
    free(json_str);
    
    cJSON* ttl_obj = cJSON_GetObjectItem(first_json, "style_ttl");
    cJSON* ttl_dims_arr = cJSON_GetObjectItem(ttl_obj, "dims");
    int64_t ttl_dim1 = cJSON_GetArrayItem(ttl_dims_arr, 1)->valueint;
    int64_t ttl_dim2 = cJSON_GetArrayItem(ttl_dims_arr, 2)->valueint;
    
    cJSON* dp_obj = cJSON_GetObjectItem(first_json, "style_dp");
    cJSON* dp_dims_arr = cJSON_GetObjectItem(dp_obj, "dims");
    int64_t dp_dim1 = cJSON_GetArrayItem(dp_dims_arr, 1)->valueint;
    int64_t dp_dim2 = cJSON_GetArrayItem(dp_dims_arr, 2)->valueint;
    
    cJSON_Delete(first_json);
    
    size_t ttl_size = count * ttl_dim1 * ttl_dim2;
    size_t dp_size = count * dp_dim1 * dp_dim2;
    float* ttl_flat = malloc(ttl_size * sizeof(float));
    float* dp_flat = malloc(dp_size * sizeof(float));
    
    for (int i = 0; i < count; i++) {
        FILE* file = fopen(voice_style_paths[i], "r");
        if (!file) {
            fprintf(stderr, "Failed to open voice style file: %s\n", voice_style_paths[i]);
            free(ttl_flat);
            free(dp_flat);
            return NULL;
        }
        
        fseek(file, 0, SEEK_END);
        file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        json_str = malloc(file_size + 1);
        if (!json_str) {
            fclose(file);
            free(ttl_flat);
            free(dp_flat);
            return NULL;
        }
        bytes_read = fread(json_str, 1, file_size, file);
        fclose(file);
        json_str[file_size] = 0;
        
        if (bytes_read != (size_t)file_size) {
            fprintf(stderr, "Failed to read voice style file %d\n", i);
            free(json_str);
            free(ttl_flat);
            free(dp_flat);
            return NULL;
        }
        
        cJSON* json = cJSON_Parse(json_str);
        free(json_str);
        
        cJSON* ttl_data = cJSON_GetObjectItem(cJSON_GetObjectItem(json, "style_ttl"), "data");
        cJSON* dp_data = cJSON_GetObjectItem(cJSON_GetObjectItem(json, "style_dp"), "data");
        
        size_t ttl_idx = i * ttl_dim1 * ttl_dim2;
        
        /* Check if data is a flat array or nested array */
        cJSON* first_ttl = cJSON_GetArrayItem(ttl_data, 0);
        int is_ttl_flat = (first_ttl && !cJSON_IsArray(first_ttl));
        
        if (is_ttl_flat) {
            /* Flat array format from voice_builder */
            cJSON* val;
            cJSON_ArrayForEach(val, ttl_data) {
                ttl_flat[ttl_idx++] = (float)val->valuedouble;
            }
        } else {
            /* Nested array format from original voice styles */
            cJSON* batch;
            cJSON_ArrayForEach(batch, ttl_data) {
                cJSON* row;
                cJSON_ArrayForEach(row, batch) {
                    cJSON* val;
                    cJSON_ArrayForEach(val, row) {
                        ttl_flat[ttl_idx++] = (float)val->valuedouble;
                    }
                }
            }
        }
        
        size_t dp_idx = i * dp_dim1 * dp_dim2;
        
        /* Check if data is a flat array or nested array */
        cJSON* first_dp = cJSON_GetArrayItem(dp_data, 0);
        int is_dp_flat = (first_dp && !cJSON_IsArray(first_dp));
        
        if (is_dp_flat) {
            /* Flat array format from voice_builder */
            cJSON* val;
            cJSON_ArrayForEach(val, dp_data) {
                dp_flat[dp_idx++] = (float)val->valuedouble;
            }
        } else {
            /* Nested array format from original voice styles */
            cJSON* batch;
            cJSON_ArrayForEach(batch, dp_data) {
                cJSON* row;
                cJSON_ArrayForEach(row, batch) {
                    cJSON* val;
                    cJSON_ArrayForEach(val, row) {
                        dp_flat[dp_idx++] = (float)val->valuedouble;
                    }
                }
            }
        }
        
        cJSON_Delete(json);
    }
    
    int64_t ttl_shape[] = {count, ttl_dim1, ttl_dim2};
    int64_t dp_shape[] = {count, dp_dim1, dp_dim2};
    
    Style* style = style_create(ttl_flat, ttl_size, ttl_shape, 3, dp_flat, dp_size, dp_shape, 3);
    
    free(ttl_flat);
    free(dp_flat);
    
    if (verbose) {
        printf("Loaded %d voice styles\n", count);
    }
    
    return style;
}

/* Global tensor buffers - kept alive for tensor lifetime */
static float** g_tensor_buffers_float = NULL;
static int64_t** g_tensor_buffers_int64 = NULL;
static size_t g_tensor_float_count = 0;
static size_t g_tensor_int64_count = 0;
static size_t g_tensor_float_capacity = 0;
static size_t g_tensor_int64_capacity = 0;

static void store_float_buffer(float* buffer) {
    if (g_tensor_float_count >= g_tensor_float_capacity) {
        size_t new_capacity = (g_tensor_float_capacity == 0) ? 16 : g_tensor_float_capacity * 2;
        float** new_buffers = realloc(g_tensor_buffers_float, new_capacity * sizeof(float*));
        if (!new_buffers) {
            fprintf(stderr, "Error: Failed to allocate memory for tensor buffers\n");
            free(buffer);
            return;
        }
        g_tensor_buffers_float = new_buffers;
        g_tensor_float_capacity = new_capacity;
    }
    g_tensor_buffers_float[g_tensor_float_count++] = buffer;
}

static void store_int64_buffer(int64_t* buffer) {
    if (g_tensor_int64_count >= g_tensor_int64_capacity) {
        size_t new_capacity = (g_tensor_int64_capacity == 0) ? 16 : g_tensor_int64_capacity * 2;
        int64_t** new_buffers = realloc(g_tensor_buffers_int64, new_capacity * sizeof(int64_t*));
        if (!new_buffers) {
            fprintf(stderr, "Error: Failed to allocate memory for tensor buffers\n");
            free(buffer);
            return;
        }
        g_tensor_buffers_int64 = new_buffers;
        g_tensor_int64_capacity = new_capacity;
    }
    g_tensor_buffers_int64[g_tensor_int64_count++] = buffer;
}

void clear_tensor_buffers() {
    for (size_t i = 0; i < g_tensor_float_count; i++) {
        free(g_tensor_buffers_float[i]);
    }
    free(g_tensor_buffers_float);
    g_tensor_buffers_float = NULL;
    g_tensor_float_count = 0;
    g_tensor_float_capacity = 0;
    
    for (size_t i = 0; i < g_tensor_int64_count; i++) {
        free(g_tensor_buffers_int64[i]);
    }
    free(g_tensor_buffers_int64);
    g_tensor_buffers_int64 = NULL;
    g_tensor_int64_count = 0;
    g_tensor_int64_capacity = 0;
}

/* Tensor utilities */
OrtValue* array_to_tensor_3d(
    OrtMemoryInfo* memory_info,
    float*** array,
    int64_t dim0, int64_t dim1, int64_t dim2
) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    size_t total_size = dim0 * dim1 * dim2;
    float* flat = malloc(total_size * sizeof(float));
    if (!flat) {
        fprintf(stderr, "Error: Failed to allocate memory for tensor data\n");
        return NULL;
    }
    
    size_t idx = 0;
    for (int64_t i = 0; i < dim0; i++) {
        for (int64_t j = 0; j < dim1; j++) {
            for (int64_t k = 0; k < dim2; k++) {
                flat[idx++] = array[i][j][k];
            }
        }
    }
    
    int64_t shape[] = {dim0, dim1, dim2};
    OrtValue* tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, flat, total_size * sizeof(float), shape, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor
    );
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error creating tensor: %s\n", msg);
        g_ort->ReleaseStatus(status);
        free(flat);
        return NULL;
    }
    
    /* Store buffer to keep it alive - it will be freed when tensor is released or manually cleared */
    store_float_buffer(flat);
    
    return tensor;
}

OrtValue* int_array_to_tensor_2d(
    OrtMemoryInfo* memory_info,
    int64_t** array,
    int64_t dim0, int64_t dim1
) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    size_t total_size = dim0 * dim1;
    int64_t* flat = malloc(total_size * sizeof(int64_t));
    if (!flat) {
        fprintf(stderr, "Error: Failed to allocate memory for tensor data\n");
        return NULL;
    }
    
    size_t idx = 0;
    for (int64_t i = 0; i < dim0; i++) {
        for (int64_t j = 0; j < dim1; j++) {
            flat[idx++] = array[i][j];
        }
    }
    
    int64_t shape[] = {dim0, dim1};
    OrtValue* tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, flat, total_size * sizeof(int64_t), shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &tensor
    );
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error creating int tensor: %s\n", msg);
        g_ort->ReleaseStatus(status);
        free(flat);
        return NULL;
    }
    
    /* Store buffer to keep it alive - it will be freed when tensor is released or manually cleared */
    store_int64_buffer(flat);
    
    return tensor;
}

/* TextToSpeech implementation */
TextToSpeech* tts_create(
    const Config* cfgs,
    UnicodeProcessor* text_processor,
    OrtSession* dp_session,
    OrtSession* text_enc_session,
    OrtSession* vector_est_session,
    OrtSession* vocoder_session
) {
    TextToSpeech* tts = malloc(sizeof(TextToSpeech));
    tts->cfgs = *cfgs;
    tts->text_processor = text_processor;
    tts->dp_session = dp_session;
    tts->text_enc_session = text_enc_session;
    tts->vector_est_session = vector_est_session;
    tts->vocoder_session = vocoder_session;
    tts->sample_rate = cfgs->ae.sample_rate;
    tts->base_chunk_size = cfgs->ae.base_chunk_size;
    tts->chunk_compress_factor = cfgs->ttl.chunk_compress_factor;
    tts->ldim = cfgs->ttl.latent_dim;
    return tts;
}

void tts_free(TextToSpeech* tts) {
    if (tts) {
        free(tts);
    }
}

static float randn() {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    float u, v, s;
    do {
        u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = 1;
    return u * s;
}

static void sample_noisy_latent(
    TextToSpeech* tts,
    const float* duration,
    int bsz,
    float**** noisy_latent_out,
    int* latent_batch,
    int* latent_dim,
    int* latent_len,
    float**** latent_mask_out,
    int* mask_batch,
    int* mask_channels,
    int* mask_len
) {
    float wav_len_max = duration[0];
    for (int i = 1; i < bsz; i++) {
        if (duration[i] > wav_len_max) wav_len_max = duration[i];
    }
    wav_len_max *= tts->sample_rate;
    
    int64_t* wav_lengths = malloc(bsz * sizeof(int64_t));
    for (int i = 0; i < bsz; i++) {
        wav_lengths[i] = (int64_t)(duration[i] * tts->sample_rate);
    }
    
    int chunk_size = tts->base_chunk_size * tts->chunk_compress_factor;
    int llen = (int)((wav_len_max + chunk_size - 1) / chunk_size);
    int ldim = tts->ldim * tts->chunk_compress_factor;
    
    float*** noisy_latent = malloc(bsz * sizeof(float**));
    for (int b = 0; b < bsz; b++) {
        noisy_latent[b] = malloc(ldim * sizeof(float*));
        for (int d = 0; d < ldim; d++) {
            noisy_latent[b][d] = malloc(llen * sizeof(float));
            for (int t = 0; t < llen; t++) {
                noisy_latent[b][d][t] = randn();
            }
        }
    }
    
    float*** latent_mask = get_latent_mask(wav_lengths, bsz, tts->base_chunk_size, tts->chunk_compress_factor);
    free(wav_lengths);
    
    for (int b = 0; b < bsz; b++) {
        for (int d = 0; d < ldim; d++) {
            for (int t = 0; t < llen; t++) {
                noisy_latent[b][d][t] *= latent_mask[b][0][t];
            }
        }
    }
    
    *noisy_latent_out = noisy_latent;
    *latent_batch = bsz;
    *latent_dim = ldim;
    *latent_len = llen;
    *latent_mask_out = latent_mask;
    *mask_batch = bsz;
    *mask_channels = 1;
    *mask_len = llen;
}

static SynthesisResult* tts_infer(
    TextToSpeech* tts,
    OrtMemoryInfo* memory_info,
    const char** text_list,
    const char** lang_list,
    int batch_size,
    const Style* style,
    int total_step,
    float speed
) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = NULL;
    
    if (batch_size != style->ttl_shape[0]) {
        fprintf(stderr, "Batch size mismatch\n");
        return NULL;
    }
    
    int64_t** text_ids;
    int text_ids_rows, text_ids_cols;
    float*** text_mask;
    int text_mask_batch, text_mask_channels, text_mask_len;
    
    if (unicode_processor_call(tts->text_processor, text_list, lang_list, batch_size,
                               &text_ids, &text_ids_rows, &text_ids_cols,
                               &text_mask, &text_mask_batch, &text_mask_channels, &text_mask_len) != 0) {
        return NULL;
    }
    
    OrtValue* text_ids_tensor = int_array_to_tensor_2d(memory_info, text_ids, text_ids_rows, text_ids_cols);
    OrtValue* text_mask_tensor = array_to_tensor_3d(memory_info, text_mask, text_mask_batch, text_mask_channels, text_mask_len);
    
    OrtValue* style_dp_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, style->dp_data, style->dp_data_size * sizeof(float),
        style->dp_shape, style->dp_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &style_dp_tensor
    );
    if (status != NULL) {
        fprintf(stderr, "Error creating style_dp tensor: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(text_ids_tensor);
        g_ort->ReleaseValue(text_mask_tensor);
        return NULL;
    }
    
    const char* dp_input_names[] = {"text_ids", "style_dp", "text_mask"};
    const char* dp_output_names[] = {"duration"};
    OrtValue* dp_inputs[] = {text_ids_tensor, style_dp_tensor, text_mask_tensor};
    OrtValue* dp_outputs = NULL;
    
    status = g_ort->Run(tts->dp_session, NULL, dp_input_names, 
                                     (const OrtValue* const*)dp_inputs, 3,
                                     dp_output_names, 1, &dp_outputs);
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "DP error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }
    
    float* dur_data;
    g_ort->GetTensorMutableData(dp_outputs, (void**)&dur_data);
    float* duration = malloc(batch_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        duration[i] = dur_data[i] / speed;
    }
    
    g_ort->ReleaseValue(dp_outputs);
    g_ort->ReleaseValue(text_ids_tensor);
    g_ort->ReleaseValue(text_mask_tensor);
    g_ort->ReleaseValue(style_dp_tensor);
    
    text_ids_tensor = int_array_to_tensor_2d(memory_info, text_ids, text_ids_rows, text_ids_cols);
    text_mask_tensor = array_to_tensor_3d(memory_info, text_mask, text_mask_batch, text_mask_channels, text_mask_len);
    
    OrtValue* style_ttl_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, style->ttl_data, style->ttl_data_size * sizeof(float),
        style->ttl_shape, style->ttl_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &style_ttl_tensor
    );
    if (status != NULL) {
        fprintf(stderr, "Error creating style_ttl tensor: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(text_ids_tensor);
        g_ort->ReleaseValue(text_mask_tensor);
        return NULL;
    }
    
    const char* text_enc_input_names[] = {"text_ids", "style_ttl", "text_mask"};
    const char* text_enc_output_names[] = {"text_emb"};
    OrtValue* text_enc_inputs[] = {text_ids_tensor, style_ttl_tensor, text_mask_tensor};
    OrtValue* text_enc_outputs = NULL;
    
    status = g_ort->Run(tts->text_enc_session, NULL, text_enc_input_names,
                        (const OrtValue* const*)text_enc_inputs, 3,
                        text_enc_output_names, 1, &text_enc_outputs);
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Text encoder error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }
    
    OrtTensorTypeAndShapeInfo* text_emb_info;
    g_ort->GetTensorTypeAndShape(text_enc_outputs, &text_emb_info);
    size_t text_emb_count;
    g_ort->GetTensorShapeElementCount(text_emb_info, &text_emb_count);
    float* text_emb_data;
    g_ort->GetTensorMutableData(text_enc_outputs, (void**)&text_emb_data);
    float* text_emb_copy = malloc(text_emb_count * sizeof(float));
    memcpy(text_emb_copy, text_emb_data, text_emb_count * sizeof(float));
    
    size_t text_emb_ndim;
    g_ort->GetDimensionsCount(text_emb_info, &text_emb_ndim);
    int64_t* text_emb_shape = malloc(text_emb_ndim * sizeof(int64_t));
    g_ort->GetDimensions(text_emb_info, text_emb_shape, text_emb_ndim);
    g_ort->ReleaseTensorTypeAndShapeInfo(text_emb_info);
    
    g_ort->ReleaseValue(text_enc_outputs);
    g_ort->ReleaseValue(text_ids_tensor);
    g_ort->ReleaseValue(text_mask_tensor);
    g_ort->ReleaseValue(style_ttl_tensor);
    
    float*** noisy_latent;
    int latent_batch, latent_dim, latent_len;
    float*** latent_mask;
    int mask_batch, mask_channels, mask_len;
    
    sample_noisy_latent(tts, duration, batch_size, &noisy_latent, &latent_batch, 
                       &latent_dim, &latent_len, &latent_mask, &mask_batch, &mask_channels, &mask_len);
    
    float* total_step_vec = malloc(batch_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        total_step_vec[i] = (float)total_step;
    }
    
    for (int step = 0; step < total_step; step++) {
        float* current_step_vec = malloc(batch_size * sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            current_step_vec[i] = (float)step;
        }
        
        OrtValue* noisy_latent_tensor = array_to_tensor_3d(memory_info, noisy_latent, latent_batch, latent_dim, latent_len);
        text_mask_tensor = array_to_tensor_3d(memory_info, text_mask, text_mask_batch, text_mask_channels, text_mask_len);
        OrtValue* latent_mask_tensor = array_to_tensor_3d(memory_info, latent_mask, mask_batch, mask_channels, mask_len);
        
        OrtValue* text_emb_tensor = NULL;
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, text_emb_copy, text_emb_count * sizeof(float),
            text_emb_shape, text_emb_ndim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &text_emb_tensor
        );
        
        style_ttl_tensor = NULL;
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, style->ttl_data, style->ttl_data_size * sizeof(float),
            style->ttl_shape, style->ttl_shape_len,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &style_ttl_tensor
        );
        
        int64_t scalar_shape[] = {batch_size};
        OrtValue* total_step_tensor = NULL;
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, total_step_vec, batch_size * sizeof(float),
            scalar_shape, 1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &total_step_tensor
        );
        
        OrtValue* current_step_tensor = NULL;
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, current_step_vec, batch_size * sizeof(float),
            scalar_shape, 1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &current_step_tensor
        );
        
        const char* vector_est_input_names[] = {
            "noisy_latent", "text_emb", "style_ttl", "text_mask", "latent_mask", "total_step", "current_step"
        };
        const char* vector_est_output_names[] = {"denoised_latent"};
        OrtValue* vector_est_inputs[] = {
            noisy_latent_tensor, text_emb_tensor, style_ttl_tensor,
            text_mask_tensor, latent_mask_tensor, total_step_tensor, current_step_tensor
        };
        OrtValue* vector_est_outputs = NULL;
        
        status = g_ort->Run(tts->vector_est_session, NULL, vector_est_input_names,
                           (const OrtValue* const*)vector_est_inputs, 7,
                           vector_est_output_names, 1, &vector_est_outputs);
        
        if (status != NULL) {
            const char* msg = g_ort->GetErrorMessage(status);
            fprintf(stderr, "Vector estimator error: %s\n", msg);
            g_ort->ReleaseStatus(status);
            return NULL;
        }
        
        float* denoised_data;
        g_ort->GetTensorMutableData(vector_est_outputs, (void**)&denoised_data);
        
        size_t idx = 0;
        for (int b = 0; b < latent_batch; b++) {
            for (int d = 0; d < latent_dim; d++) {
                for (int t = 0; t < latent_len; t++) {
                    noisy_latent[b][d][t] = denoised_data[idx++];
                }
            }
        }
        
        g_ort->ReleaseValue(vector_est_outputs);
        g_ort->ReleaseValue(noisy_latent_tensor);
        g_ort->ReleaseValue(text_emb_tensor);
        g_ort->ReleaseValue(style_ttl_tensor);
        g_ort->ReleaseValue(text_mask_tensor);
        g_ort->ReleaseValue(latent_mask_tensor);
        g_ort->ReleaseValue(total_step_tensor);
        g_ort->ReleaseValue(current_step_tensor);
        free(current_step_vec);
    }
    
    free(total_step_vec);
    free(text_emb_copy);
    free(text_emb_shape);
    
    OrtValue* latent_tensor = array_to_tensor_3d(memory_info, noisy_latent, latent_batch, latent_dim, latent_len);
    
    const char* vocoder_input_names[] = {"latent"};
    const char* vocoder_output_names[] = {"wav_tts"};
    OrtValue* vocoder_inputs[] = {latent_tensor};
    OrtValue* vocoder_outputs = NULL;
    
    status = g_ort->Run(tts->vocoder_session, NULL, vocoder_input_names,
                       (const OrtValue* const*)vocoder_inputs, 1,
                       vocoder_output_names, 1, &vocoder_outputs);
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Vocoder error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }
    
    OrtTensorTypeAndShapeInfo* wav_info;
    g_ort->GetTensorTypeAndShape(vocoder_outputs, &wav_info);
    size_t wav_count;
    g_ort->GetTensorShapeElementCount(wav_info, &wav_count);
    float* wav_data;
    g_ort->GetTensorMutableData(vocoder_outputs, (void**)&wav_data);
    g_ort->ReleaseTensorTypeAndShapeInfo(wav_info);
    
    SynthesisResult* result = malloc(sizeof(SynthesisResult));
    result->wav = malloc(wav_count * sizeof(float));
    memcpy(result->wav, wav_data, wav_count * sizeof(float));
    result->wav_size = wav_count;
    result->duration = duration;
    result->duration_count = batch_size;
    
    g_ort->ReleaseValue(vocoder_outputs);
    g_ort->ReleaseValue(latent_tensor);
    
    free_2d_int64_array(text_ids, text_ids_rows);
    free_3d_float_array(text_mask, text_mask_batch, text_mask_channels);
    free_3d_float_array(noisy_latent, latent_batch, latent_dim);
    free_3d_float_array(latent_mask, mask_batch, mask_channels);
    
    return result;
}

SynthesisResult* tts_batch(
    TextToSpeech* tts,
    OrtMemoryInfo* memory_info,
    const char** text_list,
    const char** lang_list,
    int batch_size,
    const Style* style,
    int total_step,
    float speed
) {
    return tts_infer(tts, memory_info, text_list, lang_list, batch_size, style, total_step, speed);
}

void synthesis_result_free(SynthesisResult* result) {
    if (result) {
        free(result->wav);
        free(result->duration);
        free(result);
    }
}

SynthesisResult* tts_call(
    TextToSpeech* tts,
    OrtMemoryInfo* memory_info,
    const char* text,
    const char* lang,
    const Style* style,
    int total_step,
    float speed,
    float silence_duration
) {
    if (style->ttl_shape[0] != 1) {
        fprintf(stderr, "Single speaker TTS only supports single style\n");
        return NULL;
    }
    
    int max_len = (strcmp(lang, "ko") == 0) ? 120 : 300;
    int chunk_count;
    char** text_chunks = chunk_text(text, max_len, &chunk_count);
    
    if (chunk_count > 1) {
        fprintf(stderr, "Info: Text split into %d chunks\n", chunk_count);
        for (int i = 0; i < chunk_count; i++) {
            fprintf(stderr, "  Chunk %d (len=%zu): \"%.50s%s\"\n", 
                    i, strlen(text_chunks[i]), text_chunks[i],
                    strlen(text_chunks[i]) > 50 ? "..." : "");
        }
    }
    
    float* wav_cat = NULL;
    size_t wav_cat_size = 0;
    float dur_cat = 0.0f;
    
    for (int i = 0; i < chunk_count; i++) {
        const char* chunk_text_list[] = {text_chunks[i]};
        const char* chunk_lang_list[] = {lang};
        
        SynthesisResult* chunk_result = tts_infer(tts, memory_info, chunk_text_list, chunk_lang_list, 
                                                  1, style, total_step, speed);
        if (!chunk_result) {
            free_string_array(text_chunks, chunk_count);
            if (wav_cat) free(wav_cat);
            return NULL;
        }
        
        if (wav_cat == NULL) {
            wav_cat = chunk_result->wav;
            wav_cat_size = chunk_result->wav_size;
            dur_cat = chunk_result->duration[0];
            free(chunk_result->duration);
            free(chunk_result);
        } else {
            int silence_len = (int)(silence_duration * tts->sample_rate);
            size_t new_size = wav_cat_size + silence_len + chunk_result->wav_size;
            float* new_wav = malloc(new_size * sizeof(float));
            
            memcpy(new_wav, wav_cat, wav_cat_size * sizeof(float));
            memset(new_wav + wav_cat_size, 0, silence_len * sizeof(float));
            memcpy(new_wav + wav_cat_size + silence_len, chunk_result->wav, 
                   chunk_result->wav_size * sizeof(float));
            
            free(wav_cat);
            wav_cat = new_wav;
            wav_cat_size = new_size;
            dur_cat += chunk_result->duration[0] + silence_duration;
            
            free(chunk_result->wav);
            free(chunk_result->duration);
            free(chunk_result);
        }
    }
    
    free_string_array(text_chunks, chunk_count);
    
    SynthesisResult* result = malloc(sizeof(SynthesisResult));
    result->wav = wav_cat;
    result->wav_size = wav_cat_size;
    result->duration = malloc(sizeof(float));
    result->duration[0] = dur_cat;
    result->duration_count = 1;
    
    return result;
}

TextToSpeech* load_text_to_speech(OrtEnv* env, const char* onnx_dir, int use_gpu) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    if (use_gpu) {
        fprintf(stderr, "GPU mode is not supported yet\n");
        return NULL;
    }
    printf("Using CPU for inference\n");
    
    OrtSessionOptions* opts;
    g_ort->CreateSessionOptions(&opts);
    
    Config* cfgs = load_cfgs(onnx_dir);
    if (!cfgs) return NULL;
    
    OnnxModels* models = load_onnx_all(env, onnx_dir, opts);
    if (!models) {
        free(cfgs);
        g_ort->ReleaseSessionOptions(opts);
        return NULL;
    }
    
    UnicodeProcessor* text_processor = load_text_processor(onnx_dir);
    if (!text_processor) {
        onnx_models_free(models);
        free(cfgs);
        g_ort->ReleaseSessionOptions(opts);
        return NULL;
    }
    
    TextToSpeech* tts = tts_create(cfgs, text_processor, models->dp, models->text_enc, 
                                   models->vector_est, models->vocoder);
    
    free(cfgs);
    free(models);
    g_ort->ReleaseSessionOptions(opts);
    
    return tts;
}

char* sanitize_filename(const char* text, int max_len) {
    size_t len = strlen(text);
    char* result = malloc(max_len + 1);
    if (!result) {
        fprintf(stderr, "Error: Failed to allocate memory for sanitized filename\n");
        return strdup("output");
    }
    int pos = 0;
    size_t i = 0;
    
    while (i < len && pos < max_len) {
        unsigned char c = (unsigned char)text[i];
        
        if (isalnum(c) || c == '_') {
            result[pos++] = text[i];
            i++;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len && pos + 1 < max_len) {
            result[pos++] = text[i++];
            result[pos++] = text[i++];
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len && pos + 2 < max_len) {
            result[pos++] = text[i++];
            result[pos++] = text[i++];
            result[pos++] = text[i++];
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len && pos + 3 < max_len) {
            result[pos++] = text[i++];
            result[pos++] = text[i++];
            result[pos++] = text[i++];
            result[pos++] = text[i++];
        } else {
            result[pos++] = '_';
            i++;
        }
    }
    result[pos] = 0;
    return result;
}

char** chunk_text(const char* text, int max_len, int* chunk_count) {
    size_t len = strlen(text);
    if (len <= (size_t)max_len) {
        char** chunks = malloc(sizeof(char*));
        if (!chunks) {
            *chunk_count = 0;
            return NULL;
        }
        chunks[0] = strdup(text);
        *chunk_count = 1;
        return chunks;
    }
    
    int max_chunks = 1000;
    char** chunks = malloc(max_chunks * sizeof(char*));
    if (!chunks) {
        *chunk_count = 0;
        return NULL;
    }
    int count = 0;
    size_t start = 0;
    
    while (start < len) {
        if (count >= max_chunks) {
            fprintf(stderr, "Warning: Text has more than %d chunks, truncating\n", max_chunks);
            break;
        }
        
        size_t end = start + max_len;
        if (end >= len) {
            chunks[count] = strdup(text + start);
            if (chunks[count]) count++;
            break;
        }
        
        while (end > start && text[end] != ' ' && text[end] != '.' && 
               text[end] != '!' && text[end] != '?') {
            end--;
        }
        
        if (end == start) {
            end = start + max_len;
        }
        
        size_t chunk_len = end - start;
        chunks[count] = malloc(chunk_len + 1);
        if (chunks[count]) {
            memcpy(chunks[count], text + start, chunk_len);
            chunks[count][chunk_len] = 0;
            count++;
        }
        
        start = end + 1;
        while (start < len && text[start] == ' ') start++;
    }
    
    *chunk_count = count;
    return chunks;
}

void free_string_array(char** array, int count) {
    if (array) {
        for (int i = 0; i < count; i++) {
            free(array[i]);
        }
        free(array);
    }
}
