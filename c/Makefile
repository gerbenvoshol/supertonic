# Makefile for Supertonic TTS C implementation

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -fPIC
LDFLAGS = -lm

# Detect OS
UNAME_S := $(shell uname -s)

# Platform-specific settings
ifeq ($(UNAME_S),Darwin)
    # macOS
    ONNXRUNTIME_ROOT ?= /usr/local
    CJSON_ROOT ?= /usr/local
    CFLAGS += -I$(ONNXRUNTIME_ROOT)/include -I$(CJSON_ROOT)/include
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -L$(CJSON_ROOT)/lib -lonnxruntime -lcjson
    # Add rpath for runtime linking
    LDFLAGS += -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib
else ifeq ($(UNAME_S),Linux)
    # Linux
    ONNXRUNTIME_ROOT ?= /usr/local
    CJSON_ROOT ?= /usr/local
    CFLAGS += -I$(ONNXRUNTIME_ROOT)/include -I$(CJSON_ROOT)/include
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -L$(CJSON_ROOT)/lib -lonnxruntime -lcjson
    # Add rpath for runtime linking
    LDFLAGS += -Wl,-rpath=$(ONNXRUNTIME_ROOT)/lib
endif

# Source files
SOURCES = supertonic.c wav_utils.c
OBJECTS = $(SOURCES:.c=.o)
EXAMPLE_SRC = example_onnx.c
EXAMPLE_BIN = example_onnx
AUDIOBOOK_SRC = audiobook_generator.c
AUDIOBOOK_BIN = audiobook_generator

# Targets
.PHONY: all clean

all: $(EXAMPLE_BIN) $(AUDIOBOOK_BIN)

$(EXAMPLE_BIN): $(OBJECTS) $(EXAMPLE_SRC)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_SRC) $(OBJECTS) $(LDFLAGS)
	@echo ""
	@echo "Build successful: $(EXAMPLE_BIN)"
	@echo ""

$(AUDIOBOOK_BIN): $(OBJECTS) $(AUDIOBOOK_SRC)
	$(CC) $(CFLAGS) -o $@ $(AUDIOBOOK_SRC) $(OBJECTS) $(LDFLAGS)
	@echo ""
	@echo "Build successful: $(AUDIOBOOK_BIN)"
	@echo "Run with: ./$(AUDIOBOOK_BIN) --input sample_text.txt"
	@echo ""

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXAMPLE_BIN) $(AUDIOBOOK_BIN)
	rm -rf results
	@echo "Clean complete"

# Help target
help:
	@echo "Supertonic TTS C Implementation Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build the example executable"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  ONNXRUNTIME_ROOT  - Path to ONNX Runtime installation (default: /usr/local)"
	@echo "  CJSON_ROOT        - Path to cJSON installation (default: /usr/local)"
	@echo ""
	@echo "Example:"
	@echo "  make ONNXRUNTIME_ROOT=/opt/onnxruntime CJSON_ROOT=/opt/cjson"
	@echo ""

# Dependencies
supertonic.o: supertonic.c supertonic.h wav_utils.h
wav_utils.o: wav_utils.c wav_utils.h
example_onnx.o: example_onnx.c supertonic.h wav_utils.h
audiobook_generator.o: audiobook_generator.c supertonic.h wav_utils.h
