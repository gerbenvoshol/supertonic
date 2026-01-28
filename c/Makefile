# Makefile for Supertonic TTS C implementation

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -fPIC
LDFLAGS = -lm

# Detect OS
UNAME_S := $(shell uname -s)

# ============================================================================
# Library Path Configuration
# ============================================================================
# You can specify custom library paths in two ways:
#
# 1. Create a Makefile.local file (recommended for persistent settings):
#    echo "ONNXRUNTIME_ROOT = /path/to/onnxruntime" > Makefile.local
#
# 2. Pass as command-line arguments:
#    make ONNXRUNTIME_ROOT=/path/to/onnxruntime
#
# 3. Set as environment variables:
#    export ONNXRUNTIME_ROOT=/path/to/onnxruntime
#    make
#
# Examples of ONNXRUNTIME_ROOT paths:
#   - System install: /usr/local
#   - Homebrew (macOS): /opt/homebrew (or /usr/local)
#   - Downloaded package: /home/user/Downloads/onnxruntime-linux-x64-1.23.2
#   - Custom install: /opt/onnxruntime
#
# Note: cJSON is now bundled in vendor/cjson/ and no longer requires external installation.
# ============================================================================

# Load user-specific settings if Makefile.local exists
-include Makefile.local

# Platform-specific settings
ifeq ($(UNAME_S),Darwin)
    # macOS
    ONNXRUNTIME_ROOT ?= /usr/local
    CFLAGS += -I$(ONNXRUNTIME_ROOT)/include
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime
    # Add rpath for runtime linking
    LDFLAGS += -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib
else ifeq ($(UNAME_S),Linux)
    # Linux
    ONNXRUNTIME_ROOT ?= /usr/local
    CFLAGS += -I$(ONNXRUNTIME_ROOT)/include
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime
    # Add rpath for runtime linking
    LDFLAGS += -Wl,-rpath=$(ONNXRUNTIME_ROOT)/lib
endif

# Source files
SOURCES = supertonic.c wav_utils.c vendor/cjson/cJSON.c
OBJECTS = $(SOURCES:.c=.o)
EXAMPLE_SRC = example_onnx.c
EXAMPLE_BIN = example_onnx
AUDIOBOOK_SRC = audiobook_generator.c
AUDIOBOOK_BIN = audiobook_generator

# Targets
.PHONY: all clean help show-config

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
	@echo "  make              - Build all executables"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo "  make show-config  - Display current configuration"
	@echo ""
	@echo "Configuring Library Paths:"
	@echo ""
	@echo "Method 1: Create Makefile.local (recommended for persistent settings)"
	@echo "  echo 'ONNXRUNTIME_ROOT = /home/user/Downloads/onnxruntime-linux-x64-1.23.2' > Makefile.local"
	@echo "  make"
	@echo ""
	@echo "Method 2: Command-line arguments (temporary)"
	@echo "  make ONNXRUNTIME_ROOT=/path/to/onnxruntime"
	@echo ""
	@echo "Method 3: Environment variables"
	@echo "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime"
	@echo "  make"
	@echo ""
	@echo "Common ONNXRUNTIME_ROOT paths:"
	@echo "  - System install:        /usr/local"
	@echo "  - Homebrew (macOS):      /opt/homebrew"
	@echo "  - Downloaded package:    /home/user/Downloads/onnxruntime-linux-x64-1.23.2"
	@echo "  - Custom installation:   /opt/onnxruntime"
	@echo ""
	@echo "Note: cJSON is now bundled in vendor/cjson/ - no external installation needed!"
	@echo ""
	@echo "Current defaults:"
	@echo "  ONNXRUNTIME_ROOT = $(ONNXRUNTIME_ROOT)"
	@echo ""

# Show current configuration
show-config:
	@echo "Current Build Configuration:"
	@echo "  OS:               $(UNAME_S)"
	@echo "  CC:               $(CC)"
	@echo "  ONNXRUNTIME_ROOT: $(ONNXRUNTIME_ROOT)"
	@echo "  CFLAGS:           $(CFLAGS)"
	@echo "  LDFLAGS:          $(LDFLAGS)"
	@echo ""
	@echo "Checking library paths:"
	@if [ -d "$(ONNXRUNTIME_ROOT)/include" ]; then \
		echo "  ✓ ONNX Runtime include found: $(ONNXRUNTIME_ROOT)/include"; \
	else \
		echo "  ✗ ONNX Runtime include NOT found: $(ONNXRUNTIME_ROOT)/include"; \
	fi
	@if [ -d "$(ONNXRUNTIME_ROOT)/lib" ]; then \
		echo "  ✓ ONNX Runtime lib found: $(ONNXRUNTIME_ROOT)/lib"; \
	else \
		echo "  ✗ ONNX Runtime lib NOT found: $(ONNXRUNTIME_ROOT)/lib"; \
	fi
	@if [ -f "vendor/cjson/cJSON.c" ]; then \
		echo "  ✓ cJSON source found: vendor/cjson/ (bundled)"; \
	else \
		echo "  ✗ cJSON source NOT found: vendor/cjson/"; \
	fi

# Dependencies
supertonic.o: supertonic.c supertonic.h wav_utils.h vendor/cjson/cJSON.h
wav_utils.o: wav_utils.c wav_utils.h
vendor/cjson/cJSON.o: vendor/cjson/cJSON.c vendor/cjson/cJSON.h
example_onnx.o: example_onnx.c supertonic.h wav_utils.h
audiobook_generator.o: audiobook_generator.c supertonic.h wav_utils.h
