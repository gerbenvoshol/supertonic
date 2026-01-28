#!/bin/bash

# Supertonic C Implementation - Resource Downloader
# This script downloads all necessary resources (ONNX models and voice styles)
#
# Usage:
#   ./resource.sh          - Download or update resources
#
# The script will:
#   - Check for Git LFS installation
#   - Download ONNX models and voice styles from Hugging Face
#   - Verify all required files are present
#   - Handle updates if resources already exist

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$PARENT_DIR/assets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Supertonic C - Resource Downloader"
echo "=========================================="
echo ""

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}Warning: Git LFS is not installed.${NC}"
    echo "The model files are stored using Git LFS and will not download properly without it."
    echo ""
    echo "Please install Git LFS:"
    echo "  macOS:   brew install git-lfs && git lfs install"
    echo "  Ubuntu:  sudo apt-get install git-lfs && git lfs install"
    echo "  Generic: https://git-lfs.com"
    echo ""
    echo -e "Do you want to continue anyway? (y/N): \c"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting. Please install Git LFS and run this script again."
        exit 1
    fi
    echo ""
else
    echo -e "${GREEN}✓ Git LFS is installed${NC}"
    
    # Initialize Git LFS if not already done
    if ! git lfs env &> /dev/null; then
        echo "Initializing Git LFS..."
        git lfs install
    fi
    echo ""
fi

# Check if assets directory already exists
if [ -d "$ASSETS_DIR" ]; then
    echo -e "${YELLOW}Assets directory already exists at: $ASSETS_DIR${NC}"
    echo ""
    echo "Options:"
    echo "  1) Update existing assets (git pull)"
    echo "  2) Delete and re-download all assets"
    echo "  3) Skip download and exit"
    echo -e "Enter your choice (1/2/3) [default: 1]: \c"
    read -r choice
    choice=${choice:-1}
    
    case $choice in
        1)
            echo ""
            echo "Updating existing assets..."
            cd "$ASSETS_DIR"
            if git pull; then
                echo -e "${GREEN}✓ Assets updated successfully${NC}"
            else
                echo -e "${RED}✗ Failed to update assets${NC}"
                exit 1
            fi
            ;;
        2)
            echo ""
            echo "Removing existing assets directory..."
            rm -rf "$ASSETS_DIR"
            echo "Downloading assets..."
            cd "$PARENT_DIR"
            if git clone https://huggingface.co/Supertone/supertonic-2 assets; then
                echo -e "${GREEN}✓ Assets downloaded successfully${NC}"
            else
                echo -e "${RED}✗ Failed to download assets${NC}"
                exit 1
            fi
            ;;
        3)
            echo "Skipping download."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "Assets directory not found. Downloading from Hugging Face..."
    echo "Repository: https://huggingface.co/Supertone/supertonic-2"
    echo ""
    
    cd "$PARENT_DIR"
    if git clone https://huggingface.co/Supertone/supertonic-2 assets; then
        echo ""
        echo -e "${GREEN}✓ Assets downloaded successfully${NC}"
    else
        echo ""
        echo -e "${RED}✗ Failed to download assets${NC}"
        echo "Please check your internet connection and try again."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Verifying downloaded resources..."
echo "=========================================="
echo ""

# Check for required directories and files
REQUIRED_DIRS=(
    "onnx"
    "voice_styles"
)

REQUIRED_FILES=(
    "onnx/config.json"
    "onnx/dp.onnx"
    "onnx/text_enc.onnx"
    "onnx/vector_est.onnx"
    "onnx/vocoder.onnx"
    "onnx/unicode_indexer.json"
    "voice_styles/M1.json"
    "voice_styles/F1.json"
)

all_found=true

# Check directories
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$ASSETS_DIR/$dir" ]; then
        echo -e "${GREEN}✓${NC} Found directory: $dir"
    else
        echo -e "${RED}✗${NC} Missing directory: $dir"
        all_found=false
    fi
done

echo ""

# Check files
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$ASSETS_DIR/$file" ]; then
        # Get file size
        size=$(du -h "$ASSETS_DIR/$file" | cut -f1)
        echo -e "${GREEN}✓${NC} Found file: $file ($size)"
    else
        echo -e "${RED}✗${NC} Missing file: $file"
        all_found=false
    fi
done

echo ""

if [ "$all_found" = true ]; then
    echo -e "${GREEN}=========================================="
    echo "✓ All resources verified successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "You can now build and run the C examples:"
    echo "  cd $SCRIPT_DIR"
    echo "  make"
    echo "  ./example_onnx"
    echo "  ./audiobook_generator --input sample_text.txt"
else
    echo -e "${RED}=========================================="
    echo "✗ Some resources are missing!"
    echo "==========================================${NC}"
    echo ""
    echo "This might be due to:"
    echo "  1. Incomplete download (check your internet connection)"
    echo "  2. Git LFS not properly initialized"
    echo "  3. Model files not pulled with Git LFS"
    echo ""
    echo "Try running: cd $ASSETS_DIR && git lfs pull"
    exit 1
fi

# Display summary
echo ""
echo "Resource Summary:"
echo "  Location: $ASSETS_DIR"
echo "  ONNX Models: $(ls -1 "$ASSETS_DIR/onnx"/*.onnx 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Voice Styles: $(ls -1 "$ASSETS_DIR/voice_styles"/*.json 2>/dev/null | wc -l | tr -d ' ') files"

# Calculate total size
if command -v du &> /dev/null; then
    total_size=$(du -sh "$ASSETS_DIR" 2>/dev/null | cut -f1)
    echo "  Total Size: $total_size"
fi

echo ""
echo -e "${GREEN}Done! Resources are ready to use.${NC}"
