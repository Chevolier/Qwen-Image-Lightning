#!/bin/bash

# SGLang Benchmark Script for Qwen-Image-Edit
# This script runs inference benchmarks using SGLang
#
# IMPORTANT: SGLang uses a model registry that requires HuggingFace model IDs.
# For local models, you have two options:
#   1. Use the HF model ID (e.g., "Qwen/Qwen-Image-Edit") - SGLang will download/cache
#   2. Create a symlink matching the HF path structure:
#      mkdir -p /path/to/models/Qwen && ln -s /actual/model/path /path/to/models/Qwen/Qwen-Image-Edit
#      Then use --model_path /path/to/models/Qwen/Qwen-Image-Edit
#
# The key is that the path must contain "Qwen/Qwen-Image-Edit" for SGLang to recognize it.

set -e

# Create output directories if they don't exist
mkdir -p logs
mkdir -p outputs

# Define configurations
STEPS=(50 16 8 4)
GPUS=(0 1 2 3)  # Adjust based on available GPUs

# Common parameters
PROMPT_LIST="examples/edit_prompt_list.txt"
IMAGE_PATH_LIST="examples/image_path_list.txt"
# Use HuggingFace model ID by default (SGLang registry requires this format)
# For local models, the path must contain the HF model structure (e.g., .../Qwen/Qwen-Image-Edit)
MODEL_PATH="Qwen/Qwen-Image-Edit"
MODEL_PREFIX="sglang-qwen-image-edit"
BASE_SEED=42
WIDTH=1664
HEIGHT=928
NUM_GPUS=1

# Optional: Path to local model directory for creating symlink
LOCAL_MODEL_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --local_model_dir)
            # If provided, creates a symlink structure that SGLang can recognize
            LOCAL_MODEL_DIR="$2"
            shift 2
            ;;
        --prompt_list)
            PROMPT_LIST="$2"
            shift 2
            ;;
        --image_path_list)
            IMAGE_PATH_LIST="$2"
            shift 2
            ;;
        --prefix)
            MODEL_PREFIX="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --single)
            # Run single experiment mode
            SINGLE_MODE=true
            SINGLE_STEPS="$2"
            shift 2
            ;;
        --gpu)
            SINGLE_GPU="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH      HuggingFace model ID or local path with HF structure"
            echo "                         (default: Qwen/Qwen-Image-Edit)"
            echo "  --local_model_dir DIR  Local model directory to create symlink from"
            echo "                         (creates structure: ./models/Qwen/Qwen-Image-Edit)"
            echo "  --prompt_list FILE     Path to prompt list file (default: ${PROMPT_LIST})"
            echo "  --image_path_list FILE Path to image path list file (default: ${IMAGE_PATH_LIST})"
            echo "  --prefix PREFIX        Output prefix (default: ${MODEL_PREFIX})"
            echo "  --width WIDTH          Output image width (default: ${WIDTH})"
            echo "  --height HEIGHT        Output image height (default: ${HEIGHT})"
            echo "  --num_gpus N           Number of GPUs per experiment (default: ${NUM_GPUS})"
            echo "  --single STEPS         Run single experiment with specified steps"
            echo "  --gpu GPU_ID           GPU ID for single experiment (default: 0)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Use HuggingFace model (will download if not cached)"
            echo "  $0 --single 8 --gpu 0"
            echo ""
            echo "  # Use local model directory (creates symlink)"
            echo "  $0 --local_model_dir /path/to/Qwen-Image-Edit --single 8 --gpu 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If local model directory is specified, create symlink structure
if [ -n "$LOCAL_MODEL_DIR" ]; then
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        SYMLINK_BASE="./models"
        mkdir -p "${SYMLINK_BASE}/Qwen"
        SYMLINK_PATH="${SYMLINK_BASE}/Qwen/Qwen-Image-Edit"

        if [ ! -L "$SYMLINK_PATH" ]; then
            echo "Creating symlink: ${SYMLINK_PATH} -> ${LOCAL_MODEL_DIR}"
            ln -sf "$(realpath "$LOCAL_MODEL_DIR")" "$SYMLINK_PATH"
        fi

        MODEL_PATH="$SYMLINK_PATH"
        echo "Using local model via symlink: ${MODEL_PATH}"
    else
        echo "Error: Local model directory does not exist: ${LOCAL_MODEL_DIR}"
        exit 1
    fi
fi

echo "=========================================="
echo "SGLang Benchmark for Qwen-Image-Edit"
echo "=========================================="
echo "Model Path: ${MODEL_PATH}"
echo "Prompt List: ${PROMPT_LIST}"
echo "Image Path List: ${IMAGE_PATH_LIST}"
echo "Output Prefix: ${MODEL_PREFIX}"
echo "Image Size: ${WIDTH}x${HEIGHT}"
echo "GPUs per experiment: ${NUM_GPUS}"
echo "=========================================="

if [ "$SINGLE_MODE" = true ]; then
    # Single experiment mode
    GPU=${SINGLE_GPU:-0}
    STEP=${SINGLE_STEPS}
    OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_steps"
    mkdir -p ${OUT_DIR}

    echo "Running single experiment with ${STEP} steps on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=${GPU} python generate_with_sglang.py \
        --model_path "${MODEL_PATH}" \
        --prompt_list_file ${PROMPT_LIST} \
        --image_path_list_file ${IMAGE_PATH_LIST} \
        --out_dir ${OUT_DIR} \
        --base_seed ${BASE_SEED} \
        --steps ${STEP} \
        --num_gpus ${NUM_GPUS} \
        --width ${WIDTH} \
        --height ${HEIGHT} 2>&1 | tee ${OUT_DIR}/benchmark.log

    echo "Experiment completed. Results saved to ${OUT_DIR}"
else
    # Parallel experiments mode
    echo "Starting parallel experiments..."

    for i in "${!STEPS[@]}"; do
        STEP=${STEPS[$i]}
        GPU=${GPUS[$i]}

        OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_steps"
        mkdir -p ${OUT_DIR}

        echo "Starting experiment with ${STEP} steps on GPU ${GPU}..."

        CUDA_VISIBLE_DEVICES=${GPU} nohup python generate_with_sglang.py \
            --model_path "${MODEL_PATH}" \
            --prompt_list_file ${PROMPT_LIST} \
            --image_path_list_file ${IMAGE_PATH_LIST} \
            --out_dir ${OUT_DIR} \
            --base_seed ${BASE_SEED} \
            --steps ${STEP} \
            --num_gpus ${NUM_GPUS} \
            --width ${WIDTH} \
            --height ${HEIGHT} > ${OUT_DIR}/benchmark.log 2>&1 &

        echo "  PID: $! -> ${OUT_DIR}/benchmark.log"
    done

    echo ""
    echo "All experiments started in background."
    echo "Monitor progress with: tail -f outputs/${MODEL_PREFIX}_*/benchmark.log"
    echo ""
fi
