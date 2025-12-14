#!/bin/bash

# Benchmark script for run_qwen_image_lightning.py (Qwen-Image-Edit-Lightning with cache-dit)
# This runs Qwen-Image-Edit with Lightning LoRA for fast inference

# Create output directories
mkdir -p logs
mkdir -p outputs

# Define step configurations and corresponding GPU devices
# Lightning supports 4 or 8 steps
STEPS=(8 4)
GPUS=(0 1)
# GPUS=(4 5)

# Common parameters
PROMPT_LIST="examples/edit_prompt_list.txt"
IMAGE_PATH_LIST="examples/image_path_list.txt"
MODEL_PATH="/home/ec2-user/SageMaker/efs/Models/Qwen-Image-Edit"
# Optional: custom LoRA path (if not set, uses default from lightx2v/Qwen-Image-Lightning)
LORA_PATH=""
MODEL_PREFIX="lightning-edit"
BASE_SEED=42
CFG=1.0  # Lightning typically uses CFG=1.0
WIDTH=1024
HEIGHT=1024

# Cache-dit options (set to empty string to disable)
CACHE_OPTS="--cache"
# CACHE_OPTS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
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
        --cfg)
            CFG="$2"
            shift 2
            ;;
        --no-cache)
            CACHE_OPTS=""
            shift
            ;;
        --compile)
            COMPILE_OPTS="--compile"
            shift
            ;;
        --fuse-lora)
            FUSE_LORA_OPTS="--fuse-lora"
            shift
            ;;
        --quantize)
            QUANTIZE_OPTS="--quantize"
            shift 2
            ;;
        --single)
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
            echo "Benchmark script for run_qwen_image_lightning.py (Qwen-Image-Edit-Lightning)"
            echo ""
            echo "Options:"
            echo "  --model_path PATH      Path to base model (default: ${MODEL_PATH})"
            echo "  --lora_path PATH       Path to Lightning LoRA weights (optional)"
            echo "  --prompt_list FILE     Path to prompt list file (default: ${PROMPT_LIST})"
            echo "  --image_path_list FILE Path to image path list file (default: ${IMAGE_PATH_LIST})"
            echo "  --prefix PREFIX        Output prefix (default: ${MODEL_PREFIX})"
            echo "  --width WIDTH          Output image width (default: ${WIDTH})"
            echo "  --height HEIGHT        Output image height (default: ${HEIGHT})"
            echo "  --cfg CFG              CFG scale (default: ${CFG})"
            echo "  --no-cache             Disable cache-dit"
            echo "  --compile              Enable torch.compile"
            echo "  --fuse-lora            Fuse LoRA weights into base model"
            echo "  --quantize             Enable FP8 quantization"
            echo "  --single STEPS         Run single experiment (4 or 8 steps)"
            echo "  --gpu GPU_ID           GPU ID for single experiment (default: 0)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Note: Lightning LoRA only supports 4 or 8 steps"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build LoRA path option if specified
LORA_OPTS=""
if [ -n "$LORA_PATH" ]; then
    LORA_OPTS="--lora-path ${LORA_PATH}"
fi

echo "=========================================="
echo "Qwen-Image-Edit-Lightning Benchmark"
echo "=========================================="
echo "Model Path: ${MODEL_PATH}"
echo "LoRA Path: ${LORA_PATH:-default (lightx2v/Qwen-Image-Lightning)}"
echo "Prompt List: ${PROMPT_LIST}"
echo "Image Path List: ${IMAGE_PATH_LIST}"
echo "Output Prefix: ${MODEL_PREFIX}"
echo "Image Size: ${WIDTH}x${HEIGHT}"
echo "CFG Scale: ${CFG}"
echo "Cache-dit: ${CACHE_OPTS:-disabled}"
echo "Compile: ${COMPILE_OPTS:-disabled}"
echo "Fuse LoRA: ${FUSE_LORA_OPTS:-disabled}"
echo "Quantize: ${QUANTIZE_OPTS:-disabled}"
echo "=========================================="

if [ "$SINGLE_MODE" = true ]; then
    # Single experiment mode
    GPU=${SINGLE_GPU:-0}
    STEP=${SINGLE_STEPS}

    # Validate steps (Lightning only supports 4 or 8)
    if [ "$STEP" != "4" ] && [ "$STEP" != "8" ]; then
        echo "Error: Lightning LoRA only supports 4 or 8 steps, got ${STEP}"
        exit 1
    fi

    OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_steps"
    mkdir -p ${OUT_DIR}

    echo "Running single experiment with ${STEP} steps on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=${GPU} python run_qwen_image_lightning.py \
        --model-path ${MODEL_PATH} \
        --prompt-list-file ${PROMPT_LIST} \
        --image-path-list-file ${IMAGE_PATH_LIST} \
        --out-dir ${OUT_DIR} \
        --base-seed ${BASE_SEED} \
        --steps ${STEP} \
        --cfg ${CFG} \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        ${LORA_OPTS} ${CACHE_OPTS} ${COMPILE_OPTS} ${FUSE_LORA_OPTS} ${QUANTIZE_OPTS} 2>&1 | tee ${OUT_DIR}/benchmark.log

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

        CUDA_VISIBLE_DEVICES=${GPU} nohup python run_qwen_image_lightning.py \
            --model-path ${MODEL_PATH} \
            --prompt-list-file ${PROMPT_LIST} \
            --image-path-list-file ${IMAGE_PATH_LIST} \
            --out-dir ${OUT_DIR} \
            --base-seed ${BASE_SEED} \
            --steps ${STEP} \
            --cfg ${CFG} \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            ${LORA_OPTS} ${CACHE_OPTS} ${COMPILE_OPTS} ${FUSE_LORA_OPTS} ${QUANTIZE_OPTS} > ${OUT_DIR}/benchmark.log 2>&1 &

        echo "  PID: $! -> ${OUT_DIR}/benchmark.log"
    done

    echo ""
    echo "All experiments started in background."
    echo "Monitor progress with: tail -f outputs/${MODEL_PREFIX}_*/benchmark.log"
    echo ""
fi
