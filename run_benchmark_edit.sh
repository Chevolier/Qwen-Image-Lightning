#!/bin/bash

# Benchmark script for run_qwen_image_edit.py (Qwen-Image-Edit with cache-dit)
# This runs the base Qwen-Image-Edit model with optional caching

# Create output directories
mkdir -p logs
mkdir -p outputs

# Define step configurations and corresponding GPU devices
STEPS=(4 8 16 50)
GPUS=(0 1 2 3)
# GPUS=(4 5 6 7)

# Common parameters
PROMPT_LIST="examples/edit_prompt_list.txt"
IMAGE_PATH_LIST="examples/image_path_list.txt"
MODEL_PATH="/home/ec2-user/SageMaker/efs/Models/Qwen-Image-Edit"
MODEL_PREFIX="cache-dit-edit"
BASE_SEED=42
CFG=4.0
WIDTH=1664
HEIGHT=928

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
            echo "Benchmark script for run_qwen_image_edit.py (Qwen-Image-Edit with cache-dit)"
            echo ""
            echo "Options:"
            echo "  --model_path PATH      Path to the model (default: ${MODEL_PATH})"
            echo "  --prompt_list FILE     Path to prompt list file (default: ${PROMPT_LIST})"
            echo "  --image_path_list FILE Path to image path list file (default: ${IMAGE_PATH_LIST})"
            echo "  --prefix PREFIX        Output prefix (default: ${MODEL_PREFIX})"
            echo "  --width WIDTH          Output image width (default: ${WIDTH})"
            echo "  --height HEIGHT        Output image height (default: ${HEIGHT})"
            echo "  --cfg CFG              CFG scale (default: ${CFG})"
            echo "  --no-cache             Disable cache-dit"
            echo "  --compile              Enable torch.compile"
            echo "  --single STEPS         Run single experiment with specified steps"
            echo "  --gpu GPU_ID           GPU ID for single experiment (default: 0)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Qwen-Image-Edit Benchmark (cache-dit)"
echo "=========================================="
echo "Model Path: ${MODEL_PATH}"
echo "Prompt List: ${PROMPT_LIST}"
echo "Image Path List: ${IMAGE_PATH_LIST}"
echo "Output Prefix: ${MODEL_PREFIX}"
echo "Image Size: ${WIDTH}x${HEIGHT}"
echo "CFG Scale: ${CFG}"
echo "Cache-dit: ${CACHE_OPTS:-disabled}"
echo "Compile: ${COMPILE_OPTS:-disabled}"
echo "=========================================="

if [ "$SINGLE_MODE" = true ]; then
    # Single experiment mode
    GPU=${SINGLE_GPU:-0}
    STEP=${SINGLE_STEPS}
    OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_steps"
    mkdir -p ${OUT_DIR}

    echo "Running single experiment with ${STEP} steps on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=${GPU} python run_qwen_image_edit.py \
        --model-path ${MODEL_PATH} \
        --prompt-list-file ${PROMPT_LIST} \
        --image-path-list-file ${IMAGE_PATH_LIST} \
        --out-dir ${OUT_DIR} \
        --base-seed ${BASE_SEED} \
        --steps ${STEP} \
        --cfg ${CFG} \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        ${CACHE_OPTS} ${COMPILE_OPTS} 2>&1 | tee ${OUT_DIR}/benchmark.log

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

        CUDA_VISIBLE_DEVICES=${GPU} nohup python run_qwen_image_edit.py \
            --model-path ${MODEL_PATH} \
            --prompt-list-file ${PROMPT_LIST} \
            --image-path-list-file ${IMAGE_PATH_LIST} \
            --out-dir ${OUT_DIR} \
            --base-seed ${BASE_SEED} \
            --steps ${STEP} \
            --cfg ${CFG} \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            ${CACHE_OPTS} ${COMPILE_OPTS} > ${OUT_DIR}/benchmark.log 2>&1 &

        echo "  PID: $! -> ${OUT_DIR}/benchmark.log"
    done

    echo ""
    echo "All experiments started in background."
    echo "Monitor progress with: tail -f outputs/${MODEL_PREFIX}_*/benchmark.log"
    echo ""
fi
