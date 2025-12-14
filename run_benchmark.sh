#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define step configurations and corresponding GPU devices
STEPS=(4 8 16 50)
GPUS=(0 1 2 3)

# Common parameters
PROMPT_LIST="examples/edit_prompt_list.txt"
IMAGE_PATH_LIST="examples/image_path_list.txt"
MODEL_NAME="/home/ec2-user/SageMaker/efs/Models/Qwen-Image-Edit"
LORA_PATH="/home/ec2-user/SageMaker/efs/Models/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
# MODEL_PREFIX="lightning-4step-v1-bf16"
MODEL_PREFIX="base-edit"
BASE_SEED=42
CFG=4.0

# Run experiments in parallel on different GPUs
for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    GPU=${GPUS[$i]}

    echo "Starting experiment with ${STEP} steps on GPU ${GPU}..."
    OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_step_edit_results"
    mkdir -p ${OUT_DIR}

    # --lora_path ${LORA_PATH} \
    # CUDA_VISIBLE_DEVICES=${GPU} nohup
    python generate_with_diffusers.py \
        --prompt_list_file ${PROMPT_LIST} \
        --image_path_list_file ${IMAGE_PATH_LIST} \
        --model_name ${MODEL_NAME} \
        --out_dir ${OUT_DIR} \
        --base_seed ${BASE_SEED} \
        --steps ${STEP} \
        --cfg ${CFG} > ${OUT_DIR}/step_${STEP}.out 2>&1

    echo "PID: $!"
done

echo "All experiments started. Check logs/ directory for output."
