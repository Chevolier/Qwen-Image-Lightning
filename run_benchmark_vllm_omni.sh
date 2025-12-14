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
MODEL_PREFIX="vllm_omni_edit"
BASE_SEED=42
CFG=4.0

# Run experiments in parallel on different GPUs
for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    GPU=${GPUS[$i]}

    echo "Starting experiment with ${STEP} steps on GPU ${GPU}..."
    OUT_DIR="outputs/${MODEL_PREFIX}_${STEP}_step_results"
    mkdir -p ${OUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} nohup python image_edit_vllm_omni.py \
        --prompt_list_file ${PROMPT_LIST} \
        --image_path_list_file ${IMAGE_PATH_LIST} \
        --model ${MODEL_NAME} \
        --out_dir ${OUT_DIR} \
        --seed ${BASE_SEED} \
        --num_inference_steps ${STEP} \
        --cfg_scale ${CFG} > ${OUT_DIR}/step_${STEP}.out 2>&1 &

    echo "PID: $!"
done

echo "All experiments started. Check outputs/ directory for results."
