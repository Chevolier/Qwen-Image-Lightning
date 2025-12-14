# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for image editing with Qwen-Image-Edit using vllm_omni.

Usage (single image):
    python image_edit_vllm_omni.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0

Usage (batch mode with list files):
    python image_edit_vllm_omni.py \
        --prompt_list_file prompts.txt \
        --image_path_list_file images.txt \
        --out_dir results/ \
        --num_inference_steps 50 \
        --cfg_scale 4.0

For more options, run:
    python image_edit_vllm_omni.py --help
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit an image with Qwen-Image-Edit.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit",
        help="Diffusion model name or local path.",
    )
    # Single image mode arguments
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image file (PNG, JPG, etc.) for single image mode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt describing the edit for single image mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image_edit.png",
        help="Path to save the edited image (PNG) for single image mode.",
    )
    # Batch mode arguments
    parser.add_argument(
        "--prompt_list_file",
        type=str,
        default=None,
        help="Path to file containing prompts (one per line) for batch mode.",
    )
    parser.add_argument(
        "--image_path_list_file",
        type=str,
        default=None,
        help="Path to file containing image paths (one per line) for batch mode.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for batch mode.",
    )
    # Common arguments
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image-Edit.",
    )
    parser.add_argument(
        "--num_outputs_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    return parser.parse_args()


def run_single_image(args, omni, device):
    """Run single image editing mode."""
    # Validate input image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Load input image
    input_image = Image.open(args.image).convert("RGB")
    print(f"Loaded input image from {args.image} (size: {input_image.size})")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate edited image
    images = omni.generate(
        prompt=args.prompt,
        pil_image=input_image,
        negative_prompt=args.negative_prompt,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_outputs_per_prompt,
    )

    # Save output image(s)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "output_image_edit"

    if args.num_outputs_per_prompt <= 1:
        images[0].save(output_path)
        print(f"Saved edited image to {os.path.abspath(output_path)}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved edited image to {os.path.abspath(save_path)}")


def run_batch_mode(args, omni, device):
    """Run batch image editing mode with metrics collection."""
    # Load prompt list
    with open(args.prompt_list_file, "r") as f:
        prompt_list = f.read().splitlines()

    # Load image path list
    with open(args.image_path_list_file, "r") as f:
        image_path_list = f.read().splitlines()

    assert len(prompt_list) == len(image_path_list), (
        f"Number of prompts ({len(prompt_list)}) must match "
        f"number of images ({len(image_path_list)})"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Metrics collection
    metrics = []

    for i, (prompt, image_path) in tqdm(
        enumerate(zip(prompt_list, image_path_list)), total=len(prompt_list)
    ):
        # Validate input image exists
        if not os.path.exists(image_path):
            print(f"Warning: Input image not found: {image_path}, skipping...")
            continue

        # Load input image
        input_image = Image.open(image_path).convert("RGB")
        width, height = input_image.size

        generator = torch.Generator(device=device).manual_seed(args.seed)

        # Reset GPU memory stats and measure
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()

        # Generate edited image
        images = omni.generate(
            prompt=prompt,
            pil_image=input_image,
            negative_prompt=args.negative_prompt,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_outputs_per_prompt,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time

        if device == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_memory_gb = 0.0

        metrics.append({
            "request_id": i,
            "width": width,
            "height": height,
            "inference_time_sec": inference_time,
            "peak_gpu_memory_gb": peak_memory_gb,
        })

        print(
            f"Request {i}: inference_time={inference_time:.2f}s, "
            f"peak_gpu_memory={peak_memory_gb:.2f}GB"
        )

        # Save output image(s)
        if args.num_outputs_per_prompt <= 1:
            save_path = (
                f"{args.out_dir}/{i:02d}_{args.num_inference_steps}steps_"
                f"cfg{args.cfg_scale}_example.png"
            )
            images[0].save(save_path)
        else:
            for idx, img in enumerate(images):
                save_path = (
                    f"{args.out_dir}/{i:02d}_{idx}_{args.num_inference_steps}steps_"
                    f"cfg{args.cfg_scale}_example.png"
                )
                img.save(save_path)

    # Generate statistics using pandas
    if metrics:
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(
            f"{args.out_dir}/metrics_raw_{args.num_inference_steps}steps_"
            f"cfg{args.cfg_scale}.csv",
            index=False,
        )

        # Generate statistics (min, max, mean, percentiles)
        stats = df_metrics[["inference_time_sec", "peak_gpu_memory_gb"]].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
        stats = stats.loc[["min", "mean", "50%", "75%", "90%", "95%", "99%", "max"]]
        stats.to_csv(
            f"{args.out_dir}/metrics_stats_{args.num_inference_steps}steps_"
            f"cfg{args.cfg_scale}.csv"
        )

        print("\n=== Metrics Statistics ===")
        print(stats)


def main():
    args = parse_args()

    # Determine mode: batch or single
    batch_mode = args.prompt_list_file is not None and args.image_path_list_file is not None
    single_mode = args.image is not None and args.prompt is not None

    if not batch_mode and not single_mode:
        raise ValueError(
            "Either provide --image and --prompt for single image mode, "
            "or --prompt_list_file and --image_path_list_file for batch mode."
        )

    device = detect_device_type()

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Initialize Omni with QwenImageEditPipeline
    omni = Omni(
        model=args.model,
        model_class_name="QwenImageEditPipeline",
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )
    print("Pipeline loaded")

    if batch_mode:
        run_batch_mode(args, omni, device)
    else:
        run_single_image(args, omni, device)


if __name__ == "__main__":
    main()