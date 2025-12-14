import os
import sys

sys.path.append("..")

import time
import math
import torch
import pandas as pd
from tqdm import tqdm

from PIL import Image
from diffusers import (
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from utils import GiB, get_args, strify, cachify, MemoryTracker
import cache_dit


def extend_args(parser):
    """Extend argument parser with prompt list and benchmark options."""
    parser.add_argument(
        "--prompt-list-file",
        type=str,
        default=None,
        help="Path to file containing prompts (one per line). If not provided, uses --prompt.",
    )
    parser.add_argument(
        "--image-path-list-file",
        type=str,
        default=None,
        help="Path to file containing image paths (one per line). If not provided, uses --image-path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for generated images and metrics.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="True CFG scale for generation (default 1.0 for Lightning).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save generated images.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        default=False,
        help="Skip warmup iteration when not using compile.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Custom path to LoRA weights. If not provided, uses default from lightx2v/Qwen-Image-Lightning.",
    )
    return parser


def run_single(
    pipe,
    prompt,
    image,
    height,
    width,
    seed,
    true_cfg_scale,
    num_inference_steps,
    negative_prompt=" ",
    device="cuda",
):
    """Run a single inference and return the result with timing."""
    input_args = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": num_inference_steps,
    }

    # Reset GPU memory stats
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    result = pipe(**input_args).images[0]

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # Get peak GPU memory
    if device == "cuda" and torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_memory_gb = 0.0

    return result, inference_time, peak_memory_gb


def main():
    # Get args with extended options
    parser = get_args(parse=False)
    parser = extend_args(parser)
    args = parser.parse_args()
    print(args)

    # Determine if we're in batch mode (prompt list) or single mode
    batch_mode = args.prompt_list_file is not None

    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/main/generate_with_diffusers.py
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    # Load Qwen-Image-Edit pipeline
    pipe = QwenImageEditPipeline.from_pretrained(
        (
            args.model_path
            if args.model_path is not None
            else os.environ.get(
                "QWEN_IMAGE_EDIT_DIR",
                "Qwen/Qwen-Image-Edit",
            )
        ),
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        device_map=(
            "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
        ),
    )

    steps = 8 if args.steps is None else args.steps
    assert steps in [8, 4], f"Steps must be 4 or 8 for Lightning, got {steps}"

    # Load Qwen-Image-Edit-Lightning LoRA weights
    if args.lora_path is not None:
        lora_path = args.lora_path
        pipe.load_lora_weights(lora_path)
    else:
        pipe.load_lora_weights(
            os.environ.get(
                "QWEN_IMAGE_EDIT_LIGHT_DIR",
                "lightx2v/Qwen-Image-Lightning",
            ),
            weight_name=(
                "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
                if steps > 4
                else "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
            ),
        )

    if args.fuse_lora:
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    if args.cache:
        from cache_dit import DBCacheConfig

        cachify(
            args,
            pipe,
            cache_config=DBCacheConfig(
                Fn_compute_blocks=16,
                Bn_compute_blocks=16,
                max_warmup_steps=4 if steps > 4 else 2,
                max_cached_steps=2 if steps > 4 else 1,
                max_continuous_cached_steps=1,
                enable_separate_cfg=False,  # true_cfg_scale=1.0
                residual_diff_threshold=0.50 if steps > 4 else 0.8,
            ),
        )

    # Device setup
    if torch.cuda.device_count() <= 1:
        pipe.enable_model_cpu_offload()
    elif torch.cuda.device_count() > 1 and pipe.device.type == "cpu":
        pipe.to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

    if args.quantize:
        # Apply Quantization (default: FP8 DQ) to Transformer
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,
            per_row=False,
            exclude_layers=[
                "img_in",
                "txt_in",
                "embedder",
                "embed",
                "norm_out",
                "proj_out",
            ],
        )

    if args.compile:
        cache_dit.set_compile_configs()
        pipe.transformer.compile_repeated_blocks(fullgraph=True)

    # Default dimensions
    height = 1024 if args.height is None else args.height
    width = 1024 if args.width is None else args.width
    true_cfg_scale = args.cfg
    negative_prompt = args.negative_prompt if args.negative_prompt else " "

    if batch_mode:
        # Batch mode: process multiple prompts
        os.makedirs(args.out_dir, exist_ok=True)

        # Read prompts
        with open(args.prompt_list_file, "r") as f:
            prompt_list = f.read().splitlines()

        # Read image paths
        if args.image_path_list_file is not None:
            with open(args.image_path_list_file, "r") as f:
                image_path_list = f.read().splitlines()
            assert len(prompt_list) == len(image_path_list), (
                f"Number of prompts ({len(prompt_list)}) must match "
                f"number of images ({len(image_path_list)})"
            )
        else:
            # Use single image for all prompts
            default_image_path = (
                args.image_path if args.image_path else "../data/bear.png"
            )
            image_path_list = [default_image_path] * len(prompt_list)

        # Warmup run
        if args.compile or not args.skip_warmup:
            print("Running warmup iteration...")
            warmup_image = Image.open(image_path_list[0]).convert("RGB")
            _, _, _ = run_single(
                pipe,
                prompt_list[0],
                warmup_image,
                height,
                width,
                args.base_seed,
                true_cfg_scale,
                steps,
                negative_prompt,
                device,
            )
            print("Warmup complete.")

        # Metrics collection
        metrics = []

        # Process each prompt
        for i, prompt in tqdm(
            enumerate(prompt_list), total=len(prompt_list), desc="Generating"
        ):
            image_path = image_path_list[i]
            image = Image.open(image_path).convert("RGB")

            result_image, inference_time, peak_memory_gb = run_single(
                pipe,
                prompt,
                image,
                height,
                width,
                args.base_seed,
                true_cfg_scale,
                steps,
                negative_prompt,
                device,
            )

            metrics.append(
                {
                    "request_id": i,
                    "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "inference_time_sec": inference_time,
                    "peak_gpu_memory_gb": peak_memory_gb,
                }
            )

            print(
                f"Request {i}: inference_time={inference_time:.2f}s, "
                f"peak_gpu_memory={peak_memory_gb:.2f}GB"
            )

            # Save image
            if args.save_images:
                stats = cache_dit.summary(pipe)
                save_filename = (
                    f"{i:02d}_{width}x{height}_{steps}steps_"
                    f"cfg{true_cfg_scale}.{strify(args, stats)}.png"
                )
                save_path = os.path.join(args.out_dir, save_filename)
                result_image.save(save_path)

        # Generate statistics using pandas
        df_metrics = pd.DataFrame(metrics)
        stats_suffix = f"{width}x{height}_{steps}steps_cfg{true_cfg_scale}"
        df_metrics.to_csv(
            f"{args.out_dir}/metrics_raw_{stats_suffix}.csv", index=False
        )

        # Generate statistics (min, max, mean, percentiles)
        numeric_cols = ["inference_time_sec", "peak_gpu_memory_gb"]
        stats = df_metrics[numeric_cols].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
        stats = stats.loc[["min", "mean", "50%", "75%", "90%", "95%", "99%", "max"]]
        stats.to_csv(f"{args.out_dir}/metrics_stats_{stats_suffix}.csv")

        print("\n=== Metrics Statistics ===")
        print(stats)

    else:
        # Single mode: original behavior
        image_path = args.image_path if args.image_path else "../data/bear.png"
        image = Image.open(image_path).convert("RGB")
        prompt = (
            args.prompt
            if args.prompt
            else "Only change the bear's color to purple"
        )

        # Warmup for compile mode
        if args.compile:
            print("Running warmup iteration for compilation...")
            _, _, _ = run_single(
                pipe,
                prompt,
                image,
                height,
                width,
                args.base_seed,
                true_cfg_scale,
                steps,
                negative_prompt,
                device,
            )

        # Track memory if requested
        memory_tracker = MemoryTracker() if args.track_memory else None
        if memory_tracker:
            memory_tracker.__enter__()

        result_image, time_cost, peak_memory_gb = run_single(
            pipe,
            prompt,
            image,
            height,
            width,
            args.base_seed,
            true_cfg_scale,
            steps,
            negative_prompt,
            device,
        )

        if memory_tracker:
            memory_tracker.__exit__(None, None, None)
            memory_tracker.report()

        stats = cache_dit.summary(pipe, details=True)

        save_path = f"qwen-image-edit-lightning.{steps}steps.{strify(args, stats)}.png"
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Peak GPU memory: {peak_memory_gb:.2f}GB")
        print(f"Saving image to {save_path}")
        result_image.save(save_path)


if __name__ == "__main__":
    main()
