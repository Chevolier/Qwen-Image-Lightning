import argparse
import math
import os
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel
import torch
import time
import pandas as pd
from tqdm import tqdm


def main(
    model_name,
    prompt_list_file: str,
    image_path_list_file: str | None,
    lora_path: str | None,
    out_dir: str,
    base_seed: int,
    num_inference_steps: int = 8,
    true_cfg_scale: float = 1.0,
):
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    if "2509" in model_name:
        is_edit_plus = True
    else:
        is_edit_plus = False

    if image_path_list_file is None:
        pipe_cls = DiffusionPipeline
    else:
        if is_edit_plus:
            pipe_cls = QwenImageEditPlusPipeline
        else:
            pipe_cls = QwenImageEditPipeline

    if lora_path is not None:
        model = QwenImageTransformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch_dtype
        )
        assert os.path.exists(lora_path), f"Lora path {lora_path} does not exist"
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
        pipe = pipe_cls.from_pretrained(
            model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
        )
        pipe.load_lora_weights(lora_path)
    else:
        pipe = pipe_cls.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
    }

    # # Generate with different aspect ratios
    # if image_path_list_file is None:
    #     aspect_ratios = {
    #         # "1:1": (1328, 1328),
    #         "16:9": (1664, 928),
    #         # "9:16": (928, 1664),
    #         # "4:3": (1472, 1104),
    #         # "3:4": (1104, 1472),
    #         # "3:2": (1584, 1056),
    #         # "2:3": (1056, 1584),
    #     }
    # else:
    #     aspect_ratios = {"not_used": ("auto", "auto")}
    
    aspect_ratios = {
            # "1:1": (1328, 1328),
            "16:9": (1664, 928),
            # "9:16": (928, 1664),
            # "4:3": (1472, 1104),
            # "3:4": (1104, 1472),
            # "3:2": (1584, 1056),
            # "2:3": (1056, 1584),
        }

    with open(prompt_list_file, "r") as f:
        prompt_list = f.read().splitlines()
    if image_path_list_file is not None:
        with open(image_path_list_file, "r") as f:
            image_path_list = f.read().splitlines()
        assert len(prompt_list) == len(image_path_list)
    else:
        image_path_list = None

    os.makedirs(out_dir, exist_ok=True)

    # Metrics collection
    metrics = []

    for _, (width, height) in aspect_ratios.items():
        for i, prompt in tqdm(enumerate(prompt_list)):
            if image_path_list is None:
                prompt = prompt + positive_magic["en"]
            input_args = {
                "prompt": prompt,
                "generator": torch.Generator(device=device).manual_seed(base_seed),
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": num_inference_steps,
            }
            if image_path_list is None:
                input_args["width"] = width
                input_args["height"] = height
            else:
                if is_edit_plus:
                    image_paths = image_path_list[i].split(" ")
                    input_args["image"] = [Image.open(image_path).convert("RGB") for image_path in image_paths]
                else:
                    input_args["image"] = Image.open(image_path_list[i]).convert("RGB")

            # Reset GPU memory stats and measure
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.time()
            image = pipe(**input_args).images[0]

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

            print(f"Request {i}: inference_time={inference_time:.2f}s, peak_gpu_memory={peak_memory_gb:.2f}GB")

            image.save(
                f"{out_dir}/{i:02d}_{width}x{height}_{num_inference_steps}steps_cfg{true_cfg_scale}_example.png"
            )

    # Generate statistics using pandas
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"{out_dir}/metrics_raw_{width}x{height}_{num_inference_steps}steps_cfg{true_cfg_scale}.csv", index=False)

    # Generate statistics (min, max, mean, 50%)
    stats = df_metrics[["inference_time_sec", "peak_gpu_memory_gb"]].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    stats = stats.loc[["min", "mean", "50%", "75%", "90%", "95%", "99%", "max"]]
    stats.to_csv(f"{out_dir}/metrics_stats_{width}x{height}_{num_inference_steps}steps_cfg{true_cfg_scale}.csv")

    print("\n=== Metrics Statistics ===")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_list_file", type=str, default="examples/prompt_list.txt"
    )
    parser.add_argument("--image_path_list_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    args = parser.parse_args()
    if args.steps is None:
        num_inference_steps = 50 if args.lora_path is None else 8
    else:
        num_inference_steps = args.steps
    if args.cfg is None:
        true_cfg_scale = 4.0 if args.lora_path is None else 1.0
    else:
        true_cfg_scale = args.cfg
    if args.lora_path is not None:
        assert os.path.exists(args.lora_path), (
            f"Lora path {args.lora_path} does not exist"
        )

    main(
        model_name=args.model_name,
        prompt_list_file=args.prompt_list_file,
        image_path_list_file=args.image_path_list_file,
        lora_path=args.lora_path,
        out_dir=args.out_dir,
        base_seed=args.base_seed,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
    )
