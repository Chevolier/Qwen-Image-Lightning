import argparse
import json
import os
import time
import pandas as pd
from tqdm import tqdm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sglang.multimodal_gen import DiffGenerator


# Model type mapping: local directory name patterns -> HuggingFace model IDs
# SGLang uses HF model IDs for model registry lookup
MODEL_TYPE_MAPPING = {
    "qwen-image-edit": "Qwen/Qwen-Image-Edit",
    "qwen-image": "Qwen/Qwen-Image",
}


def patch_qwen_inference_steps(num_inference_steps: int):
    """
    Patch QwenImageSamplingParams to use custom num_inference_steps.

    SGLang's design prevents users from overriding model-specific defaults
    defined in SamplingParams subclasses. This workaround patches the default
    value at the class level before initializing the generator.

    See: sglang/multimodal_gen/configs/sample/sampling_params.py
         _merge_with_user_params() skips fields in subclass_defined_fields
    """
    try:
        from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
        # Patch the class default before any instances are created
        QwenImageSamplingParams.num_inference_steps = num_inference_steps
        print(f"Patched QwenImageSamplingParams.num_inference_steps = {num_inference_steps}")
    except ImportError as e:
        print(f"Warning: Could not patch QwenImageSamplingParams: {e}")


def detect_model_type_from_path(model_path: str) -> str:
    """
    Detect the model type from a local path and return the corresponding
    HuggingFace model ID that SGLang recognizes.

    SGLang's registry uses HF model IDs for lookup. For local paths, we need
    to map them to the registered HF IDs.
    """
    # If it's already a HF model ID (contains /), return as-is
    if "/" in model_path and not os.path.exists(model_path):
        return model_path

    # For local paths, try to detect the model type
    path_lower = model_path.lower()

    # Check model_index.json for _class_name if exists
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        try:
            with open(model_index_path, "r") as f:
                config = json.load(f)
            class_name = config.get("_class_name", "").lower()

            # Map class names to HF model IDs
            if "qwenimageedit" in class_name:
                return "Qwen/Qwen-Image-Edit"
            elif "qwenimage" in class_name:
                return "Qwen/Qwen-Image"
        except Exception:
            pass

    # Fallback: check directory name patterns
    for pattern, hf_id in MODEL_TYPE_MAPPING.items():
        if pattern in path_lower:
            return hf_id

    # If no match found, return original path (may fail in SGLang)
    return model_path


def main(
    model_path: str,
    prompt_list_file: str,
    image_path_list_file: str | None,
    out_dir: str,
    base_seed: int,
    num_inference_steps: int = 8,
    num_gpus: int = 1,
    width: int = 1664,
    height: int = 928,
):
    """
    Run SGLang inference for Qwen-Image-Edit model.

    Args:
        model_path: Path to the model or HuggingFace model name
        prompt_list_file: Path to file containing prompts (one per line)
        image_path_list_file: Path to file containing image paths for editing (one per line)
        out_dir: Output directory for generated images and metrics
        base_seed: Random seed for reproducibility
        num_inference_steps: Number of inference steps
        num_gpus: Number of GPUs to use
        width: Output image width
        height: Output image height
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Read prompts
    with open(prompt_list_file, "r") as f:
        prompt_list = f.read().splitlines()

    # Read image paths if provided (for image editing)
    if image_path_list_file is not None:
        with open(image_path_list_file, "r") as f:
            image_path_list = f.read().splitlines()
        assert len(prompt_list) == len(image_path_list), \
            f"Number of prompts ({len(prompt_list)}) must match number of images ({len(image_path_list)})"
    else:
        image_path_list = None

    # Detect model type for SGLang registry
    # SGLang uses HF model IDs for registry lookup, so we need to map local paths
    detected_model_id = detect_model_type_from_path(model_path)

    # If model_path is a local directory, we need to tell SGLang both:
    # 1. The actual model path (for loading weights)
    # 2. The model type (for registry lookup)
    is_local_path = os.path.exists(model_path)

    print(f"Model path: {model_path}")
    print(f"Detected model ID for registry: {detected_model_id}")
    print(f"Is local path: {is_local_path}")

    # IMPORTANT: Patch the default num_inference_steps BEFORE creating the generator
    # SGLang's design prevents runtime override of model-specific defaults
    patch_qwen_inference_steps(num_inference_steps)

    # Initialize the DiffGenerator
    print(f"Loading model...")

    # For local models, we pass the actual path but use the HF ID for model type detection
    # SGLang will use the model_index.json to identify the pipeline
    generator_kwargs = {
        "model_path": model_path,
        "num_gpus": num_gpus,
    }

    generator = DiffGenerator.from_pretrained(**generator_kwargs)
    print("Model loaded successfully.")

    # Metrics collection
    metrics = []

    for i, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list), desc="Generating"):
        # Build sampling params
        sampling_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "seed": base_seed,
            "save_output": True,
            "output_path": out_dir,
        }

        # Add image path for image editing tasks
        if image_path_list is not None:
            sampling_params["image_path"] = image_path_list[i]

        # Add dimensions
        sampling_params["width"] = width
        sampling_params["height"] = height

        # Reset GPU memory stats and measure
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()

        # Generate image
        _ = generator.generate(sampling_params_kwargs=sampling_params)

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        inference_time = end_time - start_time

        # Get peak GPU memory
        if HAS_TORCH and torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_memory_gb = 0.0

        metrics.append({
            "request_id": i,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "inference_time_sec": inference_time,
            "peak_gpu_memory_gb": peak_memory_gb,
        })

        print(f"Request {i}: inference_time={inference_time:.2f}s, peak_gpu_memory={peak_memory_gb:.2f}GB")

    # Shutdown the generator
    generator.shutdown()

    # Generate statistics using pandas
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(
        f"{out_dir}/metrics_raw_{width}x{height}_{num_inference_steps}steps.csv",
        index=False
    )

    # Generate statistics (min, max, mean, percentiles)
    numeric_cols = ["inference_time_sec", "peak_gpu_memory_gb"]
    stats = df_metrics[numeric_cols].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    stats = stats.loc[["min", "mean", "50%", "75%", "90%", "95%", "99%", "max"]]
    stats.to_csv(f"{out_dir}/metrics_stats_{width}x{height}_{num_inference_steps}steps.csv")

    print("\n=== Metrics Statistics ===")
    print(stats)

    return df_metrics, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SGLang inference for Qwen-Image-Edit model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="Path to model or HuggingFace model name. For local paths, the model type "
             "will be auto-detected from model_index.json or directory name."
    )
    parser.add_argument(
        "--prompt_list_file",
        type=str,
        default="examples/edit_prompt_list.txt",
        help="Path to file containing prompts"
    )
    parser.add_argument(
        "--image_path_list_file",
        type=str,
        default=None,
        help="Path to file containing image paths for editing"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results_sglang",
        help="Output directory"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1664,
        help="Output image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=928,
        help="Output image height"
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        prompt_list_file=args.prompt_list_file,
        image_path_list_file=args.image_path_list_file,
        out_dir=args.out_dir,
        base_seed=args.base_seed,
        num_inference_steps=args.steps,
        num_gpus=args.num_gpus,
        width=args.width,
        height=args.height,
    )
