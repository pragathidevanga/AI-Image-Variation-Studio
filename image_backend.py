"""Backend functions for AI image-to-image transformation and image diagnostics."""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


# A compatible Stable Diffusion v1.5 checkpoint. The model is downloaded on first use.
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MAX_IMAGE_SIDE = 512


def get_device_info() -> str:
    """Return the runtime device selected for inference."""
    if torch.cuda.is_available():
        return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    return "CPU (inference may be slow)"


def preprocess_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Normalize orientation/color and resize to model-friendly dimensions."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_width = max(64, int(round((width * scale) / 8) * 8))
    new_height = max(64, int(round((height * scale) / 8) * 8))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def load_model():
    """Load and cache a real Hugging Face Diffusers image-to-image pipeline."""
    from diffusers import AutoPipelineForImage2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)

    # Reduce memory use where supported; these options are best-effort.
    if device == "cuda":
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass
    return pipeline, device


def transform_image(
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength: float = 0.45,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    seed: int = 42,
) -> Image.Image:
    """Run actual pretrained image-to-image inference and return its generated image."""
    if not prompt or not prompt.strip():
        raise ValueError("A non-empty transformation prompt is required.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("Strength must be greater than 0 and at most 1.")

    pipeline, device = load_model()
    prepared = preprocess_image(image)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    kwargs = {
        "prompt": prompt.strip(),
        "image": prepared,
        "strength": float(strength),
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": int(num_inference_steps),
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt.strip()

    with torch.inference_mode():
        result = pipeline(**kwargs)

    if not result.images:
        raise RuntimeError("The model did not return an image.")
    return result.images[0].convert("RGB")


def _aligned_arrays(original: Image.Image, generated: Image.Image) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Resize both RGB images to a common size before pixel-wise comparison."""
    original = original.convert("RGB")
    generated = generated.convert("RGB")
    # Compare at the generated output dimensions so metrics align with the displayed output.
    size = generated.size
    original_aligned = original.resize(size, Image.Resampling.LANCZOS)
    generated_aligned = generated.resize(size, Image.Resampling.LANCZOS)
    return (
        np.asarray(original_aligned, dtype=np.float32),
        np.asarray(generated_aligned, dtype=np.float32),
        size,
    )


def calculate_mse(original: Image.Image, generated: Image.Image) -> Tuple[float, Tuple[int, int]]:
    """Calculate actual mean squared error over aligned 8-bit RGB pixel values."""
    a, b, size = _aligned_arrays(original, generated)
    mse = float(np.mean((a - b) ** 2))
    return mse, size


def generate_difference_map(original: Image.Image, generated: Image.Image) -> Image.Image:
    """Return a normalized absolute pixel-difference map for visualization."""
    a, b, _ = _aligned_arrays(original, generated)
    difference = np.abs(a - b)
    max_value = float(difference.max())
    if max_value > 0:
        difference = difference / max_value * 255.0
    return Image.fromarray(np.clip(difference, 0, 255).astype(np.uint8), mode="RGB")
