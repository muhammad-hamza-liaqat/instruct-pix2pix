import os
import torch
from diffusers import StableDiffusionXLPipeline

BASE_MODEL_DIR = os.path.join(os.getcwd(), "hugging_models")
SD_XL_MODEL_NAME = "sd-xl-base"
SD_XL_MODEL_PATH = os.path.join(BASE_MODEL_DIR, SD_XL_MODEL_NAME)

os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

def load_sd_xl_pipeline():
    if os.path.exists(SD_XL_MODEL_PATH):
        sd_xl_pipeline = StableDiffusionXLPipeline.from_pretrained(
            SD_XL_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
    else:
        print("⬇️  Downloading Stable Diffusion XL Base...")
        sd_xl_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            cache_dir=BASE_MODEL_DIR,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
        sd_xl_pipeline.save_pretrained(SD_XL_MODEL_PATH)
        print(f"✅ Saved SD XL Base locally at {SD_XL_MODEL_PATH}")

    sd_xl_pipeline.to("cpu")
    return sd_xl_pipeline

sd_xl_pipeline = load_sd_xl_pipeline()
