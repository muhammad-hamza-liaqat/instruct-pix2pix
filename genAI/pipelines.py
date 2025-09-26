import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionUpscalePipeline

BASE_MODEL_DIR = os.path.join(os.getcwd(), "hugging_models")
PIX2PIX_MODEL_NAME = "instruct-pix2pix"
PIX2PIX_MODEL_PATH = os.path.join(BASE_MODEL_DIR, PIX2PIX_MODEL_NAME)

UPSCALE_MODEL_NAME = "sd-x4-upscaler"
UPSCALE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, UPSCALE_MODEL_NAME)

os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

def load_pix2pix_pipeline():
    if os.path.exists(PIX2PIX_MODEL_PATH):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            PIX2PIX_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
    else:
        print("⬇️  Downloading InstructPix2Pix model...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            cache_dir=BASE_MODEL_DIR,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
        pipe.save_pretrained(PIX2PIX_MODEL_PATH)
        print(f"✅ Saved InstructPix2Pix locally at {PIX2PIX_MODEL_PATH}")

    pipe.to("cpu")
    return pipe

def load_upscaler_pipeline():
    if os.path.exists(UPSCALE_MODEL_PATH):
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            UPSCALE_MODEL_PATH,
            torch_dtype=torch.float32
        )
    else:
        print("⬇️  Downloading Upscaler model...")
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            cache_dir=BASE_MODEL_DIR,
            torch_dtype=torch.float32
        )
        upscaler.save_pretrained(UPSCALE_MODEL_PATH)
        print(f"✅ Saved Upscaler locally at {UPSCALE_MODEL_PATH}")

    upscaler.to("cpu")
    return upscaler


pipe = load_pix2pix_pipeline()
upscaler = load_upscaler_pipeline()
