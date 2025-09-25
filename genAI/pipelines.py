import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

BASE_MODEL_DIR = os.path.join(os.getcwd(), "hugging_models")
MODEL_NAME = "instruct-pix2pix"
MODEL_PATH = os.path.join(BASE_MODEL_DIR, MODEL_NAME)

os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

def load_pix2pix_pipeline():

    if os.path.exists(MODEL_PATH):
        # print(f"🔹 Loading model from local path: {MODEL_PATH}")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
    else:
        print("⬇️  Model not found locally. Downloading from Hugging Face...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            cache_dir=BASE_MODEL_DIR,
            torch_dtype=torch.float32,
            safety_checker=dummy_safety_checker
        )
        pipe.save_pretrained(MODEL_PATH)
        print(f"✅ Model saved locally at: {MODEL_PATH}")

    pipe.to("cpu")
    return pipe

pipe = load_pix2pix_pipeline()
