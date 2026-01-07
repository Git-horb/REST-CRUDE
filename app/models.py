import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from functools import lru_cache

# --------------------------------
# Device selection
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# --------------------------------
# Stable Diffusion pipeline
# --------------------------------
@lru_cache(maxsize=1)
def txt2img_pipe():
    """
    Returns a preloaded Stable Diffusion pipeline.
    Attention slicing is enabled for CPU RAM efficiency.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None,  # saves RAM
    ).to(DEVICE)
    pipe.enable_attention_slicing()  # Hobby CPU friendly
    return pipe


# --------------------------------
# BLIP captioning model
# --------------------------------
@lru_cache(maxsize=1)
def blip_model():
    """
    Returns BLIP model + processor.
    """
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor


# --------------------------------
# CLIP zero-shot classification model
# --------------------------------
@lru_cache(maxsize=1)
def clip_model():
    """
    Returns CLIP model + processor.
    """
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


# --------------------------------
# Optional: preload all models at startup
# --------------------------------
def preload_all_models():
    """
    Call this on FastAPI startup to load all models into memory.
    Reduces first-request latency and prevents 500 errors on Hobby plan.
    """
    print("🔹 Preloading Stable Diffusion...")
    txt2img_pipe()
    print("🔹 Preloading BLIP model...")
    blip_model()
    print("🔹 Preloading CLIP model...")
    clip_model()
    print("✅ All models preloaded successfully!")
