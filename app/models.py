import torch
from diffusers import StableDiffusionPipeline
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)
from functools import lru_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@lru_cache(maxsize=1)
def txt2img_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None,  # saves RAM
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


@lru_cache(maxsize=1)
def blip_model():
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return model, processor


@lru_cache(maxsize=1)
def clip_model():
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    return model, processor
