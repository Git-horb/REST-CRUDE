from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import asyncio
import torch

from app.models import txt2img_pipe, blip_model, clip_model

app = FastAPI(title="Free AI API")

# Prevent multiple SD runs at once (VERY IMPORTANT)
sd_lock = asyncio.Lock()


@app.get("/")
def root():
    return {"status": "alive"}


@app.post("/txt2img")
async def txt2img(
    prompt: str = Form(...),
    steps: int = Form(20),
):
    prompt = prompt[:300]
    steps = min(int(steps), 25)

    async with sd_lock:
        pipe = txt2img_pipe()
        image = pipe(prompt, num_inference_steps=steps).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # BLIP caption
    blip, blip_proc = blip_model()
    inputs = blip_proc(image, return_tensors="pt").to(blip.device)
    out = blip.generate(**inputs, max_new_tokens=30)
    caption = blip_proc.decode(out[0], skip_special_tokens=True)

    # CLIP zero-shot
    clip, clip_proc = clip_model()
    labels = [
        "a photo of a person",
        "a photo of an animal",
        "a photo of food",
        "a photo of a car",
        "a photo of a landscape",
    ]

    clip_inputs = clip_proc(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(clip.device)

    outputs = clip(**clip_inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    return {
        "caption": caption,
        "classification": dict(zip(labels, probs.tolist())),
    }
