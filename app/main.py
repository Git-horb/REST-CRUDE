from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import asyncio
import torch
import gc
import traceback

from app.models import txt2img_pipe, blip_model, clip_model

app = FastAPI(title="Free AI API")

# -------------------------------
# Async lock to prevent multiple SD runs
# -------------------------------
sd_lock = asyncio.Lock()

# -------------------------------
# Preload all models on startup
# -------------------------------
@app.on_event("startup")
async def preload_models():
    print("🔹 Preloading Stable Diffusion...")
    txt2img_pipe()
    print("🔹 Preloading BLIP model...")
    blip_model()
    print("🔹 Preloading CLIP model...")
    clip_model()
    print("✅ All models loaded!")

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def root():
    return {"status": "alive"}

# -------------------------------
# Text-to-image endpoint
# -------------------------------
@app.post("/txt2img")
async def txt2img(
    prompt: str = Form(...),
    steps: int = Form(20),
):
    # -------------------------------
    # Limits to prevent RAM issues
    # -------------------------------
    prompt = prompt[:300]                     # max 300 chars
    steps = max(1, min(int(steps), 25))       # max 25 steps

    async with sd_lock:
        try:
            pipe = txt2img_pipe()
            with torch.no_grad():  # reduces memory usage
                image = pipe(prompt, num_inference_steps=steps).images[0]

            # Convert PIL image to PNG
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            response = Response(content=buf.getvalue(), media_type="image/png")

            # Cleanup memory
            del image
            gc.collect()

            # Small “breath” to prevent memory spikes
            await asyncio.sleep(2)

            return response

        except Exception as e:
            print("🛑 TXT2IMG ERROR:", e)
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal AI error")


# -------------------------------
# Image analysis endpoint
# -------------------------------
@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        # BLIP caption
        blip, blip_proc = blip_model()
        inputs = blip_proc(image, return_tensors="pt").to(blip.device)
        out = blip.generate(**inputs, max_new_tokens=30)
        caption = blip_proc.decode(out[0], skip_special_tokens=True)

        # CLIP zero-shot classification
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

        # Cleanup memory
        del image
        gc.collect()

        return {
            "caption": caption,
            "classification": dict(zip(labels, probs.tolist())),
        }

    except Exception as e:
        print("🛑 ANALYSE ERROR:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal analysis error")
