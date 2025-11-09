# app/main.py
import os
import base64
import time
import uuid
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import requests

app = FastAPI()

# --- Config (impostare via Azure App Settings) ---
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")  # es. https://<your>.services.ai.azure.com/api/projects/<project>
API_KEY = os.getenv("AZURE_API_KEY")  # chiave dell'Azure OpenAI / Foundry (o lasciare vuoto se usi managed identity)
MODEL_TEXT = os.getenv("MODEL_TEXT", "gpt-4o")         # modello per testo
MODEL_IMAGE = os.getenv("MODEL_IMAGE", "dall-e-3")     # modello per immagini
MODEL_VIDEO = os.getenv("MODEL_VIDEO", "sora")         # modello per video (es. sora/sora-2)
OUTPUT_DIR = "/tmp/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["api-key"] = API_KEY  # alcune API Azure richiedono api-key header; varia per endpoint

# small helper to call a chat/text completion on Foundry project
def call_text_model(prompt: str, model: str = MODEL_TEXT, max_tokens: int = 512):
    url = f"{PROJECT_ENDPOINT}/inference/text/generate"
    payload = {
        "model": model,
        "inputs": prompt,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()

# helper to call image generation
def call_image_model(prompt: str, model: str = MODEL_IMAGE):
    url = f"{PROJECT_ENDPOINT}/inference/images/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()

# helper to call video generation (Sora style)
def call_video_model(prompt: str, model: str = MODEL_VIDEO, duration_seconds: int = 8):
    # Example: create job, then poll for result (APIs may vary by provider)
    url = f"{PROJECT_ENDPOINT}/inference/video/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration_seconds
    }
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("job_id") or job.get("id")
    # poll
    status_url = f"{PROJECT_ENDPOINT}/inference/video/jobs/{job_id}"
    for _ in range(60):
        r = requests.get(status_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        st = r.json()
        if st.get("status") in ("succeeded","completed"):
            return st
        if st.get("status") in ("failed","error"):
            raise RuntimeError("Video generation failed: " + str(st))
        time.sleep(2)
    raise TimeoutError("Video generation timed out")

@app.post("/generate")
async def generate(
    context: str = Form(...),
    guidelines: Optional[str] = Form(""),
    style: Optional[str] = Form(""),
    uploads: Optional[List[UploadFile]] = File(None)
):
    """
    Generate 3 posts (text + image), 3 reels, 3 stories based on context/guidelines.
    Returns JSON with base64 files and metadata.
    """
    try:
        # 1) Build prompts for text (three variations)
        texts = []
        for i in range(3):
            p = f"""You are a social media copywriter. Use this context: {context}
Guidelines: {guidelines}
Style: {style}
Write a caption for Instagram/Facebook (max 120 characters), include 3 hashtags. Variation {i+1}."""
            res = call_text_model(p)
            # response parsing may differ; adapt to the JSON schema returned
            content = (res.get("content") or res.get("output") or "")
            if isinstance(content, dict):
                # example schema: {'choices':[{'text':...}]}
                content = content.get("text") or content.get("choices",[{}])[0].get("text","")
            texts.append(content.strip())

        # 2) Generate 3 images (use user uploads as style reference if provided)
        images_b64 = []
        for i in range(3):
            img_prompt = f"Create an engaging Instagram post image for: {context}. Style: {style}. Variation {i+1}."
            # if user uploaded an image, pass it as reference (base64) — Foundry supports image inputs for some models
            if uploads and len(uploads) > 0:
                first = await uploads[0].read()
                b64 = base64.b64encode(first).decode("utf-8")
                # pass image as data URI if supported
                img_prompt += f" Use the uploaded image as a reference. [image-data:{b64[:80]}...]" 
            img_resp = call_image_model(img_prompt)
            # parse response: many endpoints return base64 or url
            # Try to extract base64 from common response fields
            img_data = None
            if isinstance(img_resp, dict):
                # check typical fields
                if "b64_json" in img_resp:
                    img_data = img_resp["b64_json"]
                elif "data" in img_resp and isinstance(img_resp["data"], list):
                    d0 = img_resp["data"][0]
                    img_data = d0.get("b64_json") or d0.get("image") or d0.get("url")
                elif "output" in img_resp and isinstance(img_resp["output"], list):
                    img_data = img_resp["output"][0].get("b64")
            if not img_data:
                # fallback: try to download url
                url = None
                if "url" in img_resp:
                    url = img_resp["url"]
                elif "data" in img_resp and isinstance(img_resp["data"], list) and "url" in img_resp["data"][0]:
                    url = img_resp["data"][0]["url"]
                if url:
                    r = requests.get(url)
                    r.raise_for_status()
                    img_data = base64.b64encode(r.content).decode("utf-8")
            if not img_data:
                raise RuntimeError("Could not parse image response: " + str(img_resp)[:400])
            images_b64.append(img_data)

        # 3) Generate 3 short videos (reels) — watch out ai model limits (duration quality)
        videos = []
        for i in range(3):
            vid_prompt = f"Create a short social media reel (vertical style, up to 8s) for: {context}. Style: {style}. Variation {i+1}."
            # If user uploaded video/image, pass as reference similarly (not all models support)
            vid_job = call_video_model(vid_prompt, MODEL_VIDEO, duration_seconds=8)
            # job expected to return either a URL or base64
            vid_url = vid_job.get("result_url") or vid_job.get("download_url") or vid_job.get("output", {}).get("url")
            if vid_url:
                r = requests.get(vid_url)
                r.raise_for_status()
                b64 = base64.b64encode(r.content).decode("utf-8")
                videos.append(b64)
            else:
                # maybe job includes base64 directly
                b64 = vid_job.get("b64") or vid_job.get("b64_video")
                if b64:
                    videos.append(b64)
                else:
                    raise RuntimeError("Could not retrieve generated video: " + str(vid_job)[:400])

        # Assemble result: we return captions, images (b64), videos (b64)
        result = {
            "captions": texts,
            "images_b64": images_b64,
            "videos_b64": videos,
            "notes": "Scarica i base64 e salva come file .png o .mp4 per pubblicare manualmente."
        }
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
