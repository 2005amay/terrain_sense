"""
TerrainSense - api.py
=====================
FastAPI service wrapping the trained MobileNetV2 terrain classifier.

Endpoints:
    GET  /         -> service info + available classes
    GET  /health   -> liveness check for uptime monitors / n8n
    POST /predict  -> classify an uploaded image (multipart form field "file")

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000
Interactive docs:
    http://localhost:8000/docs

Optional API key: set the environment variable TERRAINSENSE_API_KEY
(e.g. `set TERRAINSENSE_API_KEY=secret` before starting) and clients must
then send header `X-API-Key: secret`. Leave it unset for open access.
"""

import io
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model_utils import load_model, model_available, predict_dict

# ---------------------------------------------------------------------------
# App + CORS (required so a browser/animated webpage can call it cross-origin)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TerrainSense API",
    description="Classify terrain type (Desert / Forest / Mountain / Plains) from a photo.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # any website may call it; lock to your domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("TERRAINSENSE_API_KEY")  # None -> no auth required

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------

if model_available():
    MODEL, CLASSES = load_model()   # ONNX Runtime on Render, PyTorch locally
else:
    MODEL, CLASSES = None, []


def _check_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "TerrainSense API",
        "status": "ok" if MODEL is not None else "model_missing",
        "classes": CLASSES,
        "endpoints": {"predict": "POST /predict (form field: file)", "health": "GET /health"},
    }


@app.get("/health")
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model file not found - train first")
    return {
        "status": "ok",
        "model_loaded": True,
        "backend": getattr(MODEL, "name", None),
        "classes": CLASSES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model file not found - train first")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now so corrupt files fail here, not mid-inference
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image")

    try:
        result = predict_dict(image, MODEL, CLASSES)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    return result
