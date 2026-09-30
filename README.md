# TerrainSense 🌍

Real-time image-based AI terrain recognition. Upload a landscape or ground-surface
photo and a fine-tuned MobileNetV2 instantly predicts the terrain class
(**Desert · Forest · Mountain · Plains**) with a confidence score.

**Stack:** Python 3.13 · PyTorch / Torchvision (training) · MobileNetV2 (ImageNet) ·
ONNX Runtime (serving) · FastAPI · Streamlit · Pillow · Matplotlib

## Project structure

```
Terrain_Recognition/
├── dataset/                          <- Kaggle terrain dataset, one folder per class
│   ├── Desert/       (799 jpg)
│   ├── Forest/       (799 jpg)
│   ├── Mountain/     (799 jpg)
│   └── Plains/       (799 jpg)
├── models/
│   ├── terrain_mobilenetv2.pth       <- created by train.py (PyTorch checkpoint)
│   ├── terrain_mobilenetv2.onnx      <- created by export_onnx.py (served on Render)
│   ├── classes.json                  <- class names for the ONNX path
│   ├── training_curves.png
│   └── confusion_matrix.png
├── train.py                          <- training script (CUDA/CPU)
├── export_onnx.py                    <- .pth -> .onnx + torch/onnx parity check
├── api.py                            <- FastAPI service (/predict, /health)
├── app.py                            <- local Streamlit dashboard
├── model_utils.py                    <- shared preprocessing + inference (both backends)
├── requirements.txt                  <- local dev (torch, streamlit, ...)
├── requirements-render.txt           <- Render build (no torch, ONNX Runtime only)
├── .python-version                   <- Python version pinned for Render
└── .gitignore
```

## Setup (one time)

```powershell
cd "C:\Users\amays\Documents\Default Project\Terrain_Recognition"
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt
```

(If you have an NVIDIA GPU and want CUDA training, install torch without the `--index-url` flag.)

## 1. Dataset

Kaggle terrain dataset, one folder per class inside `dataset/`:

```
dataset/
├── Desert/      799 images
├── Forest/      799 images
├── Mountain/    799 images
└── Plains/      799 images
```

- Class names are read from the folder names automatically.
- The whole `dataset/` folder is gitignored (too big to push).

## 2. Train

```powershell
.venv\Scripts\activate
python train.py
```

| Command | Effect |
|---|---|
| `python train.py --epochs 15` | more training rounds (default 12) |
| `python train.py --batch-size 16` | smaller batches if memory is tight |
| `python train.py --learning-rate 0.0005` | slower, steadier learning |
| `python train.py --unfreeze-last-block` | also train the last backbone block (more accurate, slower) |

Produces:

- `models/terrain_mobilenetv2.pth` — trained weights + class names
  (validation accuracy ≈ 97.7 %, see confusion matrix)
- `models/training_curves.png` — loss & accuracy over epochs
- `models/confusion_matrix.png` — per-class validation accuracy

Hardware is auto-detected: uses CUDA if available, otherwise CPU.

## 3. Export to ONNX (required before deploying)

```powershell
python export_onnx.py
```

Writes `models/terrain_mobilenetv2.onnx` + `models/classes.json` and verifies
the ONNX output matches PyTorch (same predicted class, probability diff < 1e-3).
Serving then needs only ONNX Runtime (~15 MB) instead of PyTorch (~2 GB).

## 4. Run the API locally

```powershell
.venv\Scripts\activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Interactive docs: <http://localhost:8000/docs>
- `GET  /` — service info + classes
- `GET  /health` — liveness check (used by n8n keep-alive)
- `POST /predict` — multipart form field `file` (the photo) →
  `{class, confidence, probabilities, latency_ms, backend}`
- Optional auth: set env var `TERRAINSENSE_API_KEY`, clients then send
  header `X-API-Key: <value>`.

The backend is chosen automatically: **ONNX Runtime** when
`models/terrain_mobilenetv2.onnx` exists, otherwise PyTorch —
identical results either way (same preprocessing code path).

## 5. Run the Streamlit dashboard (local only)

```powershell
.venv\Scripts\activate
streamlit run app.py
```

## Deploy to Render.com (free tier)

1. Push this folder to GitHub (repo root = `Terrain_Recognition/`;
   `dataset/` and `.venv/` are already gitignored, `models/` is included).
2. Render → **New → Web Service** → connect the repo.
3. Settings:
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements-render.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Environment → add:
   - `PYTHON_VERSION` = `3.13.14` (mirrors `.python-version`)
   - `TERRAINSENSE_API_KEY` = `<your secret>` (optional but recommended)
5. Deploy → the service answers at `https://<name>.onrender.com`
   (free instances spin down after 15 min idle; first request then takes ~60 s).

Quick check: `GET https://<name>.onrender.com/health` must return
`"model_loaded": true, "backend": "onnx"`.

## n8n integration

- **HTTP Request node:** `POST https://<name>.onrender.com/predict`
  - Body type: Form Data → key `file` (Binary), value = the photo
  - Timeout: **90 000 ms** (absorbs the ~60 s cold start)
  - Header `X-API-Key` if you set the key
- **Keep-alive (optional):** Schedule Trigger every 12 min →
  `GET /health` → instance stays warm and fits the 750 free hours/month.

## How it works

1. **Preprocess:** resize to 224×224 RGB, normalize with ImageNet mean/std
   (one shared implementation in `model_utils.py` used by every entry point).
2. **Transfer learning:** MobileNetV2 with frozen feature layers; the top linear
   layer is replaced with `Linear(1280, N)` for the terrain classes.
3. **Train:** Adam + CrossEntropyLoss, light augmentation (RandomHorizontalFlip),
   80/20 train/validation split, best validation checkpoint saved.
4. **Serve:** export to ONNX → FastAPI + ONNX Runtime on Render (~10–30 ms CPU).
