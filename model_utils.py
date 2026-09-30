"""
TerrainSense - model_utils.py
=============================
Shared model loading + inference used by the Streamlit app (app.py),
the FastAPI service (api.py) and export_onnx.py, so preprocessing can
never drift apart.

Two interchangeable inference backends, same weights, same preprocessing:

  "onnx"  -> ONNX Runtime   (preferred; this is what runs on Render.com,
                             needs no PyTorch, ~150 MB RAM)
  "torch" -> PyTorch        (local fallback / export tooling)

Backend order: "onnx" is picked automatically when models/terrain_mobilenetv2.onnx
exists and onnxruntime is importable, otherwise "torch".
The *only* preprocessing implementation lives in _preprocess() below and both
backends consume its output, so predictions are identical by construction
(export_onnx.py still verifies it numerically).
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "terrain_mobilenetv2.pth"     # trained checkpoint (torch)
ONNX_PATH = MODELS_DIR / "terrain_mobilenetv2.onnx"     # exported graph (onnxruntime)
CLASSES_PATH = MODELS_DIR / "classes.json"              # class names for the ONNX path

# Same preprocessing as training/evaluation in train.py.
INPUT_SIZE = (224, 224)                                  # (width, height) for PIL
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image) -> np.ndarray:
    """PIL image -> float32 NCHW batch (1, 3, 224, 224).

    Numerically equivalent to torchvision's
    Resize((224,224)) -> ToTensor() -> Normalize(mean, std) because
    torchvision also resizes PIL images with PIL bilinear interpolation.
    """
    image = image.convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0          # HWC in [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                 # broadcast over H, W
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))[None]   # NCHW
    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))  # stable softmax
    return e / e.sum(axis=-1, keepdims=True)


def _load_classes() -> list:
    """Class names for the ONNX backend (classes.json, or read from the .pth)."""
    if CLASSES_PATH.exists():
        return json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if MODEL_PATH.exists():
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                f"{CLASSES_PATH.name} not found and PyTorch is not installed to read "
                "the .pth. Run `python export_onnx.py` locally to regenerate it."
            )
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        return list(checkpoint["classes"])
    raise FileNotFoundError(f"No model found in {MODELS_DIR} (expected {ONNX_PATH.name} / {MODEL_PATH.name})")


class OnnxBackend:
    """ONNX Runtime inference (CPU). Used on Render.com - no PyTorch needed."""

    name = "onnx"

    def __init__(self, onnx_path: Path = ONNX_PATH, classes: list | None = None):
        import onnxruntime as ort  # lazy: only imported when this backend is chosen
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self._session.get_inputs()[0].name
        self.classes = classes if classes is not None else _load_classes()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """(1,3,224,224) float32 -> (1, num_classes) logits."""
        return self._session.run(None, {self.input_name: batch})[0]


class TorchBackend:
    """PyTorch inference (CPU). Local fallback and source for export_onnx.py."""

    name = "torch"

    def __init__(self, model_path: Path = MODEL_PATH):
        import torch
        from torchvision import models

        self._torch = torch
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes: list = list(checkpoint["classes"])

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, len(self.classes)
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self._model = model

    def infer(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return self._model(self._torch.from_numpy(batch)).numpy()


def model_available() -> bool:
    """True when at least one model artifact exists."""
    return ONNX_PATH.exists() or MODEL_PATH.exists()


def load_model(backend: str | None = None):
    """Return (inference_backend, classes).

    backend: None -> auto ("onnx" first, then "torch");
             "onnx" / "torch" -> force that one.
    """
    order = [backend] if backend else ["onnx", "torch"]
    problems = []

    for name in order:
        if name == "onnx":
            if not ONNX_PATH.exists():
                problems.append(f"onnx: {ONNX_PATH.name} missing")
                continue
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                problems.append("onnx: onnxruntime not installed")
                continue
            classes = _load_classes()
            return OnnxBackend(ONNX_PATH, classes), classes

        if name == "torch":
            if not MODEL_PATH.exists():
                problems.append(f"torch: {MODEL_PATH.name} missing")
                continue
            try:
                import torch  # noqa: F401
            except ImportError:
                problems.append("torch: PyTorch not installed")
                continue
            handle = TorchBackend(MODEL_PATH)
            return handle, handle.classes

        raise ValueError(f"Unknown backend {name!r} (expected 'onnx' or 'torch')")

    raise RuntimeError("No inference backend available -> " + "; ".join(problems))


def predict(image: Image.Image, model, classes) -> tuple:
    """Classify one PIL image.

    Returns (pred_index, probabilities, latency_ms); latency covers the
    forward pass only (preprocessing excluded), same as before.
    """
    batch = _preprocess(image)
    start = time.perf_counter()
    logits = model.infer(batch)
    latency_ms = (time.perf_counter() - start) * 1000.0

    probs = _softmax(logits)[0]
    pred_index = int(np.argmax(probs))
    return pred_index, [float(p) for p in probs], latency_ms


def predict_dict(image: Image.Image, model, classes) -> dict:
    """Classify one PIL image and return a JSON-friendly dict for the API."""
    pred_index, probs, latency_ms = predict(image, model, classes)
    return {
        "class": classes[pred_index],
        "confidence": probs[pred_index],
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "latency_ms": round(latency_ms, 2),
        "backend": getattr(model, "name", "unknown"),
    }
