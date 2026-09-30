"""
TerrainSense - export_onnx.py
=============================
Converts the trained PyTorch checkpoint into an ONNX graph so the API can
run on Render.com's free tier WITHOUT installing PyTorch (~2 GB build).

Outputs (both are committed to git):
    models/terrain_mobilenetv2.onnx   <- served by onnxruntime on Render
    models/classes.json               <- class names for the ONNX path

Also runs a numerical parity check: every sample must keep the same predicted
class and the probabilities must match PyTorch within 1e-3.

Run:
    python export_onnx.py
"""

import sys

import numpy as np
from PIL import Image

import model_utils as mu


def _export_graph(handle) -> None:
    """Save the loaded torchvision model as ONNX (input: 1x3x224x224 -> logits)."""
    import torch

    model = handle._model
    dummy = torch.zeros(1, 3, 224, 224)

    # Try the legacy exporter first, then whatever the installed torch defaults to.
    last_error = None
    for kwargs in ({"dynamo": False}, {}):
        try:
            torch.onnx.export(
                model,
                dummy,
                str(mu.ONNX_PATH),
                input_names=["input"],
                output_names=["logits"],
                opset_version=17,
                **kwargs,
            )
            return
        except TypeError as exc:  # e.g. unexpected 'dynamo' kwarg
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"ONNX export failed: {last_error}")


def _parity_check(torch_handle, onnx_handle, classes: list) -> None:
    """Compare both backends on real dataset photos; fail loudly on mismatch."""
    # Two real photos per class straight from the Kaggle dataset.
    samples = []
    for class_dir in sorted(mu.MODELS_DIR.parent.glob("dataset/*")):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.jpg"))[:2]:
            samples.append((class_dir.name, img_path))
    if not samples:
        sys.exit("No dataset images found - cannot verify parity. Is dataset/ present?")

    print(f"\nParity check on {len(samples)} real dataset photos:")
    print(f"  {'photo':<40} {'torch':<22} {'onnx':<22} {'max dP'}")
    ok = True
    for true_class, path in samples:
        image = Image.open(path)
        t_pred, t_probs, _ = mu.predict(image, torch_handle, classes)
        o_pred, o_probs, _ = mu.predict(image, onnx_handle, classes)
        max_diff = float(np.max(np.abs(np.array(t_probs) - np.array(o_probs))))
        same_class = t_pred == o_pred
        ok = ok and same_class and max_diff < 1e-3
        marker = " " if (same_class and max_diff < 1e-3) else "!"
        print(
            f"{marker} {path.parent.name + '/' + path.name:<38} "
            f"{classes[t_pred] + f' {t_probs[t_pred]:.4f}':<22} "
            f"{classes[o_pred] + f' {o_probs[o_pred]:.4f}':<22} {max_diff:.6f}"
        )

    if not ok:
        sys.exit("\nFAILED: ONNX output diverges from PyTorch - do NOT deploy.")
    print("\nOK: same predicted class on every photo, max probability diff < 1e-3.")


def main() -> None:
    print(f"Loading PyTorch model from {mu.MODEL_PATH} ...")
    torch_handle, classes = mu.load_model(backend="torch")
    print(f"Classes: {classes}")

    print("Exporting ONNX graph ...")
    _export_graph(torch_handle)
    mu.CLASSES_PATH.write_text(
        __import__("json").dumps(classes, indent=2), encoding="utf-8"
    )
    size_mb = mu.ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {mu.ONNX_PATH} ({size_mb:.1f} MB) + {mu.CLASSES_PATH.name}")

    # Validate the file and cross-check it against PyTorch.
    try:
        import onnx

        onnx.checker.check_model(str(mu.ONNX_PATH))
        print("onnx.checker: model OK")
    except ImportError:
        print("onnx package not installed - skipping checker (export still usable)")

    onnx_handle, _ = mu.load_model(backend="onnx")
    print(f"Loaded with {onnx_handle.name} backend")
    _parity_check(torch_handle, onnx_handle, classes)


if __name__ == "__main__":
    main()
