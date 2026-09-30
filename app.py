"""
TerrainSense - app.py
=====================
Streamlit single-page dashboard: drag and drop a landscape/ground photo and
get an instant terrain classification (class name + confidence) from the
MobileNetV2 model trained by train.py.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from model_utils import load_model as _load_model, model_available, predict as _predict

st.set_page_config(page_title="TerrainSense", page_icon="🌍", layout="centered")


# ---------------------------------------------------------------------------
# Model loading (cached - loaded once per app session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained checkpoint and rebuild the model. Returns (model, classes)."""
    return _load_model()


def predict(image: Image.Image, model, classes):
    """Run inference on a PIL image. Returns (pred_index, probabilities, latency_ms)."""
    return _predict(image, model, classes)


def plot_confidence_bar(probs: np.ndarray, classes: list[str], pred_index: int) -> plt.Figure:
    """Horizontal bar chart of per-class confidence scores."""
    order = np.argsort(probs)  # ascending, so the top class sits at the top of the chart
    colors = ["#4C9BE8"] * len(classes)
    colors[list(order).index(pred_index)] = "#2E7D32"  # highlight the predicted class

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([classes[i] for i in order], probs[order], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Class probabilities")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🌍 TerrainSense")
st.caption("Real-time image-based AI terrain recognition — MobileNetV2 transfer learning")

if not model_available():
    st.error("No trained model found.")
    st.info(
        "Train one first:\n\n"
        "1. Put your images in `dataset/<class_name>/` folders\n"
        "2. Run `python train.py`\n"
        "3. Restart this app"
    )
    st.stop()

model, classes = load_model()

st.write("**Classes:**", ", ".join(classes))

uploaded = st.file_uploader("Upload a landscape / ground surface photo",
                            type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is None:
    st.info("Drag and drop an image above to classify it.")
    st.stop()

try:
    image = Image.open(uploaded)
except Exception as exc:  # corrupt or unsupported file
    st.error(f"Could not read that image: {exc}")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.image(image, caption="Uploaded photo", use_container_width=True)

with right:
    try:
        pred_index, probs, latency_ms = predict(image, model, classes)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    confidence = float(probs[pred_index])
    st.subheader(f"Prediction: **{classes[pred_index]}**")
    st.metric("Confidence", f"{confidence:.1%}")
    st.metric("Inference time", f"{latency_ms:.1f} ms")

    st.pyplot(plot_confidence_bar(probs, classes, pred_index))

    # Top-3 breakdown as plain text for a quick read.
    top3 = np.argsort(probs)[::-1][:3]
    st.write("**Top 3:**")
    for i in top3:
        st.write(f"- {classes[i]} — {probs[i]:.1%}")

st.caption("TerrainSense · PyTorch + MobileNetV2 + Streamlit")
