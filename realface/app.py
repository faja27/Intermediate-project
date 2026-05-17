import os
import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "realface.onnx")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource
def load_session():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    # CenterCrop(224) — already 224x224, no-op
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)


def predict(session, img: Image.Image) -> tuple[str, float]:
    x     = preprocess(img)
    logit = session.run(None, {"input": x})[0][0, 0]
    prob  = float(1 / (1 + np.exp(-logit)))  # sigmoid
    label = "REAL" if prob >= 0.5 else "FAKE"
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, conf


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("About")
    st.markdown(
        """
        **Model**: EfficientNet-B0 (fine-tuned)
        **Dataset**: 10 000 faces — 5 000 real · 5 000 AI-generated (StyleGAN3)
        **Export**: ONNX opset 11, CPU inference
        """
    )
    st.divider()
    st.markdown(
        """
        **How to use**
        1. Upload a face image (JPG / PNG)
        2. Click **Analyze**
        3. Read the result
        """
    )

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("RealFace — AI Face Authenticity Detector")
st.caption("Upload a portrait and find out whether it is a real photo or an AI-generated face.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze", type="primary"):
        try:
            session        = load_session()
            label, conf    = predict(session, img)
            conf_pct       = conf * 100

            st.divider()

            color      = "#2ecc71" if label == "REAL" else "#e74c3c"
            icon       = "✅" if label == "REAL" else "🚫"
            st.markdown(
                f"<h1 style='color:{color}; text-align:center;'>{icon} {label}</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Confidence: {conf_pct:.1f}%**")
            st.progress(int(conf_pct))

            if label == "REAL":
                explanation = "The model believes this is an authentic photograph of a real person."
            else:
                explanation = "The model detected patterns typical of AI-generated (synthetic) faces."
            st.info(explanation)

        except FileNotFoundError:
            st.error(f"Model not found at `{MODEL_PATH}`. Place `realface.onnx` inside the `model/` folder.")
