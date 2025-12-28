import os
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATHS = {
    "YOLO11n (nano)": "best_yolo11n.pt",
    "YOLO11s (small)": "best_yolo11s.pt",
}

st.set_page_config(
    page_title="PPE Detection - YOLO",
    layout="wide",
    page_icon="🦺"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)

def run_inference_bgr(model, img_bgr, conf):
    return model.predict(
        source=img_bgr,
        conf=conf,
        verbose=False
    )

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "YOLO Model",
        list(MODEL_PATHS.keys())
    )

    conf = st.slider("Confidence", 0.01, 1.0, 0.25, 0.01)

    mode = st.radio("Input Mode", ["Image", "Webcam"])

# =========================
# LOAD SELECTED MODEL
# =========================
try:
    model = load_model(MODEL_PATHS[model_choice])
except Exception as e:
    st.error(e)
    st.stop()

# =========================
# TITLE
# =========================
st.markdown("## 🦺 PPE Detection System")
st.caption("Live YOLO-based PPE detection for food industry workplaces")

# =========================
# IMAGE MODE
# =========================
if mode == "Image":
    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"]
        )
        run_btn = st.button(
            "Run detection",
            type="primary",
            disabled=(uploaded is None)
        )

    if uploaded and run_btn:
        img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = run_inference_bgr(model, img_bgr, conf)

        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.image(
                annotated_rgb,
                caption="Detection Result",
                use_container_width=True
            )

# =========================
# WEBCAM MODE (LIVE)
# =========================
else:
    start = st.button("▶ Start Webcam", type="primary")
    stop = st.button("⏹ Stop Webcam")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if start:
        st.session_state.webcam_running = True
    if stop:
        st.session_state.webcam_running = False

    frame_box = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.stop()

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            results = run_inference_bgr(model, frame, conf)

            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_box.image(
                annotated_rgb,
                channels="RGB",
                use_container_width=True
            )

        cap.release()
        st.info("Webcam stopped")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed by Nicholas Victorio • PPE Detection Project")
