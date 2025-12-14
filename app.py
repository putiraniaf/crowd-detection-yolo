import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Crowd Detection using YOLOv8",
    layout="wide"
)

st.title("👥 Crowd Detection & Crowd Level Classification")
st.write(
    "A Deep Learning application using YOLOv8 to detect people and "
    "classify crowd levels from CCTV-like images."
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    return YOLO(model_path)

model = load_model()

# =========================
# UTILS
# =========================
def classify_crowd(count):
    if count <= 3:
        return "Sedikit"
    elif count <= 30:
        return "Sedang"
    else:
        return "Ramai"

def draw_boxes(image, results, conf_thres):
    count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf >= conf_thres:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, count

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05
)

input_type = st.sidebar.radio(
    "Input Type",
    ["Image", "Video"]
)

# =========================
# IMAGE MODE
# =========================
if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image (jpg / png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(image, verbose=False)[0]
        output_img, people_count = draw_boxes(
            image.copy(), results, conf_thres
        )

        crowd_level = classify_crowd(people_count)

        col1, col2 = st.columns(2)

        with col1:
            st.image(output_img, channels="BGR", caption="Detection Result")

        with col2:
            st.subheader("📊 Result")
            st.metric("People Count", people_count)
            st.metric("Crowd Level", crowd_level)

# =========================
# VIDEO MODE
# =========================
elif input_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            output_frame, people_count = draw_boxes(
                frame, results, conf_thres
            )

            crowd_level = classify_crowd(people_count)

            cv2.putText(
                output_frame,
                f"People: {people_count} | Crowd: {crowd_level}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            stframe.image(output_frame, channels="BGR")

        cap.release()
        os.remove(tfile.name)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Final Project – Deep Learning | "
    "YOLOv8 Crowd Detection | BINUS University"
)
