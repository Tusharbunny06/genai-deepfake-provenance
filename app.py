import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import tempfile
import hashlib

from utils.transforms import get_transforms
from video_generator import generate_video

st.title("GenAI Deepfake Detection with Provenance Verification")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Deepfake Detection Model
# ----------------------------

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("models/resnet_baseline.pth", map_location=device)
)

model = model.to(device)
model.eval()

transform = get_transforms()

# ----------------------------
# Provenance Hash
# ----------------------------

def generate_hash(file_bytes):

    sha256 = hashlib.sha256()
    sha256.update(file_bytes)

    return sha256.hexdigest()


# ----------------------------
# Frame Extraction
# ----------------------------

def extract_frames(video_path, num_frames=5):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_ids = np.linspace(
        0,
        max(total_frames - 1, 1),
        num_frames
    ).astype(int)

    frames = []

    for i in range(total_frames):

        ret, frame = cap.read()

        if not ret:
            break

        if i in frame_ids:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    return frames


# ----------------------------
# Deepfake Prediction
# ----------------------------

def predict_video(video_path):

    frames = extract_frames(video_path)

    processed = []

    for frame in frames:

        frame = transform(frame)
        processed.append(frame)

    tensor = torch.stack(processed).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        outputs = outputs.mean(dim=0)

        probs = torch.softmax(outputs, dim=0)

    return probs.cpu().numpy()


# ----------------------------
# Streamlit Interface
# ----------------------------

option = st.radio(
    "Choose Input",
    ["Generate AI Video (Prompt)", "Upload Video"]
)

video_path = None
video_bytes = None


# ----------------------------
# Prompt → Generated Video
# ----------------------------

if option == "Generate AI Video (Prompt)":

    prompt = st.text_input("Enter Prompt")

    if st.button("Generate Video"):

        with st.spinner("Generating video..."):

            video_path = generate_video(prompt)

            with open(video_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)


# ----------------------------
# Upload Video
# ----------------------------

if option == "Upload Video":

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4"]
    )

    if uploaded_file is not None:

        video_bytes = uploaded_file.read()

        st.video(video_bytes)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_bytes)

        video_path = tfile.name


# ----------------------------
# Analyze Video
# ----------------------------

if video_bytes is not None and st.button("Analyze Video"):

    with st.spinner("Analyzing video..."):

        hash_value = generate_hash(video_bytes)

        probs = predict_video(video_path)

        real_prob = probs[0]
        fake_prob = probs[1]

        prediction = "FAKE" if fake_prob > real_prob else "REAL"

        st.subheader(f"Prediction: {prediction}")

        st.write(f"Fake Probability: {fake_prob:.2f}")
        st.write(f"Real Probability: {real_prob:.2f}")

        st.success("Provenance hash generated successfully")