import streamlit as st
import os
from lfcc_delta_extraction import *
from spectrogram_generation import *
from pytorch_model_lfcc import *

st.title("Deepfake Audio Detection (PyTorch, LFCC+Delta)")

uploaded_file = st.file_uploader("Upload a WAV file", type="wav")
if uploaded_file:
    # Save file locally
    audio_path = os.path.join("audio_files", uploaded_file.name)
    os.makedirs("audio_files", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(audio_path)

    st.write("### Spectrogram")
    spec_img_path = save_spectrogram_image(audio_path)
    st.image(spec_img_path)

    st.write("### Classification")
    # Extract features
    lfcc_features = extract_lfcc_features(audio_path)
    # Load your PyTorch model (ensure you load once, not every time)
    model = torch.load("lfcc_with_delta_v2_updated_model_deepfake_audio_detection_model.pth", map_location="cpu")
    model.eval()
    pred, prob = predict_with_pytorch(model, lfcc_features)
    class_names = ["bonafide", "spoof"]
    st.write(f"**Prediction:** {class_names[pred]}")
    st.write(f"**Probabilities:** {prob}")
