import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --------------------------
# 1. Updated Model Class (with adaptive pooling)
# --------------------------
class SpoofDetectionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_prob):
        super(SpoofDetectionModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Adaptive pooling to handle variable lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(24)  # Fixed output size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 24, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input: [batch, 768, time_frames]
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        
        # Apply adaptive pooling to fix the size
        x = self.adaptive_pool(x)  # Output: [batch, 128, 24]
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --------------------------
# 2. Feature Extraction
# --------------------------
def limit_audio_length(waveform, sr, target_sec=4):
    target_samples = target_sec * sr
    current_samples = waveform.shape[1]
    if current_samples > target_samples:
        # Crop to the target length
        return waveform[:, :target_samples]
    else:
        # Do not pad, just return as is (or handle as you wish)
        return waveform

def extract_wav2vec_features(audio_path):
    model_name = 'facebook/wav2vec2-base'
    pre_processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    inputs = pre_processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state  # [B, T, D]
        # Normalize over the feature dimension (last dimension, usually 768)
        outputs = F.layer_norm(outputs, outputs.shape[-1:])  # normalize over dim=768
        wav2vec_tensor = outputs.permute(0, 2, 1)  # [B, 768, T]
    return wav2vec_tensor

def save_spectrogram_image(audio_path, output_path='melspectrogram.png'):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze()
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(mel_spec_db.numpy(), aspect='auto', origin='lower')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path

# --------------------------
# 3. Prediction Function
# --------------------------
def predict_with_pytorch(model, features):
    input_tensor = features.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.sigmoid(outputs).item()
    return prediction

# --------------------------
# 3. Model loading function
# --------------------------
@st.cache_resource
def load_model():
    hidden_dim = 256
    dropout_prob = 0.5
    model = SpoofDetectionModel(hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    state_dict = torch.load("wav2vec_best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --------------------------
# 4. Streamlit App
# --------------------------
st.title("Deepfake Audio Detection (PyTorch, Wav2Vec2)")

uploaded_file = st.file_uploader("Upload a WAV file", type="wav")
if uploaded_file is not None:
    os.makedirs("audio_files", exist_ok=True)
    audio_path = os.path.join("audio_files", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load the waveform
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

    # Limit the audio length
    waveform = limit_audio_length(waveform, 16000)  # Use 16000 as the sampling rate for limit_audio_length

    st.audio(audio_path)  # Display the original audio (optional)

    st.write("### Spectrogram")
    spec_img_path = save_spectrogram_image(audio_path)
    st.image(spec_img_path)

    st.write("### Classification")
    features = extract_wav2vec_features(audio_path)
    st.write(f"Wav2Vec2 features shape: {features.shape}")  # Debug: should be [1, 768, 199]
    model = load_model()
    prediction = predict_with_pytorch(model, features)
    st.write(f"**Prediction:** {prediction:.3f} (Probability of being bonafide)")

else:
    st.info("Please upload a .wav file")