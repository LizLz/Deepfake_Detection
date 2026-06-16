import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
import os
from tqdm import tqdm
import numpy as np

# --- Setup ---
model_name = 'facebook/wav2vec2-base'
pre_processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Paths and label mapping ---
label_map = {'bonafide': 1, 'spoof': 0}
audio_labels = {}
protocol_path = "D:\\Felipe\\Team Lab\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt"
audio_base_path = "D:\\Felipe\\Team Lab\\LA\\ASVspoof2019_LA_eval\\flac"
output_tensor_dir = "D:\\Felipe\\Team Lab\\teamlab-phonetics\\feature_extraction\\wav2vec_updated_segment_outputs\\wav2vec_updated_eval"
os.makedirs(output_tensor_dir, exist_ok=True)

# --- Load protocol labels ---
with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[1]
        label = parts[-1]
        audio_labels[file_name] = label_map[label]

# --- Repeat-padding function ---
def limit_audio_length(y, sr, target_length=4):
    """
    Crop to target_length seconds if longer.
    Repeat-pad if shorter.
    """
    target_samples = target_length * sr
    if len(y) > target_samples:
        return y[:target_samples]
    else:
        repeats = target_samples // len(y) + 1
        return np.tile(y, repeats)[:target_samples]

# --- Feature extraction ---
def get_wav2vec_tensor(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = limit_audio_length(y, sr)
    # No need for padding=True for single file
    inputs = pre_processor(y, sampling_rate=sr, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs).last_hidden_state  # [B, T, D]
        # Optional: normalization if used in training
        # output = torch.nn.functional.layer_norm(output, output.shape[1:])
        wav2vec_tensor = output.permute(0, 2, 1)  # [B, 768, T]
    return wav2vec_tensor.cpu()

# --- Process and save features ---
audio_files = [f for f in os.listdir(audio_base_path) if f.endswith(".flac")]

for f in tqdm(audio_files, desc="Extracting and labeling"):
    file_path = os.path.join(audio_base_path, f)
    file_id = os.path.splitext(f)[0]
    if file_id not in audio_labels:
        print(f"Warning: No label found for {file_id}, skipping.")
        continue
    try:
        wav2vec_tensor = get_wav2vec_tensor(file_path)
        label = audio_labels[file_id]
        tensor_output_path = os.path.join(output_tensor_dir, f"{file_id}.pt")
        torch.save((wav2vec_tensor, label), tensor_output_path)
    except Exception as e:
        print(f"Failed on {file_id}: {e}")

print("All audio processed and saved with labels.")
