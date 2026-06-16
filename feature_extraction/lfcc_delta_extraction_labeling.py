import os
import torch
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T
import numpy as np

# Label mapping
label_map = {'genuine': 1, 'bonafide':1, 'fake': 0, 'spoof': 0} # Add uses genuine and fake while others use bonafide and spoof
audio_labels = {}

# === Paths ===
protocol_path = "C:\\Users\\Zheng\\Downloads\\track1_label.txt"
audio_base_path = "C:\\Users\\Zheng\\Downloads\\add2022eval\\track1test\\track1test"
output_tensor_dir = "C:\\Users\\Zheng\\Desktop\\project\\team_lab_deepfake\\add_lfcc_with_delta_eval"
os.makedirs(output_tensor_dir, exist_ok=True)

# === Load protocol labels ===
with open(protocol_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        file_name = parts[0]
        label = parts[-1]

        file_id = os.path.splitext(file_name)[0]
        audio_labels[file_id] = label_map[label]

# === Utility: Limit audio length to 4 seconds ===
def limit_audio_length(waveform, sr, target_sec=4):
    target_samples = target_sec * sr
    current_samples = waveform.shape[1]
    if current_samples > target_samples:
        return waveform[:, :target_samples]
    else:
        repeats = (target_samples // current_samples) + 1
        extended = waveform.repeat(1, repeats)
        return extended[:, :target_samples]

# === LFCC extraction ===
def get_lfcc_tensor(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = limit_audio_length(waveform, sr)

    lfcc_transform = T.LFCC(
        sample_rate=sr,
        n_lfcc=40,
        speckwargs={"n_fft": 2048, "hop_length": 512}
    )
    lfcc = lfcc_transform(waveform)  # shape: [1, 40, timeframes]

    compute_delta = T.ComputeDeltas()
    delta = compute_delta(lfcc)
    delta2 = compute_delta(delta)

    lfcc_full = torch.cat([lfcc, delta, delta2], dim=1)  # shape: [1, 120, timeframes]
    return lfcc_full.squeeze(0).float()           # shape: [120, timeframes]

# === Process audio files ===
audio_files = [f for f in os.listdir(audio_base_path) if f.endswith(".wav")] # add wav files but other datasets are flac

for f in tqdm(audio_files, desc="Extracting and labeling"):
    file_path = os.path.join(audio_base_path, f)
    file_id = os.path.splitext(f)[0]
    if file_id not in audio_labels:
        print(f"No label found for {file_id}, skipping.")
        continue

    try:
        lfcc_tensor = get_lfcc_tensor(file_path)
        label = audio_labels[file_id]

        tensor_output_path = os.path.join(output_tensor_dir, f"{file_id}.pt")
        torch.save((lfcc_tensor, label), tensor_output_path)
    except Exception as e:
        print(f"Error processing {file_id}: {e}")

print("All audio processed and saved with labels.")