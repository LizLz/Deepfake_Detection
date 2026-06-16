import os
import torch
import numpy as np
import librosa
from tqdm import tqdm

# === Paths ===
input_tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/lfcc_segment_outputs/lfcc_eval"
output_tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/lfcc_segment_outputs/lfcc_tensors_eval_set_with_delta"
os.makedirs(output_tensor_dir, exist_ok=True)

# === Process all saved log-mel tensors ===
audio_files = [f for f in os.listdir(input_tensor_dir) if f.endswith(".pt")]

for f in tqdm(audio_files, desc="Adding delta features"):
    tensor_path = os.path.join(input_tensor_dir, f)
    logmel_tensor, label = torch.load(tensor_path)
    logmel_np = logmel_tensor.numpy()

    # Compute delta and delta-delta
    delta = librosa.feature.delta(logmel_np)
    delta2 = librosa.feature.delta(logmel_np, order=2)

    # Concatenate along the feature axis (first axis)
    logmel_with_deltas = np.concatenate([logmel_np, delta, delta2], axis=0)
    logmel_with_deltas_tensor = torch.from_numpy(logmel_with_deltas).float()

    # Save new tensor with label
    output_path = os.path.join(output_tensor_dir, f)
    torch.save((logmel_with_deltas_tensor, label), output_path)

print("All log-mel tensors processed and saved with delta and delta-delta features.")