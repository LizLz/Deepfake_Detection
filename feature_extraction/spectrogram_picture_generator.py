import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# === CONFIGURATION ===
input_tensor_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/mel_segment_outputs/tensors_train_set_with_delta"  # or any other feature dir
num_samples = 5  # Number of random samples to plot
output_plot_dir = "C:/Users/felip/teamlab-phonetics/feature_extraction/plots_log_mel_with_delta"
os.makedirs(output_plot_dir, exist_ok=True)

# === Get list of files ===
audio_files = [f for f in os.listdir(input_tensor_dir) if f.endswith(".pt")]
if num_samples > len(audio_files):
    num_samples = len(audio_files)
selected_files = random.sample(audio_files, num_samples)

for f in tqdm(selected_files, desc="Plotting spectrograms"):
    tensor_path = os.path.join(input_tensor_dir, f)
    features, label = torch.load(tensor_path)
    features_np = features.numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(features_np, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram: {f} | Label: {label}")
    plt.xlabel("Time Frames")
    plt.ylabel("Feature Bins")
    plt.tight_layout()
    # Save plot
    plot_path = os.path.join(output_plot_dir, f"{os.path.splitext(f)[0]}_spectrogram.png")
    plt.savefig(plot_path)
    plt.close()

print(f"✅ Plots saved to {output_plot_dir}")