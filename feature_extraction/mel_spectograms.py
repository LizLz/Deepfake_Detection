import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

audio_path = "C:/Users/felip/teamlab-phonetics/feature_extraction/LA_T_1000137.flac"
y, sr = librosa.load(audio_path, sr=None)

segment_duration_sec = 4  # as used in the paper 
segment_length = int(segment_duration_sec * sr)
n_fft = 2048
hop_length = 512
n_mels = 64

mel_segments = []
num_samples = len(y)

for start in range(0, num_samples, segment_length):
    end = start + segment_length
    segment = y[start:end]
    seg_len = len(segment)
    if seg_len < segment_length:
        # Repeat the segment enough times to reach or exceed the desired length, then trim
        n_repeats = int(np.ceil(segment_length / seg_len))
        segment = np.tile(segment, n_repeats)[:segment_length]
    elif seg_len > segment_length:
        segment = segment[:segment_length]  # Just in case, trim to exact length

    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel_segments.append(mel_spec)

# Directory to save mel spectrogram images
output_dir = "../mel_segment_outputs"
os.makedirs(output_dir, exist_ok=True)

for i, mel_spec in enumerate(mel_segments):
    # Convert mel spectrogram to log scale for better visualization
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Save as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram {i}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mel_segment_{i:03d}.png'))
    plt.close()

# Convert mel spectrogram arrays to PyTorch tensors
mel_tensors = []
for mel_spec in mel_segments:
    mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)
    mel_tensors.append(mel_tensor)

# Save tensors
output_tensor_dir = "C:/Users/felip/teamlab-phonetics/mel_segment_outputs/tensors"
os.makedirs(output_tensor_dir, exist_ok=True)
for i, mel_tensor in enumerate(mel_tensors):
    torch.save(mel_tensor, os.path.join(output_tensor_dir, f'mel_segment_{i:03d}.pt'))

print("Mel spectrograms and tensors saved successfully.")