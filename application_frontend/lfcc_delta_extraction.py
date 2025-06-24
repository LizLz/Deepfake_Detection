import librosa
import numpy as np

def extract_lfcc_features(audio_path, sr=16000, n_lfcc=20):
    y, sr = librosa.load(audio_path, sr=sr)
    # Extract LFCCs
    lfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_lfcc, dct_type=2, norm='ortho', lifter=0, n_fft=512, hop_length=160)
    # Compute deltas
    delta = librosa.feature.delta(lfccs)
    delta2 = librosa.feature.delta(lfccs, order=2)
    # Stack features: (n_features, time)
    features = np.vstack([lfccs, delta, delta2])
    return features
