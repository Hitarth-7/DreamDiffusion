%pip install torch torchvision torchaudio --quiet
%pip install --upgrade pip --quiet

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import matplotlib.pyplot as plt

data = torch.load('eeg_signals_raw_with_mean_std.pth')
eeg_data = np.array(data['dataset'])
labels = np.array(data['labels'])
means = np.array(data['means'])
stds = np.array(data['stddevs'])

print(f"Data type: {type(data)}")
print(f"Keys: {data.keys()}")
print(f"Sample dict keys: {eeg_data[0].keys()}")
for i in range(5):
    print(f"Sample {i} shape: {np.array(eeg_data[i]['eeg']).shape}")

# Pad/Truncate EEG signals

target_length = 532
fixed_signals = []
for sample in eeg_data:
    eeg = np.array(sample['eeg'])
    T = eeg.shape[1]
    if T < target_length:
        eeg = np.pad(eeg, ((0, 0), (0, target_length - T)), mode='constant')
    else:
        eeg = eeg[:, :target_length]
    fixed_signals.append(eeg)

eeg_data_array = np.stack(fixed_signals)
print("Final EEG array shape:", eeg_data_array.shape)

# Bandpass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass(data, lowcut=0.5, highcut=45.0, fs=128.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

filtered_eeg = np.zeros_like(eeg_data_array)
for i in range(eeg_data_array.shape[0]):
    for ch in range(eeg_data_array.shape[1]):
        filtered_eeg[i, ch] = apply_bandpass(eeg_data_array[i, ch])

print("Filtered EEG shape:", filtered_eeg.shape)
np.save('filtered_eeg_data.npy', filtered_eeg)

# Clean Data
cleaned = [eeg for eeg in filtered_eeg if not (np.isnan(eeg).any() or np.isinf(eeg).any()) and eeg.shape == (128, 532)]
cleaned_eeg_data = np.stack(cleaned)
print("Cleaned EEG shape:", cleaned_eeg_data.shape)

#  Epoching 
def epoch_eeg(eeg_array, epoch_len=128):
    N, C, T = eeg_array.shape
    total_epochs = T // epoch_len
    trimmed = eeg_array[:, :, :total_epochs * epoch_len]
    reshaped = trimmed.reshape(N, C, total_epochs, epoch_len)
    transposed = reshaped.transpose(0, 2, 1, 3)
    return transposed.reshape(-1, C, epoch_len)

eeg_epochs = epoch_eeg(cleaned_eeg_data)
print("Epoch EEG shape:", eeg_epochs.shape)

# Normalization
means = means.reshape(1, 128, 1)
stds = stds.reshape(1, 128, 1)
normalized_eeg = (eeg_epochs - means) / stds
print("Normalized EEG shape:", normalized_eeg.shape)

# Embedding Preparation
token_size = 4
embedding_dim = 1024
batch_size = 8
device = torch.device("cpu")

N, C, T = normalized_eeg.shape
assert C == 128 and T == 128 and T % token_size == 0
num_tokens = T // token_size


norm_means = normalized_eeg.mean(axis=(0, 2), keepdims=True)
norm_stds = normalized_eeg.std(axis=(0, 2), keepdims=True) + 1e-6
normalized_eeg = (normalized_eeg - norm_means) / norm_stds

embedding_layer = torch.nn.Linear(token_size, embedding_dim).to(device)
embedding_layer.eval()

all_embeddings = []
try:
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = normalized_eeg[start:end]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            tokens = batch_tensor.reshape(batch.shape[0], C, num_tokens, token_size)
            tokens_flat = tokens.reshape(-1, token_size)
            embedded_flat = embedding_layer(tokens_flat)
            embedded = embedded_flat.view(batch.shape[0], C, num_tokens, embedding_dim)
            all_embeddings.append(embedded.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    print("✅ Final Embeddings Shape:", final_embeddings.shape)

except Exception as e:
    print("❌ Error during embedding:", e)

# FFT Visualization
fs = 128
freqs = np.fft.fftfreq(T, d=1/fs)[:T//2]
samples_to_plot = 3
channels = [0, 10, 20]

for i in range(samples_to_plot):
    plt.figure(figsize=(15, 4))
    for j, ch in enumerate(channels):
        signal = normalized_eeg[i, ch]
        power = np.abs(fft(signal)[:T//2]) ** 2
        plt.subplot(1, len(channels), j+1)
        plt.plot(freqs, power)
        plt.title(f"Sample {i}, Channel {ch}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
