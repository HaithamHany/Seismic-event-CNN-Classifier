import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from obspy import read
from scipy.signal import spectrogram
import torch.nn.functional as F
import torchvision.transforms as transforms

def pad_spectrogram(spectrogram, max_length):
    padding_size = max_length - spectrogram.shape[-1]
    if padding_size > 0:
        # Pad only the last dimension (time axis)
        padded_spectrogram = F.pad(spectrogram, (0, padding_size), mode='constant', value=0)
    else:
        padded_spectrogram = spectrogram
    return padded_spectrogram

class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, catalog_path, labeled=True, window_size=1024, step_size=128):
        self.data = []
        self.labels = []
        self.filenames = []
        self.time_rels = []
        self.labeled = labeled
        self.window_size = window_size
        self.step_size = step_size

        # Load the catalog for cross-referencing
        self.catalog = pd.read_csv(catalog_path)

        # Load the spectrogram data
        self.load_data(folder_path)

    def load_data(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".mseed"):
                file_path = os.path.join(folder_path, filename)
                stream = read(file_path)
                trace = stream[0]

                # Extract seismic data and sampling rate
                raw_data = trace.data
                sampling_rate = trace.stats.sampling_rate

                # Generate spectrogram
                magnitude = self.generate_spectrogram(raw_data, sampling_rate)

                # Normalize spectrogram
                magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

                # Split spectrogram into windows
                spectrogram_windows = self.create_windows(magnitude)
                self.data.extend(spectrogram_windows)

                # Store filename for each window
                self.filenames.extend([filename]*len(spectrogram_windows))

                if self.labeled:
                    quake_time_rel = self.get_quake_time_from_catalog(filename)
                    label_windows = self.label_windows(spectrogram_windows, quake_time_rel, sampling_rate)
                    self.labels.extend(label_windows)

    def generate_spectrogram(self, data, fs, nperseg=256):
        frequencies, times, Sxx = spectrogram(data, fs=fs, nperseg=nperseg)
        return Sxx

    def create_windows(self, spectrogram):
        num_windows = (spectrogram.shape[1] - self.window_size) // self.step_size + 1
        windows = []
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            window = spectrogram[:, start:end]
            windows.append(window)
        return windows

    def get_quake_time_from_catalog(self, filename):
        row = self.catalog[self.catalog['filename'] == filename]
        if not row.empty:
            return float(row['time_rel(sec)'].values[0])
        return None

    def label_windows(self, windows, quake_time_rel, sampling_rate):
        labels = []
        window_duration = (self.window_size / sampling_rate)
        for i in range(len(windows)):
            window_start_time = i * (self.step_size / sampling_rate)
            window_end_time = window_start_time + window_duration

            if quake_time_rel is not None and (window_start_time <= quake_time_rel <= window_end_time):
                labels.append(1)  # Quake present in window
            else:
                labels.append(0)  # No quake in window
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(
            0)  # Shape: [1, freq_bins, time_steps]

        # Resize spectrogram to [1, 224, 224]
        spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(224, 224), mode='bilinear',
                                    align_corners=False).squeeze(0)

        if self.labeled:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return spectrogram, label
        else:
            metadata = {'filename': self.filenames[idx]}
            return spectrogram, metadata
