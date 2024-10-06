import os
import glob
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import spectrogram
import numpy as np
import re
from obspy import read  # Import ObsPy

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, catalog_file=None, window_size=8192, step_size=1024, labeled=True):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.labeled = labeled

        # Load the catalog if labeled data is expected
        if self.labeled:
            self.catalog = pd.read_csv(catalog_file)
            print("Catalog loaded.")
        else:
            self.catalog = None
            print("No catalog loaded. Data will not be labeled.")

        # List all .mseed files in the data directory
        self.filenames = sorted(glob.glob(os.path.join(self.data_dir, '*.mseed')))
        print(f"Found {len(self.filenames)} files.")

        # Load data and labels
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []
        metadata = []

        for filename in self.filenames:
            # Load the seismic data
            try:
                st = read(filename)
                tr = st[0]
                signal = tr.data.astype(np.float32)
                sampling_rate = tr.stats.sampling_rate
                print(f"Loaded {filename} with sampling rate {sampling_rate}.")
                print(f"Signal length: {len(signal)}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            # Split signal into windows
            windows = self.split_signal(signal)
            if not windows:
                print(f"No valid windows found in {filename}. Skipping file.")
                continue

            # Get labels
            if self.labeled:
                quake_time_rel = self.get_quake_time_from_catalog(filename)
                window_labels = self.label_windows(windows, quake_time_rel, sampling_rate)
            else:
                window_labels = [0] * len(windows)

            # Compute spectrograms and collect valid ones
            valid_spectrograms = []
            valid_labels = []
            for window, label in zip(windows, window_labels):
                Sxx = self.compute_spectrogram(window, sampling_rate)
                if Sxx is not None:
                    valid_spectrograms.append(Sxx)
                    valid_labels.append(label)
                    metadata.append({'filename': filename})
                else:
                    print("Skipping window due to invalid spectrogram.")

            if valid_spectrograms:
                data.extend(valid_spectrograms)
                labels.extend(valid_labels)
            else:
                print(f"No valid spectrograms found in {filename} after processing.")

        self.metadata = metadata
        return data, labels

    def split_signal(self, signal):
        windows = []
        if len(signal) < self.window_size:
            print(f"Signal length ({len(signal)}) is shorter than window size ({self.window_size}). Skipping signal.")
            return windows
        else:
            for start in range(0, len(signal) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window = signal[start:end]
                windows.append(window)
            print(f"Number of windows generated from signal: {len(windows)}")
        return windows

    def compute_spectrogram(self, window, sampling_rate):
        nperseg = 128  # Adjusted for your data
        noverlap = nperseg // 2  # 50% overlap

        if len(window) < nperseg:
            print(f"Window length ({len(window)}) is less than nperseg ({nperseg}). Skipping window.")
            return None

        frequencies, times, Sxx = spectrogram(
            window,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        print(f"Sxx shape: {Sxx.shape}")

        if Sxx.ndim != 2 or Sxx.shape[0] <= 1 or Sxx.shape[1] <= 1:
            print("Spectrogram computation failed or resulted in invalid output.")
            return None
        return Sxx

    def get_quake_time_from_catalog(self, filename):
        # Extract event ID from the filename using regex
        match = re.search(r'evid(\d+)', filename)
        if match:
            event_id = int(match.group(1))
            print(f"Extracted event ID: {event_id}")
        else:
            print(f"No event ID found in filename: {filename}")
            return None

        # Match event ID with the catalog
        row = self.catalog[self.catalog['evid'] == event_id]

        if not row.empty:
            quake_time = row['time_rel(sec)'].values[0]
            print(f"Quake time for event ID {event_id}: {quake_time}")
            return float(quake_time)
        else:
            print(f"No quake time found for event ID {event_id}")
            return None

    def label_windows(self, windows, quake_time_rel, sampling_rate):
        labels = []
        window_duration = self.window_size / sampling_rate
        step_duration = self.step_size / sampling_rate

        for i in range(len(windows)):
            window_start_time = i * step_duration
            window_end_time = window_start_time + window_duration

            if quake_time_rel is not None and (window_start_time <= quake_time_rel <= window_end_time):
                labels.append(1)  # Quake present in window
                print(f"Label 1 assigned to window {i}: Start={window_start_time:.2f}s, End={window_end_time:.2f}s")
            else:
                labels.append(0)  # No quake in window

        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram_data = self.data[idx]  # Shape: (frequencies, times)
        label = self.labels[idx]

        # Convert to torch tensor
        spectrogram_data = torch.tensor(spectrogram_data, dtype=torch.float32)
        print(f"Original spectrogram shape: {spectrogram_data.shape}")  # (65, 127)

        # Add channel dimension
        spectrogram_data = spectrogram_data.unsqueeze(0)  # Shape: (1, 65, 127)

        # No interpolation here

        return spectrogram_data, label


