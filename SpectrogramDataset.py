import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, catalog_path, labeled=True):
        self.data = []
        self.labels = []
        self.filenames = []
        self.time_rels = []
        self.labeled = labeled

        # Load the catalog for cross-referencing
        self.catalog = pd.read_csv(catalog_path)

        # Load the spectrogram data
        self.load_data(folder_path)

    def load_data(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)

                # Extract the first row (timeline) for cross-referencing
                timeline = df.iloc[0, :].values  # First row contains time

                # Calculate the magnitude of the complex numbers (from both real and imaginary parts)
                magnitude = df.iloc[1:, :].applymap(
                    lambda x: np.abs(complex(x.replace('i', 'j')))
                    # Convert string to complex number and take magnitude
                )
                magnitude = magnitude.values

                # Normalize the magnitudes
                magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
                self.data.append(magnitude)

                # Get relative time from filename
                self.filenames.append(filename)

                if self.labeled:
                    # Assign labels based on the catalog data (cross-reference using timeline)
                    quake_time_rel = self.get_quake_time_from_catalog(filename, timeline)
                    labels = self.label_time_windows(magnitude, timeline, quake_time_rel)
                    self.labels.append(labels)

    def get_quake_time_from_catalog(self, filename):
        # Match the filename with the catalog to find the quake time (time_rel)
        row = self.catalog[self.catalog['filename'] == filename]
        if not row.empty:
            return float(row['time_rel(sec)'].values[0])  # Get the time_rel of the quake
        return None

    def label_time_windows(self, data, quake_time_rel, window_size=1.0):
        """
        Label each time window: 1 if it overlaps with the quake time, otherwise 0.
        :param data: The spectrogram data for a file
        :param quake_time_rel: The time of the quake (in seconds) relative to the start of the file
        :param window_size: The duration of each time window in seconds
        :return: List of labels (1 or 0) for each time window
        """

        num_windows = data.shape[0]
        labels = []
        for i in range(num_windows):
            current_time = i * window_size
            if quake_time_rel is not None and abs(current_time - quake_time_rel) < window_size:
                labels.append(1)  # This window contains the quake
            else:
                labels.append(0)  # No quake in this window
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        metadata = {
            'filename': self.filenames[idx],
        }

        if self.labeled:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return spectrogram, label, metadata
        else:
            return spectrogram, metadata  # For test dataset