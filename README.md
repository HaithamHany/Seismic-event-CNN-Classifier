# Seismic Event Detection Using CNN and Spectrograms

This project focuses on detecting seismic events (quakes) by converting raw seismic time-series data into **spectrograms** using the **Short-Time Fourier Transform (STFT)**. The spectrograms are then fed into a **pre-trained VGG16 Convolutional Neural Network (CNN)** for binary classification, identifying time windows with seismic activity.

## Key Features

  - Converts seismic time-series data into spectrograms to capture frequency changes over time.
  
  - Uses a pre-trained VGG16 model to classify spectrograms as either "quake" or "no quake."
  
  - Labels time windows based on seismic event catalogs and applies buffer zones to capture event contexts.
  
  - Stores the model's predictions in a catalog format for easy event tracking and analysis.
