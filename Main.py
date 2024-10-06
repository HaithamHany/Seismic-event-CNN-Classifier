import torch
from torch.utils.data import DataLoader, random_split
from SpectrogramClassifier import SpectrogramClassifier
from SpectrogramDataset import SpectrogramDataset
import csv

from SpectrogramClassifier import SpectrogramClassifier
from SpectrogramDataset import SpectrogramDataset


def save_predictions_to_csv(predictions, filenames, times, output_filename="predictions.csv"):
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'time_rel', 'predicted_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pred in enumerate(predictions):
            writer.writerow({
                'filename': filenames[i],
                'time_rel': times[i],  # relative time in seconds
                'predicted_label': pred
            })


def main():
    # Folders for training and test data
    train_folder = "SpectrogramTrainData"  # Training/validation data folder
    test_folder = "SpectrogramTestData"  # Test data folder
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Initialize training dataset and data loaders
    train_dataset = SpectrogramDataset(train_folder, labeled=True)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load test dataset (without labels)
    test_dataset = SpectrogramDataset(test_folder, labeled=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the CNN model
    model = SpectrogramClassifier()

    # Train the model
    model.train_model(train_loader, val_loader, num_epochs, learning_rate)

    # Evaluate the model on the test data
    predictions, filenames, time_rels = model.predict(test_loader)

    # Save predictions to CSV in the required format
    save_predictions_to_csv(predictions, filenames, time_rels)


if __name__ == "__main__":
    main()
