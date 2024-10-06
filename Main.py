import torch
from torch.utils.data import DataLoader, random_split
from SpectrogramClassifier import SpectrogramClassifier
from SpectrogramDataset import SpectrogramDataset
import csv

# Function to save predictions to CSV in the required format
def save_predictions_to_csv(predictions, filenames, output_filename="predictions.csv"):
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'predicted_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pred in enumerate(predictions):
            writer.writerow({
                'filename': filenames[i],
                'predicted_label': pred
            })

# Main function to handle training and testing
def main():
    # Folder paths for training, test, and catalog data
    train_folder = "SpectrogramTrainData"  # Folder for training/validation data
    test_folder = "SpectrogramTestData"  # Folder for test data
    catalog_path = "Catalog/apollo12_catalog_GradeA_final.csv"  # Catalog file path

    # Parameters for training
    num_epochs = 3
    batch_size = 8
    learning_rate = 0.001

    # Initialize training dataset and split into training/validation sets
    train_dataset_full = SpectrogramDataset(train_folder, catalog_path, labeled=True)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize test dataset (unlabeled)
    test_dataset = SpectrogramDataset(test_folder, catalog_path, labeled=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the CNN model for binary classification
    model = SpectrogramClassifier()

    # Train the model using the training and validation data
    model.train_model(train_loader, val_loader, num_epochs, learning_rate)

    # Evaluate the model on test data and get predictions
    predictions, filenames = model.predict(test_loader)

    # Save the predictions to a CSV file
    save_predictions_to_csv(predictions, filenames)

if __name__ == "__main__":
    main()
