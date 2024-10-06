import torch
from torch.utils.data import DataLoader, random_split
from SpectrogramClassifier import SpectrogramClassifier
from SpectrogramDataset import SpectrogramDataset
import csv
from collections import Counter

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

    try:
        # Initialize training dataset and split into training/validation sets
        train_dataset_full = SpectrogramDataset(train_folder, catalog_path, labeled=True)
        if len(train_dataset_full) == 0:
            print("No training data available after preprocessing.")
            return

        train_size = int(0.8 * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Check class distribution
        train_labels = [label for _, label in train_dataset]
        print('Training set class distribution:', Counter(train_labels))
        val_labels = [label for _, label in val_dataset]
        print('Validation set class distribution:', Counter(val_labels))

        # Initialize the CNN model for binary classification
        model = SpectrogramClassifier()

        # Train the model
        model.train_model(train_loader, val_loader, num_epochs, learning_rate)

        # Evaluate on test data
        test_dataset = SpectrogramDataset(test_folder, labeled=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        predictions, filenames = model.predict(test_loader)
        save_predictions_to_csv(predictions, filenames)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
