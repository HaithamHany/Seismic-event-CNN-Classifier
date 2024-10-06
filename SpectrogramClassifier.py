import torch
import torch.nn as nn
import torchvision.models as models

class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(SpectrogramClassifier, self).__init__()

        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=pretrained)

        # Modify the first convolutional layer to accept 1-channel input (grayscale/spectrogram)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Optionally freeze the feature layers
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False

        # Modify the classifier to output the desired number of classes (binary classification)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Forward pass through the modified VGG16 model
        x = self.vgg16(x)
        return x

    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        # Define optimizer only for parameters that require gradient (i.e., unfrozen layers)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

            # Validation loop
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.forward(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'Validation Accuracy: {val_accuracy:.2f}%')

    def predict(self, test_loader):
        self.eval()
        predictions = []
        filenames = []
        time_rels = []

        with torch.no_grad():
            for inputs, metadata in test_loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

                # Extract filenames and time_rel from metadata
                filenames.extend(metadata['filename'])
                time_rels.extend(metadata['time_rel'])

        return predictions, filenames, time_rels