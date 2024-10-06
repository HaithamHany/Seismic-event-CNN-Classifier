import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(SpectrogramClassifier, self).__init__()

        # Load pre-trained VGG16 and modify to accept 1-channel input (grayscale)
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Optionally freeze the feature layers
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False

        # Modify the classifier for binary classification
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    # Forward pass through the model
    def forward(self, x):
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg16.classifier(x)
        return x

    # Function to train the model
    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the appropriate device

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()

                # Perform interpolation after batching
                inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            # Validation phase
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.long()
                    inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
                    outputs = self.forward(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'Validation Accuracy: {val_accuracy:.2f}%')


    # Function to predict test data
    def predict(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        predictions, filenames = [], []

        with torch.no_grad():
            for inputs, filenames_batch in test_loader:
                inputs = inputs.to(device)
                inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                filenames.extend(filenames_batch)

        return predictions, filenames

