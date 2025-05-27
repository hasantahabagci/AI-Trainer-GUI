# Hasan Taha Bağcı
# 150210338
# Custom CNN Model Definition

import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module): # Custom CNN class
    def __init__(self, num_classes=10): # Constructor
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(32)                             # Batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(64)                             # Batch normalization
        self.pool1 = nn.MaxPool2d(2, 2)                           # Max pooling
        self.dropout1 = nn.Dropout(0.25)                          # Dropout

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Third convolutional layer
        self.bn3 = nn.BatchNorm2d(128)                            # Batch normalization
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)# Fourth convolutional layer
        self.bn4 = nn.BatchNorm2d(128)                            # Batch normalization
        self.pool2 = nn.MaxPool2d(2, 2)                           # Max pooling
        self.dropout2 = nn.Dropout(0.25)                          # Dropout

        # Calculate the flattened size dynamically (assuming 32x32 input)
        # After conv1, bn1: 32x32
        # After conv2, bn2: 32x32
        # After pool1: 16x16
        # After conv3, bn3: 16x16
        # After conv4, bn4: 16x16
        # After pool2: 8x8
        # Flattened size = 128 * 8 * 8
        self.fc1_input_features = 128 * 8 * 8
        self.fc1 = nn.Linear(self.fc1_input_features, 512)       # First fully connected layer
        self.bn_fc1 = nn.BatchNorm1d(512)                         # Batch normalization
        self.dropout3 = nn.Dropout(0.5)                           # Dropout
        self.fc2 = nn.Linear(512, num_classes)                    # Output layer

    def forward(self, x): # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, self.fc1_input_features) # Flatten the tensor
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def get_custom_cnn(num_classes=10, pretrained=False): # Function to get Custom CNN
    if pretrained:
        print("Warning: Pretrained weights are not available for CustomCNN. Returning an untrained model.")
    return CustomCNN(num_classes=num_classes)

if __name__ == '__main__':
    import torch
    model = get_custom_cnn()
    dummy_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
    output = model(dummy_input)
    print(f"CustomCNN output shape: {output.shape}") # Should be (1, 10)