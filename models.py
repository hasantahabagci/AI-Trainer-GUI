# Hasan Taha Bağcı 150210338
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CustomCNN(nn.Module): # Define a custom CNN model
    def __init__(self, num_classes=10): # Constructor
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # First convolutional layer
        self.bn1 = nn.BatchNorm2d(32)                               # Batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(64)                               # Batch normalization
        self.pool1 = nn.MaxPool2d(2, 2)                             # Max pooling
        self.dropout1 = nn.Dropout(0.25)                            # Dropout

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Third convolutional layer
        self.bn3 = nn.BatchNorm2d(128)                              # Batch normalization
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # Fourth convolutional layer
        self.bn4 = nn.BatchNorm2d(128)                              # Batch normalization
        self.pool2 = nn.MaxPool2d(2, 2)                             # Max pooling
        self.dropout2 = nn.Dropout(0.25)                            # Dropout

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Fifth convolutional layer
        self.bn5 = nn.BatchNorm2d(256)                              # Batch normalization
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # Sixth convolutional layer
        self.bn6 = nn.BatchNorm2d(256)                              # Batch normalization
        self.pool3 = nn.MaxPool2d(2, 2)                             # Max pooling
        self.dropout3 = nn.Dropout(0.25)                            # Dropout
        
        # Calculate the flattened size dynamically
        # Input to conv layers is 32x32. After 3 pooling layers (32 -> 16 -> 8 -> 4)
        # The size becomes 4x4.
        self.flattened_size = 256 * 4 * 4                           # Size after flattening

        self.fc1 = nn.Linear(self.flattened_size, 1024)             # First fully connected layer
        self.dropout4 = nn.Dropout(0.5)                             # Dropout
        self.fc2 = nn.Linear(1024, num_classes)                     # Output layer

    def forward(self, x): # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))                         # Activation
        x = F.relu(self.bn2(self.conv2(x)))                         # Activation
        x = self.pool1(x)                                           # Pooling
        x = self.dropout1(x)                                        # Dropout

        x = F.relu(self.bn3(self.conv3(x)))                         # Activation
        x = F.relu(self.bn4(self.conv4(x)))                         # Activation
        x = self.pool2(x)                                           # Pooling
        x = self.dropout2(x)                                        # Dropout

        x = F.relu(self.bn5(self.conv5(x)))                         # Activation
        x = F.relu(self.bn6(self.conv6(x)))                         # Activation
        x = self.pool3(x)                                           # Pooling
        x = self.dropout3(x)                                        # Dropout

        x = x.view(-1, self.flattened_size)                         # Flatten the tensor
        x = F.relu(self.fc1(x))                                     # Activation
        x = self.dropout4(x)                                        # Dropout
        x = self.fc2(x)                                             # Output
        return x

def get_resnet50(num_classes=10, pretrained=True): # Function to get ResNet50 model
    """Loads a pre-trained ResNet50 model and modifies the final layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None) # Load ResNet50
    # Modify the final fully connected layer for CIFAR-10
    num_ftrs = model.fc.in_features                             # Get number of input features
    model.fc = nn.Linear(num_ftrs, num_classes)                 # Replace final layer
    return model

def get_vgg16(num_classes=10, pretrained=True): # Function to get VGG16 model
    """Loads a pre-trained VGG16 model and modifies the final classifier layer."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None) # Load VGG16
    # Modify the classifier for CIFAR-10
    num_ftrs = model.classifier[6].in_features                  # Get number of input features for the last layer
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)      # Replace final layer
    return model

if __name__ == '__main__':
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # Check for MPS (Mac M1/M2)
    print(f"Using device: {device}")

    # Test CustomCNN
    custom_model = CustomCNN(num_classes=10).to(device)
    dummy_input_custom = torch.randn(1, 3, 32, 32).to(device) # Batch size 1, 3 channels, 32x32 image
    output_custom = custom_model(dummy_input_custom)
    print(f"CustomCNN output shape: {output_custom.shape}") # Should be [1, 10]

    # Test ResNet50
    resnet_model = get_resnet50(num_classes=10, pretrained=False).to(device) # Set pretrained=False for quick test without download
    dummy_input_resnet = torch.randn(1, 3, 32, 32).to(device) # ResNet expects at least 32x32, but typically 224x224
                                                            # For CIFAR-10, we use it as is, or add an upsampling layer if needed.
                                                            # torchvision models adapt to smaller inputs like 32x32.
    output_resnet = resnet_model(dummy_input_resnet)
    print(f"ResNet50 output shape: {output_resnet.shape}") # Should be [1, 10]

    # Test VGG16
    vgg_model = get_vgg16(num_classes=10, pretrained=False).to(device) # Set pretrained=False for quick test
    dummy_input_vgg = torch.randn(1, 3, 32, 32).to(device)
    output_vgg = vgg_model(dummy_input_vgg)
    print(f"VGG16 output shape: {output_vgg.shape}") # Should be [1, 10]
