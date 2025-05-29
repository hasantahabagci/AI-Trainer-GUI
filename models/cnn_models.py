# Hasan Taha Bağcı 150210338
# models/cnn_models.py

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CIFAR10_CLASSES = 10 # Number of classes in CIFAR-10

class CustomCNN(nn.Module): # A simple custom CNN for CIFAR-10
    def __init__(self, num_classes=NUM_CIFAR10_CLASSES):
        super(CustomCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # 32x32x3 -> 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 32x32x32 -> 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32x64 -> 16x16x64
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 16x16x64 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # 16x16x128 -> 16x16x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16x128 -> 8x8x128
            nn.Dropout(0.05) # Added dropout
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # 8x8x128 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # 8x8x256 -> 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8x256 -> 4x4x256
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1), # Added dropout
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1), # Added dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layer
        x = self.fc_layer(x)
        return x

def get_resnet50(num_classes=NUM_CIFAR10_CLASSES, pretrained=True): # Get ResNet50 model
    """
    Loads a ResNet50 model, optionally with pretrained weights,
    and modifies the final fully connected layer for the specified number of classes.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
        
    num_ftrs = model.fc.in_features # Get the number of input features of the original fc layer
    model.fc = nn.Linear(num_ftrs, num_classes) # Replace the fc layer
    return model

def get_vgg16(num_classes=NUM_CIFAR10_CLASSES, pretrained=True): # Get VGG16 model
    """
    Loads a VGG16 model, optionally with pretrained weights,
    and modifies the final classifier layer for the specified number of classes.
    """
    if pretrained:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        model = models.vgg16(weights=None)

    num_ftrs = model.classifier[6].in_features # VGG's classifier is a Sequential module, the last Linear layer is at index 6
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) # Replace the last layer
    return model

# Dictionary to easily access models by name
AVAILABLE_MODELS = {
    "CustomCNN": CustomCNN,
    "ResNet50": get_resnet50,
    "VGG16": get_vgg16,
}

def get_model(model_name, num_classes=NUM_CIFAR10_CLASSES, pretrained=True): # Factory function to get a model
    """
    Returns a model instance based on its name.
    For ResNet50 and VGG16, 'pretrained' argument is used.
    CustomCNN does not use 'pretrained' argument directly in its constructor call here.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(AVAILABLE_MODELS.keys())}")

    if model_name == "CustomCNN":
        return AVAILABLE_MODELS[model_name](num_classes=num_classes)
    else: # For ResNet50, VGG16
        return AVAILABLE_MODELS[model_name](num_classes=num_classes, pretrained=pretrained)

if __name__ == '__main__':
    print("Testing CustomCNN...")
    custom_model = get_model("CustomCNN")
    dummy_input_custom = torch.randn(2, 3, 32, 32) # Batch size 2, 3 channels, 32x32 image
    output_custom = custom_model(dummy_input_custom)
    print("CustomCNN output shape:", output_custom.shape) # Expected: torch.Size([2, 10])
    assert output_custom.shape == (2, NUM_CIFAR10_CLASSES)

    # Test ResNet50
    print("\nTesting ResNet50 (pretrained)...")
    resnet_model_pt = get_model("ResNet50", pretrained=True)
    dummy_input_resnet = torch.randn(2, 3, 32, 32) 
    
    output_resnet_pt = resnet_model_pt(dummy_input_resnet)
    print("ResNet50 (pretrained) output shape:", output_resnet_pt.shape)
    assert output_resnet_pt.shape == (2, NUM_CIFAR10_CLASSES)

    print("\nTesting ResNet50 (not pretrained)...")
    resnet_model_scratch = get_model("ResNet50", pretrained=False)
    output_resnet_scratch = resnet_model_scratch(dummy_input_resnet)
    print("ResNet50 (not pretrained) output shape:", output_resnet_scratch.shape)
    assert output_resnet_scratch.shape == (2, NUM_CIFAR10_CLASSES)

    # Test VGG16
    print("\nTesting VGG16 (pretrained)...")
    vgg_model_pt = get_model("VGG16", pretrained=True)
    dummy_input_vgg = torch.randn(2, 3, 32, 32) # Similar note as ResNet for input size
    output_vgg_pt = vgg_model_pt(dummy_input_vgg)
    print("VGG16 (pretrained) output shape:", output_vgg_pt.shape)
    assert output_vgg_pt.shape == (2, NUM_CIFAR10_CLASSES)

    print("\nTesting VGG16 (not pretrained)...")
    vgg_model_scratch = get_model("VGG16", pretrained=False)
    output_vgg_scratch = vgg_model_scratch(dummy_input_vgg)
    print("VGG16 (not pretrained) output shape:", output_vgg_scratch.shape)
    assert output_vgg_scratch.shape == (2, NUM_CIFAR10_CLASSES)
    
    print("\nAll model tests passed!")
