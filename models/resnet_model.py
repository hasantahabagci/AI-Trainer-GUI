# Hasan Taha Bağcı
# 150210338
# ResNet50 Model Definition

import torchvision.models as models
import torch.nn as nn

def get_resnet50(num_classes=10, pretrained=True): # Function to get ResNet50 model
    """
    Loads a ResNet50 model, optionally with pretrained weights,
    and modifies the final fully connected layer for the specified number of classes.
    """
    if pretrained:
        print("Loading PRETRAINED ResNet50 model.")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Load pretrained ResNet50
    else:
        print("Loading ResNet50 model WITHOUT pretrained weights.")
        model = models.resnet50(weights=None) # Load ResNet50 without pretrained weights

    num_ftrs = model.fc.in_features # Get the number of input features of the original fc layer
    model.fc = nn.Linear(num_ftrs, num_classes) # Replace the fc layer with a new one for num_classes
    return model

if __name__ == '__main__':
    import torch
    model = get_resnet50(num_classes=10, pretrained=False) # Example: 10 classes, not pretrained
    dummy_input = torch.randn(1, 3, 32, 32) # CIFAR-10 like input (ResNet usually expects 224x224, but works with smaller)
    output = model(dummy_input)
    print(f"ResNet50 output shape: {output.shape}") # Should be (1, 10)

    model_pretrained = get_resnet50(num_classes=10, pretrained=True)
    output_pretrained = model_pretrained(dummy_input)
    print(f"Pretrained ResNet50 output shape: {output_pretrained.shape}")