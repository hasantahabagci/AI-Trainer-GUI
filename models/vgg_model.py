# Hasan Taha Bağcı
# 150210338
# VGG16 Model Definition

import torchvision.models as models
import torch.nn as nn

def get_vgg16(num_classes=10, pretrained=True): # Function to get VGG16 model
    """
    Loads a VGG16 model, optionally with pretrained weights,
    and modifies the final classifier layer for the specified number of classes.
    """
    if pretrained:
        print("Loading PRETRAINED VGG16 model.")
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) # Load pretrained VGG16
    else:
        print("Loading VGG16 model WITHOUT pretrained weights.")
        model = models.vgg16(weights=None) # Load VGG16 without pretrained weights

    # VGG16's classifier is a Sequential module. The last layer needs to be replaced.
    num_ftrs = model.classifier[6].in_features # Get in_features of the last Linear layer
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) # Replace the last layer
    return model

if __name__ == '__main__':
    import torch
    model = get_vgg16(num_classes=10, pretrained=False) # Example: 10 classes, not pretrained
    dummy_input = torch.randn(1, 3, 32, 32) # CIFAR-10 like input (VGG usually expects 224x224)
    output = model(dummy_input)
    print(f"VGG16 output shape: {output.shape}") # Should be (1, 10)

    model_pretrained = get_vgg16(num_classes=10, pretrained=True)
    output_pretrained = model_pretrained(dummy_input)
    print(f"Pretrained VGG16 output shape: {output_pretrained.shape}")