# src/models/architectures.py

import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes, pretrained=True):
    """
    Creates a ResNet50 model with a modified fully connected layer for a given number of classes.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        nn.Module: ResNet50 model.
    """

    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == "__main__":
    # Example usage
    num_classes = 2  # Replace with your number of classes
    model = get_resnet50(num_classes)
    print(model)

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)  # Example input shape
    output = model(dummy_input)
    print("Output shape:", output.shape)