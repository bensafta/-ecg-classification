# src/models/resnet.py
import torch
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Adapter première couche pour 1 canal si nécessaire
    if model.conv1.in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, 0, :, :] = old_conv.weight.mean(dim=1)
        model.conv1 = new_conv

    # Adapter la dernière couche fully connected
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
