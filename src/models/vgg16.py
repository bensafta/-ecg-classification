# src/models/vgg16.py
import torch
import torch.nn as nn
from torchvision import models

def build_vgg16(num_classes=2):
    # Charger VGG16 pré-entraîné
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    # Adapter première couche pour 1 canal si nécessaire
    old_conv = model.features[0]
    if old_conv.in_channels != 3:
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, 0, :, :] = old_conv.weight.mean(dim=1)
        model.features[0] = new_conv

    # Adapter la dernière couche pour 2 classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
