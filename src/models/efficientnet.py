# src/models/efficientnet.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_efficientnetb0(num_classes=2):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Adapter première couche pour 1 canal
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, 0, :, :] = old_conv.weight.mean(dim=1)
    model.features[0][0] = new_conv

    # Adapter la dernière couche pour 2 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
