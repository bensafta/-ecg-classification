# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, arch='baseline', input_shape=(1, 224, 224)):
        super(SimpleCNN, self).__init__()
        self.arch = arch

        # Convolutions
        if arch == 'baseline':
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        elif arch == 'wide':
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        elif arch == 'deep':
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        # Calcul dynamique de la taille de fc1
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self._forward_features(dummy)
            n_size = out.view(1, -1).size(1)
        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.arch == 'deep':
            x = F.relu(self.conv3(x))
            x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fonction utilitaire pour créer le modèle
def create_model(num_classes=4, arch='baseline', input_shape=(1, 224, 224)):
    return SimpleCNN(num_classes=num_classes, arch=arch, input_shape=input_shape)
