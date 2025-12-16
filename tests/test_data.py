# tests/test_data.py

import pytest
import torch
import numpy as np
from PIL import Image
import os
import sys
import tempfile
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import preprocess_image
from src.data.dataset import ECGImageDataset


class TestData:
    """Tests pour la gestion des données avec raw mélangé"""

    def test_preprocess_output(self):
        img = Image.new("L", (300, 300))
        output = preprocess_image(img)

        assert isinstance(output, np.ndarray)
        assert output.shape == (224, 224, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_dataset_with_csv_labels(self):
        """
        Dataset avec images mélangées + fichier labels.csv
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            os.makedirs(raw_dir)

            # Créer images
            for i in range(2):
                img = Image.new("L", (224, 224))
                img.save(os.path.join(raw_dir, f"img_{i}.png"))

            # Créer labels.csv
            labels_path = os.path.join(tmpdir, "labels.csv")
            with open(labels_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "label"])
                writer.writerow(["img_0.png", 0])
                writer.writerow(["img_1.png", 1])

            dataset = ECGImageDataset(
                image_dir=raw_dir,
                labels_file=labels_path
            )

            assert len(dataset) == 2

            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (1, 224, 224)
            assert label in [0, 1]

    def test_missing_labels_file(self):
        """CSV labels manquant"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            os.makedirs(raw_dir)

            img = Image.new("L", (224, 224))
            img.save(os.path.join(raw_dir, "img.png"))

            with pytest.raises(Exception):
                ECGImageDataset(image_dir=raw_dir, labels_file="missing.csv")
