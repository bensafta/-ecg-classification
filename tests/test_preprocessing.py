
import pytest
import numpy as np
from PIL import Image
import os
import sys

# Ajouter le dossier racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import preprocess_image


class TestPreprocessing:
    """Tests pour le module de preprocessing"""

    def test_image_resize(self):
        """Vérifie que les images sont bien redimensionnées"""
        img = Image.new("L", (512, 512))
        processed = preprocess_image(img)
        assert processed.shape == (224, 224, 1)

    def test_normalization(self):
        """Vérifie que la normalisation est bien entre 0 et 1"""
        img = Image.new("L", (224, 224))
        processed = preprocess_image(img)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_invalid_input(self):
        """Vérifie la gestion d'entrée invalide"""
        with pytest.raises(Exception):
            preprocess_image(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
