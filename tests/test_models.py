
import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.architecture import build_model


class TestModels:
    """Tests des architectures de modèles"""

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 1, 224, 224)

    def test_vgg16_forward(self, sample_input):
        model = build_model("vgg16", num_classes=2)
        output = model(sample_input)
        assert output.shape == (4, 2)

    def test_efficientnet_forward(self, sample_input):
        model = build_model("efficientnet", num_classes=2)
        output = model(sample_input)
        assert output.shape == (4, 2)

    def test_trainable_parameters(self):
        model = build_model("vgg16", num_classes=2)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
