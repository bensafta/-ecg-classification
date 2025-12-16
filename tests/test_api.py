import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)


class TestAPI:
    """Tests d'intégration pour l'API FastAPI"""

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_predict_valid_image(self):
        img = Image.new("L", (224, 224), color=128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("test.png", buf, "image/png")}
        response = client.post("/api/predict", files=files)

        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_invalid_file(self):
        files = {"file": ("test.txt", BytesIO(b"invalid"), "text/plain")}
        response = client.post("/api/predict", files=files)
        assert response.status_code >= 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
