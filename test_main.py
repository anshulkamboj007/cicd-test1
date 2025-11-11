from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json=[5.1, 3.5, 1.4, 0.2])
    assert response.status_code == 200
    assert "prediction" in response.json()