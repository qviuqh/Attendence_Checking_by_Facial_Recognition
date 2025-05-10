from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Facial Recognition API" in response.text

def test_load_model():
    response = client.post("/load_model")
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert "Model loaded successfully" in response.json().get("message", "")

def test_load_json_data():
    response = client.get("/load_json_data")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert isinstance(response.json(), dict)

def test_load_data():
    response = client.get("/load_data")
    assert response.status_code in (200, 404)

def test_predict():
    payload = {
        "embedding": [0.01] * 512
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        data = response.json()
        assert "student_id" in data
        assert "confidence" in data
