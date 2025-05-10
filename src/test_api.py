import pytest
from httpx import AsyncClient
from api.main import app  

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert "Facial Recognition API" in response.text

@pytest.mark.asyncio
async def test_load_model():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/load_model")
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            assert "Model loaded successfully" in response.json().get("message", "")

@pytest.mark.asyncio
async def test_load_json_data():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/load_json_data")
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            assert isinstance(response.json(), dict)

@pytest.mark.asyncio
async def test_load_data():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/load_data")
        assert response.status_code in (200, 404)

@pytest.mark.asyncio
async def test_predict():
    payload = {
        "embedding": [0.01] * 512
    }
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json=payload)
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.json()
            assert "student_id" in data
            assert "confidence" in data
