import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemma_active" in data
    assert "sentry_active" in data

def test_compress_api_default():
    response = client.post("/api/v1/compress", json={"prompt": "Can you please help me write tests?"})
    assert response.status_code == 200
    data = response.json()
    assert "compressed_prompt" in data
    assert "tokens_saved" in data

def test_compress_api_entropy_mode():
    response = client.post("/api/v1/compress", json={
        "prompt": "Can you please write a complex algorithm? Thanks.",
        "mode": "gemma_entropy",
        "drop_rate": 0.40
    })
    assert response.status_code == 200
    data = response.json()
    assert data["mode_used"] == "gemma_entropy"
    assert "engine_used" in data
    assert "latency_ms" in data

def test_handoff_api():
    response = client.post("/api/v1/handoff", json={
        "project_name": "TestRun",
        "raw_transcript": "We decided to keep tests passing."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "TestRun"
    assert "formatted_markdown" in data
