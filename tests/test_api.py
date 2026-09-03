import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "battery_percentage" in data

def test_api_battery():
    response = client.get("/api/battery")
    assert response.status_code == 200
    data = response.json()
    assert "battery_percentage" in data
    assert "recharge_needed" in data

def test_api_detections_endpoint():
    response = client.get("/api/detections?order=asc")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)

def test_api_sample_now():
    response = client.post("/api/sample_now")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "active_camera" in data
