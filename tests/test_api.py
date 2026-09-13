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

def test_api_system_nodes():
    response = client.get("/api/system/nodes")
    assert response.status_code == 200
    data = response.json()
    assert "detection_node" in data
    assert data["detection_node"]["node"] == "roland1"
    assert "supabase_node" in data
    assert data["supabase_node"]["node"] == "roland1"
    assert "inference_node" in data
    assert data["inference_node"]["node"] == "roland3"

def test_api_zone_summary_and_picture_endpoints():
    # Test zone summary contains uses_pictures flag for mobile devices
    response = client.get("/api/cameras/zone_summary")
    assert response.status_code == 200
    data = response.json()
    assert "cameras" in data
    cams = data["cameras"]
    assert len(cams) > 0
    mobile_cams = [c for c in cams if "s21" in c["name"].lower() or "tab" in c["name"].lower()]
    for mc in mobile_cams:
        assert mc["uses_pictures"] is True or mc["is_streaming"] is True

    # Test picture download endpoint
    res_pic = client.get("/api/camera/Samsung Galaxy S21 Ultra/picture")
    original_bytes = res_pic.content if res_pic.status_code == 200 else None

    try:
        # Test picture upload endpoint with mock jpeg content
        mock_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 600 + b"\xff\xd9"
        res_upload = client.post(
            "/api/camera/Samsung Galaxy S21 Ultra/upload_picture",
            files={"file": ("test.jpg", mock_jpeg, "image/jpeg")}
        )
        assert res_upload.status_code == 200
        upload_data = res_upload.json()
        assert upload_data["success"] is True

        # Verify updated picture can now be fetched
        res_pic2 = client.get("/api/camera/Samsung Galaxy S21 Ultra/picture")
        assert res_pic2.status_code == 200
        assert len(res_pic2.content) == len(mock_jpeg)
    finally:
        # Restore original picture
        if original_bytes:
            client.post(
                "/api/camera/Samsung Galaxy S21 Ultra/upload_picture",
                files={"file": ("orig.jpg", original_bytes, "image/jpeg")}
            )


