import base64
import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from spark.ratwatch_api import app, build_watch_thumbnail, SYSTEM_PROMPT

client = TestClient(app)

def test_spark_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "model" in data
    assert "stats" in data

def test_thumbnail_generator():
    from io import BytesIO
    from PIL import Image
    im = Image.new("RGB", (640, 480), color=(255, 0, 0))
    buf = BytesIO()
    im.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    thumb_b64 = build_watch_thumbnail(img_bytes, max_width=240, quality=50)
    assert len(thumb_b64) > 0
    raw_thumb = base64.b64decode(thumb_b64)
    thumb_im = Image.open(BytesIO(raw_thumb))
    assert thumb_im.width <= 240

@pytest.mark.asyncio
async def test_sighting_intake_confirmed():
    from io import BytesIO
    from PIL import Image
    im = Image.new("RGB", (400, 300), color=(100, 100, 100))
    buf = BytesIO()
    im.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    meta = {
        "device_id": "s21-garden",
        "ts_ms": 1777750000000,
        "conf": 0.42,
        "box_frame": [100, 100, 250, 200],
        "frame_wh": [1280, 720],
        "track_id": 9
    }

    mock_vllm_resp = {
        "verdict": {
            "rat": True,
            "confidence": 0.88,
            "location": "under compost bin plinth",
            "reason": "brown rat body and tail visible",
            "watch_text": "Rat under compost plinth"
        },
        "vllm_ms": 1350
    }

    with patch("spark.ratwatch_api.call_vllm_gemma26b", new=AsyncMock(return_value=mock_vllm_resp)):
        with patch("spark.ratwatch_api.push_watch_alert", new=AsyncMock(return_value=True)):
            res = client.post(
                "/v1/sighting",
                data={"meta": json.dumps(meta)},
                files={"image": ("crop.jpg", img_bytes, "image/jpeg")}
            )
            assert res.status_code == 200
            data = res.json()
            assert data["confirmed"] is True
            assert data["watch_delivered"] is True
            assert data["verdict"]["rat"] is True
            assert data["verdict"]["location"] == "under compost bin plinth"

@pytest.mark.asyncio
async def test_sighting_intake_rejected():
    from io import BytesIO
    from PIL import Image
    im = Image.new("RGB", (400, 300), color=(50, 50, 50))
    buf = BytesIO()
    im.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    meta = {
        "device_id": "s21-garden",
        "conf": 0.30,
        "box_frame": [50, 50, 120, 120],
        "frame_wh": [1280, 720]
    }

    mock_vllm_resp = {
        "verdict": {
            "rat": False,
            "confidence": 0.10,
            "location": "garden paving",
            "reason": "garden hose curved on wet paving",
            "watch_text": "No rat"
        },
        "vllm_ms": 1100
    }

    with patch("spark.ratwatch_api.call_vllm_gemma26b", new=AsyncMock(return_value=mock_vllm_resp)):
        res = client.post(
            "/v1/sighting",
            data={"meta": json.dumps(meta)},
            files={"image": ("crop.jpg", img_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["confirmed"] is False
        assert data["verdict"]["rat"] is False
