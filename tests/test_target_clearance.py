import cv2
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from src.sampler import SamplerEngine
from src.inference_client import DetectionResult

@pytest.mark.asyncio
async def test_sampler_object_cleared_on_animal_exit():
    """Verify that when an animal exits the scene, SamplerEngine dispatches 'object_cleared'."""
    ring_mock = MagicMock()
    ring_mock.camera_name = "Garden"
    ring_mock.get_health_status.return_value = {"battery_percentage": 90}
    ring_mock._is_mock = True
    
    inf_mock = MagicMock()
    inf_mock.detection_polygon = None
    inf_mock.target_object = "horse"
    
    motion_mock = MagicMock()
    motion_mock.compute_zone_delta.return_value = {
        "has_material_delta": True,
        "delta_percent": 10.0,
        "focused_crop_bytes": b"fake_crop",
        "crop_bbox": [100, 100, 300, 300]
    }
    motion_mock.get_reference_baseline.return_value = None
    
    fast_mock = MagicMock()
    storage_mock = MagicMock()
    
    sampler = SamplerEngine(
        ring_manager=ring_mock,
        inference_client=inf_mock,
        storage_manager=storage_mock
    )
    sampler.motion_pipeline = motion_mock
    sampler.fast_detector = fast_mock
    sampler._run_async_ai_inference = AsyncMock()
    
    events = []
    async def fake_notify(event_type, payload):
        events.append((event_type, payload))
        
    sampler._notify_subscribers = AsyncMock(side_effect=fake_notify)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy_frame)
    ring_mock.async_fetch_snapshot = AsyncMock(return_value=(enc.tobytes(), None, False, True))
    
    # Tick 1: Animal detected
    fast_mock.detect_boxes.return_value = {
        "has_candidates": True,
        "is_animal": True,
        "candidate_boxes": [[100, 100, 300, 300]],
        "tracked_objects": [
            {"id": 1, "label": "Horse #1", "class_name": "horse", "confidence": 0.92, "box": [100, 100, 300, 300]}
        ],
        "crops": [b"crop1"],
        "primary_crop_bytes": b"crop1",
        "primary_box": [100, 100, 300, 300]
    }
    
    await sampler.sample_once(force_ai=False)
    assert sampler._tracked_animal_active is True
    assert sampler._empty_animal_ticks == 0
    assert any(ev[0] == "object_detected" for ev in events)
    
    # Tick 2: Animal moves away (no candidate)
    fast_mock.detect_boxes.return_value = {
        "has_candidates": False,
        "is_animal": False,
        "candidate_boxes": [],
        "tracked_objects": [],
        "crops": [],
        "primary_crop_bytes": None,
        "primary_box": None
    }
    events.clear()
    await sampler.sample_once(force_ai=False)
    assert sampler._empty_animal_ticks == 1
    assert sampler._tracked_animal_active is True
    assert not any(ev[0] == "object_cleared" for ev in events)
    
    # Tick 3: Second consecutive empty tick -> triggers object_cleared and resets
    events.clear()
    await sampler.sample_once(force_ai=False)
    assert sampler._empty_animal_ticks == 0  # reset on cleared
    assert sampler._tracked_animal_active is False
    
    cleared_events = [ev for ev in events if ev[0] == "object_cleared"]
    assert len(cleared_events) == 1
    assert cleared_events[0][1]["cleared"] is True
    assert cleared_events[0][1]["object_boundary"] is None


@pytest.mark.asyncio
async def test_sampler_ai_negative_clears_boosted_and_bounding_box():
    """Verify that negative AI inference dispatches cleared status and resets tracking state."""
    ring_mock = MagicMock()
    ring_mock.camera_name = "Garden"
    ring_mock.get_health_status.return_value = {"battery_percentage": 90}
    inf_mock = MagicMock()
    inf_mock.target_object = "horse"
    inf_mock.confidence_threshold = 0.70
    inf_mock.detection_polygon = None
    storage_mock = MagicMock()
    
    sampler = SamplerEngine(
        ring_manager=ring_mock,
        inference_client=inf_mock,
        storage_manager=storage_mock
    )
    sampler._is_rat_active = True
    sampler._tracked_animal_active = True
    
    events = []
    async def fake_notify(event_type, payload):
        events.append((event_type, payload))
    sampler._notify_subscribers = AsyncMock(side_effect=fake_notify)
    
    # Simulate negative AI inference result
    negative_result = DetectionResult(
        detected=False,
        is_detected=False,
        confidence=0.12,
        bounding_box=None,
        label="None",
        description="No horse visible in scene."
    )
    inf_mock.analyze_image = AsyncMock(return_value=negative_result)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy_frame)
    frame_bytes = enc.tobytes()
    
    await sampler._run_async_ai_inference(
        crop_bytes=b"crop",
        ref_bytes=None,
        is_subcrop=False,
        crop_bbox=None,
        zone_info={"original_shape": (640, 480)},
        object_boundary=[100, 100, 300, 300],
        snapshot_bytes=frame_bytes,
        start_ts=datetime.now(),
        timestamp_str="2026-09-13 12:00:00",
        health={},
        delta_pct=5.0
    )
    
    assert sampler._is_rat_active is False
    assert sampler._tracked_animal_active is False
    
    sample_completed_events = [ev for ev in events if ev[0] == "sample_completed"]
    assert len(sample_completed_events) == 1
    payload = sample_completed_events[0][1]
    assert payload["cleared"] is True
    assert payload["is_boosted"] is False
    assert payload["bounding_box"] is None
    assert payload["label"] == "None"
    assert payload["object_type"] == "none"


def test_screen_cam_clearance_endpoints():
    """Verify that screen cam ingestion properly clears state when animal is absent."""
    import base64
    from fastapi.testclient import TestClient
    from src.app import app
    import src.app as app_module
    
    test_client = TestClient(app)
    
    # Set screen cam animal tracked active
    app_module._stream_animal_tracked = True
    app_module._stream_empty_ticks = 0
    
    # Send empty frames (black image)
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy)
    b64_str = base64.b64encode(enc.tobytes()).decode("utf-8")
    payload = {"image_base64": f"data:image/jpeg;base64,{b64_str}", "device_name": "Screen Cam (Live)"}
    
    # 1st empty tick
    resp1 = test_client.post("/api/screen_cam/analyze", json=payload)
    assert resp1.status_code == 200
    assert app_module._stream_empty_ticks >= 1
    
    # 2nd empty tick
    resp2 = test_client.post("/api/screen_cam/analyze", json=payload)
    assert resp2.status_code == 200
    
    # 3rd empty tick -> triggers clearance
    resp3 = test_client.post("/api/screen_cam/analyze", json=payload)
    assert resp3.status_code == 200
    assert app_module._stream_animal_tracked is False
    assert app_module._stream_empty_ticks == 0
