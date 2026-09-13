import cv2
import numpy as np
import pytest
from src.fast_detector import FastObjectDetector, ANIMAL_CLASSES

def create_synthetic_frame_with_shapes(num_objects=2):
    """Creates a synthetic frame with distinct high-contrast shapes to trigger candidate detection."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background texture
    cv2.randn(img, 40, 5)
    
    boxes = []
    for i in range(num_objects):
        cx = 150 + i * 220
        cy = 240
        # Draw high contrast oval/rectangle simulating animal body
        cv2.ellipse(img, (cx, cy), (45, 30), 0, 0, 360, (220, 220, 220), -1)
        # Head
        cv2.circle(img, (cx + 35, cy - 10), 16, (200, 200, 200), -1)
        boxes.append((cx - 50, cy - 35, cx + 55, cy + 35))
        
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes(), boxes

def test_fast_detector_multi_object_structure():
    detector = FastObjectDetector(min_box_area_px=100)
    frame_bytes, _ = create_synthetic_frame_with_shapes(num_objects=3)
    
    res = detector.detect_boxes(frame_bytes, target_object="horse")
    assert "tracked_objects" in res
    assert "candidate_boxes" in res
    assert "primary_box" in res
    assert "is_animal" in res
    
    # Verify backward compatibility
    if res["has_candidates"]:
        assert len(res["candidate_boxes"]) > 0
        assert res["primary_box"] == res["candidate_boxes"][0]
        assert len(res["tracked_objects"]) >= len(res["candidate_boxes"])
        for obj in res["tracked_objects"]:
            assert "id" in obj
            assert "box" in obj
            assert "class_name" in obj
            assert "label" in obj
            assert len(obj["box"]) == 4

def test_multi_candidate_build_response():
    detector = FastObjectDetector()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Two simulated horse detections
    items = [
        {"box": (50, 50, 200, 200), "id": 1, "class_id": 17, "class_name": "horse", "confidence": 0.94},
        {"box": (300, 100, 450, 250), "id": 2, "class_id": 17, "class_name": "horse", "confidence": 0.91}
    ]
    
    res = detector._build_candidate_response(dummy_img, items, padding_ratio=0.10, is_animal=True, target_object="horse")
    assert res["has_candidates"] is True
    assert len(res["tracked_objects"]) == 2
    
    # Check Horse #1 and Horse #2
    t1 = res["tracked_objects"][0]
    t2 = res["tracked_objects"][1]
    assert t1["id"] == 1
    assert t1["label"] == "Horse #1"
    assert t1["class_name"] == "horse"
    assert t1["confidence"] == 0.94
    
    assert t2["id"] == 2
    assert t2["label"] == "Horse #2"
    assert t2["class_name"] == "horse"
    assert t2["confidence"] == 0.91
    
    # Verify candidate boxes and crops
    assert len(res["candidate_boxes"]) == 2
    assert len(res["crops"]) == 2
    assert res["primary_box"] == t1["box"]

def test_contour_tracking_id_persistence():
    detector = FastObjectDetector(min_box_area_px=80)
    
    # Frame 1 with 2 objects
    frame1, _ = create_synthetic_frame_with_shapes(num_objects=2)
    res1 = detector.detect_boxes(frame1, target_object="horse")
    
    # Frame 2 with 2 objects
    frame2, _ = create_synthetic_frame_with_shapes(num_objects=2)
    res2 = detector.detect_boxes(frame2, target_object="horse")
    
    if res1["has_candidates"] and res2["has_candidates"]:
        ids1 = {o["id"] for o in res1["tracked_objects"]}
        ids2 = {o["id"] for o in res2["tracked_objects"]}
        # Verify tracks maintained overlap/continuity
        assert len(ids1.intersection(ids2)) > 0

@pytest.mark.asyncio
async def test_sampler_multi_object_payload():
    from unittest.mock import AsyncMock, MagicMock
    from src.sampler import SamplerEngine
    
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
        "delta_percent": 12.5,
        "focused_crop_bytes": b"fake_crop",
        "crop_bbox": [100, 100, 300, 300]
    }
    motion_mock.get_reference_baseline.return_value = None
    
    fast_mock = MagicMock()
    fast_mock.detect_boxes.return_value = {
        "has_candidates": True,
        "is_animal": True,
        "candidate_boxes": [[100, 100, 300, 300], [400, 400, 600, 600]],
        "tracked_objects": [
            {"id": 1, "label": "Horse #1", "class_name": "horse", "confidence": 0.92, "box": [100, 100, 300, 300]},
            {"id": 2, "label": "Horse #2", "class_name": "horse", "confidence": 0.89, "box": [400, 400, 600, 600]}
        ],
        "crops": [b"crop1", b"crop2"],
        "primary_crop_bytes": b"crop1",
        "primary_box": [100, 100, 300, 300]
    }
    
    storage_mock = MagicMock()
    
    sampler = SamplerEngine(
        ring_manager=ring_mock,
        inference_client=inf_mock,
        storage_manager=storage_mock
    )
    sampler.motion_pipeline = motion_mock
    sampler.fast_detector = fast_mock
    
    notified_events = []
    async def fake_notify(event_type, payload):
        notified_events.append((event_type, payload))
        
    sampler._notify_subscribers = AsyncMock(side_effect=fake_notify)
    
    # Run single iteration with force_ai=False
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", dummy_frame)
    ring_mock.async_fetch_snapshot = AsyncMock(return_value=(enc.tobytes(), None, False, True))
    
    await sampler.sample_once(force_ai=False)
    
    # Check that object_detected was emitted with tracked_objects
    obj_events = [ev for ev in notified_events if ev[0] == "object_detected"]
    assert len(obj_events) >= 1
    ev_type, payload = obj_events[0]
    assert "tracked_objects" in payload
    assert len(payload["tracked_objects"]) == 2
    assert payload["tracked_objects"][0]["label"] == "Horse #1"
    assert payload["tracked_objects"][1]["label"] == "Horse #2"
    assert "2 Horses Tracked" in payload["status_text"]
