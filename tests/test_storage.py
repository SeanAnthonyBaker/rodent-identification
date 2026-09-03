import os
import shutil
import pytest
from datetime import datetime, timedelta
from src.storage import StorageManager

@pytest.fixture
def temp_storage(tmp_path):
    det_dir = tmp_path / "detections"
    db_file = tmp_path / "test_detections.db"
    return StorageManager(detections_dir=str(det_dir), db_path=str(db_file))

def test_save_and_list_detections_chronological(temp_storage):
    img_data = b"fake_jpeg_data_bytes"
    t1 = datetime(2026, 8, 25, 10, 0, 0)
    t2 = datetime(2026, 8, 25, 10, 5, 0)

    # Save two detections at different times
    rec1 = temp_storage.save_detection(
        image_bytes=img_data,
        confidence=0.92,
        description="First rat detected near feeder",
        battery_percentage=85,
        device_name="Ring Cam Test",
        dt=t1
    )

    rec2 = temp_storage.save_detection(
        image_bytes=img_data,
        confidence=0.88,
        description="Second rat detected near fence",
        battery_percentage=84,
        device_name="Ring Cam Test",
        dt=t2
    )

    # Verify chronological ascending order (oldest first)
    asc_list = temp_storage.list_detections(order="asc")
    assert len(asc_list) == 2
    assert asc_list[0].id == rec1.id
    assert asc_list[1].id == rec2.id
    assert asc_list[0].confidence == 0.92
    assert asc_list[0].battery_percentage == 85

    # Verify descending order (newest first)
    desc_list = temp_storage.list_detections(order="desc")
    assert desc_list[0].id == rec2.id
    assert desc_list[1].id == rec1.id

def test_delete_detection(temp_storage):
    img_data = b"fake_jpeg"
    rec = temp_storage.save_detection(
        image_bytes=img_data,
        confidence=0.95,
        description="Rat spotted",
        battery_percentage=75,
        device_name="Ring Cam"
    )

    assert temp_storage.get_detection(rec.id) is not None
    assert temp_storage.delete_detection(rec.id) is True
    assert temp_storage.get_detection(rec.id) is None
