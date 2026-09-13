import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.storage import StorageManager, RolandSupabaseClient, DetectionRecord

def test_supabase_client_initialization():
    client = RolandSupabaseClient(base_url="http://roland1:54321", api_key="test_key")
    assert client.base_url == "http://roland1:54321"
    assert client.rest_url == "http://roland1:54321/rest/v1"
    assert client.storage_url == "http://roland1:54321/storage/v1"
    assert client.headers["apikey"] == "test_key"
    assert "Bearer test_key" in client.headers["Authorization"]

def test_storage_manager_offline_fallback(tmp_path):
    """When Supabase on roland1 is offline, StorageManager should fall back to SQLite gracefully."""
    det_dir = tmp_path / "detections"
    db_file = tmp_path / "test.db"

    # roland1 is non-resolvable in test environment, so check_health will fail
    storage = StorageManager(
        detections_dir=str(det_dir),
        db_path=str(db_file),
        backend="auto",
        supabase_url="http://roland1:54321",
        fallback_to_sqlite=True
    )

    info = storage.backend_info
    assert info["supabase_node"] == "roland1"
    assert info["supabase_url"] == "http://roland1:54321"
    assert info["supabase_active"] is False
    assert info["active_backend"] == "sqlite"

    # Ensure save and list work in fallback mode
    img_data = b"test_frame_bytes"
    rec = storage.save_detection(
        image_bytes=img_data,
        confidence=0.91,
        description="Rat spotted near shed",
        battery_percentage=90,
        device_name="Roland 1 Camera"
    )
    assert rec.id > 0
    assert rec.description == "Rat spotted near shed"

    dets = storage.list_detections()
    assert len(dets) == 1
    assert dets[0].id == rec.id

def test_storage_manager_mocked_supabase_mode(tmp_path):
    """When Supabase on roland1 is reachable, StorageManager should use Supabase."""
    det_dir = tmp_path / "detections"
    db_file = tmp_path / "test.db"

    # Mock health check and Supabase operations
    with patch.object(RolandSupabaseClient, "check_health", return_value=True), \
         patch.object(RolandSupabaseClient, "insert") as mock_insert, \
         patch.object(RolandSupabaseClient, "select") as mock_select, \
         patch.object(RolandSupabaseClient, "get_by_id") as mock_get_by_id, \
         patch.object(RolandSupabaseClient, "delete_by_id", return_value=True), \
         patch.object(RolandSupabaseClient, "get_stats") as mock_stats:

        mock_insert.return_value = {
            "id": 101,
            "event_id": "evt_101",
            "frame_index": 1,
            "timestamp": "2026-09-12T12:00:00",
            "formatted_time": "Sep 12, 2026 - 12:00:00 PM",
            "filename": "rat_detection_20260912_120000.jpg",
            "confidence": 0.95,
            "description": "Rat spotted on Roland 1",
            "object_type": "rat",
            "label": "Rat",
            "battery_percentage": 100,
            "device_name": "Roland 1 Local Cam",
            "bounding_box": "[100, 200, 300, 400]",
            "created_at": 1789214400.0
        }

        mock_select.return_value = [mock_insert.return_value]
        mock_get_by_id.return_value = mock_insert.return_value
        mock_stats.return_value = {
            "total_detections": 1,
            "latest_detection_timestamp": "2026-09-12T12:00:00"
        }

        storage = StorageManager(
            detections_dir=str(det_dir),
            db_path=str(db_file),
            backend="supabase",
            supabase_url="http://roland1:54321"
        )

        assert storage.is_supabase_active is True
        assert storage.backend_info["active_backend"] == "supabase"

        # Save detection
        rec = storage.save_detection(
            image_bytes=b"sample_image",
            confidence=0.95,
            description="Rat spotted on Roland 1",
            battery_percentage=100,
            device_name="Roland 1 Local Cam"
        )
        assert rec.id == 101
        assert mock_insert.called

        # List detections
        dets = storage.list_detections(order="desc")
        assert len(dets) == 1
        assert dets[0].id == 101
        assert mock_select.called

        # Get detection
        single = storage.get_detection(101)
        assert single is not None
        assert single.id == 101
        assert mock_get_by_id.called

        # Delete detection
        deleted = storage.delete_detection(101)
        assert deleted is True

        # Stats
        stats = storage.get_stats()
        assert stats["total_detections"] == 1
