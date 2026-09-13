import pytest
import numpy as np
import cv2
from src.app import apply_live_cctv_hud
from src.ring_client import GalaxyTabWindowsCamera


def test_apply_live_cctv_hud_live_vs_standby():
    # Create a 640x480 test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    raw_bytes = encoded.tobytes()

    # 1. Live stream frame
    hud_live = apply_live_cctv_hud(raw_bytes, "Galaxy Tab A11+", battery_pct=90, is_live=True)
    assert isinstance(hud_live, bytes)
    assert len(hud_live) > 0

    # 2. Standby / offline snapshot frame
    hud_standby = apply_live_cctv_hud(
        raw_bytes,
        "Galaxy Tab A11+",
        battery_pct=80,
        is_live=False,
        status_label="STANDBY (SNAPSHOT)",
        timestamp_str="2026-09-13 12:49:38"
    )
    assert isinstance(hud_standby, bytes)
    assert len(hud_standby) > 0
    # The output bytes will differ because of distinct text overlays and banner colors
    assert hud_live != hud_standby


def test_galaxy_tab_offline_health():
    tab = GalaxyTabWindowsCamera(name="Galaxy Tab A11+", camera_index=-1)
    health = tab.get_health()
    # When no active web stream has sent frames, it must not report is_streaming = True
    assert health["is_streaming"] is False
    assert health["uses_pictures"] is True
    assert health["is_phone"] is True
    assert health["is_local"] is False
