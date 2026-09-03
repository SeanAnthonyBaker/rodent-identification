import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from src.inference_client import RolandInferenceClient
from src.storage import StorageManager
from src.app import app

client = TestClient(app)

def test_prompt_generation_for_all_targets():
    client_tree = RolandInferenceClient(target_object="tree")
    p_tree = client_tree._build_prompt("tree", has_reference=False)
    assert "tree" in p_tree.lower()
    assert "trunk" in p_tree.lower()

    client_bird = RolandInferenceClient(target_object="bird")
    p_bird = client_bird._build_prompt("bird", has_reference=True)
    assert "bird" in p_bird.lower()

    client_horse = RolandInferenceClient(target_object="horse")
    p_horse = client_horse._build_prompt("horse", has_reference=False)
    assert "horse" in p_horse.lower()
    assert "equine" in p_horse.lower()

    client_poo = RolandInferenceClient(target_object="horses_poo")
    p_poo = client_poo._build_prompt("horses_poo", has_reference=False)
    assert "horse" in p_poo.lower()
    assert "manure" in p_poo.lower() or "poo" in p_poo.lower() or "droppings" in p_poo.lower()

    client_all = RolandInferenceClient(target_object="all")
    p_all = client_all._build_prompt("all", has_reference=False)
    assert "tree" in p_all.lower()
    assert "bird" in p_all.lower()
    assert "rodent" in p_all.lower()
    assert "horse" in p_all.lower()
    assert "horses_poo" in p_all.lower()


def test_storage_multi_object_saving_and_filtering(tmp_path):
    storage = StorageManager(
        detections_dir=str(tmp_path / "detections"),
        db_path=str(tmp_path / "test.db")
    )
    img_data = b"dummy_img_bytes"

    storage.save_detection(img_data, 0.95, "Tree detected", 90, "Cam1", object_type="tree", label="Tree")
    storage.save_detection(img_data, 0.90, "Bird detected", 90, "Cam1", object_type="bird", label="Bird")
    storage.save_detection(img_data, 0.92, "Rat detected", 90, "Cam1", object_type="rodent", label="Rat")
    storage.save_detection(img_data, 0.96, "Horse detected", 90, "Cam1", object_type="horse", label="Horse")
    storage.save_detection(img_data, 0.94, "Horses poo detected", 90, "Cam1", object_type="horses_poo", label="Horses poo")

    all_dets = storage.list_detections(order="asc")
    assert len(all_dets) == 5

    tree_dets = storage.list_detections(object_type="tree")
    assert len(tree_dets) == 1
    assert tree_dets[0].label == "Tree"

    horse_dets = storage.list_detections(object_type="horse")
    assert len(horse_dets) == 1
    assert horse_dets[0].label == "Horse"

    poo_dets = storage.list_detections(object_type="horses_poo")
    assert len(poo_dets) == 1
    assert poo_dets[0].label == "Horses poo"


def test_api_target_object_endpoints():
    # Test setting target object
    for target in ["tree", "bird", "rat", "horse", "horses_poo", "all"]:
        res = client.post("/api/target_object", json={"target_object": target})
        assert res.status_code == 200
        assert res.json()["target_object"] == target

    # Test settings endpoint returns current target object
    res = client.get("/api/settings")
    assert res.status_code == 200
    assert "target_object" in res.json()


def test_api_simulate_all_targets():
    for target in ["tree", "bird", "rat", "horse", "horses_poo"]:
        res = client.post(f"/api/simulate_detection?animal={target}")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["detection"]["object_type"] in ["tree", "bird", "rodent", "rat", "horse", "horses_poo"]
