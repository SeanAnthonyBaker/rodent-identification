import pytest
from src.inference_client import RolandInferenceClient

def test_extract_json_markdown():
    client = RolandInferenceClient()
    markdown_output = """```json
    {
      "rat_detected": true,
      "confidence": 0.95,
      "description": "Rat spotted along the bottom patio edge",
      "bounding_box": [500, 400, 600, 550]
    }
    ```"""
    parsed = client._extract_json(markdown_output)
    assert parsed is not None
    assert parsed["rat_detected"] is True
    assert parsed["confidence"] == 0.95
    assert len(parsed["bounding_box"]) == 4

def test_extract_json_raw():
    client = RolandInferenceClient()
    raw_output = '{"rat_detected": false, "confidence": 0.05, "description": "Empty yard"}'
    parsed = client._extract_json(raw_output)
    assert parsed is not None
    assert parsed["rat_detected"] is False

@pytest.mark.asyncio
async def test_fallback_simulation():
    client = RolandInferenceClient(endpoint_url="http://invalid-non-existent-host:9999")
    # Simulation should return gracefully without crashing
    res = await client.analyze_image(b"some_random_image_bytes")
    assert res is not None
    assert isinstance(res.is_rat_detected, bool)
    assert res.confidence >= 0.0
