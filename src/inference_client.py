import asyncio
import base64
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("inference_client")

class DetectionResult(BaseModel):
    detected: bool = Field(default=False, description="Whether a valid rodent or pheasant is detected")
    is_detected: bool = Field(default=False, description="Alias for detected")
    is_rat_detected: bool = Field(default=False, description="Backwards compatibility alias for is_detected")
    subject_type: str = Field(default="none", description="'none', 'rodent', 'pheasant', 'other_animal', or 'false_positive_clutter'")
    object_type: str = Field(default="none", description="Alias for subject_type")
    label: str = Field(default="None", description="Display label ('Pheasant', 'Rat', 'Clutter', etc.)")
    confidence_score: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
    confidence: float = Field(default=0.0, description="Alias for confidence_score")
    description: str = Field(default="", description="Factual explanation")
    bounding_box: Optional[list] = Field(default=None, description="Bounding perimeter coordinates [ymin, xmin, ymax, xmax]")
    inference_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    model_name: str = Field(default="gemma-4-e12b")
    raw_response: Optional[str] = Field(default=None)


class RolandInferenceClient:
    """Client for querying Gemma 4 E12b with structured sampling (temp=0.0), closed JSON schema, and dual-image contrast reference."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434",
        endpoint_type: str = "ollama",
        model_name: str = "tulkah_gemma4_12b:latest",
        confidence_threshold: float = 0.45,
        timeout_seconds: float = 45.0,
        detection_polygon: Optional[List[List[float]]] = None,
        camera_polygons: Optional[Dict[str, list]] = None,
        target_object: str = "rat",
        temperature: float = 0.0,
        top_p: float = 0.1,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-3.7-flash"
    ):
        import os
        self.endpoint_url = endpoint_url.rstrip("/")
        self.endpoint_type = endpoint_type.lower()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.detection_polygon = detection_polygon
        self.camera_polygons: Dict[str, list] = camera_polygons or {}
        self.target_object = (target_object or "rat").lower()
        self.temperature = temperature
        self.top_p = top_p
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model = gemini_model
        self._lock = asyncio.Lock()

    def get_camera_polygon(self, camera_name: Optional[str]) -> Optional[List[List[float]]]:
        """Gets polygon for specific camera name, with case-insensitive fallback and default fallback."""
        if camera_name and self.camera_polygons:
            if camera_name in self.camera_polygons:
                return self.camera_polygons[camera_name]
            for k, v in self.camera_polygons.items():
                if k.lower() == camera_name.lower():
                    return v
        return self.detection_polygon

    def apply_polygon_mask(self, image_bytes: bytes, polygon: Optional[List[List[float]]] = None, camera_name: Optional[str] = None) -> bytes:
        """Applies a polygon ROI mask. Everything outside the polygon is blacked out."""
        poly = polygon or self.get_camera_polygon(camera_name)
        if not poly or len(poly) < 3:
            return image_bytes

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return image_bytes

            h, w = img.shape[:2]
            pts = []
            for pt in poly:
                px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
                py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])
                pts.append([px, py])

            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_np], 255)

            masked_img = cv2.bitwise_and(img, img, mask=mask)
            success, encoded = cv2.imencode(".jpg", masked_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if success:
                return encoded.tobytes()
        except Exception as e:
            logger.error(f"Error applying polygon mask: {e}")
        return image_bytes

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from model response text."""
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass

        return None

    def _simulate_inference(self, image_bytes: bytes) -> DetectionResult:
        """Safe fallback: Returns clean scene on connection error."""
        return DetectionResult(
            detected=False,
            is_detected=False,
            is_rat_detected=False,
            subject_type="none",
            object_type="none",
            label="None",
            confidence_score=0.0,
            confidence=0.0,
            description="Clear scene (Fallback).",
            bounding_box=None,
            inference_time_ms=30.0,
            model_name=self.model_name,
            raw_response="Fallback"
        )

    def _build_prompt(self, target_object: str, has_reference: bool) -> str:
        """Builds tailored, high-precision vision prompt for Gemma 4 12B based on target object selection."""
        target = target_object.lower().strip()

        if target in ["tree", "trees"]:
            if has_reference:
                return (
                    "You are an exact computer vision detector for environmental and tree monitoring.\n"
                    "Image 1: Reference baseline image.\n"
                    "Image 2: Current camera frame to evaluate.\n\n"
                    "Task: Is there a tree, tree trunk, fallen tree branch/log, or distinct tree canopy/foliage in Image 2?\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"tree\": Tree trunk, whole tree, large branch, or prominent tree foliage.\n"
                    "- \"false_positive_clutter\": Statues, ornaments, garden furniture, plain lawn, empty patio, walls.\n"
                    "- \"none\": No tree or branch in frame.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "tree" or "none" or "false_positive_clutter",\n'
                    '  "label": "Tree" or "None" or "Clutter",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Factual description of detected tree or branches"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact computer vision detector for environmental and tree monitoring.\n"
                    "Task: Carefully analyze this image to identify any tree, tree trunk, tree branch, or major tree foliage.\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"tree\": Tree trunk, whole tree, large branch, or prominent tree foliage.\n"
                    "- \"false_positive_clutter\": Statues, ornaments, garden furniture, plain lawn, empty patio, walls.\n"
                    "- \"none\": No tree or branch in frame.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "tree" or "none" or "false_positive_clutter",\n'
                    '  "label": "Tree" or "None" or "Clutter",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Factual description of detected tree or branches"\n'
                    "}"
                )

        elif target in ["bird", "birds", "pheasant"]:
            if has_reference:
                return (
                    "You are an exact wildlife detection verifier.\n"
                    "Image 1: Reference baseline image (empty lawn/garden with NO birds).\n"
                    "Image 2: Current high-resolution camera frame to evaluate.\n\n"
                    "Task: Compare Image 2 against Image 1. Is there an actual live bird (such as a pheasant, pigeon, magpie, garden songbird, or waterfowl) in Image 2?\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"bird\": Live bird or pheasant (visible plumage, beak/head, wings, feathers, or tail).\n"
                    "- \"other_animal\": Mammals such as cats, dogs, or rodents.\n"
                    "- \"false_positive_clutter\": Statues, stone ornaments, garden chairs, shadows, leaves.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "bird" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Bird" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact wildlife detection verifier.\n"
                    "Task: Carefully analyze this image to detect any live bird (pheasant, pigeon, songbird, waterfowl, raptor).\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"bird\": Live bird or pheasant (visible plumage, beak/head, wings, feathers, or tail).\n"
                    "- \"other_animal\": Mammals such as cats, dogs, or rodents.\n"
                    "- \"false_positive_clutter\": Statues, stone ornaments, garden chairs, shadows, leaves.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "bird" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Bird" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )

        elif target in ["horses_poo", "horse_poo", "horses poo", "poo", "manure", "dung"]:
            if has_reference:
                return (
                    "You are an exact computer vision inspector for paddock, lawn, and equine monitoring.\n"
                    "Image 1: Reference baseline image (clean ground with NO manure).\n"
                    "Image 2: Current high-resolution camera frame to evaluate.\n\n"
                    "Task: Compare Image 2 against Image 1. Is there horse droppings, horse poo, manure piles, or dung on the ground in Image 2?\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"horses_poo\": Horse droppings, manure pile, horse dung, or fresh/dried equine manure on the lawn/ground.\n"
                    "- \"false_positive_clutter\": Normal soil patches, stones, shadows, leaf clusters, garden clutter.\n"
                    "- \"none\": Clean ground / no horse poo.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "horses_poo" or "none" or "false_positive_clutter",\n'
                    '  "label": "Horses poo" or "None" or "Clutter",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Factual explanation of horse poo or clean ground"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact computer vision inspector for paddock, lawn, and equine monitoring.\n"
                    "Task: Carefully analyze this image to detect any horse droppings, horse poo, manure piles, or dung on the ground/lawn.\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"horses_poo\": Horse droppings, manure pile, horse dung, or fresh/dried equine manure on the lawn/ground.\n"
                    "- \"false_positive_clutter\": Normal soil patches, stones, shadows, leaf clusters, garden clutter.\n"
                    "- \"none\": Clean ground / no horse poo.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "horses_poo" or "none" or "false_positive_clutter",\n'
                    '  "label": "Horses poo" or "None" or "Clutter",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Factual explanation of horse poo or clean ground"\n'
                    "}"
                )

        elif target in ["horse", "horses", "pony", "equine"]:
            if has_reference:
                return (
                    "You are an exact computer vision detector for equine and paddock monitoring.\n"
                    "Image 1: Reference baseline image (empty paddock/lawn with NO horses).\n"
                    "Image 2: Current high-resolution camera frame to evaluate.\n\n"
                    "Task: Compare Image 2 against Image 1. Is there an actual live horse, pony, or equine animal in Image 2?\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"horse\": Live horse, pony, foal, or equine animal (large quadruped mammal with mane, tail, hooves, equine anatomy).\n"
                    "- \"other_animal\": Small animals like dogs, cats, rodents, birds.\n"
                    "- \"false_positive_clutter\": Statues, fences, trees, garden clutter, shadows.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "horse" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Horse" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact computer vision detector for equine and paddock monitoring.\n"
                    "Task: Carefully analyze this image to detect any live horse, pony, or equine animal.\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"horse\": Live horse, pony, foal, or equine animal (large quadruped mammal with mane, tail, hooves, equine anatomy).\n"
                    "- \"other_animal\": Small animals like dogs, cats, rodents, birds.\n"
                    "- \"false_positive_clutter\": Statues, fences, trees, garden clutter, shadows.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "horse" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Horse" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )

        elif target in ["rat", "rodent", "rats", "mouse"]:
            if has_reference:
                return (
                    "You are an exact wildlife detection verifier.\n"
                    "Image 1: Reference baseline image of the empty garden/lawn (NO animals present).\n"
                    "Image 2: Current high-resolution camera frame to evaluate.\n\n"
                    "Task: Compare Image 2 against Image 1. Is there an actual live rat or mouse in Image 2 that is absent in Image 1?\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"rodent\": Live rat or mouse (mammal with 3D body, head, snout, ears, paws, and tail).\n"
                    "- \"other_animal\": Cats, dogs, birds, horses, or other non-rodent animals.\n"
                    "- \"false_positive_clutter\": Statues, stone ornaments, garden chairs, shadows, leaves, plant pots.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "rodent" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Rat" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact wildlife detection verifier.\n"
                    "Carefully analyze this high-resolution camera image.\n\n"
                    "You MUST choose exactly one subject_type category:\n"
                    "- \"rodent\": Live rat or mouse (mammal with 3D body, head, snout, ears, paws, and tail).\n"
                    "- \"other_animal\": Cats, dogs, birds, horses, or other non-rodent animals.\n"
                    "- \"false_positive_clutter\": Statues, stone ornaments, garden chairs, shadows, leaves, plant pots.\n"
                    "- \"none\": Empty scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "rodent" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Rat" or "Other Animal" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )

        else:
            # Default "all" multi-object detection mode
            if has_reference:
                return (
                    "You are an exact multi-object computer vision and wildlife detector.\n"
                    "Image 1: Reference baseline image (empty background).\n"
                    "Image 2: Current high-resolution camera frame to evaluate.\n\n"
                    "Task: Compare Image 2 against Image 1. Identify which target object is present in Image 2:\n"
                    "1. \"tree\": Tree, tree trunk, fallen tree branch/log, or tree canopy.\n"
                    "2. \"bird\": Live bird, pheasant, pigeon, or songbird.\n"
                    "3. \"rodent\": Live rat or mouse (mammal with body, snout, paws, tail).\n"
                    "4. \"horse\": Live horse, pony, or equine animal.\n"
                    "5. \"horses_poo\": Horse droppings, manure pile, or dung on the ground.\n"
                    "6. \"other_animal\": Domestic cats, dogs, pets, humans, or non-target animals (NOT rodents).\n"
                    "7. \"false_positive_clutter\": Statues, stones, sofa/chairs, shadows, leaves, paper, cushions, plant pots.\n"
                    "8. \"none\": Empty / clean scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "tree" or "bird" or "rodent" or "horse" or "horses_poo" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Tree" or "Bird" or "Rat" or "Horse" or "Horses poo" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )
            else:
                return (
                    "You are an exact multi-object computer vision and wildlife detector.\n"
                    "Task: Carefully analyze this camera frame and identify which target object is present:\n"
                    "1. \"tree\": Tree, tree trunk, fallen tree branch/log, or tree canopy.\n"
                    "2. \"bird\": Live bird, pheasant, pigeon, or songbird.\n"
                    "3. \"rodent\": Live rat or mouse (mammal with body, snout, paws, tail).\n"
                    "4. \"horse\": Live horse, pony, or equine animal.\n"
                    "5. \"horses_poo\": Horse droppings, manure pile, or dung on the ground.\n"
                    "6. \"other_animal\": Domestic cats, dogs, pets, humans, or non-target animals (NOT rodents).\n"
                    "7. \"false_positive_clutter\": Statues, stones, sofa/chairs, shadows, leaves, paper, cushions, plant pots.\n"
                    "8. \"none\": Empty / clean scene.\n\n"
                    "Respond strictly in JSON format:\n"
                    "{\n"
                    '  "detected": true or false,\n'
                    '  "confidence_score": float between 0.0 and 1.0,\n'
                    '  "subject_type": "tree" or "bird" or "rodent" or "horse" or "horses_poo" or "other_animal" or "false_positive_clutter" or "none",\n'
                    '  "label": "Tree" or "Bird" or "Rat" or "Horse" or "Horses poo" or "Clutter" or "None",\n'
                    '  "bounding_box": [ymin, xmin, ymax, xmax] or null,\n'
                    '  "explanation": "Concise factual reason for classification"\n'
                    "}"
                )


    async def analyze_image(
        self,
        image_bytes: bytes,
        polygon: Optional[List[List[float]]] = None,
        reference_image_bytes: Optional[bytes] = None,
        is_subcrop: bool = False,
        target_object: Optional[str] = None
    ) -> DetectionResult:
        """Runs locked-temperature structured inference with dual-image baseline contrast verification and multi-object classification."""
        start_time = time.perf_counter()
        target_mode = (target_object or self.target_object).lower().strip()

        # If analyzing full frame, apply zone mask; if sub-crop, analyze high-res pixels directly
        target_bytes = image_bytes if is_subcrop else self.apply_polygon_mask(image_bytes, polygon)
        b64_curr = base64.b64encode(target_bytes).decode("utf-8")

        images_payload = []
        has_ref = reference_image_bytes is not None
        if reference_image_bytes:
            ref_masked = reference_image_bytes if is_subcrop else self.apply_polygon_mask(reference_image_bytes, polygon)
            b64_ref = base64.b64encode(ref_masked).decode("utf-8")
            images_payload = [b64_ref, b64_curr]
        else:
            images_payload = [b64_curr]

        prompt = self._build_prompt(target_mode, has_reference=has_ref)

        try:
            async with self._lock:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    if self.endpoint_type == "ollama":
                        endpoint = self.endpoint_url
                        if "roland3" in endpoint:
                            endpoint = "http://localhost:11434"
                        url = f"{endpoint}/api/generate"
                        payload = {
                            "model": self.model_name,
                            "prompt": prompt,
                            "images": images_payload,
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "top_k": 1,
                                "seed": 42
                            }
                        }
                        try:
                            resp = await client.post(url, json=payload)
                            resp.raise_for_status()
                        except Exception as e:
                            if "localhost" not in url:
                                logger.warning(f"Failed reaching {url}, falling back to local Ollama on http://localhost:11434: {e}")
                                resp = await client.post("http://localhost:11434/api/generate", json=payload)
                                resp.raise_for_status()
                            else:
                                raise e
                        data = resp.json()
                        raw_text = data.get("response", "")

                    elif self.endpoint_type == "openai":
                        url = f"{self.endpoint_url}/v1/chat/completions"
                        content_items = [{"type": "text", "text": prompt}]
                        for b64 in images_payload:
                            content_items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        payload = {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": content_items}],
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "response_format": {"type": "json_object"}
                        }
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        raw_text = data["choices"][0]["message"]["content"]

                    elif self.endpoint_type == "gemini":
                        import os
                        api_key = getattr(self, "gemini_api_key", None) or os.environ.get("GEMINI_API_KEY", "")
                        if not api_key:
                            raise ValueError("Gemini API Key not found. Please provide GEMINI_API_KEY in Settings.")

                        model = getattr(self, "gemini_model", None) or self.model_name or "gemini-3.7-flash"
                        if model.startswith("models/"):
                            model = model[len("models/"):]

                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

                        parts = [{"text": prompt}]
                        for b64 in images_payload:
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64
                                }
                            })

                        payload = {
                            "contents": [{"parts": parts}],
                            "generationConfig": {
                                "temperature": 0.1,
                                "topP": 0.95,
                                "responseMimeType": "application/json"
                            }
                        }
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        raw_text = ""
                        candidates = data.get("candidates", [])
                        if candidates:
                            c_parts = candidates[0].get("content", {}).get("parts", [])
                            if c_parts:
                                raw_text = c_parts[0].get("text", "")

                    else:
                        raise ValueError(f"Unsupported endpoint type: {self.endpoint_type}")

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    parsed = self._extract_json(raw_text)

                if parsed:
                    raw_type = str(parsed.get("subject_type", parsed.get("object_type", "none"))).lower()
                    raw_detected = bool(parsed.get("detected", False))
                    conf = float(parsed.get("confidence_score", parsed.get("confidence", 0.0)))
                    desc = parsed.get("explanation", parsed.get("description", "Analyzed by Gemma"))
                    bbox = parsed.get("bounding_box")

                    # Normalize classification based on subject_type and explanation
                    is_valid = False
                    norm_type = "none"
                    norm_label = "None"
                    desc_low = desc.lower()

                    # Check for explicit negation in the model's explanation
                    negation_phrases = [
                        "not a rat", "not a rodent", "no rat", "no rodent",
                        "not a live rat", "not a live rodent", "not a specified target",
                        "lacks the distinct anatomical", "lacks discernible", "lacks the anatomical",
                        "classified as false positive", "false positive clutter", "is general clutter",
                        "domestic cat", "is a cat", "cat is present", "a cat and", "cat on a",
                        "a dog", "is a dog", "domestic dog", "pet"
                    ]
                    is_negation = any(neg in desc_low for neg in negation_phrases)

                    # 1. Non-target categories or detected false positives
                    if raw_type in ["false_positive_clutter", "clutter", "statue", "furniture", "sofa"]:
                        is_valid = False
                        norm_type = "false_positive_clutter"
                        norm_label = "Clutter"
                        conf = 0.0
                    elif raw_type in ["other_animal", "cat", "dog", "pet", "human"]:
                        is_valid = False
                        norm_type = "other_animal"
                        norm_label = "Other Animal (Cat/Pet)"
                        conf = 0.0
                    elif raw_type in ["none", "empty"] or not raw_detected or is_negation:
                        is_valid = False
                        norm_type = "other_animal" if ("cat" in desc_low or "dog" in desc_low) else ("false_positive_clutter" if is_negation else "none")
                        norm_label = "Other Animal (Cat/Pet)" if ("cat" in desc_low or "dog" in desc_low) else ("Clutter" if is_negation else "None")
                        conf = 0.0
                    # 2. Positive candidate categories (only if raw_detected is True and not negated)
                    elif "rodent" in raw_type or "rat" in raw_type or "mouse" in raw_type:
                        norm_type = "rodent"
                        norm_label = "Rat"
                        matches_target = target_mode in ["all", "rat", "rodent", "rats", "mouse"]
                        is_valid = (conf >= self.confidence_threshold) and matches_target
                    elif "horses_poo" in raw_type or "horse_poo" in raw_type or "manure" in raw_type or "dung" in raw_type:
                        norm_type = "horses_poo"
                        norm_label = "Horses poo"
                        matches_target = target_mode in ["all", "horses_poo", "horse_poo", "horses poo", "poo", "manure", "dung"]
                        is_valid = (conf >= self.confidence_threshold) and matches_target
                    elif "horse" in raw_type or "pony" in raw_type or "equine" in raw_type:
                        norm_type = "horse"
                        norm_label = "Horse"
                        matches_target = target_mode in ["all", "horse", "horses", "pony", "equine"]
                        is_valid = (conf >= self.confidence_threshold) and matches_target
                    elif "tree" in raw_type or "branch" in raw_type or "trunk" in raw_type:
                        norm_type = "tree"
                        norm_label = "Tree"
                        matches_target = target_mode in ["all", "tree", "trees"]
                        is_valid = (conf >= self.confidence_threshold) and matches_target
                    elif "bird" in raw_type or "pheasant" in raw_type or "pigeon" in raw_type:
                        norm_type = "bird"
                        norm_label = "Pheasant" if "pheasant" in raw_type or "pheasant" in desc_low else "Bird"
                        matches_target = target_mode in ["all", "bird", "birds", "pheasant"]
                        is_valid = (conf >= self.confidence_threshold) and matches_target
                    else:
                        is_valid = False
                        norm_type = "none"
                        norm_label = "None"
                        conf = 0.0

                    return DetectionResult(
                        detected=is_valid,
                        is_detected=is_valid,
                        is_rat_detected=is_valid,
                        subject_type=norm_type,
                        object_type=norm_type,
                        label=norm_label,
                        confidence_score=conf if is_valid else 0.0,
                        confidence=conf if is_valid else 0.0,
                        description=desc,
                        bounding_box=bbox if is_valid else None,
                        inference_time_ms=elapsed_ms,
                        model_name=self.model_name,
                        raw_response=raw_text
                    )
                else:
                    logger.warning(f"Could not parse JSON from model output: {raw_text}")
                    return DetectionResult(
                        detected=False,
                        is_detected=False,
                        is_rat_detected=False,
                        subject_type="none",
                        object_type="none",
                        label="None",
                        confidence_score=0.0,
                        confidence=0.0,
                        description=raw_text[:200] if raw_text else "Clean scene",
                        inference_time_ms=elapsed_ms,
                        model_name=self.model_name,
                        raw_response=raw_text
                    )

        except Exception as e:
            logger.warning(f"Unable to connect to Roland 3 ({self.endpoint_url}): {e}. Using fallback simulation.")
            return self._simulate_inference(image_bytes)

