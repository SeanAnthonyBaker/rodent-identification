import asyncio
import base64
import json
import logging
import os
import time
from typing import Dict, Any, Optional, Set
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ratwatch-spark")

# Environment Configuration
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "gemma4-26b")
WATCH_URL = os.getenv("WATCH_URL", "http://watch.local:8099/alert")
PHONE_BRIDGE_URL = os.getenv("PHONE_BRIDGE_URL", "http://s21.local:8098/bridge-alert")
CONFIRM_MIN = float(os.getenv("CONFIRM_MIN", "0.55"))
VISION_TOKENS = int(os.getenv("VISION_TOKENS", "280"))
THINKING_MODE = os.getenv("THINKING", "off")

app = FastAPI(title="RatWatch DGX Spark Validator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket Clients
active_ws_clients: Set[WebSocket] = set()

# Internal Sighting Job Queue
job_queue = asyncio.Queue(maxsize=8)
stats = {
    "total_received": 0,
    "total_confirmed": 0,
    "total_rejected": 0,
    "avg_vllm_ms": 0.0,
    "is_warm": False,
    "last_sighting_ts": None
}

SYSTEM_PROMPT = """You validate pest sightings from a phone crop. Reply with a single JSON object. No markdown.
Keys:
  rat: boolean
  confidence: number 0 to 1
  location: string, max 8 words, room-relative from the crop
  reason: string, max 12 words
  watch_text: string, max 10 words, present tense
Set rat=true only if a live rat is visible.
Shoes, cables, bags, cats, toys, droppings-only, and shadows are false.
The phone box is a hint, not proof."""

async def broadcast_ws_event(event_type: str, data: Dict[str, Any]):
    """Broadcasts real-time events to all connected web front ends."""
    if not active_ws_clients:
        return
    msg = json.dumps({"type": event_type, "data": data})
    disconnected = set()
    for ws in list(active_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        active_ws_clients.discard(ws)

def build_watch_thumbnail(image_bytes: bytes, max_width: int = 240, quality: int = 50) -> str:
    """Builds a lightweight 240px JPEG thumbnail base64 string for Wear OS watch."""
    try:
        im = Image.open(BytesIO(image_bytes))
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            im = im.resize((max_width, new_h), Image.Resampling.BILINEAR)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.error(f"Thumbnail generation error: {e}")
        return ""

async def call_vllm_gemma26b(image_b64: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Queries Gemma 4 26B-A4B on vLLM without chain-of-thought delay."""
    t0 = time.perf_counter()
    user_text = (
        f"Phone conf={meta.get('conf', 0.0)} box={meta.get('box_frame', [])} "
        f"frame={meta.get('frame_wh', [1280, 720])} device={meta.get('device_id', 's21')}\n"
        f"Is there a live rat, and where in this crop?"
    )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient(timeout=4.0) as client:
        resp = await client.post(f"{VLLM_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        vllm_ms = int((time.perf_counter() - t0) * 1000)

        # Parse JSON
        try:
            verdict = json.loads(raw_content)
        except Exception:
            # Fallback cleaning
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            verdict = json.loads(cleaned.strip())

        return {
            "verdict": verdict,
            "vllm_ms": vllm_ms
        }

async def push_watch_alert(watch_text: str, confidence: float, thumb_b64: str, location: str) -> bool:
    """Dispatches direct LAN alert to Wear OS watch, with fallback bridge to phone."""
    alert_payload = {
        "id": f"sighting-{int(time.time()*1000)}",
        "ts_ms": int(time.time() * 1000),
        "watch_text": watch_text,
        "confidence": confidence,
        "location": location,
        "thumb_jpeg_b64": thumb_b64
    }

    # Attempt 1: Direct Wi-Fi Wear OS Push
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                res = await client.post(WATCH_URL, json=alert_payload)
                if res.status_code == 200:
                    logger.info(f"✅ Watch alert delivered via direct Wi-Fi in attempt {attempt+1}")
                    return True
        except Exception as e:
            logger.debug(f"Watch direct attempt {attempt+1} failed: {e}")
            await asyncio.sleep(0.1)

    # Attempt 2: Fallback to Phone Bluetooth Bridge
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            res = await client.post(PHONE_BRIDGE_URL, json=alert_payload)
            if res.status_code == 200:
                logger.info("✅ Watch alert delivered via S21 Phone Bluetooth Bridge")
                return True
    except Exception as e:
        logger.warning(f"Phone bridge fallback failed: {e}")

    return False

@app.on_event("startup")
async def startup_event():
    """Warms the Gemma 4 26B model at boot to eliminate first-hit latency."""
    logger.info("Initializing RatWatch DGX Spark Validator...")
    # Generate a lightweight dummy 640px test image
    try:
        im = Image.new("RGB", (640, 360), color=(80, 80, 80))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=70)
        dummy_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        dummy_meta = {"conf": 0.35, "box_frame": [100, 100, 200, 200], "device_id": "warmup-check"}
        logger.info("Sending model warm-up probe to vLLM...")
        res = await call_vllm_gemma26b(dummy_b64, dummy_meta)
        stats["is_warm"] = True
        stats["avg_vllm_ms"] = res["vllm_ms"]
        logger.info(f"✅ Gemma 4 26B warm-up complete in {res['vllm_ms']} ms")
    except Exception as e:
        logger.warning(f"⚠️ Model warm-up deferred: {e}")

@app.get("/health")
async def health_check():
    return {
        "ok": True,
        "model": VLLM_MODEL,
        "warm": stats["is_warm"],
        "infer_p50_ms": stats["avg_vllm_ms"],
        "stats": stats
    }

@app.websocket("/v1/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws_clients.add(websocket)
    logger.info(f"Front-end client connected to /v1/stream (Total: {len(active_ws_clients)})")
    try:
        while True:
            # Keepalive listener
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        active_ws_clients.discard(websocket)
        logger.info("Front-end client disconnected from /v1/stream")

@app.post("/v1/sighting")
async def process_sighting(
    image: UploadFile = File(...),
    meta: str = Form(...)
):
    """
    Core S21 sighting intake endpoint.
    Processes crop JPEG <= 80KB, validates with Gemma 4 26B on Spark,
    broadcasts on WebSocket, and pushes alerts to Wear OS watch.
    """
    t_recv = int(time.time() * 1000)
    stats["total_received"] += 1
    
    # 1. Parse Metadata & Validate File Size
    try:
        meta_dict = json.loads(meta)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    img_bytes = await image.read()
    if len(img_bytes) > 300 * 1024:
        raise HTTPException(status_code=413, detail="Image size exceeds 300 KB limit")

    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    # 2. Emit 'possible' event to Web Front End immediately
    box_frame = meta_dict.get("box_frame", [0, 0, 100, 100])
    frame_wh = meta_dict.get("frame_wh", [1280, 720])
    
    # Calculate center point
    if len(box_frame) >= 4:
        cx_norm = ((box_frame[0] + box_frame[2]) / 2.0) / frame_wh[0]
        cy_norm = ((box_frame[1] + box_frame[3]) / 2.0) / frame_wh[1]
    else:
        cx_norm, cy_norm = 0.5, 0.5

    possible_payload = {
        "track_id": meta_dict.get("track_id", 0),
        "device_id": meta_dict.get("device_id", "s21"),
        "timestamp": meta_dict.get("ts_ms", t_recv),
        "confidence": meta_dict.get("conf", 0.0),
        "center_point": [cx_norm, cy_norm],
        "bounding_box": box_frame,
        "status": "Validating with Gemma 4 26B on Spark..."
    }
    await broadcast_ws_event("possible", possible_payload)

    # 3. Query Gemma 4 26B NVFP4 on vLLM
    try:
        res = await call_vllm_gemma26b(img_b64, meta_dict)
        verdict = res["verdict"]
        vllm_ms = res["vllm_ms"]
        stats["avg_vllm_ms"] = vllm_ms
    except Exception as e:
        logger.error(f"vLLM inference error: {e}")
        # Default fallback to unconfirmed
        verdict = {
            "rat": False,
            "confidence": 0.0,
            "location": "unknown",
            "reason": f"Spark error: {str(e)}",
            "watch_text": "Validation error"
        }
        vllm_ms = 0

    is_confirmed = verdict.get("rat", False) and (float(verdict.get("confidence", 0.0)) >= CONFIRM_MIN)
    watch_delivered = False

    # 4. Dispatch Wear OS Alert if Confirmed
    if is_confirmed:
        stats["total_confirmed"] += 1
        thumb_b64 = build_watch_thumbnail(img_bytes)
        watch_text = verdict.get("watch_text", "Rat detected")
        location = verdict.get("location", "Garden run")
        conf = float(verdict.get("confidence", 0.0))
        watch_delivered = await push_watch_alert(watch_text, conf, thumb_b64, location)

        # Broadcast confirmed verdict event
        verdict_event = {
            "track_id": meta_dict.get("track_id", 0),
            "rat": True,
            "confidence": conf,
            "location": location,
            "reason": verdict.get("reason", ""),
            "watch_text": watch_text,
            "center_point": [cx_norm, cy_norm],
            "spark_ms": vllm_ms,
            "watch_sent": watch_delivered,
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }
        await broadcast_ws_event("verdict", verdict_event)
    else:
        stats["total_rejected"] += 1
        rejected_event = {
            "track_id": meta_dict.get("track_id", 0),
            "rat": False,
            "confidence": float(verdict.get("confidence", 0.0)),
            "reason": verdict.get("reason", "Not confirmed"),
            "spark_ms": vllm_ms
        }
        await broadcast_ws_event("rejected", rejected_event)

    return {
        "recv_ms": t_recv,
        "vllm_ms": vllm_ms,
        "confirmed": is_confirmed,
        "watch_delivered": watch_delivered,
        "verdict": verdict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
