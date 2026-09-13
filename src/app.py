import asyncio
import sys

if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _safe_call_connection_lost(self, exc=None):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass

import base64
import json
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import httpx

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import config
from src.ring_client import RingManager, is_blank_or_disabled_frame
from src.inference_client import RolandInferenceClient
from src.storage import StorageManager
from src.sampler import SamplerEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")

# Initialize modules
ring_manager = RingManager(
    token_file=config.ring.token_file,
    device_name=config.ring.device_name,
    mock_fallback=config.ring.mock_if_unavailable
)

inference_client = RolandInferenceClient(
    endpoint_url=config.inference.endpoint_url,
    endpoint_type=config.inference.endpoint_type,
    model_name=config.inference.model_name,
    confidence_threshold=config.inference.confidence_threshold,
    timeout_seconds=config.inference.timeout_seconds,
    detection_polygon=config.inference.detection_polygon,
    gemini_api_key=config.inference.gemini_api_key,
    gemini_model=config.inference.gemini_model
)

storage_manager = StorageManager(
    detections_dir=config.storage.detections_dir,
    db_path=config.storage.db_path,
    backend=config.storage.backend,
    supabase_url=config.storage.supabase_url,
    supabase_key=config.storage.supabase_key,
    supabase_table=config.storage.supabase_table,
    supabase_bucket=config.storage.supabase_bucket,
    fallback_to_sqlite=config.storage.fallback_to_sqlite
)

sampler_engine = SamplerEngine(
    ring_manager=ring_manager,
    inference_client=inference_client,
    storage_manager=storage_manager,
    interval_seconds=config.ring.sample_interval_seconds,
    active_detection_interval_seconds=config.ring.active_detection_interval_seconds
)

# Connected WebSocket clients
active_websockets: List[WebSocket] = []

async def ws_broadcaster(message: dict):
    for ws in list(active_websockets):
        try:
            await ws.send_json(message)
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)

sampler_engine.register_subscriber(ws_broadcaster)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect Ring, start phone, tablet & local camera broadcasters, and start background sampler
    await ring_manager.async_connect()
    if hasattr(ring_manager, "_phone_cam") and hasattr(ring_manager._phone_cam, "broadcaster"):
        ring_manager._phone_cam.broadcaster.start()
    if hasattr(ring_manager, "_tab_cam") and hasattr(ring_manager._tab_cam, "broadcaster"):
        ring_manager._tab_cam.broadcaster.start()
    if hasattr(ring_manager, "_local_cam") and hasattr(ring_manager._local_cam, "broadcaster"):
        ring_manager._local_cam.broadcaster.start()

    # Prewarm snapshot cache: immediate live Ring daylight snapshots and crisp mobile pictures
    if hasattr(ring_manager, "_phone_cam") and hasattr(ring_manager._phone_cam, "get_picture"):
        p = ring_manager._phone_cam.get_picture()
        if p:
            ring_manager._snapshot_cache["Samsung Galaxy S21 Ultra"] = p
            ring_manager._snapshot_cache["S21"] = p
    if hasattr(ring_manager, "_tab_cam") and hasattr(ring_manager._tab_cam, "get_picture"):
        p = ring_manager._tab_cam.get_picture()
        if p:
            ring_manager._snapshot_cache["Galaxy Tab A11+"] = p
            ring_manager._snapshot_cache["Tab A11+"] = p

    for c in ring_manager._all_cameras:
        c_name = getattr(c, "name", "")
        is_ring = hasattr(c, "family") and c.family in ["stickup_cams", "doorbots", "doorbells"]
        if is_ring:
            asyncio.create_task(ring_manager.async_fetch_snapshot(camera_name=c_name))

    sampler_engine.start()
    logger.info(f"Application started on Roland 1. Inference target: '{config.inference.endpoint_url}' (Roland 3). Supabase: '{config.storage.supabase_url}'. Camera: '{ring_manager.camera_name}'.")
    yield
    # Shutdown
    if hasattr(ring_manager, "_phone_cam") and hasattr(ring_manager._phone_cam, "broadcaster"):
        ring_manager._phone_cam.broadcaster.stop()
    if hasattr(ring_manager, "_tab_cam") and hasattr(ring_manager._tab_cam, "broadcaster"):
        ring_manager._tab_cam.broadcaster.stop()
    if hasattr(ring_manager, "_local_cam") and hasattr(ring_manager._local_cam, "broadcaster"):
        ring_manager._local_cam.broadcaster.stop()
    sampler_engine.stop()
    logger.info("Application shutdown.")

app = FastAPI(
    title="Roland 1 Rodent Detection Appliance",
    description="Real-time edge detection & Supabase services on Roland 1, multimodal AI inference on Roland 3",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Dashboard frontend is loading...</h2>")

@app.get("/mobile_cam", response_class=HTMLResponse)
async def serve_mobile_cam():
    """Serves wireless mobile camera broadcaster interface for Samsung S21 Ultra."""
    mobile_file = static_dir / "mobile_cam.html"
    if mobile_file.exists():
        return FileResponse(str(mobile_file))
    return HTMLResponse("<h2>Mobile camera broadcaster not found</h2>")

@app.get("/api/status")
async def get_status():
    """Returns general system status, battery, and current statistics."""
    status = sampler_engine.get_status()
    status["storage_backend"] = storage_manager.backend_info
    status["nodes"] = {
        "detection_node": config.node.detection_node,
        "supabase_node": config.node.supabase_node,
        "inference_node": config.node.inference_node
    }
    return status

@app.get("/api/system/nodes")
async def get_system_nodes():
    """Returns the distributed topology status across Roland 1 and Roland 3."""
    return {
        "detection_node": {
            "node": config.node.detection_node,
            "role": "Real-Time Rodent Detection, Motion Gate & Camera Ingestion",
            "active_camera": ring_manager.camera_name,
            "cadence": "realtime" if sampler_engine._is_rat_active else "idle",
            "interval_seconds": sampler_engine.current_interval_seconds
        },
        "supabase_node": {
            "node": config.node.supabase_node,
            "role": "Supabase Database & Storage Services",
            "url": config.storage.supabase_url,
            "table": config.storage.supabase_table,
            "active_backend": storage_manager.backend_info["active_backend"],
            "supabase_active": storage_manager.is_supabase_active,
            "fallback_to_sqlite": storage_manager.fallback_to_sqlite
        },
        "inference_node": {
            "node": config.node.inference_node,
            "role": "AI Multimodal Sighting Verification",
            "endpoint_url": config.inference.endpoint_url,
            "endpoint_type": config.inference.endpoint_type,
            "model_name": config.inference.model_name
        }
    }

@app.get("/api/cameras")
async def get_cameras():
    """Returns all discovered Ring cameras."""
    return {
        "active_camera": ring_manager.camera_name,
        "cameras": ring_manager.list_cameras()
    }

class SelectCameraPayload(BaseModel):
    camera_name: str

@app.post("/api/cameras/select")
async def select_camera(payload: SelectCameraPayload):
    """Switches the active camera and moves S21 to continuous real-time analysis."""
    success = ring_manager.select_camera(payload.camera_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera '{payload.camera_name}' not found")
    
    # Update active detection polygon to the newly selected camera's specific zone
    active_zone = inference_client.get_camera_polygon(payload.camera_name)
    inference_client.detection_polygon = active_zone

    is_fast_cam = any(k in payload.camera_name.lower() for k in ["s21", "tab", "a11", "galaxy tab", "phone", "galaxy", "android", "local"])
    if is_fast_cam:
        sampler_engine.current_interval_seconds = sampler_engine.active_detection_interval_seconds
        logger.info(f"⚡ Fast camera selected ({payload.camera_name}) -> Switched to REAL-TIME continuous analysis ({sampler_engine.active_detection_interval_seconds}s)")
    else:
        sampler_engine.current_interval_seconds = sampler_engine.base_interval_seconds

    try:
        config_file = Path("config.yaml")
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            if "ring" not in raw_cfg: raw_cfg["ring"] = {}
            if "inference" not in raw_cfg: raw_cfg["inference"] = {}
            raw_cfg["ring"]["device_name"] = payload.camera_name
            raw_cfg["inference"]["detection_polygon"] = active_zone
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_cfg, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"Failed to persist selected camera in config: {e}")

    # Trigger an immediate sample on the new camera
    asyncio.create_task(sampler_engine.sample_once())
    
    await ws_broadcaster({
        "type": "camera_switched",
        "data": {
            "active_camera": ring_manager.camera_name,
            "health": ring_manager.get_health_status(),
            "detection_polygon": active_zone,
            "is_realtime": is_fast_cam,
            "interval_seconds": sampler_engine.current_interval_seconds
        }
    })
    
    return {
        "success": True,
        "active_camera": ring_manager.camera_name,
        "detection_polygon": active_zone,
        "health": ring_manager.get_health_status()
    }

class CameraZonePayload(BaseModel):
    polygon: Optional[List[List[float]]] = None

def _normalize_cam_key(name: str) -> str:
    n = name.lower().strip()
    if any(k in n for k in ["tab", "a11", "galaxy tab", "tablet", "outhouse"]):
        return "Galaxy Tab A11+"
    if any(k in n for k in ["s21", "s1", "phone"]):
        return "S21"
    if "garden" in n:
        return "Garden"
    if "cam1" in n or "cam 1" in n:
        return "cam1"
    return name

@app.get("/api/camera/{camera_name}/zone")
async def get_camera_zone(camera_name: str):
    """Retrieves the detection zone polygon configured specifically for this camera."""
    key = _normalize_cam_key(camera_name)
    poly = inference_client.get_camera_polygon(key) or inference_client.get_camera_polygon(camera_name)
    target_cam = ring_manager.find_camera(camera_name)
    is_active = bool(ring_manager._active_camera and target_cam and ring_manager._active_camera == target_cam)
    return {
        "camera_name": key,
        "polygon": poly,
        "is_active_camera": is_active
    }

@app.post("/api/camera/{camera_name}/zone")
async def set_camera_zone(camera_name: str, payload: CameraZonePayload):
    """Saves an independent detection zone polygon for this specific camera."""
    key = _normalize_cam_key(camera_name)
    poly = payload.polygon if (payload.polygon and len(payload.polygon) >= 3) else None
    inference_client.camera_polygons[key] = poly
    inference_client.camera_polygons[camera_name] = poly
    
    # If this camera is currently the Active camera, apply to live inference immediately
    target_cam = ring_manager.find_camera(camera_name)
    is_active = bool(ring_manager._active_camera and target_cam and ring_manager._active_camera == target_cam)
    if is_active:
        inference_client.detection_polygon = poly

    # Persist to config.yaml
    try:
        config_file = Path("config.yaml")
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        else:
            raw_cfg = {}
        if "inference" not in raw_cfg: raw_cfg["inference"] = {}
        if "camera_polygons" not in raw_cfg["inference"]: raw_cfg["inference"]["camera_polygons"] = {}
        
        raw_cfg["inference"]["camera_polygons"][key] = poly
        if is_active:
            raw_cfg["inference"]["detection_polygon"] = poly
            
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw_cfg, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"Failed to persist camera zone to config.yaml: {e}")

    await ws_broadcaster({
        "type": "camera_zone_updated",
        "data": {
            "camera_name": camera_name,
            "polygon": poly,
            "is_active_camera": is_active
        }
    })

    return {
        "success": True,
        "camera_name": camera_name,
        "polygon": poly,
        "is_active_camera": is_active
    }

@app.get("/api/camera/{camera_name}/snapshot")
async def get_camera_snapshot(camera_name: str):
    """Serves latest snapshot specifically for the requested camera name (Garden, cam1, etc.)."""
    image_bytes, error, _, _ = await ring_manager.async_fetch_snapshot(camera_name=camera_name)
    if not image_bytes:
        raise HTTPException(status_code=404, detail=error or f"No snapshot available for {camera_name}")
    return Response(content=image_bytes, media_type="image/jpeg")

@app.get("/api/battery")
async def get_battery():
    """Returns current Ring camera battery percentage and health."""
    health = ring_manager.get_health_status()
    battery = health.get("battery_percentage")
    recharge_needed = battery is not None and battery <= 20

    return {
        "battery_percentage": battery,
        "battery_status": health.get("battery_status"),
        "recharge_needed": recharge_needed,
        "device_name": health.get("device_name"),
        "is_mock": health.get("is_mock"),
        "last_updated": health.get("last_updated")
    }

@app.get("/api/detections")
async def list_detections(
    order: str = Query("asc", description="Sort order: 'asc' for chronological or 'desc' for newest first"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    object_type: Optional[str] = Query(None, description="Filter by object type ('tree', 'bird', 'rat', 'horses_poo', 'all')")
):
    """Retrieves detections formatted for chronological carousel display, optionally filtered by target object."""
    detections = storage_manager.list_detections(order=order, limit=limit, offset=offset, object_type=object_type)
    stats = storage_manager.get_stats()
    return {
        "total": stats["total_detections"],
        "order": order,
        "object_type": object_type or "all",
        "detections": [d.model_dump() for d in detections]
    }

@app.get("/api/detections/{detection_id}")
async def get_detection(detection_id: int):
    rec = storage_manager.get_detection(detection_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Detection not found")
    return rec.model_dump()

@app.get("/api/detections/{detection_id}/image")
async def get_detection_image(detection_id: int):
    rec = storage_manager.get_detection(detection_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Detection not found")

    filepath = Path(config.storage.detections_dir) / rec.filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(str(filepath), media_type="image/jpeg")

@app.delete("/api/detections/{detection_id}")
async def delete_detection(detection_id: int):
    deleted = storage_manager.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {"success": True, "id": detection_id}

class BatchDeletePayload(BaseModel):
    ids: List[int]

@app.post("/api/detections/delete_batch")
async def delete_detections_batch(payload: BatchDeletePayload):
    """Deletes an arbitrary list of frame IDs."""
    count = storage_manager.delete_detections_batch(payload.ids)
    return {"success": True, "deleted_count": count, "ids": payload.ids}

@app.get("/api/events")
async def get_events(
    order: str = Query("desc"),
    limit: int = Query(50),
    offset: int = Query(0),
    object_type: Optional[str] = Query(None)
):
    """Returns grouped events (continuous sightings) over time."""
    events = storage_manager.list_events(order=order, limit=limit, offset=offset, object_type=object_type)
    return {"events": events, "total": len(events)}

@app.get("/api/events/{event_id}/frames")
async def get_event_frames(event_id: str):
    """Returns all frames belonging to an event session."""
    frames = storage_manager.get_event_frames(event_id)
    return {"event_id": event_id, "frames": [f.model_dump() for f in frames], "count": len(frames)}

@app.delete("/api/events/{event_id}")
async def delete_event(event_id: str):
    """Deletes ALL frames and records belonging to an entire event session."""
    count = storage_manager.delete_event(event_id)
    return {"success": True, "event_id": event_id, "deleted_count": count}

@app.post("/api/detections/clear_all")
async def clear_all_detections():
    """Wipes all detections from database and discards test captures."""
    count = storage_manager.clear_all_detections()
    return {"success": True, "deleted_count": count}

@app.get("/api/camera/latest_snapshot")
async def get_latest_snapshot():
    """Serves the latest camera frame captured by the sampler."""
    image_bytes = sampler_engine.get_latest_snapshot()
    if not image_bytes:
        # Generate on the fly
        image_bytes, _, _, _ = await ring_manager.async_fetch_snapshot()
        if not image_bytes:
            raise HTTPException(status_code=503, detail="No snapshot available")
    return Response(content=image_bytes, media_type="image/jpeg")

@app.get("/api/audio/alert.wav")
async def get_alert_audio():
    """Serves the generated voice alert audio file."""
    audio_path = sampler_engine.notifier.alert_wav_path
    if not audio_path.exists():
        sampler_engine.notifier.generate_voice_file("Warning: Target detected on camera.")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio alert file not available")
    return FileResponse(str(audio_path), media_type="audio/wav")

@app.post("/api/notifications/test")
async def test_watch_notification():
    """Triggers an immediate test push & voice notification to the Zepp OS watch & Android."""
    res = await sampler_engine.notifier.notify_animal_detected(
        confidence=0.98,
        description="TEST ALERT: Multi-object alert test to verify your notification stream.",
        device_name=ring_manager.camera_name,
        object_type=inference_client.target_object if inference_client.target_object != "all" else "rat",
        timestamp=datetime.now()
    )
    return {"success": True, "result": res}

@app.post("/api/sample_now")
async def trigger_sample_now():
    """Triggers an immediate sample and inference cycle."""
    asyncio.create_task(sampler_engine.sample_once())
    return {"status": "Sampling triggered", "active_camera": ring_manager.camera_name}

class TargetObjectPayload(BaseModel):
    target_object: str

@app.post("/api/target_object")
async def set_target_object(payload: TargetObjectPayload):
    """Sets the active target object type (tree, bird, rat, horses_poo, all)."""
    target = payload.target_object.lower().strip()
    inference_client.target_object = target
    try:
        config_file = Path("config.yaml")
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        else:
            raw_cfg = {}
        if "inference" not in raw_cfg: raw_cfg["inference"] = {}
        raw_cfg["inference"]["target_object"] = target
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw_cfg, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"Error persisting target_object: {e}")

    await ws_broadcaster({
        "type": "target_object_changed",
        "data": {
            "target_object": target
        }
    })
    return {"success": True, "target_object": target}

@app.post("/api/simulate_detection")
async def simulate_detection(animal: str = Query("rat", description="'tree', 'bird', 'rat', 'horses_poo', or 'pheasant'")):
    """Generates a simulated detection for Tree, Bird, Rat, or Horses poo to verify carousel and notifications."""
    from src.ring_client import MockRingCamera
    from datetime import datetime
    import base64

    target = animal.lower().strip()
    if target in ["tree", "trees"]:
        obj_type = "tree"
        label = "Tree"
        desc = "Simulated Tree detection: Fallen branch / tree trunk spotted in target area."
        bbox = [120, 200, 880, 780]
    elif target in ["bird", "birds", "pheasant"]:
        obj_type = "bird"
        label = "Pheasant" if "pheasant" in target else "Bird"
        desc = "Simulated Bird detection: Gamebird / songbird spotted foraging in garden zone."
        bbox = [350, 420, 680, 750]
    elif target in ["horse", "horses", "pony", "equine"]:
        obj_type = "horse"
        label = "Horse"
        desc = "Simulated Horse detection: Equine animal spotted in paddock / field area."
        bbox = [200, 150, 850, 850]
    elif target in ["horses_poo", "horse_poo", "horses poo", "poo", "manure"]:
        obj_type = "horses_poo"
        label = "Horses poo"
        desc = "Simulated Horses Poo detection: Fresh horse manure pile detected on lawn."
        bbox = [600, 450, 750, 620]
    else:
        obj_type = "rodent"
        label = "Rat"
        desc = "Simulated Rat detection: Rodent detected moving along fence line."
        bbox = [520, 580, 590, 720]

    sim_cam = MockRingCamera("cam1 (Simulation Test)")
    sim_cam._battery_level = ring_manager.get_battery_level() or 75
    img_bytes = await sim_cam.async_get_snapshot()

    now_dt = datetime.now()
    sim_event_id = f"evt_sim_{int(now_dt.timestamp())}"
    record = storage_manager.save_detection(
        image_bytes=img_bytes,
        confidence=0.95,
        description=desc,
        object_type=obj_type,
        label=label,
        battery_percentage=sim_cam.battery_life,
        device_name=ring_manager.camera_name,
        bounding_box=bbox,
        dt=now_dt,
        event_id=sim_event_id,
        frame_index=1
    )

    b64_thumb = base64.b64encode(img_bytes).decode("utf-8")
    status_payload = {
        "sample_index": sampler_engine._sample_count + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_health": ring_manager.get_health_status(),
        "battery_percentage": sim_cam.battery_life,
        "detected": True,
        "rat_detected": True,
        "object_type": obj_type,
        "label": label,
        "bounding_box": bbox,
        "confidence": 0.95,
        "description": record.description,
        "inference_time_ms": 115.0,
        "detection_saved": record.model_dump(),
        "latest_image_base64": f"data:image/jpeg;base64,{b64_thumb}"
    }
    await sampler_engine._notify_subscribers("sample_completed", status_payload)
    return {"success": True, "detection": record.model_dump()}


def apply_live_cctv_hud(image_bytes: bytes, camera_name: str, battery_pct: Optional[int] = None) -> bytes:
    """Applies a crisp real-time CCTV HUD overlay with ticking clock and pulsing live REC indicator."""
    if not image_bytes:
        return image_bytes
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        h, w, _ = img.shape
        now = datetime.now()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 100000:01d}"
        bat_str = f"{battery_pct}%" if battery_pct is not None else "87%"

        # Pulse state (toggles dot every 500ms)
        is_pulse_on = (now.microsecond // 500000) == 0

        # Top banner background glass box
        box_h = 38
        box_w = min(w - 20, 620)
        sub_img = img[10:10+box_h, 10:10+box_w]
        rect = np.full(sub_img.shape, (15, 23, 42), dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.35, rect, 0.65, 0, sub_img)

        # Draw pulsing Red REC dot
        dot_color = (0, 0, 240) if is_pulse_on else (40, 40, 110)
        cv2.circle(img, (28, 29), 6, dot_color, -1)
        if is_pulse_on:
            cv2.circle(img, (28, 29), 9, (0, 0, 255), 1)

        # Header text
        txt = f"LIVE | {camera_name.upper()} | BAT: {bat_str} | {ts_str}"
        cv2.putText(img, txt, (44, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

        # Bottom status bar box
        b_sub = img[h-35:h-10, 10:360]
        b_rect = np.full(b_sub.shape, (10, 15, 26), dtype=np.uint8)
        cv2.addWeighted(b_sub, 0.3, b_rect, 0.7, 0, b_sub)
        cv2.putText(img, "AI DETECTOR: ARMED (GEMMA 4 E12B)", (20, h-17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

        # Viewfinder corner ticks
        b_len = 24
        b_col = (148, 163, 184)
        # Top-Left
        cv2.line(img, (14, 14), (14 + b_len, 14), b_col, 2)
        cv2.line(img, (14, 14), (14, 14 + b_len), b_col, 2)
        # Top-Right
        cv2.line(img, (w - 14, 14), (w - 14 - b_len, 14), b_col, 2)
        cv2.line(img, (w - 14, 14), (w - 14, 14 + b_len), b_col, 2)
        # Bottom-Left
        cv2.line(img, (14, h - 14), (14 + b_len, h - 14), b_col, 2)
        cv2.line(img, (14, h - 14), (14, h - 14 - b_len), b_col, 2)
        # Bottom-Right
        cv2.line(img, (w - 14, h - 14), (w - 14 - b_len, h - 14), b_col, 2)
        cv2.line(img, (w - 14, h - 14), (w - 14, h - 14 - b_len), b_col, 2)

        success, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            return encoded.tobytes()
    except Exception as e:
        logger.debug(f"HUD overlay error: {e}")
    return image_bytes


@app.get("/api/camera/{camera_name}/live_stream")
async def live_camera_stream(camera_name: str):
    """Serves continuous live MJPEG video stream with standard multipart framing for all cameras."""
    target_cam = ring_manager.find_camera(camera_name)
    has_broadcaster = target_cam and hasattr(target_cam, "broadcaster") and target_cam.broadcaster is not None

    # For cameras equipped with a live broadcaster (S21, Local Webcam): stream native smooth FPS
    if has_broadcaster:
        q = target_cam.broadcaster.subscribe()
        async def fast_broadcaster_stream():
            try:
                real_cam_name = getattr(target_cam, "name", camera_name)
                pic_fallback = target_cam.get_picture() if hasattr(target_cam, "get_picture") else None
                first_frame = target_cam.broadcaster.latest_frame or getattr(target_cam, "_last_frame_bytes", None)
                if is_blank_or_disabled_frame(first_frame) and pic_fallback:
                    first_frame = pic_fallback
                if not first_frame and hasattr(target_cam, "async_get_snapshot"):
                    first_frame = await target_cam.async_get_snapshot()
                if is_blank_or_disabled_frame(first_frame) and pic_fallback:
                    first_frame = pic_fallback
                if not first_frame:
                    first_frame = ring_manager.create_standby_frame(f"Connecting {real_cam_name}...", real_cam_name)

                bat = ring_manager.get_battery_level(target_cam) or 100
                init_hud = apply_live_cctv_hud(first_frame, real_cam_name, bat)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(init_hud)}\r\n\r\n".encode("ascii")
                    + init_hud
                    + b"\r\n"
                )
                
                # Smooth broadcast loop capped at ~20 FPS (prevents socket reset & browser decode choke)
                last_sent = time.time()
                while True:
                    try:
                        frame = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        frame = target_cam.broadcaster.latest_frame
                    if is_blank_or_disabled_frame(frame) and pic_fallback:
                        frame = pic_fallback
                    if not frame:
                        frame = ring_manager.create_standby_frame(f"{real_cam_name} Reconnecting (Check USB)...", real_cam_name)

                    now = time.time()
                    if frame and (now - last_sent >= 0.05):
                        last_sent = now
                        live_hud = apply_live_cctv_hud(frame, real_cam_name, bat)
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(live_hud)}\r\n\r\n".encode("ascii")
                            + live_hud
                            + b"\r\n"
                        )
                    else:
                        await asyncio.sleep(0.01)
            except (asyncio.CancelledError, GeneratorExit, ConnectionResetError, BrokenPipeError):
                pass
            finally:
                target_cam.broadcaster.unsubscribe(q)

        return StreamingResponse(
            fast_broadcaster_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive"
            }
        )

    async def safe_bg_fetch(cam_name: str):
        try:
            await ring_manager.async_fetch_snapshot(camera_name=cam_name)
        except Exception as e:
            logger.debug(f"Background snapshot fetch failed for {cam_name}: {e}")

    # Frame generator for Ring cameras yielding discrete, complete JPEGs with standard boundary=frame
    async def frame_generator():
        try:
            # 1. Immediately yield cached or standby frame so browser image switches with 0 latency
            init_frame = ring_manager._snapshot_cache.get(camera_name)
            if not init_frame and is_phone and hasattr(ring_manager, "_phone_cam"):
                init_frame = ring_manager._phone_cam._last_frame_bytes
            if not init_frame:
                init_frame = ring_manager.create_standby_frame(f"Connecting {camera_name}...", camera_name)
            
            bat = ring_manager.get_battery_level(target_cam) if target_cam else ring_manager.get_battery_level()
            real_cam_name = getattr(target_cam, "name", camera_name) if target_cam else camera_name
            init_hud = apply_live_cctv_hud(init_frame, real_cam_name, bat)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + init_hud + b"\r\n")

            # Trigger immediate fresh snapshot fetch from Ring cloud
            asyncio.create_task(safe_bg_fetch(camera_name))

            # 2. Continuous non-blocking real-time stream loop (10 FPS with live HUD)
            last_bg_refresh = time.time()
            while True:
                frame = ring_manager._snapshot_cache.get(camera_name)
                if not frame and is_phone and hasattr(ring_manager, "_phone_cam"):
                    frame = ring_manager._phone_cam._last_frame_bytes

                now = time.time()
                # If no frame cached, or if 25s elapsed, refresh in background without blocking stream
                if not frame or (not is_phone and (now - last_bg_refresh > 25.0)):
                    last_bg_refresh = now
                    asyncio.create_task(safe_bg_fetch(camera_name))

                if not frame:
                    frame = ring_manager.create_standby_frame(f"Connecting {camera_name}...", camera_name)

                if frame:
                    bat = ring_manager.get_battery_level(target_cam) if target_cam else ring_manager.get_battery_level()
                    real_cam_name = getattr(target_cam, "name", camera_name) if target_cam else camera_name
                    live_frame = apply_live_cctv_hud(frame, real_cam_name, bat)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + live_frame + b"\r\n")

                sleep_duration = 0.04 if is_phone else 0.1
                await asyncio.sleep(sleep_duration)
        except (asyncio.CancelledError, GeneratorExit, ConnectionResetError, BrokenPipeError):
            logger.debug(f"Client disconnected from stream for {camera_name}")
            return
        except Exception as e:
            logger.error(f"Error in stream generator for {camera_name}: {e}")
            return

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.post("/api/camera/S21/rotate")
async def rotate_s21_camera(orientation: Optional[str] = "landscape"):
    """Sets orientation of S21 camera hardware (landscape = 90 deg clockwise to right)."""
    target_cam = ring_manager.find_camera("S21")
    if target_cam and hasattr(target_cam, "async_set_orientation"):
        await target_cam.async_set_orientation(orientation)
        return {"success": True, "orientation": orientation}
    return {"success": False, "error": "S21 camera not connected"}


@app.get("/api/camera/live_stream")
async def live_video_stream():
    """Serves high-frame-rate MJPEG video stream to any web viewer."""
    async def frame_generator():
        last_fetch_time = 0.0
        while True:
            frame = sampler_engine.get_latest_snapshot()
            if not frame and hasattr(ring_manager, "_phone_cam"):
                frame = ring_manager._phone_cam._last_frame_bytes
            
            now = time.time()
            if not frame or (now - last_fetch_time > 3.0):
                frame, _, _, _ = await ring_manager.async_fetch_snapshot()
                last_fetch_time = now

            if not frame:
                frame = ring_manager.create_standby_frame("Armed & Monitoring")

            if frame:
                cam_n = ring_manager.camera_name
                bat = ring_manager.get_battery_level()
                live_frame = apply_live_cctv_hud(frame, cam_n, bat)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + live_frame + b"\r\n")
            await asyncio.sleep(0.2)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


class ScreenCamFramePayload(BaseModel):
    image_base64: str
    device_name: Optional[str] = "Screen Cam (Live)"
    polygon: Optional[List[List[float]]] = None
    battery_percentage: Optional[int] = None
    is_charging: Optional[bool] = None

class PhoneBatteryReportPayload(BaseModel):
    battery_percentage: int
    is_charging: Optional[bool] = None
    device_name: Optional[str] = "Samsung Galaxy S21 Ultra"

@app.post("/api/camera/{camera_name}/battery")
@app.post("/api/camera/s21/battery")
@app.post("/api/phone/battery")
async def report_phone_battery(camera_name: str = "S21", payload: PhoneBatteryReportPayload = None):
    """Allows phone app / Shortcut / Tasker / curl to report current battery percentage."""
    if payload is None:
        raise HTTPException(status_code=400, detail="Missing payload")
    cam = ring_manager.find_camera(camera_name)
    if cam and hasattr(cam, "battery_life"):
        cam.battery_life = payload.battery_percentage
        logger.info(f"🔋 S21 Battery reported directly from phone: {payload.battery_percentage}%")
        return {"success": True, "camera": camera_name, "battery_percentage": payload.battery_percentage}
    return {"success": False, "error": f"Camera '{camera_name}' not found"}

@app.get("/api/camera/{camera_name}/refresh_battery")
@app.get("/api/camera/s21/refresh_battery")
async def refresh_camera_battery(camera_name: str = "S21"):
    """Actively polls the camera / phone to read hardware battery."""
    cam = ring_manager.find_camera(camera_name)
    if cam and hasattr(cam, "refresh_battery_from_phone"):
        val = await asyncio.to_thread(cam.refresh_battery_from_phone)
        return {"success": True, "battery_percentage": val}
    return {"success": False, "error": "Camera does not support dynamic battery refresh"}

_last_stream_ai_time = 0.0
_is_analyzing_stream = False
_prev_stream_gray = None

def detect_stream_motion(image_bytes: bytes) -> float:
    """Fast grayscale pixel difference for instant motion detection."""
    global _prev_stream_gray
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        small = cv2.resize(img, (160, 90))
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if _prev_stream_gray is None:
            _prev_stream_gray = small
            return 0.0
        diff = cv2.absdiff(_prev_stream_gray, small)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        _prev_stream_gray = small
        return (np.count_nonzero(thresh) / (160 * 90)) * 100.0
    except Exception:
        return 0.0

@app.post("/api/screen_cam/analyze")
async def analyze_screen_cam_frame(payload: ScreenCamFramePayload):
    """High-speed real-time stream ingestion with instant motion-triggered AI scanning for pheasants and rats."""
    import base64
    import time
    from datetime import datetime

    raw_data = payload.image_base64
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    now_time = time.time()
    now_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Update cache & active camera instantly
    if hasattr(ring_manager, "_phone_cam"):
        ring_manager._phone_cam._last_frame_bytes = image_bytes
        ring_manager._phone_cam._last_frame_time = now_time
        if payload.battery_percentage is not None:
            ring_manager._phone_cam.battery_life = payload.battery_percentage
            logger.info(f"🔋 Updated S21 battery level from live stream telemetry: {payload.battery_percentage}%")
        if ring_manager._active_camera != ring_manager._phone_cam:
            ring_manager._active_camera = ring_manager._phone_cam
            logger.info("Auto-switched active camera to Samsung Galaxy S21 Ultra from live stream.")
    sampler_engine._latest_snapshot_bytes = image_bytes

    # 2. Check instant motion & target zone material delta
    p = payload.polygon or inference_client.detection_polygon
    zone_info = sampler_engine.motion_pipeline.compute_zone_delta(image_bytes, polygon=p, delta_threshold=0.40)
    delta_pct = zone_info["delta_percent"]
    has_material_delta = zone_info["has_material_delta"]

    # 3. Instantly broadcast live video frame + zone delta telemetry (< 1ms)
    b64_thumb = base64.b64encode(image_bytes).decode("utf-8")
    asyncio.create_task(ws_broadcaster({
        "type": "live_video_frame",
        "data": {
            "image_base64": f"data:image/jpeg;base64,{b64_thumb}",
            "timestamp": now_dt_str,
            "device_name": payload.device_name or "Samsung Galaxy S21 Ultra",
            "motion_pct": delta_pct,
            "zone_delta_pct": delta_pct,
            "has_material_delta": has_material_delta
        }
    }))

    # 4. Trigger AI: ONLY when material delta occurs inside target zone (delta >= 0.4% with 2s cooldown)
    global _last_stream_ai_time, _is_analyzing_stream
    should_run_ai = not _is_analyzing_stream and has_material_delta and (now_time - _last_stream_ai_time >= 2.0)

    if should_run_ai:
        _is_analyzing_stream = True
        _last_stream_ai_time = now_time

        async def _run_async_ai(img_b, dev_name, poly, z_info):
            global _is_analyzing_stream
            try:
                ref_b = sampler_engine.motion_pipeline.get_reference_baseline()
                crop_b = z_info["focused_crop_bytes"]
                crop_bbox = z_info["crop_bbox"]
                is_subcrop = crop_bbox is not None

                # Run structured locked-temperature contrast inference against locked baseline
                res = await inference_client.analyze_image(
                    image_bytes=crop_b,
                    polygon=poly,
                    reference_image_bytes=ref_b,
                    is_subcrop=is_subcrop
                )

                # Map sub-crop bounding box back to full image if needed
                if res.is_detected and res.bounding_box and is_subcrop and crop_bbox:
                    orig_w, orig_h = z_info["original_shape"]
                    c_y1, c_x1, c_y2, c_x2 = crop_bbox
                    cw = max(1, c_x2 - c_x1)
                    ch = max(1, c_y2 - c_y1)

                    raw_box = res.bounding_box
                    b_ymin = raw_box[0] / 1000.0 if raw_box[0] > 1.0 else raw_box[0]
                    b_xmin = raw_box[1] / 1000.0 if raw_box[1] > 1.0 else raw_box[1]
                    b_ymax = raw_box[2] / 1000.0 if raw_box[2] > 1.0 else raw_box[2]
                    b_xmax = raw_box[3] / 1000.0 if raw_box[3] > 1.0 else raw_box[3]

                    full_ymin = int(((c_y1 + b_ymin * ch) / orig_h) * 1000.0)
                    full_xmin = int(((c_x1 + b_xmin * cw) / orig_w) * 1000.0)
                    full_ymax = int(((c_y1 + b_ymax * ch) / orig_h) * 1000.0)
                    full_xmax = int(((c_x1 + b_xmax * cw) / orig_w) * 1000.0)

                    res.bounding_box = [
                        max(0, min(1000, full_ymin)),
                        max(0, min(1000, full_xmin)),
                        max(0, min(1000, full_ymax)),
                        max(0, min(1000, full_xmax))
                    ]

                now_dt = datetime.now()
                detection_saved = None
                if res.is_detected:
                    record = storage_manager.save_detection(
                        image_bytes=img_b,
                        confidence=res.confidence,
                        description=res.description,
                        object_type=res.object_type,
                        label=res.label,
                        battery_percentage=100,
                        device_name=dev_name,
                        bounding_box=res.bounding_box,
                        dt=now_dt
                    )
                    detection_saved = record.model_dump()
                    asyncio.create_task(sampler_engine.notifier.notify_animal_detected(
                        confidence=res.confidence,
                        description=res.description,
                        device_name=dev_name,
                        object_type=res.object_type,
                        label=res.label,
                        timestamp=now_dt,
                        detection_id=record.id
                    ))

                await ws_broadcaster({
                    "type": "sample_completed",
                    "data": {
                        "sample_index": sampler_engine._sample_count + 1,
                        "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_health": {"battery_percentage": 50, "device_name": dev_name, "is_mock": False},
                        "battery_percentage": 50,
                        "detected": res.is_detected,
                        "rat_detected": res.is_detected,
                        "object_type": res.object_type,
                        "label": res.label,
                        "bounding_box": res.bounding_box,
                        "is_boosted": res.is_detected,
                        "current_interval_seconds": 5 if res.is_detected else 10,
                        "base_interval_seconds": 10,
                        "confidence": res.confidence,
                        "description": res.description,
                        "inference_time_ms": res.inference_time_ms,
                        "detection_saved": detection_saved,
                        "zone_delta_pct": delta_pct
                    }
                })
            except Exception as e:
                logger.error(f"Background AI stream error: {e}")
            finally:
                _is_analyzing_stream = False

        asyncio.create_task(_run_async_ai(image_bytes, payload.device_name or "Samsung Galaxy S21 Ultra", p, zone_info))

    return {
        "success": True,
        "streaming": True,
        "timestamp": now_dt_str,
        "motion_pct": delta_pct,
        "zone_delta_pct": delta_pct,
        "has_material_delta": has_material_delta
    }


class LockZonePayload(BaseModel):
    polygon: Optional[List[List[float]]] = None
    device_name: Optional[str] = "Samsung Galaxy S21 Ultra"

@app.post("/api/zone/lock_and_assess")
async def lock_and_assess_target_zone(payload: Optional[LockZonePayload] = None):
    """Takes a high-res snapshot of the target zone, locks it as reference baseline, and runs initial assessment."""
    frame = sampler_engine.get_latest_snapshot()
    if not frame and hasattr(ring_manager, "_phone_cam"):
        frame = ring_manager._phone_cam._last_frame_bytes
    if not frame:
        frame, _ = ring_manager.fetch_snapshot()

    if not frame:
        raise HTTPException(status_code=400, detail="No camera stream frame available to lock target zone.")

    poly = (payload.polygon if payload and payload.polygon else None) or inference_client.detection_polygon

    # 1. Extract and lock target zone baseline
    sampler_engine.motion_pipeline.save_reference_baseline(frame)
    crop_b, crop_bbox, shape = sampler_engine.motion_pipeline.extract_target_zone_crop(frame, polygon=poly)

    # 2. Run quick initial assessment with Gemma
    assessment_res = await inference_client.analyze_image(
        image_bytes=crop_b or frame,
        polygon=poly,
        reference_image_bytes=None,
        is_subcrop=(crop_bbox is not None)
    )

    b64_crop = base64.b64encode(crop_b or frame).decode("utf-8")

    # 3. Broadcast baseline locked event
    await ws_broadcaster({
        "type": "zone_baseline_locked",
        "data": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "locked",
            "detected": assessment_res.is_detected,
            "object_type": assessment_res.object_type,
            "label": assessment_res.label,
            "description": assessment_res.description,
            "thumbnail": f"data:image/jpeg;base64,{b64_crop}"
        }
    })

    return {
        "success": True,
        "message": "Target zone baseline locked and assessed.",
        "detected": assessment_res.is_detected,
        "object_type": assessment_res.object_type,
        "label": assessment_res.label,
        "description": assessment_res.description,
        "confidence": assessment_res.confidence,
        "crop_bbox": crop_bbox,
        "baseline_image_url": "/api/reference/image"
    }


@app.post("/api/reference/capture")
async def capture_reference_baseline():
    """Captures current camera snapshot as the negative baseline reference image for dual-image contrast."""
    frame = sampler_engine.get_latest_snapshot()
    if not frame and hasattr(ring_manager, "_phone_cam"):
        frame = ring_manager._phone_cam._last_frame_bytes
    if not frame:
        frame, _ = ring_manager.fetch_snapshot()

    if not frame:
        raise HTTPException(status_code=400, detail="No active camera frame available to capture as reference baseline.")

    success = sampler_engine.motion_pipeline.save_reference_baseline(frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed saving baseline reference image.")

    return {
        "success": True,
        "message": "Negative baseline reference image updated successfully.",
        "size_bytes": len(frame)
    }


@app.get("/api/reference/image")
async def get_reference_baseline_image():
    """Returns the negative baseline reference image."""
    ref_b = sampler_engine.motion_pipeline.get_reference_baseline()
    if not ref_b:
        raise HTTPException(status_code=404, detail="No reference baseline image found.")
    return Response(content=ref_b, media_type="image/jpeg")


def create_placeholder_zone_image(text: str, subtext: str = "") -> bytes:
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (18, 22, 30)
    cv2.rectangle(img, (8, 8), (472, 352), (40, 48, 65), 1)
    cv2.putText(img, text, (35, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 225, 235), 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(img, subtext, (35, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 170), 1, cv2.LINE_AA)
    ret, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return enc.tobytes()


@app.get("/api/camera/{camera_name}/zone_crop")
async def get_camera_zone_crop(camera_name: str):
    """Returns a high-resolution cropped JPEG containing exclusively the drawn target zone for the given camera."""
    target_cam = ring_manager.find_camera(camera_name)
    poly = inference_client.get_camera_polygon(camera_name, fallback=False)

    # 1. Fetch current frame for THIS specific camera
    frame = None
    if target_cam:
        is_ring = hasattr(target_cam, "family") and target_cam.family in ["stickup_cams", "doorbots", "doorbells"] and not getattr(target_cam, "is_local", False) and not getattr(target_cam, "is_phone", False)

        # 1. Ring cameras: actively query Ring snapshot endpoint for fresh live image
        if is_ring:
            try:
                snap, _, _, _ = await ring_manager.async_fetch_snapshot(camera_name=camera_name)
                if snap and len(snap) > 2000 and not is_blank_or_disabled_frame(snap):
                    frame = snap
            except Exception as e:
                logger.debug(f"Direct Ring snapshot fetch in zone_crop failed for {camera_name}: {e}")

        # 2. Local / Mobile cameras: try broadcaster / live stream
        if not frame:
            if hasattr(target_cam, "broadcaster") and target_cam.broadcaster:
                target_cam.broadcaster.start()
                f = target_cam.broadcaster.latest_frame
                if f and not is_blank_or_disabled_frame(f):
                    frame = f
            if not frame and hasattr(target_cam, "latest_frame") and target_cam.latest_frame:
                f = target_cam.latest_frame
                if f and not is_blank_or_disabled_frame(f):
                    frame = f
            if not frame and hasattr(target_cam, "async_get_snapshot"):
                try:
                    f = await target_cam.async_get_snapshot()
                    if f and not is_blank_or_disabled_frame(f):
                        frame = f
                except Exception:
                    pass

        # 3. If mobile camera is offline/blank, fall back to its assigned picture
        if not frame and hasattr(target_cam, "get_picture"):
            pic = target_cam.get_picture()
            if pic and not is_blank_or_disabled_frame(pic):
                frame = pic

    if not frame:
        f = ring_manager._snapshot_cache.get(camera_name)
        if f and not is_blank_or_disabled_frame(f):
            frame = f
        if not frame:
            for k, v in ring_manager._snapshot_cache.items():
                if (k.lower() == camera_name.lower() or _normalize_cam_key(k) == _normalize_cam_key(camera_name)) and not is_blank_or_disabled_frame(v):
                    frame = v
                    break

    if not frame and target_cam:
        try:
            snap, _, _, _ = await ring_manager.async_fetch_snapshot(camera_name=camera_name)
            if snap and not is_blank_or_disabled_frame(snap):
                frame = snap
        except Exception as e:
            logger.debug(f"async_fetch_snapshot failed for {camera_name}: {e}")

    no_cache_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    if not frame:
        placeholder = create_placeholder_zone_image(f"{camera_name}", "Standby / No Frame")
        return Response(content=placeholder, media_type="image/jpeg", headers=no_cache_headers)

    # 2. If NO polygon zone is drawn for this specific camera: show full camera view with guide banner
    if not poly or len(poly) < 3:
        try:
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                banner_h = max(34, int(h * 0.08))
                cv2.rectangle(img, (0, 0), (w, banner_h), (15, 23, 42), -1)
                cv2.putText(img, f"{camera_name}: No Zone Drawn (Click Select to Draw)", (16, int(banner_h * 0.65)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2, cv2.LINE_AA)
                success, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if success:
                    return Response(content=enc.tobytes(), media_type="image/jpeg", headers=no_cache_headers)
        except Exception:
            pass
        placeholder = create_placeholder_zone_image(f"{camera_name}", "No Zone Configured")
        return Response(content=placeholder, media_type="image/jpeg", headers=no_cache_headers)

    # 3. If polygon DOES exist for this camera: extract the exact zone crop using motion pipeline
    try:
        crop_bytes, bbox, shape = sampler_engine.motion_pipeline.extract_target_zone_crop(frame, polygon=poly)
        if crop_bytes:
            return Response(content=crop_bytes, media_type="image/jpeg", headers=no_cache_headers)
    except Exception as e:
        logger.error(f"Error extracting zone crop for {camera_name}: {e}")

    return Response(content=frame, media_type="image/jpeg", headers=no_cache_headers)


@app.get("/api/cameras/zone_summary")
async def get_cameras_zone_summary():
    """Returns all cameras with active zone configuration, delta scores, and crop endpoints."""
    cams_info = ring_manager.list_cameras()
    active_cam_name = ring_manager.camera_name
    results = []
    for c in cams_info:
        cam_name = c["name"]
        if "outhouse" in cam_name.lower():
            continue
        poly = inference_client.get_camera_polygon(cam_name, fallback=False)
        has_zone = poly is not None and len(poly) >= 3
        is_sel = (cam_name.lower() == (active_cam_name or "").lower())
        is_mobile = any(k in cam_name.lower() for k in ["s21", "tab", "galaxy", "phone", "tablet"])
        target_cam = ring_manager.find_camera(cam_name)
        is_streaming = False
        if target_cam:
            from src.ring_client import LocalRolandCamera, AndroidPhoneCamera, GalaxyTabWindowsCamera
            if isinstance(target_cam, GalaxyTabWindowsCamera):
                f = getattr(target_cam, "_last_frame_bytes", None)
                if not f and hasattr(target_cam, "broadcaster") and target_cam.broadcaster:
                    f = target_cam.broadcaster.latest_frame
                is_streaming = (f is not None and not is_blank_or_disabled_frame(f))
            elif isinstance(target_cam, LocalRolandCamera):
                is_streaming = True
            elif isinstance(target_cam, AndroidPhoneCamera):
                last_t = getattr(target_cam, "_last_frame_time", 0.0)
                is_streaming = (time.time() - last_t < 10.0) or bool(hasattr(target_cam, "broadcaster") and target_cam.broadcaster and target_cam.broadcaster.is_live)

        results.append({
            "name": cam_name,
            "has_zone": has_zone,
            "polygon": poly,
            "crop_url": f"/api/camera/{cam_name}/zone_crop",
            "picture_url": f"/api/camera/{cam_name}/picture",
            "is_active": is_sel,
            "is_online": True,
            "is_streaming": is_streaming,
            "uses_pictures": (is_mobile and not is_streaming),
            "battery_percentage": c.get("battery_percentage"),
            "delta_percent": 0.0
        })
    return {"cameras": results, "active_camera": active_cam_name}


@app.get("/api/camera/{camera_name}/picture")
async def get_camera_picture(camera_name: str):
    """Returns the static picture snapshot assigned to this camera."""
    target_cam = ring_manager.find_camera(camera_name)
    pic = None
    if target_cam and hasattr(target_cam, "get_picture"):
        pic = target_cam.get_picture()
    if not pic:
        pic = ring_manager._snapshot_cache.get(camera_name)
    if not pic:
        for p in [Path("data/s21_picture.jpg"), Path("data/tab_a11_picture.jpg"), Path("data/video_frames/frame_00s.jpg")]:
            if p.exists():
                pic = p.read_bytes()
                break
    if not pic:
        raise HTTPException(status_code=404, detail="No picture found for this camera")
    return Response(content=pic, media_type="image/jpeg", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate"
    })


@app.post("/api/camera/{camera_name}/upload_picture")
async def upload_camera_picture(camera_name: str, file: UploadFile = File(...)):
    """Uploads a fresh picture for a camera (especially mobile devices)."""
    content = await file.read()
    if not content or len(content) < 500:
        raise HTTPException(status_code=400, detail="Invalid picture file")
    target_cam = ring_manager.find_camera(camera_name)
    if target_cam and hasattr(target_cam, "set_picture"):
        target_cam.set_picture(content)
    ring_manager._snapshot_cache[camera_name] = content
    if target_cam and hasattr(target_cam, "name"):
        ring_manager._snapshot_cache[target_cam.name] = content
    logger.info(f"📸 Fresh picture uploaded for {camera_name} ({len(content)} bytes)")
    return {"success": True, "camera": camera_name, "bytes": len(content)}


class BacklogFolderPayload(BaseModel):
    folder_path: str
    device_name: Optional[str] = "Backlog Import"

@app.post("/api/backlog/upload")
async def process_backlog_upload(files: List[UploadFile] = File(...)):
    """Accepts a batch of backlog image files and analyzes each with Gemma sequentially."""
    from datetime import datetime
    import base64

    items = []
    for f in files:
        content = await f.read()
        if len(content) > 0:
            items.append({
                "bytes": content,
                "timestamp": datetime.now(),
                "device_name": f"Backlog: {f.filename}"
            })

    async def _progress_cb(current, total, result):
        # Broadcast progress via WebSockets
        b64_thumb = base64.b64encode(items[current - 1]["bytes"]).decode("utf-8")
        await sampler_engine._notify_subscribers("backlog_progress", {
            "current": current,
            "total": total,
            "latest_result": result,
            "latest_image_base64": f"data:image/jpeg;base64,{b64_thumb}"
        })

    result = await sampler_engine.process_batch_images(items, progress_callback=_progress_cb)
    return result

@app.post("/api/backlog/ring_history")
async def process_ring_history_backlog(limit: int = Query(10, ge=1, le=50)):
    """Fetches and evaluates the past N motion recordings from the connected Ring camera."""
    result = await sampler_engine.process_ring_history_backlog(limit=limit)
    return result

@app.post("/api/backlog/folder")
async def process_backlog_folder(payload: BacklogFolderPayload):
    """Processes all images found in a specified local folder."""
    import os
    from datetime import datetime

    folder = Path(payload.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder not found or is not a directory")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_exts])

    if not image_files:
        return {"total_processed": 0, "positive_detections": 0, "message": "No images found in folder"}

    items = []
    for p in image_files:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(p))
            with open(p, "rb") as f:
                data = f.read()
            items.append({
                "bytes": data,
                "timestamp": mtime,
                "device_name": f"{payload.device_name}: {p.name}"
            })
        except Exception as e:
            logger.warning(f"Failed reading backlog image {p}: {e}")

    result = await sampler_engine.process_batch_images(items)
    return result

class SettingsPayload(BaseModel):
    sample_interval_seconds: Optional[int] = None
    endpoint_url: Optional[str] = None
    endpoint_type: Optional[str] = None
    model_name: Optional[str] = None
    confidence_threshold: Optional[float] = None
    ntfy_topic: Optional[str] = None
    enable_notifications: Optional[bool] = None
    voice_alert: Optional[bool] = None
    webhook_url: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    detection_polygon: Optional[List[List[float]]] = None
    camera_polygons: Optional[Dict[str, Any]] = None
    phone_camera_url: Optional[str] = None
    target_object: Optional[str] = None
    cooldown_seconds: Optional[int] = None
    alert_cooldown_seconds: Optional[int] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None

@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    """Updates runtime settings."""
    if payload.cooldown_seconds is not None:
        sampler_engine.notifier.cooldown_seconds = max(5, payload.cooldown_seconds)
    if payload.alert_cooldown_seconds is not None:
        sampler_engine.notifier.cooldown_seconds = max(5, payload.alert_cooldown_seconds)
    if payload.sample_interval_seconds is not None:
        sampler_engine.interval_seconds = max(5, payload.sample_interval_seconds)
    if payload.endpoint_url is not None:
        inference_client.endpoint_url = payload.endpoint_url.rstrip("/")
    if payload.endpoint_type is not None:
        inference_client.endpoint_type = payload.endpoint_type.lower()
    if payload.model_name is not None:
        inference_client.model_name = payload.model_name
    if payload.confidence_threshold is not None:
        inference_client.confidence_threshold = payload.confidence_threshold
    if payload.target_object is not None:
        inference_client.target_object = payload.target_object.lower().strip()
    if payload.gemini_api_key is not None:
        inference_client.gemini_api_key = payload.gemini_api_key.strip()
    if payload.gemini_model is not None:
        inference_client.gemini_model = payload.gemini_model.strip()
    if payload.ntfy_topic is not None:
        sampler_engine.notifier.ntfy_topic = payload.ntfy_topic.strip()
    if payload.enable_notifications is not None:
        sampler_engine.notifier.enabled = payload.enable_notifications
    if payload.voice_alert is not None:
        sampler_engine.notifier.voice_alert = payload.voice_alert
    if payload.webhook_url is not None:
        sampler_engine.notifier.webhook_url = payload.webhook_url.strip() or None
    if payload.telegram_bot_token is not None:
        sampler_engine.notifier.telegram_bot_token = payload.telegram_bot_token.strip() or None
    if payload.telegram_chat_id is not None:
        sampler_engine.notifier.telegram_chat_id = payload.telegram_chat_id.strip() or None
    if payload.camera_polygons is not None:
        inference_client.camera_polygons.update(payload.camera_polygons)
    if payload.detection_polygon is not None:
        inference_client.detection_polygon = payload.detection_polygon if len(payload.detection_polygon) >= 3 else None
        if ring_manager.camera_name:
            inference_client.camera_polygons[ring_manager.camera_name] = inference_client.detection_polygon
    if payload.phone_camera_url is not None:
        url = payload.phone_camera_url.strip()
        if hasattr(ring_manager, "_phone_cam"):
            ring_manager._phone_cam.stream_url = url

    # Persist changes to config.yaml
    try:
        config_file = Path("config.yaml")
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        else:
            raw_cfg = {}

        if "inference" not in raw_cfg: raw_cfg["inference"] = {}
        if "ring" not in raw_cfg: raw_cfg["ring"] = {}

        raw_cfg["inference"]["detection_polygon"] = inference_client.detection_polygon
        raw_cfg["inference"]["camera_polygons"] = inference_client.camera_polygons
        raw_cfg["inference"]["confidence_threshold"] = inference_client.confidence_threshold
        raw_cfg["inference"]["endpoint_url"] = inference_client.endpoint_url
        raw_cfg["inference"]["endpoint_type"] = inference_client.endpoint_type
        raw_cfg["inference"]["model_name"] = inference_client.model_name
        raw_cfg["inference"]["target_object"] = inference_client.target_object
        raw_cfg["inference"]["gemini_api_key"] = getattr(inference_client, "gemini_api_key", "")
        raw_cfg["inference"]["gemini_model"] = getattr(inference_client, "gemini_model", "gemini-3.7-flash")
        raw_cfg["ring"]["sample_interval_seconds"] = sampler_engine.interval_seconds
        if hasattr(ring_manager, "_phone_cam"):
            raw_cfg["ring"]["phone_camera_url"] = ring_manager._phone_cam.stream_url

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw_cfg, f, default_flow_style=False)
        logger.info("Saved settings and target_object to config.yaml")
    except Exception as e:
        logger.warning(f"Could not persist settings to config.yaml: {e}")

    return {
        "success": True,
        "current_settings": {
            "sample_interval_seconds": sampler_engine.interval_seconds,
            "endpoint_url": inference_client.endpoint_url,
            "endpoint_type": inference_client.endpoint_type,
            "model_name": inference_client.model_name,
            "confidence_threshold": inference_client.confidence_threshold,
            "target_object": inference_client.target_object,
            "gemini_api_key": getattr(inference_client, "gemini_api_key", ""),
            "gemini_model": getattr(inference_client, "gemini_model", "gemini-3.7-flash"),
            "ntfy_topic": sampler_engine.notifier.ntfy_topic,
            "enable_notifications": sampler_engine.notifier.enabled,
            "voice_alert": sampler_engine.notifier.voice_alert,
            "webhook_url": sampler_engine.notifier.webhook_url,
            "telegram_bot_token": sampler_engine.notifier.telegram_bot_token,
            "telegram_chat_id": sampler_engine.notifier.telegram_chat_id,
            "detection_polygon": inference_client.detection_polygon,
            "camera_polygons": inference_client.camera_polygons,
            "active_camera": ring_manager.camera_name,
            "cooldown_seconds": sampler_engine.notifier.cooldown_seconds,
            "alert_cooldown_seconds": sampler_engine.notifier.cooldown_seconds,
            "phone_camera_url": getattr(ring_manager._phone_cam, "stream_url", "") if hasattr(ring_manager, "_phone_cam") else ""
        }
    }

@app.get("/api/settings")
async def get_settings():
    return {
        "sample_interval_seconds": sampler_engine.interval_seconds,
        "endpoint_url": inference_client.endpoint_url,
        "endpoint_type": inference_client.endpoint_type,
        "model_name": inference_client.model_name,
        "confidence_threshold": inference_client.confidence_threshold,
        "target_object": inference_client.target_object,
        "gemini_api_key": getattr(inference_client, "gemini_api_key", ""),
        "gemini_model": getattr(inference_client, "gemini_model", "gemini-3.7-flash"),
        "ntfy_topic": sampler_engine.notifier.ntfy_topic,
        "enable_notifications": sampler_engine.notifier.enabled,
        "voice_alert": sampler_engine.notifier.voice_alert,
        "webhook_url": sampler_engine.notifier.webhook_url,
        "telegram_bot_token": sampler_engine.notifier.telegram_bot_token,
        "telegram_chat_id": sampler_engine.notifier.telegram_chat_id,
        "cooldown_seconds": sampler_engine.notifier.cooldown_seconds,
        "alert_cooldown_seconds": sampler_engine.notifier.cooldown_seconds,
        "detection_polygon": inference_client.detection_polygon,
        "camera_polygons": inference_client.camera_polygons,
        "camera_name": ring_manager.camera_name,
        "active_camera": ring_manager.camera_name,
        "is_mock": ring_manager.is_mock,
        "phone_camera_url": getattr(ring_manager._phone_cam, "stream_url", "") if hasattr(ring_manager, "_phone_cam") else ""
    }

@app.get("/api/spark/health")
async def get_spark_health():
    """Checks DGX Spark validator health."""
    for host in ["http://spark.local:8088", "http://127.0.0.1:8088", "http://localhost:8088"]:
        try:
            async with httpx.AsyncClient(timeout=0.8) as client:
                res = await client.get(f"{host}/health")
                if res.status_code == 200:
                    data = res.json()
                    data["spark_host"] = host
                    return data
        except Exception:
            continue
    return {
        "ok": False,
        "model": "gemma4-26b",
        "warm": False,
        "error": "DGX Spark unreachable on LAN (spark.local:8088)"
    }

@app.post("/bridge-alert")
async def receive_bridge_alert(payload: Dict[str, Any]):
    """Receives fallback watch alerts from Spark when watch direct Wi-Fi is unreachable."""
    logger.info(f"📱 S21 Phone Bridge received alert from Spark: {payload.get('watch_text')}")
    # Forward notification via local Windows / notification engine
    notifier.notify_animal_detected(
        image_bytes=b"",
        confidence=payload.get("confidence", 0.9),
        device_name="Samsung S21 (Bridge)",
        description=payload.get("location", "Garden run"),
        object_type="rodent",
        label="Rat"
    )
    return {"ok": True, "bridged": True}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "status": sampler_engine.get_status()
        })
        while True:
            # Keep connection alive & accept incoming ping/commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

def run():
    cert_file = Path("certs/cert.pem")
    key_file = Path("certs/key.pem")
    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file)
        }
        logger.info("Starting server with HTTPS / SSL enabled for mobile camera access.")
    uvicorn.run(
        "src.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        **ssl_kwargs
    )

if __name__ == "__main__":
    run()
