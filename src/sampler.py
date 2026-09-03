import asyncio
import base64
import inspect
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from src.ring_client import RingManager, LocalRolandCamera, MockRingCamera
from src.inference_client import RolandInferenceClient, DetectionResult
from src.storage import StorageManager, DetectionRecord
from src.notifier import WatchNotificationManager
from src.motion_pipeline import MotionCascadePipeline
from src.fast_detector import FastObjectDetector

logger = logging.getLogger("sampler")

class SamplerEngine:
    """Orchestrates periodic snapshot sampling with fast object detection and multimodal Gemma verification."""

    def __init__(
        self,
        ring_manager: RingManager,
        inference_client: RolandInferenceClient,
        storage_manager: StorageManager,
        notifier: Optional[WatchNotificationManager] = None,
        interval_seconds: int = 10,
        active_detection_interval_seconds: int = 1
    ):
        self.ring = ring_manager
        self.inference = inference_client
        self.storage = storage_manager
        self.notifier = notifier or WatchNotificationManager()
        self.motion_pipeline = MotionCascadePipeline()
        self.fast_detector = FastObjectDetector()
        self.base_interval_seconds = interval_seconds
        self.active_detection_interval_seconds = active_detection_interval_seconds
        self.current_interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._latest_snapshot_bytes: Optional[bytes] = None
        self._latest_snapshot_time: Optional[str] = None
        self._latest_inference_result: Optional[DetectionResult] = None
        self._sample_count: int = 0
        self._last_battery_level: Optional[int] = None
        self._is_rat_active: bool = False
        # Event Session Grouping State (records continuous sighting as a single event)
        self._active_event_id: Optional[str] = None
        self._active_event_object: Optional[str] = None
        self._active_event_camera: Optional[str] = None
        self._active_event_last_seen: float = 0.0
        self._active_event_frame_idx: int = 0
        self.event_inactivity_timeout_seconds: float = 8.0
        self._ai_task_running: bool = False
        self._last_ai_dispatch: float = 0.0
        self._determination_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._identification_task: Optional[asyncio.Task] = None
        self._determination_task: Optional[asyncio.Task] = None
        self._subscribers: List[Callable[[Dict[str, Any]], Any]] = []

    @property
    def interval_seconds(self) -> int:
        return self.current_interval_seconds

    @interval_seconds.setter
    def interval_seconds(self, value: int):
        self.base_interval_seconds = max(5, value)
        if not self._is_rat_active:
            self.current_interval_seconds = self.base_interval_seconds

    @property
    def is_running(self) -> bool:
        return self._running

    def register_subscriber(self, callback: Callable[[Dict[str, Any]], Any]):
        """Registers a callback for realtime telemetry / detection events."""
        self._subscribers.append(callback)

    def unregister_subscriber(self, callback: Callable[[Dict[str, Any]], Any]):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        msg = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        for sub in list(self._subscribers):
            try:
                try:
                    res = sub(msg)
                except TypeError:
                    res = sub(event_type, data)
                if inspect.iscoroutine(res):
                    await res
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

    async def sample_once(self) -> Dict[str, Any]:
        """Executes a single sampling cycle with OpenCV background subtraction and high-res sub-crop inference."""
        self._sample_count += 1
        start_ts = datetime.now()
        timestamp_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Fetch snapshot and battery status
        health = self.ring.get_health_status()
        self._last_battery_level = health.get("battery_percentage")

        snapshot_bytes, error, is_standby, is_new = await self.ring.async_fetch_snapshot()
        if error or not snapshot_bytes:
            logger.error(f"Failed to fetch snapshot: {error}")
            return {
                "success": False,
                "error": error or "Empty snapshot",
                "timestamp": timestamp_str,
                "battery": health
            }

        self._latest_snapshot_bytes = snapshot_bytes
        self._latest_snapshot_time = timestamp_str

        detection_saved = None

        if is_standby:
            inference_result = DetectionResult(
                detected=False,
                is_detected=False,
                is_rat_detected=False,
                subject_type="none",
                object_type="none",
                label="None",
                confidence=0.0,
                description="Camera Standby / Recharging",
                inference_time_ms=0.0
            )
            self._latest_inference_result = inference_result
            self._is_rat_active = False
            self.current_interval_seconds = self.base_interval_seconds
        elif not is_new and not self.ring._is_mock and not isinstance(self.ring._active_camera, LocalRolandCamera):
            # Same static event file from before: skip redundant inference
            inference_result = DetectionResult(
                detected=False,
                is_detected=False,
                is_rat_detected=False,
                subject_type="none",
                object_type="none",
                label="None",
                confidence=0.0,
                description="Monitoring (Awaiting fresh motion event)",
                inference_time_ms=0.0
            )
            self._latest_inference_result = inference_result
            self._is_rat_active = False
            self.current_interval_seconds = self.base_interval_seconds
        else:
            # 1. Fast Detector: Eyes that never blink (finds candidate objects, draws boxes, extracts crops)
            fast_result = self.fast_detector.detect_boxes(
                snapshot_bytes,
                polygon=self.inference.detection_polygon
            )
            has_candidates = fast_result["has_candidates"]
            primary_box = fast_result["primary_box"]
            primary_crop = fast_result["primary_crop_bytes"]

            # 2. Zone delta check for motion activity
            zone_info = self.motion_pipeline.compute_zone_delta(
                snapshot_bytes,
                polygon=self.inference.detection_polygon,
                delta_threshold=0.35
            )
            has_animal_candidate = fast_result["has_candidates"] and fast_result.get("is_animal", False)
            has_delta = zone_info["has_material_delta"] or has_animal_candidate
            delta_pct = zone_info["delta_percent"]

            # ONLY bound an object when it could be an animal
            object_boundary = primary_box if has_animal_candidate else None

            ref_bytes = self.motion_pipeline.get_reference_baseline()
            if not ref_bytes and not self.ring._is_mock:
                self.motion_pipeline.save_reference_baseline(snapshot_bytes)
                ref_bytes = snapshot_bytes

            # Crop sent to Gemma: tight bounding box crop from the fast detector or zone crop
            crop_bytes = primary_crop or zone_info["focused_crop_bytes"]
            crop_bbox = zone_info["crop_bbox"]
            is_subcrop = primary_crop is not None or crop_bbox is not None

            # --- STAGE 1: IMMEDIATE TRANSITION TO REAL-TIME & OBJECT BOUNDARY EMISSION ---
            if has_delta:
                was_idle = not self._is_rat_active
                self._is_rat_active = True
                self.current_interval_seconds = self.active_detection_interval_seconds
                self._active_event_last_seen = start_ts.timestamp()

                if was_idle:
                    logger.info(f"⚡ Activity in target zone ({delta_pct:.1f}% delta) -> Accelerating sampling to REAL-TIME ({self.active_detection_interval_seconds}s)")

                # Notify clients: only send object_boundary when it could be an animal
                await self._notify_subscribers("object_detected", {
                    "device_name": self.ring.camera_name,
                    "timestamp": timestamp_str,
                    "object_boundary": object_boundary,
                    "is_animal": has_animal_candidate,
                    "delta_percent": delta_pct,
                    "sampling_cadence": "realtime",
                    "interval_seconds": self.current_interval_seconds,
                    "status_text": f"⚡ Activity in Zone ({delta_pct:.1f}% delta) — Real-Time Active" if not object_boundary else "⚡ Potential Animal in Zone"
                })

            # --- STAGE 2: BRING AI VISION ENGINE INTO PLAY (NON-BLOCKING ASYNC CASCADE) ---
            if has_delta:
                # Dispatch focused high-res crop to Gemini / Gemma Vision Engine asynchronously without blocking fast detector
                now_t = time.time()
                if not self._ai_task_running and (now_t - self._last_ai_dispatch >= 1.5):
                    self._last_ai_dispatch = now_t
                    asyncio.create_task(self._run_async_ai_inference(
                        crop_bytes=crop_bytes,
                        ref_bytes=ref_bytes,
                        is_subcrop=is_subcrop,
                        crop_bbox=crop_bbox,
                        zone_info=zone_info,
                        object_boundary=object_boundary,
                        snapshot_bytes=snapshot_bytes,
                        start_ts=start_ts,
                        timestamp_str=timestamp_str,
                        health=health,
                        delta_pct=delta_pct
                    ))
            else:
                # Scene quiet: check inactivity timeout
                if self._is_rat_active:
                    now_ts = start_ts.timestamp()
                    if (now_ts - self._active_event_last_seen) >= self.event_inactivity_timeout_seconds:
                        logger.info(f"Target zone quiet for {self.event_inactivity_timeout_seconds}s. Sampling rate REVERTED back to stated interval ({self.base_interval_seconds}s).")
                        self._is_rat_active = False
                        self._active_event_id = None
                        self.current_interval_seconds = self.base_interval_seconds
                        await self._notify_subscribers("cadence_changed", {
                            "sampling_cadence": "idle",
                            "interval_seconds": self.base_interval_seconds,
                            "status_text": f"Monitoring (Sampling every {self.base_interval_seconds}s)"
                        })

        return {
            "sample_index": self._sample_count,
            "timestamp": timestamp_str,
            "device_health": health,
            "battery_percentage": self._last_battery_level,
            "is_boosted": self._is_rat_active,
            "current_interval_seconds": self.current_interval_seconds,
            "base_interval_seconds": self.base_interval_seconds
        }

    async def _run_async_ai_inference(
        self,
        crop_bytes: bytes,
        ref_bytes: Optional[bytes],
        is_subcrop: bool,
        crop_bbox: Optional[Tuple[int, int, int, int]],
        zone_info: Dict[str, Any],
        object_boundary: Optional[List[int]],
        snapshot_bytes: bytes,
        start_ts: datetime,
        timestamp_str: str,
        health: Dict[str, Any],
        delta_pct: float
    ):
        """Asynchronously calls Gemini / Gemma multimodal AI, saves detection, and updates UI."""
        self._ai_task_running = True
        try:
            inference_result = await self.inference.analyze_image(
                image_bytes=crop_bytes,
                polygon=self.inference.detection_polygon,
                reference_image_bytes=ref_bytes,
                is_subcrop=is_subcrop
            )

            # Map sub-crop bounding box to full-frame normalized coordinates [0-1000]
            if inference_result.is_detected and inference_result.bounding_box and is_subcrop and crop_bbox:
                orig_w, orig_h = zone_info["original_shape"]
                c_y1, c_x1, c_y2, c_x2 = crop_bbox
                cw = max(1, c_x2 - c_x1)
                ch = max(1, c_y2 - c_y1)

                raw_box = inference_result.bounding_box
                if raw_box and len(raw_box) >= 4:
                    b_ymin = raw_box[0] / 1000.0 if raw_box[0] > 1.0 else raw_box[0]
                    b_xmin = raw_box[1] / 1000.0 if raw_box[1] > 1.0 else raw_box[1]
                    b_ymax = raw_box[2] / 1000.0 if raw_box[2] > 1.0 else raw_box[2]
                    b_xmax = raw_box[3] / 1000.0 if raw_box[3] > 1.0 else raw_box[3]

                    full_ymin = int(((c_y1 + b_ymin * ch) / orig_h) * 1000.0)
                    full_xmin = int(((c_x1 + b_xmin * cw) / orig_w) * 1000.0)
                    full_ymax = int(((c_y1 + b_ymax * ch) / orig_h) * 1000.0)
                    full_xmax = int(((c_x1 + b_xmax * cw) / orig_w) * 1000.0)

                    inference_result.bounding_box = [
                        max(0, min(1000, full_ymin)),
                        max(0, min(1000, full_xmin)),
                        max(0, min(1000, full_ymax)),
                        max(0, min(1000, full_xmax))
                    ]
            # --- STRICT 85%+ ANIMAL DETERMINATION GATE ---
            # Only confirm animal and persist/alert if:
            # 1. Gemma/Gemini confirmed an animal (not foliage, shadow, tree, ground, or clutter)
            # 2. Confidence score is >= 85% (0.85)
            min_conf_threshold = 0.85
            non_animal_types = {"none", "clutter", "shadow", "foliage", "tree", "plant", "ground", "grass"}
            is_valid_animal = (
                inference_result.is_detected
                and inference_result.object_type.lower() not in non_animal_types
                and (inference_result.confidence >= min_conf_threshold)
            )

            if not is_valid_animal:
                # Discard low-confidence or non-animal detections (no alerts, no DB records)
                inference_result.is_detected = False
                inference_result.is_rat_detected = False
                inference_result.bounding_box = None
            elif not inference_result.bounding_box and object_boundary:
                inference_result.bounding_box = object_boundary

            self._latest_inference_result = inference_result
            detection_saved = None

            # Persist detection & manage alerts
            if inference_result.is_detected:
                import uuid
                now_ts = start_ts.timestamp()
                self._active_event_last_seen = now_ts
                self._is_rat_active = True
                self.current_interval_seconds = self.active_detection_interval_seconds

                same_cam = (self._active_event_camera or "").lower() == (self.ring.camera_name or "").lower()
                same_obj = (self._active_event_object or "").lower() == inference_result.object_type.lower()
                session_valid = (
                    self._active_event_id is not None
                    and same_cam
                    and same_obj
                    and (now_ts - self._active_event_last_seen) < self.event_inactivity_timeout_seconds
                )

                if not session_valid:
                    self._active_event_id = f"evt_{int(now_ts)}_{uuid.uuid4().hex[:6]}"
                    self._active_event_object = inference_result.object_type
                    self._active_event_camera = self.ring.camera_name
                    self._active_event_frame_idx = 1
                    is_new_event = True
                else:
                    self._active_event_frame_idx += 1
                    is_new_event = False

                obj_emojis = {
                    "tree": "🌲", "bird": "🐦", "pheasant": "🦚",
                    "rodent": "🐀", "rat": "🐀", "horse": "🐴", "horses_poo": "🐴💩",
                    "dog": "🐕", "cat": "🐈", "other_animal": "🐾", "animal": "🐾",
                    "fox": "🦊", "badger": "🦡", "deer": "🦌"
                }
                emoji = obj_emojis.get(inference_result.object_type, "🎯")

                record = self.storage.save_detection(
                    image_bytes=snapshot_bytes,
                    confidence=inference_result.confidence,
                    description=inference_result.description,
                    object_type=inference_result.object_type,
                    label=inference_result.label,
                    battery_percentage=self._last_battery_level,
                    device_name=self.ring.camera_name,
                    bounding_box=inference_result.bounding_box or object_boundary,
                    dt=start_ts,
                    event_id=self._active_event_id,
                    frame_index=self._active_event_frame_idx
                )
                detection_saved = record.model_dump()
                detection_saved["event_id"] = self._active_event_id
                detection_saved["frame_index"] = self._active_event_frame_idx

                if is_new_event:
                    logger.info(f"{emoji} {inference_result.label.upper()} IDENTIFIED BY AI! (Event #{self._active_event_id})")
                    asyncio.create_task(self.notifier.notify_animal_detected(
                        confidence=inference_result.confidence,
                        description=inference_result.description,
                        device_name=self.ring.camera_name,
                        object_type=inference_result.object_type,
                        label=inference_result.label,
                        timestamp=start_ts,
                        detection_id=record.id
                    ))
                else:
                    logger.info(f"{emoji} Attached frame #{self._active_event_frame_idx} to ongoing event {self._active_event_id}")

            # Notify UI subscribers with AI identification result
            b64_thumb = base64.b64encode(snapshot_bytes).decode("utf-8")
            status_payload = {
                "sample_index": self._sample_count,
                "timestamp": timestamp_str,
                "device_health": health,
                "battery_percentage": self._last_battery_level,
                "detected": inference_result.is_detected,
                "rat_detected": inference_result.is_detected,
                "object_type": inference_result.object_type,
                "label": inference_result.label,
                "bounding_box": inference_result.bounding_box,
                "is_boosted": self._is_rat_active,
                "current_interval_seconds": self.current_interval_seconds,
                "base_interval_seconds": self.base_interval_seconds,
                "confidence": inference_result.confidence,
                "description": inference_result.description,
                "inference_time_ms": inference_result.inference_time_ms,
                "detection_saved": detection_saved,
                "latest_image_base64": f"data:image/jpeg;base64,{b64_thumb}"
            }
            await self._notify_subscribers("sample_completed", status_payload)
        except Exception as e:
            logger.error(f"Error in async AI inference task: {e}")
        finally:
            self._ai_task_running = False

    async def _identification_worker(self):
        """Agent 2: Object Identification Agent ('The Eyes') - Runs at 20 FPS locally without blocking."""
        logger.info("Agent 2 (Object Identification Agent) started at 20 FPS.")
        last_frame_bytes = None
        while self._running:
            try:
                # Pull latest frame from lockless camera buffer
                active_cam = self.ring._active_camera
                frame_bytes = None
                if active_cam and hasattr(active_cam, "broadcaster") and active_cam.broadcaster:
                    frame_bytes = active_cam.broadcaster.latest_frame
                if not frame_bytes and active_cam:
                    frame_bytes = getattr(active_cam, "_last_frame_bytes", None)
                if not frame_bytes:
                    frame_bytes = self._latest_snapshot_bytes

                if frame_bytes and frame_bytes != last_frame_bytes:
                    last_frame_bytes = frame_bytes
                    self._latest_snapshot_bytes = frame_bytes
                    poly = self.inference.detection_polygon

                    # 1. Instant zone delta check against locked baseline (~1ms)
                    zone_info = self.motion_pipeline.compute_zone_delta(frame_bytes, polygon=poly)
                    delta_pct = zone_info.get("delta_percent", 0.0)
                    has_material_delta = zone_info.get("has_material_delta", False)

                    # Only wake up Object Identification Agent if zone has changed from locked baseline
                    if not has_material_delta and not self._is_rat_active:
                        await asyncio.sleep(0.05)
                        continue

                    # 2. Fast YOLO animal detector (~15ms)
                    detector_res = self.fast_detector.detect_boxes(frame_bytes, polygon=poly)
                    object_boundary = detector_res.get("primary_box")
                    has_animal_candidate = detector_res.get("is_animal", False)
                    has_delta = has_material_delta or has_animal_candidate

                    if object_boundary and has_animal_candidate:
                        # Immediately emit real-time bounding box to browser WebSocket (< 1ms)
                        await self._notify_subscribers("object_detected", {
                            "object_boundary": object_boundary,
                            "is_animal": True,
                            "delta_percent": delta_pct,
                            "sampling_cadence": "realtime",
                            "interval_seconds": 1,
                            "status_text": "⚡ Animal Tracked in Zone — Real-Time"
                        })

                        # Enqueue high-res crop for Agent 3 (Determination Agent) if cooldown has passed
                        now_t = time.time()
                        if (now_t - self._last_ai_dispatch >= 1.5) and self._determination_queue.qsize() < 2:
                            self._last_ai_dispatch = now_t
                            crop_bytes = zone_info.get("focused_crop_bytes", frame_bytes)
                            crop_bbox = zone_info.get("crop_bbox")
                            ref_bytes = self.motion_pipeline.get_reference_baseline()
                            await self._determination_queue.put({
                                "crop_bytes": crop_bytes,
                                "crop_bbox": crop_bbox,
                                "ref_bytes": ref_bytes,
                                "zone_info": zone_info,
                                "object_boundary": object_boundary,
                                "snapshot_bytes": frame_bytes,
                                "start_ts": datetime.now(),
                                "timestamp_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "delta_pct": delta_pct,
                                "health": self.ring.get_health_status()
                            })
                    elif not has_delta and self._is_rat_active:
                        now_ts = time.time()
                        if (now_ts - self._active_event_last_seen) >= self.event_inactivity_timeout_seconds:
                            self._is_rat_active = False
                            self._active_event_id = None
                            await self._notify_subscribers("cadence_changed", {
                                "sampling_cadence": "idle",
                                "interval_seconds": self.base_interval_seconds,
                                "status_text": f"Monitoring (Sampling every {self.base_interval_seconds}s)"
                            })
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Identification worker tick error: {e}")

            await asyncio.sleep(0.05)  # 20 FPS

    async def _determination_worker(self):
        """Agent 3: Object Determination Agent ('The Brain') - Multimodal reasoning & species verification."""
        logger.info("Agent 3 (Object Determination Agent) queue worker started.")
        while self._running:
            try:
                job = await self._determination_queue.get()
                await self._run_async_ai_inference(
                    crop_bytes=job["crop_bytes"],
                    ref_bytes=job["ref_bytes"],
                    is_subcrop=job["crop_bbox"] is not None,
                    crop_bbox=job["crop_bbox"],
                    zone_info=job["zone_info"],
                    object_boundary=job["object_boundary"],
                    snapshot_bytes=job["snapshot_bytes"],
                    start_ts=job["start_ts"],
                    timestamp_str=job["timestamp_str"],
                    health=job["health"],
                    delta_pct=job["delta_pct"]
                )
                self._determination_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Determination worker error: {e}")

    async def _loop(self):
        logger.info(f"Starting adaptive sampling engine (Baseline: {self.base_interval_seconds}s, Active Target: {self.active_detection_interval_seconds}s)")
        while self._running:
            try:
                # Ring cameras / battery devices check snapshots periodically
                is_phone = "s21" in self.ring.camera_name.lower() or "phone" in self.ring.camera_name.lower()
                if not is_phone:
                    await self.sample_once()
            except Exception as e:
                logger.error(f"Error during sampling tick: {e}", exc_info=True)

            try:
                await asyncio.sleep(self.current_interval_seconds)
            except asyncio.CancelledError:
                break

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())
            self._identification_task = asyncio.create_task(self._identification_worker())
            self._determination_task = asyncio.create_task(self._determination_worker())
            logger.info("3-Agent vision pipeline running (VideoStreamAgent, ObjectIdentificationAgent, ObjectDeterminationAgent)")

    def stop(self):
        if self._running:
            self._running = False
            for t in [self._task, self._identification_task, self._determination_task]:
                if t:
                    t.cancel()
            self._task = None
            self._identification_task = None
            self._determination_task = None
            logger.info("Sampling engine stopped.")

    def get_status(self) -> Dict[str, Any]:
        health = self.ring.get_health_status()
        return {
            "running": self._running,
            "interval_seconds": self.current_interval_seconds,
            "base_interval_seconds": self.base_interval_seconds,
            "active_detection_interval_seconds": self.active_detection_interval_seconds,
            "is_boosted": self._is_rat_active,
            "sample_count": self._sample_count,
            "latest_sample_time": self._latest_snapshot_time,
            "device_health": health,
            "battery_percentage": health.get("battery_percentage"),
            "target_object": self.inference.target_object,
            "inference_engine": {
                "endpoint_type": self.inference.endpoint_type,
                "model_name": self.inference.model_name,
                "gemini_model": getattr(self.inference, "gemini_model", "gemini-3.7-flash")
            },
            "latest_inference": (
                self._latest_inference_result.model_dump() if self._latest_inference_result else None
            ),
            "storage_stats": self.storage.get_stats()
        }

    async def process_custom_image(
        self,
        image_bytes: bytes,
        timestamp: Optional[datetime] = None,
        device_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyzes an individual backlog image with Gemma and persists if positive."""
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        dev_name = device_name or self.ring.camera_name

        inference_result = await self.inference.analyze_image(image_bytes)
        detection_saved = None

        if inference_result.is_detected:
            emoji = "🦚" if inference_result.object_type == "pheasant" else "🐀"
            logger.info(f"{emoji} {inference_result.label.upper()} DETECTED in backlog image ({ts_str})! Confidence: {inference_result.confidence:.2f}")
            record = self.storage.save_detection(
                image_bytes=image_bytes,
                confidence=inference_result.confidence,
                description=inference_result.description,
                object_type=inference_result.object_type,
                label=inference_result.label,
                battery_percentage=self.ring.get_battery_level(),
                device_name=dev_name,
                bounding_box=inference_result.bounding_box,
                dt=ts
            )
            detection_saved = record.model_dump()

        return {
            "timestamp": ts_str,
            "device_name": dev_name,
            "detected": inference_result.is_detected,
            "rat_detected": inference_result.is_detected,
            "object_type": inference_result.object_type,
            "label": inference_result.label,
            "bounding_box": inference_result.bounding_box,
            "confidence": inference_result.confidence,
            "description": inference_result.description,
            "inference_time_ms": inference_result.inference_time_ms,
            "detection_saved": detection_saved
        }

    async def process_batch_images(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """Processes a list of backlog image items sequentially."""
        total = len(items)
        results = []
        positive_count = 0

        for i, item in enumerate(items):
            img_bytes = item["bytes"]
            ts = item.get("timestamp")
            dev = item.get("device_name")

            res = await self.process_custom_image(img_bytes, timestamp=ts, device_name=dev)
            if res["rat_detected"]:
                positive_count += 1
            results.append(res)

            if progress_callback:
                try:
                    cb_res = progress_callback(i + 1, total, res)
                    if inspect.iscoroutine(cb_res):
                        await cb_res
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        return {
            "total_processed": total,
            "positive_detections": positive_count,
            "results": results
        }

    async def process_ring_history_backlog(self, limit: int = 10) -> Dict[str, Any]:
        """Scans the past N Ring motion recordings, downloads each frame, and evaluates backlog."""
        if not self.ring._active_camera or self.ring.is_mock:
            return {"error": "Ring camera not connected or in simulation mode", "processed": 0}

        try:
            history = await self.ring._active_camera.async_history(limit=limit)
            if not history:
                return {"message": "No historical events found in Ring account", "processed": 0}

            import cv2, tempfile, httpx
            from pathlib import Path

            batch_items = []
            for event in history:
                try:
                    url = await self.ring._active_camera.async_recording_url(event["id"])
                    if not url:
                        continue

                    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                        resp = await client.get(url)
                        if resp.status_code != 200 or len(resp.content) == 0:
                            continue

                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                            tmp.write(resp.content)
                            tmp_path = tmp.name

                        try:
                            cap = cv2.VideoCapture(tmp_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
                            ret, frame = cap.read()
                            cap.release()

                            if ret and frame is not None:
                                success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                if success:
                                    event_dt = event.get("created_at") or datetime.now()
                                    batch_items.append({
                                        "bytes": encoded.tobytes(),
                                        "timestamp": event_dt,
                                        "device_name": f"{self.ring.camera_name} (Event {event['id']})"
                                    })
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error processing Ring event {event.get('id')}: {e}")

            return await self.process_batch_images(batch_items)

        except Exception as e:
            logger.error(f"Failed processing Ring history backlog: {e}")
            return {"error": str(e), "processed": 0}

    def get_latest_snapshot(self) -> Optional[bytes]:
        return self._latest_snapshot_bytes
