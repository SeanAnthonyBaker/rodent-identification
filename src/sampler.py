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

logger = logging.getLogger("sampler")

class SamplerEngine:
    """Orchestrates periodic snapshot sampling with OpenCV background subtraction, high-res sub-crops, and dual-image contrast verification."""

    def __init__(
        self,
        ring_manager: RingManager,
        inference_client: RolandInferenceClient,
        storage_manager: StorageManager,
        notifier: Optional[WatchNotificationManager] = None,
        interval_seconds: int = 10,
        active_detection_interval_seconds: int = 5
    ):
        self.ring = ring_manager
        self.inference = inference_client
        self.storage = storage_manager
        self.notifier = notifier or WatchNotificationManager()
        self.motion_pipeline = MotionCascadePipeline()
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
        self.event_inactivity_timeout_seconds: float = 60.0
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
            # 1. Compute target zone delta against locked baseline reference
            zone_info = self.motion_pipeline.compute_zone_delta(
                snapshot_bytes,
                polygon=self.inference.detection_polygon,
                delta_threshold=0.40
            )
            has_delta = zone_info["has_material_delta"]
            delta_pct = zone_info["delta_percent"]

            ref_bytes = self.motion_pipeline.get_reference_baseline()
            if not ref_bytes and not self.ring._is_mock:
                self.motion_pipeline.save_reference_baseline(snapshot_bytes)
                ref_bytes = snapshot_bytes

            crop_bytes = zone_info["focused_crop_bytes"]
            crop_bbox = zone_info["crop_bbox"]
            is_subcrop = crop_bbox is not None

            # 2. VLM Gating: If target zone is quiet (0 delta) and no active tracking, bypass Gemma (0 GPU compute)
            if not has_delta and not self._is_rat_active:
                inference_result = DetectionResult(
                    detected=False,
                    is_detected=False,
                    is_rat_detected=False,
                    subject_type="none",
                    object_type="none",
                    label="None",
                    confidence_score=0.0,
                    confidence=0.0,
                    description=f"Target zone quiet ({delta_pct:.1f}% delta). Gemma inference bypassed.",
                    inference_time_ms=5.0
                )
                self._latest_inference_result = inference_result
                self.current_interval_seconds = self.base_interval_seconds
            else:
                # 3. Run locked-temperature contrast inference on Gemma
                inference_result = await self.inference.analyze_image(
                    image_bytes=crop_bytes,
                    polygon=self.inference.detection_polygon,
                    reference_image_bytes=ref_bytes,
                    is_subcrop=is_subcrop
                )

                # 4. Map sub-crop bounding box to full-frame normalized coordinates [0-1000]
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

                self._latest_inference_result = inference_result

                # 5. Dynamic Rate Switching & Single Event Session Grouping
                if inference_result.is_detected:
                    import uuid
                    now_ts = start_ts.timestamp()
                    same_cam = (self._active_event_camera or "").lower() == (self.ring.camera_name or "").lower()
                    same_obj = (self._active_event_object or "").lower() == inference_result.object_type.lower()
                    session_valid = (
                        self._active_event_id is not None
                        and same_cam
                        and same_obj
                        and (now_ts - self._active_event_last_seen) < self.event_inactivity_timeout_seconds
                    )

                    if not session_valid:
                        # Start a brand new single Event Session
                        self._active_event_id = f"evt_{int(now_ts)}_{uuid.uuid4().hex[:6]}"
                        self._active_event_object = inference_result.object_type
                        self._active_event_camera = self.ring.camera_name
                        self._active_event_frame_idx = 1
                        is_new_event = True
                    else:
                        # Append frame to the existing ongoing Event Session
                        self._active_event_frame_idx += 1
                        is_new_event = False

                    self._active_event_last_seen = now_ts
                    self._is_rat_active = True
                    self.current_interval_seconds = self.active_detection_interval_seconds
                    obj_emojis = {
                        "tree": "🌲",
                        "bird": "🐦",
                        "pheasant": "🦚",
                        "rodent": "🐀",
                        "rat": "🐀",
                        "horse": "🐴",
                        "horses_poo": "🐴💩"
                    }
                    emoji = obj_emojis.get(inference_result.object_type, "🎯")

                    # Record frame linked to the single event_id
                    record = self.storage.save_detection(
                        image_bytes=snapshot_bytes,
                        confidence=inference_result.confidence,
                        description=inference_result.description,
                        object_type=inference_result.object_type,
                        label=inference_result.label,
                        battery_percentage=self._last_battery_level,
                        device_name=self.ring.camera_name,
                        bounding_box=inference_result.bounding_box,
                        dt=start_ts,
                        event_id=self._active_event_id,
                        frame_index=self._active_event_frame_idx
                    )
                    detection_saved = record.model_dump()
                    detection_saved["event_id"] = self._active_event_id
                    detection_saved["frame_index"] = self._active_event_frame_idx

                    if is_new_event:
                        logger.info(f"{emoji} {inference_result.label.upper()} DETECTED! (New Event #{self._active_event_id}) -> Sampling BOOSTED to {self.current_interval_seconds}s")
                        # Trigger immediate push & voice alert only ONCE per event
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
                else:
                    if self._is_rat_active:
                        now_ts = start_ts.timestamp()
                        if (now_ts - self._active_event_last_seen) >= self.event_inactivity_timeout_seconds:
                            logger.info(f"Event {self._active_event_id} concluded after {self._active_event_frame_idx} frames. Sampling rate REVERTED back to baseline ({self.base_interval_seconds}s).")
                            self._is_rat_active = False
                            self._active_event_id = None
                            self.current_interval_seconds = self.base_interval_seconds

        # 4. Notify UI subscribers of new frame, active mode & status
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
        return status_payload

    async def _loop(self):
        logger.info(f"Starting adaptive sampling engine (Baseline: {self.base_interval_seconds}s, Active Target: {self.active_detection_interval_seconds}s)")
        while self._running:
            try:
                # Check if live camera stream is active (e.g. S21 streaming directly)
                is_live_streaming = False
                if hasattr(self.ring, "_phone_cam") and self.ring._phone_cam:
                    if time.time() - getattr(self.ring._phone_cam, "_last_frame_time", 0) < 4.0:
                        is_live_streaming = True

                # When live stream is active, delta gating is handled in real-time by the stream pipeline.
                # Background periodic polling is paused during live streaming to prevent 10/15s screen refreshes.
                if not is_live_streaming:
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

    def stop(self):
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                self._task = None
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
