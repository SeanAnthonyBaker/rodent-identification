import asyncio
import io
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set
import cv2
import httpx
from PIL import Image, ImageDraw

logger = logging.getLogger("ring_client")


class MjpegStreamBroadcaster:
    """Maintains a single persistent connection to an MJPEG camera stream and broadcasts parsed JPEGs at full 30 FPS."""
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self._subscribers: Set[asyncio.Queue] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._latest_frame: Optional[bytes] = None
        self._is_running = False

    def start(self):
        if not self._is_running or not self._worker_task or self._worker_task.done():
            self._is_running = True
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._stream_loop())
            except RuntimeError:
                pass

    def stop(self):
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()

    @property
    def latest_frame(self) -> Optional[bytes]:
        return self._latest_frame

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=2)
        self._subscribers.add(q)
        self.start()
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    async def _stream_loop(self):
        while self._is_running:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", self.stream_url) as resp:
                        if resp.status_code != 200:
                            await asyncio.sleep(2.0)
                            continue
                        
                        buffer = bytearray()
                        async for chunk in resp.aiter_raw():
                            if not self._is_running:
                                break
                            buffer.extend(chunk)
                            while True:
                                start = buffer.find(b"\xff\xd8")
                                if start == -1:
                                    buffer.clear()
                                    break
                                end = buffer.find(b"\xff\xd9", start + 2)
                                if end == -1:
                                    if start > 0:
                                        del buffer[:start]
                                    break
                                
                                jpeg_frame = bytes(buffer[start:end+2])
                                del buffer[:end+2]
                                
                                self._latest_frame = jpeg_frame
                                
                                # Broadcast to all active browser queues
                                dead_queues = []
                                for q in list(self._subscribers):
                                    try:
                                        if q.full():
                                            try:
                                                q.get_nowait()
                                            except asyncio.QueueEmpty:
                                                pass
                                        q.put_nowait(jpeg_frame)
                                    except Exception:
                                        dead_queues.append(q)
                                for dq in dead_queues:
                                    self._subscribers.discard(dq)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"S21 MJPEG stream broadcaster reconnecting: {e}")
                await asyncio.sleep(1.0)


class LocalRolandCamera:
    """Captures live frames directly from the webcam/camera attached to Roland 1."""
    def __init__(self, name: str = "Local Camera (Roland 1)", camera_index: int = 1):
        self.name = name
        self.device_id = f"local-roland-cam-{camera_index}"
        self.family = "local_cameras"
        self.model = "Roland 1 Direct Camera (USB/Webcam)"
        self.camera_index = camera_index
        self._battery_level = 100

    @property
    def battery_life(self) -> int:
        return 100

    @property
    def wifi_signal_strength(self) -> int:
        return 0

    def get_health(self) -> Dict[str, Any]:
        return {
            "battery_percentage": 100,
            "battery_percentage_category": "good",
            "wifi_signal_strength": 0,
            "device_name": self.name,
            "device_id": self.device_id,
            "is_mock": False,
            "is_local": True
        }

    async def async_get_snapshot(self, **kwargs) -> Optional[bytes]:
        """Captures an instant real-time frame directly from Roland 1's camera."""
        def _capture():
            # Try specified camera_index first, then fallback across 1, 0
            indices = [self.camera_index] + [i for i in [1, 0] if i != self.camera_index]
            for idx in indices:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    continue
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    # Check if frame is not completely pitch black
                    if frame.mean() > 5.0:
                        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if success:
                            return encoded.tobytes()
            # If no non-black frame, try the first working cap
            for idx in indices:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if success:
                            return encoded.tobytes()
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _capture)


class AndroidPhoneCamera:
    """Ingests live stream from Samsung Galaxy S21 Ultra via Webcam, IP Stream, or Browser."""
    def __init__(self, name: str = "Samsung Galaxy S21 Ultra", stream_url: Optional[str] = None, camera_index: int = 1):
        self.name = name
        self.device_id = "phone-cam-s21-ultra"
        self.family = "phone_cameras"
        self.model = "Samsung Galaxy S21 Ultra (Webcam / Wireless Stream)"
        self.stream_url = stream_url or "http://192.168.1.165:8080/video"
        self.camera_index = camera_index
        self._last_frame_bytes: Optional[bytes] = None
        self._last_frame_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None
        self._orientation_initialized: bool = False
        self.broadcaster = MjpegStreamBroadcaster(self.stream_url)
        self.broadcaster.start()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def async_set_orientation(self, orientation: str = "landscape"):
        """Sets hardware camera orientation on S21 phone ('landscape' = 90 deg clockwise to the right)."""
        if self.stream_url:
            base_url = self.stream_url.split("/video")[0].split("/shot.jpg")[0]
            try:
                client = self._get_client()
                await client.get(f"{base_url}/settings/orientation?set={orientation}", timeout=2.0)
                logger.info(f"S21 phone camera orientation set to '{orientation}' (90 deg to the right)")
            except Exception as e:
                logger.debug(f"Error setting phone camera orientation: {e}")

    @property
    def battery_life(self) -> int:
        return 95

    @property
    def wifi_signal_strength(self) -> int:
        return -45

    def get_health(self) -> Dict[str, Any]:
        return {
            "battery_percentage": 95,
            "battery_percentage_category": "good",
            "wifi_signal_strength": -45,
            "device_name": self.name,
            "device_id": self.device_id,
            "is_mock": False,
            "is_phone": True,
            "stream_url": self.stream_url
        }

    async def async_get_snapshot(self, **kwargs) -> Optional[bytes]:
        """Fetches a snapshot from S21 Ultra via Browser stream, IP Webcam HTTP stream, or USB Webcam."""
        if not self._orientation_initialized and self.stream_url:
            self._orientation_initialized = True
            asyncio.create_task(self.async_set_orientation("landscape"))

        # 0. Broadcaster 30 FPS buffer (Real-time Instant Snapshot)
        if hasattr(self, "broadcaster") and self.broadcaster.latest_frame:
            return self.broadcaster.latest_frame

        # 1. PRIORITY 1: Use live frame uploaded via /mobile_cam browser stream!
        if self._last_frame_bytes and (time.time() - self._last_frame_time < 60.0):
            return self._last_frame_bytes

        # 2. PRIORITY 2: Try direct HTTP snapshot endpoint (IP Webcam app: /shot.jpg)
        if self.stream_url:
            shot_url = self.stream_url.replace("/video", "/shot.jpg")
            try:
                client = self._get_client()
                resp = await client.get(shot_url)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    self._last_frame_bytes = resp.content
                    self._last_frame_time = time.time()
                    return resp.content
            except Exception:
                pass

            # Try OpenCV RTSP / MJPEG capture
            def _capture_url():
                try:
                    cap = cv2.VideoCapture(self.stream_url)
                    if not cap.isOpened():
                        return None
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        if success:
                            return encoded.tobytes()
                except Exception as e:
                    logger.debug(f"OpenCV phone stream capture error: {e}")
                return None

            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, _capture_url)
            if res:
                self._last_frame_bytes = res
                self._last_frame_time = time.time()
                return res

        # 3. PRIORITY 3: Check if frame cached
        if self._last_frame_bytes:
            return self._last_frame_bytes

        # 4. Fallback to DirectShow webcam
        def _capture_webcam():
            # Return cached frame or None cleanly if no physical webcam is present
            if self._last_frame_bytes:
                return self._last_frame_bytes
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _capture_webcam)


class MockRingCamera:
    """Simulates a Ring Camera when hardware/token is not present."""
    def __init__(self, name: str = "Garden Stick Up Cam"):
        self.name = name
        self.device_id = "mock-ring-cam-01"
        self.family = "stickup_cams"
        self.model = "Stick Up Cam Battery (Mock)"
        self._battery_level = 82
        self._wifi_rssi = -58

    @property
    def battery_life(self) -> int:
        return max(5, self._battery_level)

    @property
    def wifi_signal_strength(self) -> int:
        return self._wifi_rssi

    def get_health(self) -> Dict[str, Any]:
        return {
            "battery_percentage": self.battery_life,
            "battery_percentage_category": "good" if self.battery_life > 30 else "low",
            "wifi_signal_strength": self.wifi_signal_strength,
            "device_name": self.name,
            "device_id": self.device_id,
            "is_mock": True
        }

    async def async_get_snapshot(self, **kwargs) -> bytes:
        img = Image.new("RGB", (1280, 720), color=(30, 45, 35))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 1280, 240], fill=(20, 30, 45))
        draw.rectangle([0, 240, 1280, 440], fill=(55, 45, 38))
        draw.rectangle([0, 440, 1280, 720], fill=(35, 50, 30))

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        draw.rectangle([0, 0, 1280, 48], fill=(0, 0, 0, 180))
        draw.text((20, 14), f"RING CAM: {self.name} | BATTERY: {self.battery_life}% | {timestamp_str}", fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


class RingManager:
    """Manages connections to Ring Cameras (Garden, cam1, etc.) and Local Roland 1 Cameras."""

    def __init__(self, token_file: str = "ring_token.json", device_name: Optional[str] = None, mock_fallback: bool = True):
        self.token_file = Path(token_file)
        self.device_name = device_name or "Garden"
        self.mock_fallback = mock_fallback
        self._auth = None
        self._ring = None
        self._active_camera = None
        self._all_cameras = []
        self._is_mock = False
        self._local_cam = LocalRolandCamera("Local Camera (Roland 1)", 0)
        from src.config import config
        phone_url = getattr(config.ring, "phone_camera_url", "http://192.168.1.150:8080/video")
        self._phone_cam = AndroidPhoneCamera("Samsung Galaxy S21 Ultra", stream_url=phone_url)
        self._snapshot_cache: Dict[str, bytes] = {}
        self._last_event_ids: Dict[str, str] = {}
        self._last_vod_trigger_times: Dict[str, float] = {}

    async def async_trigger_on_demand_recording(self, camera=None):
        """Forces Ring camera to record a fresh on-demand video clip."""
        cam = camera or self._active_camera
        if not cam or self._is_mock or not self._ring or isinstance(cam, (LocalRolandCamera, AndroidPhoneCamera)):
            return
        try:
            dev_id = getattr(cam, "_attrs", {}).get("id") or getattr(cam, "id", None)
            if dev_id:
                url = f"https://api.ring.com/clients_api/doorbots/{dev_id}/vod"
                await self._ring.auth.async_query(url, method="POST")
                logger.info(f"Triggered on-demand VOD live recording for {getattr(cam, 'name', 'Camera')}")
        except Exception as e:
            logger.debug(f"VOD trigger error: {e}")

    def _token_updater(self, token: Dict[str, Any]):
        """Persists refreshed token automatically."""
        try:
            with open(self.token_file, "w", encoding="utf-8") as f:
                json.dump(token, f, indent=2)
            logger.info("Ring OAuth token updated and saved.")
        except Exception as e:
            logger.error(f"Failed saving updated Ring token: {e}")

    async def async_connect(self):
        """Connects to Ring API and prioritizes Garden and cam1."""
        self._all_cameras = [self._local_cam, self._phone_cam]

        if not self.token_file.exists():
            logger.warning(f"Ring token file '{self.token_file}' not found. Defaulting to Local Camera.")
            self._active_camera = self._local_cam
            return

        try:
            from ring_doorbell import Auth, Ring
            with open(self.token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)

            self._auth = Auth("RodentIdentification/1.0", token_data, self._token_updater)
            self._ring = Ring(self._auth)
            await self._ring.async_update_data()

            devices = self._ring.devices()
            ring_cams = list(devices.stickup_cams) + list(devices.doorbells)

            # Sort Ring cameras so Garden and cam1 are first
            def _sort_key(c):
                name = getattr(c, "name", "").lower()
                if "garden" in name: return 0
                if "cam1" in name or "cam 1" in name: return 1
                if "outhouse" in name: return 2
                return 3

            ring_cams.sort(key=_sort_key)

            # Combine Ring cameras (first) with Local Roland 1 Camera and Phone Camera
            self._all_cameras = ring_cams + [self._local_cam, self._phone_cam]

            # Match active camera
            if self.device_name:
                matched = next((c for c in self._all_cameras if c.name.lower() == self.device_name.lower()), None)
                self._active_camera = matched or ring_cams[0] if ring_cams else self._all_cameras[0]
            else:
                self._active_camera = ring_cams[0] if ring_cams else self._all_cameras[0]

            self._is_mock = False
            logger.info(f"Connected to Ring API. Discovered {len(ring_cams)} Ring devices: {[c.name for c in ring_cams]}. Active: '{self.camera_name}'")

        except Exception as e:
            logger.error(f"Error connecting to Ring API: {e}", exc_info=True)
            self._active_camera = self._local_cam

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Returns all available cameras (Ring Garden, cam1, Outhouse + Local/Phone)."""
        results = []
        for cam in self._all_cameras:
            is_local = isinstance(cam, LocalRolandCamera)
            is_phone = isinstance(cam, AndroidPhoneCamera)
            is_ring = not is_local and not is_phone and not self._is_mock
            bat = getattr(cam, "battery_life", None)
            if is_local: bat = 100
            elif is_phone: bat = 95
            
            results.append({
                "name": getattr(cam, "name", "Camera"),
                "id": getattr(cam, "id", None) or getattr(cam, "device_id", None),
                "model": getattr(cam, "model", "Stick Up Cam" if is_ring else "Camera"),
                "battery_percentage": int(bat) if bat is not None else None,
                "wifi_signal_strength": getattr(cam, "wifi_signal_strength", None),
                "is_ring": is_ring,
                "is_local": is_local,
                "is_phone": is_phone,
                "is_active": (self._active_camera and self._active_camera.name.lower() == cam.name.lower())
            })
        return results

    def find_camera(self, camera_name: Optional[str]) -> Optional[Any]:
        """Resolves a camera object by direct name or common aliases (Garden, Cam1, S21, S1, Phone)."""
        if not camera_name:
            return self._active_camera
        c_low = camera_name.lower().strip()
        matched = next((c for c in self._all_cameras if c.name.lower() == c_low), None)
        if matched:
            return matched
        if any(k in c_low for k in ["s21", "s1", "phone", "galaxy", "android"]):
            return getattr(self, "_phone_cam", None)
        if "garden" in c_low:
            return next((c for c in self._all_cameras if "garden" in c.name.lower()), None)
        if "cam1" in c_low or "cam 1" in c_low:
            return next((c for c in self._all_cameras if "cam1" in c.name.lower() or "cam 1" in c.name.lower()), None)
        return None

    def select_camera(self, camera_name: str) -> bool:
        """Switches active camera to the specified camera name."""
        matched = self.find_camera(camera_name)
        if matched:
            self._active_camera = matched
            self.device_name = matched.name
            logger.info(f"Switched active camera to: '{matched.name}'")
            return True
        return False

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def camera_name(self) -> str:
        if self._active_camera:
            return getattr(self._active_camera, "name", "Unknown Camera")
        return "Not Connected"

    def get_battery_level(self, camera=None) -> Optional[int]:
        """Returns battery percentage (0-100) or None if wired/unavailable."""
        cam = camera or self._active_camera
        if not cam:
            return None
        if isinstance(cam, LocalRolandCamera):
            return 100
        if isinstance(cam, AndroidPhoneCamera):
            return 95
        try:
            bat = getattr(cam, "battery_life", None)
            if bat is not None:
                return int(bat)
            return None
        except Exception as e:
            logger.error(f"Failed to read battery level: {e}")
            return None

    def get_health_status(self, camera=None) -> Dict[str, Any]:
        """Returns comprehensive device telemetry including battery and WiFi."""
        cam = camera or self._active_camera
        if not cam:
            return {
                "connected": False,
                "battery_percentage": None,
                "status": "Disconnected",
                "is_mock": False
            }

        is_local = isinstance(cam, LocalRolandCamera)
        is_phone = isinstance(cam, AndroidPhoneCamera)
        is_ring = not is_local and not is_phone and not self._is_mock
        battery = self.get_battery_level(cam)
        wifi_rssi = getattr(cam, "wifi_signal_strength", None)
        name = getattr(cam, "name", "Camera")

        return {
            "connected": True,
            "device_name": name,
            "device_id": getattr(cam, "device_id", "unknown") if (is_local or is_phone) else getattr(cam, "id", "unknown"),
            "model": getattr(cam, "model", "Camera"),
            "battery_percentage": battery,
            "battery_status": "AC / Continuous Power" if is_local else (
                "Critical (< 15%)" if battery is not None and battery <= 15
                else "Low (< 30%)" if battery is not None and battery <= 30
                else "Good" if battery is not None
                else "Wired / Unknown"
            ),
            "wifi_signal_strength": wifi_rssi,
            "is_mock": self._is_mock,
            "is_ring": is_ring,
            "is_local": is_local,
            "is_phone": is_phone,
            "available_cameras": [c.name for c in self._all_cameras] if self._all_cameras else [name],
            "last_updated": datetime.now().isoformat()
        }

    def create_standby_frame(self, reason: str = "Armed & Monitoring", camera_name: Optional[str] = None) -> bytes:
        """Generates a clean CCTV monitor screen when waiting for next motion event."""
        cam_name = camera_name or self.camera_name
        img = Image.new("RGB", (1280, 720), color=(10, 15, 26))
        draw = ImageDraw.Draw(img)

        # Sleek dark CCTV frame border
        draw.rectangle([0, 0, 1280, 720], fill=(12, 18, 30))
        draw.rectangle([20, 20, 1260, 700], outline=(30, 45, 65), width=2)

        # Crosshairs / Viewfinder markings
        draw.line([(600, 360), (680, 360)], fill=(51, 65, 85), width=1)
        draw.line([(640, 320), (640, 400)], fill=(51, 65, 85), width=1)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        battery = self.get_battery_level()
        bat_str = f"🔋 Battery: {battery}%" if battery is not None else "🔋 Battery: 90%"

        # Header bar
        draw.rectangle([20, 20, 1260, 70], fill=(15, 23, 42))
        draw.text((40, 35), f"🔴 LIVE FEED: {cam_name.upper()} | {bat_str} | {timestamp_str}", fill=(248, 250, 252))

        # Main Info
        draw.text((120, 280), f"📹 {cam_name.upper()} — ONLINE & ARMED", fill=(52, 211, 153))
        draw.text((120, 330), f"Status: {reason} & Listening for Motion / Wildlife", fill=(226, 232, 240))
        draw.text((120, 380), f"• Wildlife Detection: ACTIVE (Human-Only Filter Disabled)", fill=(245, 158, 11))
        draw.text((120, 420), f"• HD Frame Stream will activate instantly when motion occurs.", fill=(148, 163, 184))
        draw.text((120, 460), f"• Select between 'Garden' and 'cam1' in the camera switcher above.", fill=(100, 116, 139))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    async def _fetch_frame_from_latest_recording(self, camera=None) -> Tuple[Optional[bytes], bool, bool]:
        """Downloads latest Ring camera event video. Returns (frame_bytes, is_standby, is_new_event)."""
        cam = camera or self._active_camera
        if not cam or self._is_mock or isinstance(cam, (LocalRolandCamera, AndroidPhoneCamera)):
            return None, False, False

        cam_name = getattr(cam, "name", "unknown")
        try:
            if self._ring:
                try:
                    await self._ring.async_update_data()
                except Exception:
                    pass

            history = await cam.async_history(limit=2)
            if not history:
                return self.create_standby_frame("No recorded events found", cam_name), True, False

            latest_event = history[0]
            event_id = str(latest_event.get("id"))
            prev_event_id = self._last_event_ids.get(cam_name)
            is_new = (prev_event_id != event_id)

            # If this event was already downloaded and processed, reuse cached frame instantly
            if not is_new and cam_name in self._snapshot_cache:
                last_vod = self._last_vod_trigger_times.get(cam_name, 0.0)
                if time.time() - last_vod > 20:
                    self._last_vod_trigger_times[cam_name] = time.time()
                    asyncio.create_task(self.async_trigger_on_demand_recording(cam))
                return self._snapshot_cache[cam_name], False, False

            self._last_event_ids[cam_name] = event_id

            url = await cam.async_recording_url(latest_event["id"])
            if not url:
                cached = self._snapshot_cache.get(cam_name)
                return cached or self.create_standby_frame("Awaiting new recording", cam_name), False, False

            async with httpx.AsyncClient(follow_redirects=True, timeout=12.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200 or len(resp.content) == 0:
                    cached = self._snapshot_cache.get(cam_name)
                    return cached or self.create_standby_frame("Connecting to camera...", cam_name), False, False

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name

                try:
                    cap = cv2.VideoCapture(tmp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
                    mid_frame_idx = max(0, total_frames // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None:
                        success, encoded_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if success:
                            frame_bytes = encoded_jpg.tobytes()
                            self._snapshot_cache[cam_name] = frame_bytes
                            if is_new:
                                logger.info(f"New Ring motion event on {cam_name} (ID: {event_id}). Extracted frame ({frame.shape[1]}x{frame.shape[0]}).")
                            return frame_bytes, False, is_new
                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed extracting frame from Ring recording for {cam_name}: {e}")
        
        cached = self._snapshot_cache.get(cam_name)
        return cached, False, False

    async def async_fetch_snapshot(self, camera_name: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str], bool, bool]:
        """Fetches latest snapshot from specified camera or active camera. Returns (bytes, error, is_standby, is_new)."""
        if not self._active_camera:
            await self.async_connect()

        target_cam = self._active_camera
        if camera_name:
            target_cam = self.find_camera(camera_name) or self._active_camera

        if not target_cam:
            return None, "No active camera connected", False, False

        cam_name = getattr(target_cam, "name", "Camera")

        try:
            # Case 1: Local Roland 1 Camera or Android Phone Camera
            if isinstance(target_cam, (LocalRolandCamera, AndroidPhoneCamera)):
                snap = await target_cam.async_get_snapshot()
                if snap:
                    self._snapshot_cache[cam_name] = snap
                    if camera_name:
                        self._snapshot_cache[camera_name] = snap
                    if isinstance(target_cam, AndroidPhoneCamera):
                        self._snapshot_cache["S21"] = snap
                    return snap, None, False, True
                return None, f"Could not open stream for {target_cam.name}", False, False

            # Case 2: Mock Camera
            if self._is_mock or isinstance(target_cam, MockRingCamera):
                snap = await target_cam.async_get_snapshot()
                self._snapshot_cache[cam_name] = snap
                return snap, None, False, True

            # Case 3: Live Ring Camera event frame
            rec_frame, is_standby, is_new = await self._fetch_frame_from_latest_recording(target_cam)
            if rec_frame:
                self._snapshot_cache[cam_name] = rec_frame
                return rec_frame, None, is_standby, is_new

            if cam_name in self._snapshot_cache:
                return self._snapshot_cache[cam_name], None, False, False

            standby = self.create_standby_frame("Standby", cam_name)
            return standby, None, True, False

        except Exception as e:
            logger.error(f"Error capturing snapshot for {cam_name}: {e}")
            if cam_name in self._snapshot_cache:
                return self._snapshot_cache[cam_name], None, False, False
            return None, str(e), False, False

    def fetch_snapshot(self, camera_name: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """Synchronous wrapper for fetch_snapshot."""
        cam_name = camera_name or self.camera_name
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._snapshot_cache.get(cam_name), None
            snap, err, _, _ = loop.run_until_complete(self.async_fetch_snapshot(camera_name))
            return snap, err
        except Exception:
            return self._snapshot_cache.get(cam_name), None
