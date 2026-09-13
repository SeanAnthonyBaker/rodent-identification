import asyncio
import io
import json
import os
import subprocess
import shutil
import re
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set
import threading
import cv2
import httpx
from PIL import Image, ImageDraw
import numpy as np

logger = logging.getLogger("ring_client")


def is_blank_or_disabled_frame(frame_bytes: Optional[bytes]) -> bool:
    """Checks whether a frame is pure black or a Windows Link disabled-camera placeholder."""
    if not frame_bytes or len(frame_bytes) < 100:
        return True
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return True
        mean_val = float(img.mean())
        if mean_val < 8.0:
            return True
        black_ratio = float(np.count_nonzero(img < 15)) / float(img.size)
        if black_ratio > 0.90 and mean_val < 20.0:
            return True
        return False
    except Exception:
        return True


def capture_desktop_window_frame(keyword: str) -> Optional[bytes]:
    """Captures the visible window frame matching keyword from the Default interactive desktop.
    Excludes small dialogs/prompts (< 380x350) and minimized windows.
    """
    try:
        import win32service, win32gui, win32ui, win32con, ctypes
        from PIL import Image
        from io import BytesIO

        hdesk = win32service.OpenDesktop('Default', 0, False, 0x01FF)
        ctypes.windll.user32.SetThreadDesktop(int(hdesk))
        target_hwnd = None
        for h in hdesk.EnumDesktopWindows():
            if win32gui.IsWindowVisible(h) and not win32gui.IsIconic(h):
                t = win32gui.GetWindowText(h)
                if t and keyword.lower() in t.lower():
                    rect = win32gui.GetWindowRect(h)
                    w = rect[2] - rect[0]
                    h_len = rect[3] - rect[1]
                    if w > 380 and h_len > 350:
                        target_hwnd = int(h)
                        break
        if not target_hwnd:
            return None

        rect = win32gui.GetWindowRect(target_hwnd)
        w = rect[2] - rect[0]
        h_len = rect[3] - rect[1]
        hwndDC = win32gui.GetWindowDC(target_hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h_len)
        saveDC.SelectObject(saveBitMap)
        ctypes.windll.user32.PrintWindow(target_hwnd, saveDC.GetSafeHdc(), 2)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(target_hwnd, hwndDC)

        buf = BytesIO()
        im.save(buf, format='JPEG', quality=85)
        raw_bytes = buf.getvalue()
        if is_blank_or_disabled_frame(raw_bytes):
            return None
        return raw_bytes
    except Exception as e:
        return None


def ensure_adb_forward(local_port: int = 8085, remote_port: int = 8080, target_serial: Optional[str] = None) -> bool:
    """Ensures that ADB port forwarding from PC local_port to Android remote_port is actively configured."""
    adb_path = r"C:\Users\seanb\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v3.3.4\adb.exe"
    if not os.path.exists(adb_path):
        import shutil
        adb_path = shutil.which("adb") or adb_path
    if os.path.exists(adb_path):
        try:
            list_res = subprocess.run(
                [adb_path, "forward", "--list"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4
            )
            if list_res.returncode == 0 and f"tcp:{local_port} tcp:{remote_port}" in list_res.stdout:
                return True

            fwd_cmd = [adb_path]
            if target_serial:
                fwd_cmd.extend(["-s", target_serial])
            fwd_cmd.extend(["forward", f"tcp:{local_port}", f"tcp:{remote_port}"])

            res = subprocess.run(
                fwd_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=8
            )
            return res.returncode == 0
        except Exception as e:
            logger.debug(f"ensure_adb_forward({local_port}) error: {e}")
    return False

def ensure_adb_forward_8085() -> bool:
    return ensure_adb_forward(8085, 8080)

def ensure_adb_forward_8086() -> bool:
    return ensure_adb_forward(8086, 8080, "192.168.1.194:5555")


class MjpegStreamBroadcaster:
    """Maintains a single persistent connection to an MJPEG camera stream and broadcasts parsed JPEGs at full 30 FPS."""
    def __init__(
        self,
        stream_url: str,
        adb_port: int = 8085,
        wifi_candidates: Optional[List[str]] = None,
        target_device_serial: Optional[str] = None
    ):
        self.stream_url = stream_url
        self.adb_port = adb_port
        self.wifi_candidates = wifi_candidates or ["http://192.168.1.165:8080/video"]
        self.target_device_serial = target_device_serial
        self._subscribers: Set[asyncio.Queue] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._latest_frame: Optional[bytes] = None
        self._last_frame_time: float = 0.0
        self._is_running = False

    def start(self):
        try:
            loop = asyncio.get_running_loop()
            if not self._worker_task or self._worker_task.done():
                self._is_running = True
                self._worker_task = loop.create_task(self._stream_loop())
        except RuntimeError:
            self._is_running = False

    def stop(self):
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()

    @property
    def is_live(self) -> bool:
        return (time.time() - self._last_frame_time) < 3.5

    @property
    def latest_frame(self) -> Optional[bytes]:
        if (time.time() - self._last_frame_time) < 3.5:
            return self._latest_frame
        return None

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
                if f"127.0.0.1:{self.adb_port}" in self.stream_url:
                    await asyncio.to_thread(ensure_adb_forward, self.adb_port, 8080, self.target_device_serial)

                stream_timeout = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=None)
                async with httpx.AsyncClient(timeout=stream_timeout) as client:
                    async with client.stream("GET", self.stream_url) as resp:
                        if resp.status_code != 200:
                            await asyncio.sleep(1.0)
                            continue
                        
                        buffer = bytearray()
                        async for chunk in resp.aiter_raw():
                            if not self._is_running:
                                break
                            buffer.extend(chunk)
                            while True:
                                # 1. Try deterministic Content-Length framing from IP Webcam
                                cl_idx = buffer.find(b"Content-Length: ")
                                if cl_idx == -1:
                                    cl_idx = buffer.find(b"content-length: ")
                                if cl_idx != -1:
                                    h_end = buffer.find(b"\r\n\r\n", cl_idx)
                                    if h_end != -1:
                                        try:
                                            length_bytes = buffer[cl_idx+16:h_end].split(b"\r\n")[0].strip()
                                            length = int(length_bytes)
                                            f_start = h_end + 4
                                            if len(buffer) < f_start + length:
                                                break
                                            jpeg_frame = bytes(buffer[f_start:f_start+length])
                                            del buffer[:f_start+length]
                                        except (ValueError, IndexError):
                                            del buffer[:h_end+4]
                                            continue
                                    else:
                                        break
                                else:
                                    # Fallback marker search if server doesn't send Content-Length
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
                                self._last_frame_time = time.time()
                                
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
                logger.debug(f"MJPEG stream broadcaster reconnecting ({self.stream_url}): {e}")
                # Auto-failover between USB/ADB port and Wi-Fi LAN ports
                if f"127.0.0.1:{self.adb_port}" in self.stream_url:
                    fwd_ok = await asyncio.to_thread(ensure_adb_forward, self.adb_port, 8080, self.target_device_serial)
                    if not fwd_ok and self.wifi_candidates:
                        self.stream_url = self.wifi_candidates[0]
                elif any(cand in self.stream_url for cand in self.wifi_candidates):
                    fwd_ok = await asyncio.to_thread(ensure_adb_forward, self.adb_port, 8080, self.target_device_serial)
                    if fwd_ok:
                        self.stream_url = f"http://127.0.0.1:{self.adb_port}/video"
                    elif len(self.wifi_candidates) > 1:
                        cur_idx = self.wifi_candidates.index(self.stream_url) if self.stream_url in self.wifi_candidates else 0
                        next_idx = (cur_idx + 1) % len(self.wifi_candidates)
                        self.stream_url = self.wifi_candidates[next_idx]
                await asyncio.sleep(0.5)


class LocalWebcamBroadcaster:
    """Maintains a persistent connection to a local USB webcam or Windows Virtual Camera and broadcasts frames at ~25 FPS."""
    def __init__(self, camera_index: int = 0, backend: Optional[int] = None, strict_index: bool = True):
        self.camera_index = camera_index
        self.backend = backend
        self.strict_index = strict_index
        self._subscribers: Set[asyncio.Queue] = set()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[bytes] = None
        self._is_running = False
        self._lock = threading.Lock()

    def start(self):
        if not self._is_running or not self._thread or not self._thread.is_alive():
            self._is_running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._is_running = False

    @property
    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=2)
        self._subscribers.add(q)
        self.start()
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    def _capture_loop(self):
        backends = [self.backend] if self.backend is not None else ([cv2.CAP_MSMF, cv2.CAP_DSHOW] if self.camera_index in (1, 2) else [cv2.CAP_DSHOW, cv2.CAP_MSMF])
        indices = [self.camera_index] if self.strict_index else [self.camera_index] + [i for i in [2, 1, 0] if i != self.camera_index]

        while self._is_running:
            cap = None
            try:
                for b in backends:
                    for idx in indices:
                        try:
                            temp_cap = cv2.VideoCapture(idx, b)
                            if temp_cap.isOpened():
                                ret, test_f = temp_cap.read()
                                if ret and test_f is not None:
                                    cap = temp_cap
                                    self.camera_index = idx
                                    self.backend = b
                                    break
                            temp_cap.release()
                        except Exception:
                            pass
                    if cap and cap.isOpened():
                        break

                if not cap or not cap.isOpened():
                    time.sleep(1.0)
                    continue

                logger.info(f"LocalWebcamBroadcaster active on camera index {self.camera_index} (backend: {self.backend})")
                consecutive_failures = 0
                while self._is_running:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        consecutive_failures += 1
                        if consecutive_failures > 50:
                            logger.debug(f"Broadcaster on camera index {self.camera_index} experienced read timeout, reconnecting...")
                            break
                        time.sleep(0.04)
                        continue
                    consecutive_failures = 0

                    success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not success:
                        continue

                    jpeg_bytes = enc.tobytes()
                    with self._lock:
                        self._latest_frame = jpeg_bytes

                    dead_queues = []
                    for q in list(self._subscribers):
                        try:
                            if q.full():
                                try:
                                    q.get_nowait()
                                except Exception:
                                    pass
                            q.put_nowait(jpeg_bytes)
                        except Exception:
                            dead_queues.append(q)
                    for dq in dead_queues:
                        self._subscribers.discard(dq)

                    time.sleep(0.04)
            except Exception as e:
                logger.debug(f"LocalWebcamBroadcaster loop error: {e}")
                time.sleep(1.0)
            finally:
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass


class LocalRolandCamera:
    """Captures live frames directly from the webcam/camera attached to Roland 1."""
    def __init__(self, name: str = "Local Camera (Roland 1)", camera_index: int = 0, backend: Optional[int] = None):
        self.name = name
        self.device_id = f"local-roland-cam-{camera_index}"
        self.family = "local_cameras"
        self.model = "Roland 1 Direct Camera (USB/Webcam)"
        self.camera_index = camera_index
        self._battery_level = 100
        self.broadcaster = LocalWebcamBroadcaster(camera_index=camera_index, backend=backend, strict_index=True)
        self.broadcaster.start()

    @property
    def latest_frame(self) -> Optional[bytes]:
        if self.broadcaster and self.broadcaster.latest_frame:
            return self.broadcaster.latest_frame
        return None

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
        if self.broadcaster:
            self.broadcaster.start()
            if self.broadcaster.latest_frame:
                return self.broadcaster.latest_frame
            for _ in range(5):
                await asyncio.sleep(0.05)
                if self.broadcaster.latest_frame:
                    return self.broadcaster.latest_frame
        return None


class GalaxyTabWindowsCamera(LocalRolandCamera):
    """Ingests live stream or high-resolution picture snapshot from Samsung Galaxy Tab A11+."""
    def __init__(self, name: str = "Galaxy Tab A11+", camera_index: int = 2, picture_path: Optional[str] = None):
        super().__init__(name=name, camera_index=camera_index, backend=cv2.CAP_DSHOW)
        self.device_id = "tablet-cam-tab-a11"
        self.family = "tablet_cameras"
        self.model = "Samsung Galaxy Tab A11+ (Picture / Windows Link)"
        self._battery_level = 80
        self.picture_path = Path(picture_path or "data/tab_a11_picture.jpg")
        self._cached_picture: Optional[bytes] = None
        self._last_frame_bytes: Optional[bytes] = None
        self._last_frame_time: float = 0.0

    @property
    def battery_life(self) -> int:
        return self._battery_level

    @battery_life.setter
    def battery_life(self, val: int):
        self._battery_level = val

    @property
    def wifi_signal_strength(self) -> int:
        return -40

    def get_health(self) -> Dict[str, Any]:
        has_fresh_web = bool(self._last_frame_bytes and (time.time() - self._last_frame_time < 6.0))
        has_dshow = bool(self.broadcaster and self.broadcaster.latest_frame and not is_blank_or_disabled_frame(self.broadcaster.latest_frame))
        is_stream = has_fresh_web or has_dshow
        return {
            "battery_percentage": self._battery_level,
            "battery_percentage_category": "good",
            "wifi_signal_strength": -40,
            "device_name": self.name,
            "device_id": self.device_id,
            "is_mock": False,
            "is_phone": True,
            "is_local": True,
            "is_windows_link": True,
            "is_streaming": is_stream,
            "uses_pictures": not is_stream
        }

    def get_picture(self) -> Optional[bytes]:
        """Loads assigned high-res surveillance picture for Galaxy Tab A11+."""
        if self._cached_picture:
            return self._cached_picture
        candidates = [
            self.picture_path,
            Path("data/tab_a11_picture.jpg"),
            Path("scratch/live_stream_from_user.jpg"),
            Path("data/video_frames/frame_00s.jpg"),
            Path("data/current_feed_debug.jpg")
        ]
        for p in candidates:
            if p.exists():
                try:
                    b = p.read_bytes()
                    if len(b) > 1000 and not is_blank_or_disabled_frame(b):
                        self._cached_picture = b
                        return b
                except Exception:
                    pass
        return None

    def set_picture(self, image_bytes: bytes) -> bool:
        """Sets/updates the active picture for Galaxy Tab A11+."""
        try:
            self.picture_path.parent.mkdir(parents=True, exist_ok=True)
            self.picture_path.write_bytes(image_bytes)
            self._cached_picture = image_bytes
            return True
        except Exception as e:
            logger.error(f"Failed setting picture for {self.name}: {e}")
            return False

    async def async_get_snapshot(self, **kwargs) -> Optional[bytes]:
        """Captures frame from stream if valid/live; otherwise returns assigned picture."""
        # 1. Direct Web Browser stream (from /mobile_cam?cam=tab)
        if self._last_frame_bytes and (time.time() - self._last_frame_time < 6.0):
            if not is_blank_or_disabled_frame(self._last_frame_bytes):
                return self._last_frame_bytes

        # 2. Windows Virtual Camera (DirectShow Index 2)
        if self.broadcaster:
            self.broadcaster.start()
            frame = self.broadcaster.latest_frame
            if frame and not is_blank_or_disabled_frame(frame):
                return frame
            for _ in range(3):
                await asyncio.sleep(0.04)
                frame = self.broadcaster.latest_frame
                if frame and not is_blank_or_disabled_frame(frame):
                    return frame

        # 3. Desktop window screen capture of Phone Link mirroring ("Galaxy Tab A11+")
        win_frame = capture_desktop_window_frame("Galaxy Tab A11+") or capture_desktop_window_frame("Tab A11")
        if win_frame and not is_blank_or_disabled_frame(win_frame):
            return win_frame

        # 4. Fallback to cached browser frame if valid
        if self._last_frame_bytes and not is_blank_or_disabled_frame(self._last_frame_bytes):
            return self._last_frame_bytes

        # 5. Fallback to clear assigned picture
        pic = self.get_picture()
        if pic:
            return pic
        return None


class AndroidPhoneCamera:
    """Ingests live stream from Android device (S21 Ultra, Galaxy Tab A11+, etc.) via Webcam, IP Stream, or Browser."""
    def __init__(
        self,
        name: str = "Samsung Galaxy S21 Ultra",
        stream_url: Optional[str] = None,
        camera_index: int = 1,
        device_id: str = "phone-cam-s21-ultra",
        model: str = "Samsung Galaxy S21 Ultra (Webcam / Wireless Stream)",
        adb_port: int = 8085,
        target_device_serial: Optional[str] = None,
        wifi_candidates: Optional[List[str]] = None,
        picture_path: Optional[str] = None
    ):
        self.name = name
        self.device_id = device_id
        self.family = "phone_cameras"
        self.model = model
        self.camera_index = camera_index
        self.adb_port = adb_port
        self.target_device_serial = target_device_serial
        self.wifi_candidates = wifi_candidates or ["http://192.168.1.165:8080/video"]
        self.picture_path = Path(picture_path or "data/s21_picture.jpg")
        self._cached_picture: Optional[bytes] = None
        
        # Dynamically probe and select active working endpoint (USB/ADB or Wi-Fi)
        self.stream_url = self._resolve_active_stream_url(stream_url)
        
        self._battery_level: int = 80
        self._last_frame_bytes: Optional[bytes] = None
        self._last_frame_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None
        self._orientation_initialized: bool = False
        self.broadcaster = MjpegStreamBroadcaster(
            self.stream_url,
            adb_port=self.adb_port,
            wifi_candidates=self.wifi_candidates,
            target_device_serial=self.target_device_serial
        )
        self.broadcaster.start()
        self.dshow_broadcaster = LocalWebcamBroadcaster(camera_index=self.camera_index, backend=cv2.CAP_DSHOW)
        self.dshow_broadcaster.start()
        self._poll_thread = threading.Thread(target=self._battery_poll_loop, daemon=True)
        self._poll_thread.start()

    def _resolve_active_stream_url(self, explicit_url: Optional[str] = None) -> str:
        """Fast-probes USB and Wi-Fi to select whichever endpoint is actively responding."""
        if explicit_url:
            return explicit_url

        # 1. Prioritize USB direct cable connection
        usb_url = self._check_and_setup_usb_forward()
        if usb_url:
            return usb_url

        import socket
        def _is_port_open(host: str, port: int, timeout: float = 0.15) -> bool:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    s.connect((host, port))
                    return True
            except Exception:
                return False

        # 2. Fast probe Wi-Fi candidates
        for cand in self.wifi_candidates:
            try:
                base = cand.replace("http://", "").replace("https://", "").split("/")[0]
                host, port_s = base.split(":")
                if _is_port_open(host, int(port_s), timeout=0.15):
                    logger.info(f"⚡ {self.name} actively connected via Wi-Fi ({cand})!")
                    return cand
            except Exception:
                pass

        return f"http://127.0.0.1:{self.adb_port}/video"

    def _check_and_setup_usb_forward(self) -> Optional[str]:
        if ensure_adb_forward(self.adb_port, 8080, self.target_device_serial):
            logger.info(f"⚡ {self.name} actively connected via ADB forward (http://127.0.0.1:{self.adb_port})!")
            return f"http://127.0.0.1:{self.adb_port}/video"
        return None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def async_set_orientation(self, orientation: str = "landscape"):
        """Sets hardware camera orientation on phone/tablet and sets quality for 30 FPS."""
        if self.stream_url:
            base_url = self.stream_url.split("/video")[0].split("/shot.jpg")[0]
            try:
                client = self._get_client()
                await client.get(f"{base_url}/settings/orientation?set={orientation}", timeout=2.0)
                logger.info(f"{self.name} camera orientation set to '{orientation}'")
                await client.get(f"{base_url}/settings/quality?set=35", timeout=2.0)
                logger.info(f"{self.name} camera stream quality set to 35% for low-latency 30 FPS bandwidth")
            except Exception as e:
                logger.debug(f"Error configuring camera settings for {self.name}: {e}")

    def read_battery_from_adb(self) -> Optional[int]:
        """Reads hardware battery percentage directly via ADB dumpsys."""
        adb = r"C:\Users\seanb\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v3.3.4\adb.exe"
        if not os.path.exists(adb):
            adb = shutil.which("adb") or adb
        if not os.path.exists(adb):
            return None

        try:
            res = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=1.0)
            devs = []
            for line in res.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devs.append(parts[0])

            # Auto-connect wireless ADB IP only if port is open
            if not devs and self.target_device_serial and ":" in self.target_device_serial:
                import socket
                try:
                    h, p = self.target_device_serial.split(":")
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.06)
                        if s.connect_ex((h, int(p))) == 0:
                            subprocess.run([adb, "connect", self.target_device_serial], capture_output=True, text=True, timeout=1.0)
                except Exception:
                    pass
                res2 = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=1.0)
                for line in res2.stdout.strip().splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "device":
                        devs.append(parts[0])

            if self.target_device_serial:
                if self.target_device_serial in devs:
                    devs = [self.target_device_serial]
                else:
                    return None
            else:
                devs = [d for d in devs if "192.168.1.194" not in d]

            for dev in devs:
                b_res = subprocess.run([adb, "-s", dev, "shell", "dumpsys", "battery"], capture_output=True, text=True, timeout=1.5)
                if b_res.returncode == 0:
                    m = re.search(r"level:\s*(\d+)", b_res.stdout)
                    if m:
                        val = int(m.group(1))
                        logger.info(f"🔋 Successfully read {self.name} battery level via ADB ({dev}): {val}%")
                        return val
        except Exception as e:
            logger.debug(f"read_battery_from_adb error for {self.name}: {e}")
        return None

    def read_battery_from_http(self) -> Optional[int]:
        """Queries camera stream endpoints (IP Webcam / broadcaster) for battery telemetry with fast socket pre-checks."""
        candidates = []
        if self.stream_url and self.stream_url.startswith("http"):
            parts = self.stream_url.split("/")
            if len(parts) >= 3:
                candidates.append(f"{parts[0]}//{parts[2]}")
        adb_endpoint = f"http://127.0.0.1:{self.adb_port}"
        if adb_endpoint not in candidates:
            candidates.append(adb_endpoint)
        for u in self.wifi_candidates:
            parts = u.split("/")
            if len(parts) >= 3:
                base = f"{parts[0]}//{parts[2]}"
                if base not in candidates:
                    candidates.append(base)

        import socket
        for base in candidates:
            # 60ms socket guard before attempting HTTP GET
            try:
                hp = base.replace("http://", "").replace("https://", "")
                if ":" in hp:
                    host, port_s = hp.split(":")
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.06)
                        if s.connect_ex((host, int(port_s))) != 0:
                            continue
            except Exception:
                continue

            for ep in ["/status.json", "/battery.json", "/sensors.json"]:
                try:
                    resp = httpx.get(f"{base}{ep}", timeout=0.6)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict):
                            for k in ["battery", "cur_battery", "battery_level", "level", "battery_pct"]:
                                if k in data and data[k] is not None:
                                    val = int(data[k])
                                    logger.info(f"🔋 Successfully read {self.name} battery level via HTTP ({base}{ep}): {val}%")
                                    return val
                            if "battery" in data and isinstance(data["battery"], dict):
                                val = data["battery"].get("val") or data["battery"].get("level")
                                if val is not None:
                                    val = int(val)
                                    logger.info(f"🔋 Successfully read {self.name} battery level via HTTP sensor: {val}%")
                                    return val
                except Exception:
                    continue
        return None

    def refresh_battery_from_phone(self) -> int:
        """Dynamically polls the device to read battery, caching the latest reading."""
        # 1. Try ADB (direct hardware OS telemetry)
        adb_val = self.read_battery_from_adb()
        if adb_val is not None:
            self._battery_level = adb_val
            return adb_val

        # 2. Try HTTP (IP Webcam / broadcaster)
        http_val = self.read_battery_from_http()
        if http_val is not None:
            self._battery_level = http_val
            return http_val

        return self._battery_level

    def _battery_poll_loop(self):
        """Background daemon thread checking device battery every 20 seconds."""
        while True:
            try:
                self.refresh_battery_from_phone()
            except Exception:
                pass
            time.sleep(20)

    @property
    def battery_life(self) -> int:
        return self._battery_level

    @battery_life.setter
    def battery_life(self, val: int):
        self._battery_level = val

    @property
    def wifi_signal_strength(self) -> int:
        return -45

    def get_health(self) -> Dict[str, Any]:
        has_fresh_web = bool(self._last_frame_bytes and (time.time() - self._last_frame_time < 6.0))
        has_dshow = bool(hasattr(self, "dshow_broadcaster") and self.dshow_broadcaster and self.dshow_broadcaster.latest_frame and not is_blank_or_disabled_frame(self.dshow_broadcaster.latest_frame))
        has_mjpeg = bool(hasattr(self, "broadcaster") and self.broadcaster and self.broadcaster.is_live and self.broadcaster.latest_frame and not is_blank_or_disabled_frame(self.broadcaster.latest_frame))
        is_stream = has_fresh_web or has_dshow or has_mjpeg
        return {
            "battery_percentage": self._battery_level,
            "battery_percentage_category": "good" if self._battery_level > 20 else "low",
            "wifi_signal_strength": -45,
            "device_name": self.name,
            "device_id": self.device_id,
            "is_mock": False,
            "is_phone": True,
            "stream_url": self.stream_url,
            "is_streaming": is_stream,
            "uses_pictures": not is_stream
        }

    def get_picture(self) -> Optional[bytes]:
        """Loads assigned high-res surveillance picture for S21 Ultra."""
        if self._cached_picture:
            return self._cached_picture
        candidates = [
            self.picture_path,
            Path("data/s21_picture.jpg"),
            Path("scratch/s21_landscape.jpg"),
            Path("scratch/s21_test.jpg")
        ]
        for p in candidates:
            if p.exists():
                try:
                    b = p.read_bytes()
                    if len(b) > 1000 and not is_blank_or_disabled_frame(b):
                        self._cached_picture = b
                        return b
                except Exception:
                    pass
        return None

    def set_picture(self, image_bytes: bytes) -> bool:
        """Sets/updates the active picture for S21 Ultra."""
        try:
            self.picture_path.parent.mkdir(parents=True, exist_ok=True)
            self.picture_path.write_bytes(image_bytes)
            self._cached_picture = image_bytes
            self._last_frame_bytes = image_bytes
            self._last_frame_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed setting picture for {self.name}: {e}")
            return False

    def _generate_standby_frame(self) -> bytes:
        img = Image.new("RGB", (1280, 720), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)
        # Background subtle grid lines
        for y in range(0, 720, 40):
            draw.line([(0, y), (1280, y)], fill=(30, 41, 59), width=1)
        for x in range(0, 1280, 40):
            draw.line([(x, 0), (x, 720)], fill=(30, 41, 59), width=1)
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        draw.rectangle([0, 0, 1280, 52], fill=(2, 6, 23, 230))
        draw.text((24, 16), f"CAMERA FEED: {self.name.upper()} | STANDBY", fill=(251, 191, 36))
        draw.text((950, 16), timestamp_str, fill=(148, 163, 184))
        
        draw.text((380, 310), f"{self.name} (IP Webcam)", fill=(241, 245, 249))
        primary_link = self.wifi_candidates[0] if self.wifi_candidates else f"http://127.0.0.1:{self.adb_port}"
        draw.text((320, 350), f"Listening on http://127.0.0.1:{self.adb_port} / {primary_link}", fill=(148, 163, 184))
        draw.text((350, 390), "Launch IP Webcam app on device to begin live stream", fill=(100, 116, 139))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    async def async_get_snapshot(self, **kwargs) -> Optional[bytes]:
        """Fetches fresh snapshot: prioritizes live web stream, DirectShow virtual cam, IP broadcaster; falls back to picture."""
        # 1. Direct Web Browser stream (from /mobile_cam?cam=s21)
        if self._last_frame_bytes and (time.time() - self._last_frame_time < 6.0):
            if not is_blank_or_disabled_frame(self._last_frame_bytes):
                return self._last_frame_bytes

        # 2. Windows Virtual Camera (DirectShow Index 1 - Sean's S22 Ultra (Windows Virtual Camera))
        if hasattr(self, "dshow_broadcaster") and self.dshow_broadcaster:
            self.dshow_broadcaster.start()
            f = self.dshow_broadcaster.latest_frame
            if f and not is_blank_or_disabled_frame(f):
                return f
            for _ in range(2):
                await asyncio.sleep(0.03)
                f = self.dshow_broadcaster.latest_frame
                if f and not is_blank_or_disabled_frame(f):
                    return f

        # 3. Broadcaster 30 FPS buffer (Real-time Instant Snapshot from IP Webcam)
        if hasattr(self, "broadcaster") and self.broadcaster and self.broadcaster.is_live:
            if self.broadcaster.latest_frame and not is_blank_or_disabled_frame(self.broadcaster.latest_frame):
                return self.broadcaster.latest_frame

        # 4. Desktop window screen capture of Phone Link mirroring ("S21" or "S22")
        win_frame = capture_desktop_window_frame("S21") or capture_desktop_window_frame("S22")
        if win_frame and not is_blank_or_disabled_frame(win_frame):
            return win_frame

        # 5. Last frame bytes if available
        if self._last_frame_bytes and not is_blank_or_disabled_frame(self._last_frame_bytes):
            return self._last_frame_bytes

        # 6. Always fallback to the assigned picture
        pic = self.get_picture()
        if pic:
            return pic

        # 7. Standby frame
        return self._generate_standby_frame()


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
        candidates = [
            Path(f"scratch/{self.name}_direct.jpg"),
            Path(f"scratch/{self.name.lower()}_direct.jpg"),
            Path(f"scratch/{self.name.lower()}_test.jpg"),
            Path("scratch/Garden_direct.jpg") if "garden" in self.name.lower() else Path("scratch/cam1_direct.jpg"),
            Path("scratch/garden_test.jpg") if "garden" in self.name.lower() else Path("scratch/cam1_test.jpg")
        ]
        for c in candidates:
            if c.exists():
                try:
                    data = c.read_bytes()
                    if len(data) > 1000 and not is_blank_or_disabled_frame(data):
                        return data
                except Exception:
                    pass

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
        self._garden_cam = MockRingCamera("Garden")
        self._cam1_cam = MockRingCamera("cam1")
        self._local_cam = LocalRolandCamera("Local Camera (Roland 1)", 0)
        from src.config import config
        phone_url = getattr(config.ring, "phone_camera_url", "http://192.168.1.150:8080/video")
        self._phone_cam = AndroidPhoneCamera(
            name="Samsung Galaxy S21 Ultra",
            device_id="phone-cam-s21-ultra",
            model="Samsung Galaxy S21 Ultra (Webcam / Wireless Stream)",
            adb_port=8085,
            wifi_candidates=["http://192.168.1.165:8080/video", "http://192.168.1.150:8080/video"],
            stream_url=phone_url
        )
        self._tab_cam = GalaxyTabWindowsCamera(
            name="Galaxy Tab A11+",
            camera_index=2
        )
        # Garden and cam1 are ALWAYS present in _all_cameras from millisecond 0
        self._all_cameras = [self._garden_cam, self._cam1_cam, self._local_cam, self._phone_cam, self._tab_cam]
        self._active_camera = self._garden_cam
        self._snapshot_cache: Dict[str, bytes] = {}
        self._last_event_ids: Dict[str, str] = {}
        self._last_vod_trigger_times: Dict[str, float] = {}
        self._http_session = None
        self._reconnect_task = None

        # Pre-seed snapshot cache from scratch directory if files exist
        for cam_key, fpath in [
            ("Garden", "scratch/Garden_direct.jpg"),
            ("Garden", "scratch/garden_test.jpg"),
            ("cam1", "scratch/cam1_direct.jpg"),
            ("cam1", "scratch/cam1_test.jpg"),
        ]:
            p = Path(fpath)
            if p.exists() and cam_key not in self._snapshot_cache:
                try:
                    b = p.read_bytes()
                    if len(b) > 1000 and not is_blank_or_disabled_frame(b):
                        self._snapshot_cache[cam_key] = b
                except Exception:
                    pass

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

    async def _background_ring_reconnect_loop(self):
        """Continuously retries connecting to Ring API in background until successful."""
        logger.info("Starting background Ring reconnect loop...")
        while True:
            await asyncio.sleep(10)
            try:
                import socket
                import aiohttp
                import ring_doorbell.auth as auth_mod
                from ring_doorbell import Auth, Ring

                auth_mod.TIMEOUT = 45.0

                if self._http_session is None or self._http_session.closed:
                    connector = aiohttp.TCPConnector(family=socket.AF_INET)
                    timeout = aiohttp.ClientTimeout(total=45, connect=25)
                    self._http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

                with open(self.token_file, "r", encoding="utf-8") as f:
                    token_data = json.load(f)

                self._auth = Auth("RodentIdentification/1.0", token_data, self._token_updater, http_client_session=self._http_session)
                self._ring = Ring(self._auth)
                await self._ring.async_update_data()

                devices = self._ring.devices()
                ring_cams = list(devices.stickup_cams) + list(devices.doorbells)
                ring_cams = [c for c in ring_cams if "outhouse" not in getattr(c, "name", "").lower()]

                for rc in ring_cams:
                    if "garden" in rc.name.lower():
                        self._garden_cam = rc
                    elif "cam1" in rc.name.lower() or "cam 1" in rc.name.lower():
                        self._cam1_cam = rc

                self._all_cameras = [self._garden_cam, self._cam1_cam, self._local_cam, self._phone_cam, self._tab_cam]
                self._is_mock = False
                logger.info(f"✅ Successfully reconnected to Ring API in background! Found {len(ring_cams)} live devices.")
                break
            except Exception as e:
                logger.debug(f"Background Ring reconnect attempt: {e}")

    async def async_connect(self):
        """Connects to Ring API and prioritizes Garden and cam1 while filtering out Outhouse."""
        if not self.token_file.exists():
            logger.warning(f"Ring token file '{self.token_file}' not found.")
            return

        try:
            import socket
            import aiohttp
            import ring_doorbell.auth as auth_mod
            from ring_doorbell import Auth, Ring

            auth_mod.TIMEOUT = 45.0

            if self._http_session is None or self._http_session.closed:
                connector = aiohttp.TCPConnector(family=socket.AF_INET)
                timeout = aiohttp.ClientTimeout(total=45, connect=25)
                self._http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            with open(self.token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)

            self._auth = Auth("RodentIdentification/1.0", token_data, self._token_updater, http_client_session=self._http_session)
            self._ring = Ring(self._auth)
            await self._ring.async_update_data()

            devices = self._ring.devices()
            ring_cams = list(devices.stickup_cams) + list(devices.doorbells)

            # Filter out Outhouse device completely (replaced by Galaxy Tab A11+)
            ring_cams = [c for c in ring_cams if "outhouse" not in getattr(c, "name", "").lower()]

            # Sort Ring cameras so Garden and cam1 are first
            def _sort_key(c):
                name = getattr(c, "name", "").lower()
                if "garden" in name: return 0
                if "cam1" in name or "cam 1" in name: return 1
                return 2

            ring_cams.sort(key=_sort_key)

            for rc in ring_cams:
                if "garden" in rc.name.lower():
                    self._garden_cam = rc
                elif "cam1" in rc.name.lower() or "cam 1" in rc.name.lower():
                    self._cam1_cam = rc

            # Combine Ring cameras (first) with Local Roland 1 Camera, S21 Ultra, and Galaxy Tab A11+
            self._all_cameras = [self._garden_cam, self._cam1_cam, self._local_cam, self._phone_cam, self._tab_cam]

            # Match active camera
            if self.device_name:
                matched = self.find_camera(self.device_name)
                self._active_camera = matched or self._garden_cam
            else:
                self._active_camera = self._garden_cam

            self._is_mock = False
            logger.info(f"Connected to Ring API. Discovered {len(ring_cams)} Ring devices: {[c.name for c in ring_cams]}. Total cameras: {[c.name for c in self._all_cameras]}. Active: '{self.camera_name}'")

        except Exception as e:
            logger.warning(f"Could not connect to Ring API at startup (will retry in background): {e}")
            # Ensure Garden and cam1 are never dropped
            if not any("garden" in getattr(c, "name", "").lower() for c in self._all_cameras):
                self._all_cameras.insert(0, self._garden_cam)
            if not any("cam1" in getattr(c, "name", "").lower() for c in self._all_cameras):
                self._all_cameras.insert(1, self._cam1_cam)

            # Launch background reconnect loop
            if not self._reconnect_task or self._reconnect_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self._reconnect_task = loop.create_task(self._background_ring_reconnect_loop())
                except RuntimeError:
                    pass

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Returns all available cameras (Ring Garden, cam1, Galaxy Tab A11+, S21 Ultra, Local Roland)."""
        results = []
        for cam in self._all_cameras:
            name = getattr(cam, "name", "Camera")
            if "outhouse" in name.lower():
                continue
            is_tab = isinstance(cam, GalaxyTabWindowsCamera) or "tab" in name.lower()
            is_local = isinstance(cam, LocalRolandCamera) and not is_tab
            is_phone = isinstance(cam, AndroidPhoneCamera)
            is_ring = not is_local and not is_phone and not is_tab and not self._is_mock
            bat = getattr(cam, "battery_life", None)
            if is_local: bat = 100
            elif is_phone or is_tab: bat = getattr(cam, "battery_life", 80)
            
            is_streaming = False
            if is_tab:
                has_fresh_web = bool(getattr(cam, "_last_frame_bytes", None) and (time.time() - getattr(cam, "_last_frame_time", 0.0) < 6.0))
                has_dshow = bool(hasattr(cam, "broadcaster") and cam.broadcaster and cam.broadcaster.latest_frame and not is_blank_or_disabled_frame(cam.broadcaster.latest_frame))
                is_streaming = has_fresh_web or has_dshow
            elif is_local:
                is_streaming = True
            elif is_phone:
                has_fresh_web = bool(getattr(cam, "_last_frame_bytes", None) and (time.time() - getattr(cam, "_last_frame_time", 0.0) < 6.0))
                has_dshow = bool(hasattr(cam, "dshow_broadcaster") and cam.dshow_broadcaster and cam.dshow_broadcaster.latest_frame and not is_blank_or_disabled_frame(cam.dshow_broadcaster.latest_frame))
                has_mjpeg = bool(hasattr(cam, "broadcaster") and cam.broadcaster and cam.broadcaster.is_live and cam.broadcaster.latest_frame and not is_blank_or_disabled_frame(cam.broadcaster.latest_frame))
                is_streaming = has_fresh_web or has_dshow or has_mjpeg

            results.append({
                "name": name,
                "id": getattr(cam, "id", None) or getattr(cam, "device_id", None),
                "model": getattr(cam, "model", "Stick Up Cam" if is_ring else "Camera"),
                "battery_percentage": int(bat) if bat is not None else None,
                "wifi_signal_strength": getattr(cam, "wifi_signal_strength", None),
                "is_ring": is_ring,
                "is_local": is_local,
                "is_phone": is_phone or is_tab,
                "is_streaming": is_streaming,
                "uses_pictures": (is_phone or is_tab) and not is_streaming,
                "is_active": (self._active_camera and self._active_camera.name.lower() == cam.name.lower())
            })
        return results

    def find_camera(self, camera_name: Optional[str]) -> Optional[Any]:
        """Resolves a camera object by direct name or common aliases (Garden, Cam1, S21, Tab, Outhouse)."""
        if not camera_name:
            return self._active_camera
        c_low = camera_name.lower().strip()
        matched = next((c for c in self._all_cameras if c.name.lower() == c_low), None)
        if matched:
            return matched
        # Alias Outhouse and Tab queries to Galaxy Tab A11+
        if any(k in c_low for k in ["tab", "a11", "galaxy tab", "tablet", "outhouse"]):
            return getattr(self, "_tab_cam", None)
        if any(k in c_low for k in ["s21", "s1", "phone"]):
            return getattr(self, "_phone_cam", None)
        if "garden" in c_low:
            m = next((c for c in self._all_cameras if "garden" in c.name.lower()), None)
            return m or getattr(self, "_garden_cam", None)
        if "cam1" in c_low or "cam 1" in c_low:
            m = next((c for c in self._all_cameras if "cam1" in c.name.lower() or "cam 1" in c.name.lower()), None)
            return m or getattr(self, "_cam1_cam", None)
        if any(k in c_low for k in ["local", "roland", "usb", "webcam"]):
            return getattr(self, "_local_cam", None)
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
            return getattr(cam, "battery_life", 50)
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

            history = await cam.async_history(limit=5)
            if not history:
                return self.create_standby_frame("No recorded events found", cam_name), True, False

            # Find the most recent event with an accessible recording URL
            valid_event = None
            url = None
            for event in history:
                e_id = event.get("id")
                if not e_id:
                    continue
                try:
                    u = await cam.async_recording_url(e_id)
                    if u:
                        valid_event = event
                        url = u
                        break
                except Exception:
                    pass

            if not valid_event or not url:
                cached = self._snapshot_cache.get(cam_name)
                return cached or self.create_standby_frame("Awaiting new recording", cam_name), False, False

            event_id = str(valid_event.get("id"))
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
            # Case 1: Local Roland 1 Camera, Android Phone Camera, or Galaxy Tab Camera
            if isinstance(target_cam, (LocalRolandCamera, AndroidPhoneCamera)):
                snap = await target_cam.async_get_snapshot()
                if snap and not is_blank_or_disabled_frame(snap):
                    self._snapshot_cache[cam_name] = snap
                    if camera_name:
                        self._snapshot_cache[camera_name] = snap
                    if isinstance(target_cam, AndroidPhoneCamera):
                        if "s21" in cam_name.lower():
                            self._snapshot_cache["S21"] = snap
                        elif any(k in cam_name.lower() for k in ["tab", "a11"]):
                            self._snapshot_cache["Galaxy Tab A11+"] = snap
                            self._snapshot_cache["Tab A11+"] = snap
                    return snap, None, False, True

                # If snap returned blank/disabled, attempt picture fallback
                if hasattr(target_cam, "get_picture"):
                    pic = target_cam.get_picture()
                    if pic and not is_blank_or_disabled_frame(pic):
                        self._snapshot_cache[cam_name] = pic
                        if camera_name:
                            self._snapshot_cache[camera_name] = pic
                        return pic, None, False, True

                return None, f"Could not open stream for {target_cam.name}", False, False

            # Case 2: Mock Camera
            if self._is_mock or isinstance(target_cam, MockRingCamera):
                snap = await target_cam.async_get_snapshot()
                self._snapshot_cache[cam_name] = snap
                return snap, None, False, True

            # Case 3: Live Ring Camera snapshot via Ring Snapshot Cloud Endpoint
            doorbot_id = getattr(target_cam, "_attrs", {}).get("id") or getattr(target_cam, "id", None)
            if doorbot_id and self._ring:
                try:
                    from ring_doorbell.const import SNAPSHOT_ENDPOINT, SNAPSHOT_TIMESTAMP_ENDPOINT
                    snap_resp = await self._ring.async_query(SNAPSHOT_ENDPOINT.format(doorbot_id))
                    sc = getattr(snap_resp, "status_code", None) or getattr(snap_resp, "status", None)
                    if sc == 200 and len(snap_resp.content) > 1000:
                        snap_bytes = snap_resp.content
                        self._snapshot_cache[cam_name] = snap_bytes
                        if camera_name:
                            self._snapshot_cache[camera_name] = snap_bytes
                        try:
                            Path(f"scratch/{cam_name}_direct.jpg").write_bytes(snap_bytes)
                        except Exception:
                            pass
                        try:
                            asyncio.create_task(self._ring.async_query(SNAPSHOT_TIMESTAMP_ENDPOINT, method="POST", json={"doorbot_ids": [doorbot_id]}))
                        except Exception:
                            pass
                        logger.info(f"📸 Live daylight Ring snapshot refreshed for {cam_name} ({len(snap_bytes)} bytes)")
                        return snap_bytes, None, False, True
                except Exception as e:
                    logger.debug(f"Direct Ring snapshot query failed for {cam_name}: {e}")

            # Case 4: Live Ring Camera event frame fallback
            rec_frame, is_standby, is_new = await self._fetch_frame_from_latest_recording(target_cam)
            if rec_frame and not is_blank_or_disabled_frame(rec_frame):
                self._snapshot_cache[cam_name] = rec_frame
                return rec_frame, None, is_standby, is_new

            # Check in-memory cache
            if cam_name in self._snapshot_cache and not is_blank_or_disabled_frame(self._snapshot_cache[cam_name]):
                return self._snapshot_cache[cam_name], None, False, False

            # Check disk fallback from scratch
            candidates = [
                Path(f"scratch/{cam_name}_direct.jpg"),
                Path(f"scratch/{cam_name.lower()}_direct.jpg"),
                Path(f"scratch/{cam_name.lower()}_test.jpg"),
                Path("scratch/Garden_direct.jpg") if "garden" in cam_name.lower() else Path("scratch/cam1_direct.jpg"),
            ]
            for p in candidates:
                if p.exists():
                    try:
                        b = p.read_bytes()
                        if len(b) > 1000 and not is_blank_or_disabled_frame(b):
                            self._snapshot_cache[cam_name] = b
                            return b, None, False, False
                    except Exception:
                        pass

            standby = self.create_standby_frame("Standby", cam_name)
            return standby, None, True, False

        except Exception as e:
            logger.error(f"Error capturing snapshot for {cam_name}: {e}")
            if cam_name in self._snapshot_cache and not is_blank_or_disabled_frame(self._snapshot_cache[cam_name]):
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
