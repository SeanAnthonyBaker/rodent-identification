import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger("notifier")

class WatchNotificationManager:
    """Manages push, Telegram, Webhook, and voice notifications to Android phones & WAN-connected Zepp OS smartwatches."""

    def __init__(
        self,
        enabled: bool = True,
        voice_alert: bool = True,
        webhook_url: Optional[str] = None,
        ntfy_topic: str = "rodentwatch_alerts",
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        cooldown_seconds: int = 60
    ):
        self.enabled = enabled
        self.voice_alert = voice_alert
        self.webhook_url = webhook_url
        self.ntfy_topic = ntfy_topic
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_timestamps: Dict[str, float] = {}
        self.audio_dir = Path("data/audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.alert_wav_path = self.audio_dir / "rat_alert.wav"

    def should_alert(self, device_name: str, object_type: str = "rat") -> bool:
        """Returns True if the cooldown period has expired for this camera/object pair."""
        key = f"{device_name.lower().strip()}:{object_type.lower().strip()}"
        last_ts = self._last_alert_timestamps.get(key, 0.0)
        return (time.time() - last_ts) >= self.cooldown_seconds

    def record_alert(self, device_name: str, object_type: str = "rat"):
        """Records the timestamp of an emitted alert for deduplication."""
        key = f"{device_name.lower().strip()}:{object_type.lower().strip()}"
        self._last_alert_timestamps[key] = time.time()

    def clear_cooldown(self, device_name: Optional[str] = None, object_type: Optional[str] = None):
        """Clears cooldown timestamps when a scene is confirmed clear."""
        if device_name and object_type:
            key = f"{device_name.lower().strip()}:{object_type.lower().strip()}"
            self._last_alert_timestamps.pop(key, None)
        elif device_name:
            for k in list(self._last_alert_timestamps.keys()):
                if k.startswith(f"{device_name.lower().strip()}:"):
                    self._last_alert_timestamps.pop(k, None)
        else:
            self._last_alert_timestamps.clear()

    def generate_voice_file(self, text: str = "Warning! A target has been detected on the camera.") -> Optional[Path]:
        """Generates a WAV speech file using native Windows SAPI TTS."""
        try:
            ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{str(self.alert_wav_path.resolve())}')
$synth.Speak('{text}')
$synth.Dispose()
"""
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True, timeout=10)
            if self.alert_wav_path.exists() and self.alert_wav_path.stat().st_size > 0:
                logger.info(f"Voice alert audio generated ({self.alert_wav_path.stat().st_size} bytes)")
                return self.alert_wav_path
            else:
                logger.warning(f"Voice generation script finished without file: {res.stderr}")
        except Exception as e:
            logger.warning(f"Voice synthesis error: {e}")
        return None

    async def notify_animal_detected(
        self,
        confidence: float,
        description: str,
        device_name: str,
        object_type: str = "rat",
        label: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        detection_id: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """Sends immediate push and voice notifications for Pheasant or Rat across WAN."""
        if not self.enabled:
            return {"sent": False, "reason": "Notifications disabled"}

        if not force and not self.should_alert(device_name, object_type):
            logger.info(f"Notification suppressed for {device_name} ({object_type}) - alert cooldown active ({self.cooldown_seconds}s)")
            return {"sent": False, "reason": "Alert suppressed by deduplication cooldown", "suppressed": True}

        self.record_alert(device_name, object_type)

        ts_str = (timestamp or datetime.now()).strftime("%I:%M:%S %p")
        confidence_pct = int(confidence * 100)
        
        ot = object_type.lower()
        if ot in ["tree", "trees"]:
            display_label = label or "Tree"
            emoji = "🌲"
        elif ot in ["bird", "birds", "pheasant"]:
            display_label = label or ("Pheasant" if "pheasant" in ot else "Bird")
            emoji = "🦚" if "pheasant" in display_label.lower() else "🐦"
        elif ot in ["horse", "horses", "pony", "equine"]:
            display_label = label or "Horse"
            emoji = "🐴"
        elif ot in ["horses_poo", "horse_poo", "horses poo", "poo", "manure"]:
            display_label = label or "Horses poo"
            emoji = "🐴💩"
        else:
            display_label = label or "Rat"
            emoji = "🐀"

        title = f"{display_label.upper()} DETECTED"
        body = f"{emoji} {display_label.upper()} DETECTED on {device_name} at {ts_str} ({confidence_pct}% confidence).\n{description[:120]}"

        # 1. Generate Voice Audio
        if self.voice_alert:
            voice_text = f"Notice: {display_label} has been detected on {device_name}."
            await asyncio.get_event_loop().run_in_executor(None, self.generate_voice_file, voice_text)

        results = {}

        # 2. Push via ntfy (WAN cloud push with TTS)
        if self.ntfy_topic:
            try:
                headers = {
                    "Title": title,
                    "Priority": "urgent",
                    "Tags": f"warning,{object_type.lower()},loudspeaker",
                    "X-TTS": f"Warning: {display_label} detected on camera!"
                }
                if detection_id:
                    headers["Click"] = f"http://localhost:8000"

                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(
                        f"https://ntfy.sh/{self.ntfy_topic}",
                        data=body.encode("utf-8"),
                        headers=headers
                    )
                    results["ntfy"] = (resp.status_code == 200)
                    logger.info(f"Sent watch push notification to ntfy.sh/{self.ntfy_topic} (Status: {resp.status_code})")
            except Exception as e:
                logger.warning(f"Failed sending ntfy watch notification: {e}")
                results["ntfy_error"] = str(e)

        # 3. Push via Telegram Bot (Direct WAN cloud delivery to smartwatch)
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                tg_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                tg_payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"{emoji} *{display_label.upper()} DETECTED!*\n\n📹 *Camera:* {device_name}\n🕒 *Time:* {ts_str}\n🎯 *Confidence:* {confidence_pct}%\n\n_{description[:150]}_",
                    "parse_mode": "Markdown"
                }
                async with httpx.AsyncClient(timeout=8.0) as client:
                    tg_resp = await client.post(tg_url, json=tg_payload)
                    results["telegram"] = (tg_resp.status_code == 200)
                    logger.info(f"Sent Telegram watch alert (Status: {tg_resp.status_code})")
            except Exception as e:
                logger.warning(f"Failed sending Telegram alert: {e}")
                results["telegram_error"] = str(e)

        # 4. Custom Webhook (Home Assistant, Tasker, Node-RED, MacroDroid)
        if self.webhook_url:
            try:
                payload = {
                    "event": f"{object_type.lower()}_detected",
                    "object_type": object_type,
                    "label": display_label,
                    "title": title,
                    "message": body,
                    "voice_message": f"Warning: {display_label} detected on camera!",
                    "confidence": confidence,
                    "device_name": device_name,
                    "timestamp": (timestamp or datetime.now()).isoformat()
                }
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(self.webhook_url, json=payload)
                    results["webhook"] = (resp.status_code in (200, 201, 204))
            except Exception as e:
                logger.warning(f"Failed sending custom webhook: {e}")
                results["webhook_error"] = str(e)

        return {"sent": True, "details": results}

    async def notify_rat_detected(
        self,
        confidence: float,
        description: str,
        device_name: str,
        timestamp: Optional[datetime] = None,
        detection_id: Optional[int] = None,
        object_type: str = "rat",
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """Backwards-compatible wrapper."""
        return await self.notify_animal_detected(
            confidence=confidence,
            description=description,
            device_name=device_name,
            object_type=object_type,
            label=label,
            timestamp=timestamp,
            detection_id=detection_id
        )
