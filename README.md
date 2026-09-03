# Sovereign Rat Watch: S21 + DGX Spark Gemma 4 26B + Wear OS

A 100% on-premises, zero-cloud rat detection and validation appliance that achieves an end-to-end detection lead time of **under 2 seconds**:

1. **Samsung Galaxy S21 (`uk.local.ratwatch.phone`)**: 
   - Camera2 1280x720 @ 15fps with hardware stabilization bypassed.
   - 160x90 grayscale motion gate ($\le 15\text{ ms}$).
   - On-device single-class `rat` YOLO11n INT8 ($416\times 416$) executing in $\le 60\text{ ms}$ on Exynos 2100 / $\le 35\text{ ms}$ on Snapdragon 888.
   - 40% expanded context crop ($\le 640\text{px}$, JPEG q72, $\le 80\text{ KB}$).
   - Non-blocking HTTP multipart POST to DGX Spark in $\le 150\text{ ms}$ from trigger frame.
2. **NVIDIA DGX Spark 128 GB (`ratwatch-spark` + vLLM Port 8000 & 8088)**:
   - vLLM serving `Gemma-4-26B-A4B-IT-NVFP4` with `--moe-backend marlin`, `--kv-cache-dtype fp8`, vision token budget $= 280$, and thinking mode **OFF**.
   - Validates sighting in $1.2 - 1.65\text{ s}$ with room/garden-relative spatial location text.
   - Pushes direct LAN alerts to Wear OS watch in $\le 45\text{ ms}$.
3. **Wear OS Watch (`uk.local.ratwatch.watch`)**:
   - Listens on `http://0.0.0.0:8099/alert` over local Wi-Fi.
   - Triggers 400ms haptic vibration with a 240px thumbnail and 1-line location banner (`"Rat along shed plinth"`).
4. **Existing Web Front End Dashboard**:
   - Reuses current interactive UI with live feed, large scrollable target dropdown, chronological carousel, and **small center target reticle**.
   - Receives instant `possible` ($\le 130\text{ ms}$) and confirmed `verdict` / `rejected` WebSocket events.

---

## 1. System Topology & Latency Budget

```
[Rat Enters Scene]
       │
       ▼ (≤10 ms)
[Camera2 1280x720 @ 15fps]
       │
       ▼ (≤15 ms)
[Motion Gate (160x90 AbsDiff > 4%)]
       │
       ▼ (≤60 ms on Exynos / ≤35 ms on Snapdragon)
[YOLO11n INT8 (416x416 Input, conf_send ≥ 0.28)]
       │
       ├─────────────────────────────────────────► Web Front End: "possible" (≤130 ms)
       ▼ (≤25 ms)
[40% Context Crop (max edge 640px, JPEG q72 ≤80KB)]
       │
       ▼ (≤20 ms LAN HTTP POST)
[NVIDIA DGX Spark: ratwatch-spark Port 8088]
       │
       ▼ (1200 - 1650 ms)
[vLLM: Gemma 4 26B-A4B NVFP4, Marlin MoE, Vision=280 tokens, Thinking OFF]
       │
       ├─────────────────────────────────────────► Web Front End: "verdict" (≤15 ms)
       ▼ (≤45 ms LAN HTTP POST)
[Wear OS Watch: 400ms Haptic Buzz + 240px Crop]
```

**Total Lead Time: $\mathbf{1.38 - 1.84\text{ seconds}}$** (Hard Target: $< 2.0\text{s}$).

---

## 2. Directory Structure

```
.
├── android/          # S21 Android app (uk.local.ratwatch.phone)
│   ├── app/src/main/
│   │   ├── AndroidManifest.xml
│   │   └── java/uk/local/ratwatch/phone/
│   │       ├── CameraService.kt   # Screen-off foreground Camera2 service
│   │       ├── MotionGate.kt      # 160x90 absdiff motion filter
│   │       ├── YoloDetector.kt    # Dual-delegate YOLO11n INT8 detector
│   │       ├── NetworkPoster.kt   # 40% expanded crop builder & POST
│   │       └── MainActivity.kt    # 3-screen minimal view: Arm, Mount, Log
│   └── build.gradle.kts
├── wear/             # Wear OS watch app (uk.local.ratwatch.watch)
│   ├── app/src/main/
│   │   ├── AndroidManifest.xml
│   │   └── java/uk/local/ratwatch/watch/
│   │       ├── AlertServer.kt     # Port 8099 embedded LAN alert server
│   │       └── MainActivity.kt    # 400ms haptic buzz + 240px thumbnail
│   └── build.gradle.kts
├── spark/            # DGX Spark 128GB validation service
│   ├── ratwatch_api.py            # FastAPI service + WebSocket stream
│   ├── docker-compose.yml         # vLLM Gemma 4 26B + API stack
│   ├── Dockerfile.api
│   ├── requirements.txt
│   └── systemd/                   # Optional systemd service definitions
│       ├── vllm.service
│       └── ratwatch-api.service
├── models/           # Calibration recipes, dataset gen & checksums
│   └── README.md
├── src/              # Current Python application server & endpoints
├── static/           # Current Web Front End UI & center target reticle
├── tests/            # Test suite (17/17 tests passing)
└── README.md
```

---

## 3. Installation & Deployment Order

### Step 1: Deploy DGX Spark 128 GB
On your NVIDIA DGX Spark machine:
```bash
cd spark
docker compose up -d --build
```
Verify health:
```bash
curl http://localhost:8088/health
# {"ok": true, "model": "gemma4-26b", "warm": true, "infer_p50_ms": 1380}
```

### Step 2: Build & Install S21 Phone App
```bash
cd android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```
Open **Rat Watch S21** and tap **ARM DETECTOR**.

### Step 3: Build & Install Wear OS Watch App
```bash
cd wear
./gradlew assembleDebug
adb -s <watch-ip>:5555 install app/build/outputs/apk/debug/app-debug.apk
```

---

## 4. Garden Mounting & Calibration Sheet

1. **Height:** Mount the phone on a garden stand **0.4 m to 1.2 m** off the ground.
2. **Angle:** Point downward across ground runs (compost bin base, shed plinth, decking edge, or fence line).
3. **Lighting:** Lock Auto-Exposure (AE) and Focus (AF) to avoid night pumping in low-light environments.
4. **Night Boost:** Use the Web Dashboard's **Night Boost** clarity filter for enhanced contrast on dark pavement.

---

## 5. Running the Test Suite

```bash
uv run pytest
```
All 17 automated tests pass (API, Vision Engine, DGX Spark Validator, Storage, Multi-Object Filter).
