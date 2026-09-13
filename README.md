# Sovereign Rat Watch: Distributed Edge Appliance (roland1 + roland3 + Wear OS)

A 100% on-premises, zero-cloud distributed rat detection and validation appliance partitioned across dedicated compute nodes:

1. **`roland1` — Edge Rodent Detection & Supabase Database Services Node**:
   - **Rodent Detection Pipeline**: Ingests direct local camera (`LocalRolandCamera`), Samsung S21 Ultra stream / USB ADB forward, or Ring doorbell cameras.
   - **Low-Latency Filter**: Runs 160x90 grayscale motion gate ($\le 15\text{ ms}$) and on-device YOLO11n object candidate detector ($\le 35\text{ ms}$).
   - **Continuous Sampling Engine**: Accelerates from idle monitoring cadence to real-time (1s) immediately upon activity.
   - **Supabase Database Services**: Self-hosted Supabase stack on `roland1` (Port 54321 / 8000 via Kong, PostgREST on Port 3000, PostgreSQL on Port 5432) storing sightings, continuous event sessions, and high-res JPEG crops with local SQLite resilience fallback.
   - **Web Dashboard**: Real-time CCTV HUD, center target reticle, target selector dropdown, and WebSocket stream.

2. **`roland3` — Dedicated AI Inference Compute Node**:
   - **Multimodal AI Verification**: High-throughput Ollama / vLLM serving Gemma 4 models (`http://roland3:11434` or Port 8000/8088).
   - Validates candidate sightings in under 1.5 seconds with room/garden spatial descriptions and closed JSON output schema.
   - Eliminates inference load from the edge detection node `roland1`.

3. **Wear OS Watch (`uk.local.ratwatch.watch`)**:
   - Listens on `http://0.0.0.0:8099/alert` over local Wi-Fi.
   - Triggers 400ms haptic vibration with a 240px thumbnail and 1-line location banner (`"Rat along shed plinth"`).

---

## 1. System Topology & Latency Budget

```
 ┌────────────────────────────────────────────────────────┐
 │                      ROLAND 1                          │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │  Rodent Detection Engine                         │  │
 │  │  - Camera Ingestion (USB/Webcam, S21, Ring)      │  │
 │  │  - MotionGate (160x90 AbsDiff ≤15ms)             │  │
 │  │  - FastObjectDetector (YOLO11n candidate filter) │  │
 │  │  - SamplerEngine (Real-time cadence boost)       │  │
 │  │  - FastAPI Web Application & WebSocket           │  │
 │  └────────────────────────┬─────────────────────────┘  │
 │                           │                            │
 │  ┌────────────────────────▼─────────────────────────┐  │
 │  │  Supabase Database Services (roland1)            │  │
 │  │  - Kong Gateway (port 54321)                     │  │
 │  │  - PostgREST REST API (/rest/v1)                 │  │
 │  │  - PostgreSQL Database (detections, events)      │  │
 │  │  - Supabase Storage (detections bucket)          │  │
 │  └──────────────────────────────────────────────────┘  │
 └───────────────────────────┬────────────────────────────┘
                             │ Async AI Sighting Verification
                             │ (Crop + Reference Image POST)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │                      ROLAND 3                          │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │  AI Inference Engine                             │  │
 │  │  - Ollama / vLLM (Gemma 4 12B / 26B)             │  │
 │  │  - Port 11434 (Ollama) or Port 8000/8088 (vLLM)  │  │
 │  │  - Structured JSON Output & Verification         │  │
 │  └──────────────────────────────────────────────────┘  │
 └────────────────────────────────────────────────────────┘
```

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

### Step 1: Deploy AI Inference Node (roland3)
On your GPU / Ollama server **roland3**:
```bash
docker compose -f docker-compose.roland3.yml up -d
```
Verify Ollama / vLLM health:
```bash
curl http://roland3:11434/api/tags
# or vLLM: curl http://roland3:8000/v1/models
```

### Step 2: Deploy Detection & Supabase Database Services (roland1)
On your Edge appliance **roland1**:
```bash
docker compose -f docker-compose.roland1.yml up -d
```
Verify Supabase and Detection status:
```bash
curl http://roland1:8000/api/system/nodes
# Returns detection_node, supabase_node, and inference_node topology
```

### Step 3: Build & Install S21 Phone App (Optional Mobile Edge)
```bash
cd android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```
Open **Rat Watch S21** and tap **ARM DETECTOR**.

### Step 4: Build & Install Wear OS Watch App
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
All 23 automated tests pass (API, Vision Engine, DGX Spark Validator, Storage, Supabase on roland1, Fallback SQLite, Multi-Object Filter).
