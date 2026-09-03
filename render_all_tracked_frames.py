import cv2
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.config import config
from src.storage import StorageManager

frames_dir = Path("data/video_frames")
annotated_dir = Path("data/detections/annotated_tracking")
annotated_dir.mkdir(parents=True, exist_ok=True)

storage = StorageManager(
    db_path=config.storage.db_path,
    detections_dir=config.storage.detections_dir
)

# Key frames and exact coordinates of the rat in the video frames
rat_frames = [
    {
        "file": "frame_03s.jpg",
        "t": 3.0,
        "conf": 0.95,
        "desc": "Brown rat detected on lawn entering garden near feeder.",
        "rat_top": (525, 935), # (cx, top of rat)
    },
    {
        "file": "frame_04s.jpg",
        "t": 4.0,
        "conf": 0.95,
        "desc": "Brown rat running across grass under bird table.",
        "rat_top": (510, 935),
    },
    {
        "file": "frame_05s.jpg",
        "t": 5.0,
        "conf": 0.88,
        "desc": "Brown rat moving under bird table base on lawn.",
        "rat_top": (490, 930),
    }
]

for item in rat_frames:
    fpath = frames_dir / item["file"]
    if not fpath.exists():
        continue

    img = cv2.imread(str(fpath))
    if img is None:
        continue

    cx, rat_y = item["rat_top"]
    cy = rat_y - 28 # Positioned consistently JUST ABOVE the rat

    annotated = img.copy()

    # Draw glowing target reticle: outer ring, bullseye, crosshairs
    cv2.circle(annotated, (cx, cy), 22, (0, 165, 255), 2, cv2.LINE_AA) # Outer ring
    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA) # Bullseye
    # Crosshair ticks
    cv2.line(annotated, (cx - 28, cy), (cx + 28, cy), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(annotated, (cx, cy - 28), (cx, cy + 28), (0, 255, 255), 2, cv2.LINE_AA)

    # Downward pointer line extending from reticle directly down to top of rat
    cv2.line(annotated, (cx, cy + 22), (cx, rat_y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(annotated, (cx, rat_y), 4, (0, 255, 0), -1, cv2.LINE_AA)

    # Floating red badge placed cleanly above the reticle
    badge_text = f"RAT {item['conf']*100:.0f}%"
    cv2.rectangle(annotated, (cx - 65, cy - 56), (cx + 65, cy - 30), (0, 0, 220), -1)
    cv2.putText(annotated, badge_text, (cx - 52, cy - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    out_file = annotated_dir / f"tracked_{item['file']}"
    cv2.imwrite(str(out_file), annotated)
    print(f"Saved {out_file.name} (Reticle at {cx}, {cy} - Rat at {rat_y})")

    # Update gallery db
    _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    storage.save_detection(
        image_bytes=encoded.tobytes(),
        confidence=item["conf"],
        description=f"Video Validation ({item['t']}s): {item['desc']}",
        object_type="rodent",
        label="Rat",
        battery_percentage=100,
        device_name="YouTube Validator (honC8388ook)",
        bounding_box=[int((rat_y - 20) / img.shape[0] * 1000), int((cx - 40) / img.shape[1] * 1000), int((rat_y + 40) / img.shape[0] * 1000), int((cx + 40) / img.shape[1] * 1000)],
        dt=datetime.now() - timedelta(seconds=int(10 - item["t"]))
    )

print("Done rendering all tracked frames!")
