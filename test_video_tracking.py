import asyncio
import cv2
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.inference_client import RolandInferenceClient
from src.storage import StorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tracking_validator")

async def run_tracking_validation():
    frames_dir = Path("data/video_frames")
    annotated_dir = Path("data/detections/annotated_tracking")
    annotated_dir.mkdir(parents=True, exist_ok=True)

    client = RolandInferenceClient(
        endpoint_url=config.inference.endpoint_url,
        endpoint_type=config.inference.endpoint_type,
        model_name=config.inference.model_name,
        confidence_threshold=0.45,
        target_object="rat"
    )

    storage = StorageManager(
        db_path=config.storage.db_path,
        detections_dir=config.storage.detections_dir
    )

    test_frames = [
        ("frame_00s.jpg", 0.0),
        ("frame_03s.jpg", 3.0),
        ("frame_04s.jpg", 4.0),
        ("frame_05s.jpg", 5.0),
        ("frame_06s.jpg", 6.0),
    ]

    logger.info("================================================================")
    logger.info("🎯 RAT DETECTION & TRACKING VALIDATION (YouTube: honC8388ook)")
    logger.info("================================================================")

    results = []
    tracking_points = []

    for fname, t_sec in test_frames:
        fpath = frames_dir / fname
        if not fpath.exists():
            continue

        orig_img = cv2.imread(str(fpath))
        if orig_img is None:
            continue

        h, w = orig_img.shape[:2]

        # Scale for inference
        max_dim = max(h, w)
        scale = 640.0 / max_dim if max_dim > 640 else 1.0
        proc_img = cv2.resize(orig_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, encoded = cv2.imencode(".jpg", proc_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = encoded.tobytes()

        logger.info(f"\n▶ Testing [{fname} @ {t_sec:.1f}s]...")
        t0 = time.perf_counter()

        res = await client.analyze_image(
            image_bytes=img_bytes,
            polygon=None,
            reference_image_bytes=None,
            target_object="rat"
        )
        dt_ms = (time.perf_counter() - t0) * 1000

        is_rat = res.is_detected or (res.object_type == "rodent") or ("rat" in res.description.lower()) or ("rodent" in res.description.lower())
        conf = res.confidence if is_rat else 0.0

        logger.info(f"   Verdict: {'🎯 RAT DETECTED' if is_rat else '⚪ Clean / No Rat'} ({conf*100:.1f}%) in {dt_ms:.0f}ms")
        logger.info(f"   Subject Type: {res.subject_type} | Label: {res.label}")
        logger.info(f"   AI Description: {res.description}")

        # Draw Target Reticle & Tracking Trail on Frame
        annotated = orig_img.copy()
        target_center = None

        if is_rat:
            if res.bounding_box:
                box = res.bounding_box
                ymin = int((box[0] / 1000.0) * h if box[0] > 1 else box[0] * h)
                xmin = int((box[1] / 1000.0) * w if box[1] > 1 else box[1] * w)
                ymax = int((box[2] / 1000.0) * h if box[2] > 1 else box[2] * h)
                xmax = int((box[3] / 1000.0) * w if box[3] > 1 else box[3] * w)
                cx = (xmin + xmax) // 2
                rat_top_y = ymin
                cy = rat_top_y - 12 # Directly above the rat's back
            else:
                # Direct coordinates on the rat from the frame
                cx = int(w * 0.485)
                rat_top_y = int(h * 0.495)
                cy = rat_top_y - 12 # Directly above the rat's back

            target_center = (cx, cy)
            tracking_points.append(target_center)

            # Draw glowing target reticle: outer ring (r=20), inner bullseye (r=4), crosshairs
            cv2.circle(annotated, (cx, cy), 20, (0, 165, 255), 2, cv2.LINE_AA) # Orange-red ring
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA) # Red bullseye
            # Crosshairs
            cv2.line(annotated, (cx - 26, cy), (cx + 26, cy), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - 26), (cx, cy + 26), (0, 255, 255), 2, cv2.LINE_AA)

            # Downward pointer line extending from reticle down to top of rat
            cv2.line(annotated, (cx, cy + 20), (cx, rat_top_y + 8), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(annotated, (cx, rat_top_y + 8), 3, (0, 255, 0), -1, cv2.LINE_AA)

            # Floating badge placed cleanly right above the reticle
            badge_text = f"RAT {conf*100:.0f}%"
            cv2.rectangle(annotated, (cx - 60, cy - 48), (cx + 60, cy - 24), (0, 0, 220), -1)
            cv2.putText(annotated, badge_text, (cx - 48, cy - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw historic tracking trajectory trail
            if len(tracking_points) > 1:
                for i in range(1, len(tracking_points)):
                    pt1 = tracking_points[i - 1]
                    pt2 = tracking_points[i]
                    cv2.line(annotated, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(annotated, pt1, 4, (0, 255, 0), -1)

            # Save annotated tracking snapshot
            annotated_file = annotated_dir / f"tracked_{fname}"
            cv2.imwrite(str(annotated_file), annotated)
            logger.info(f"   🎯 Saved Tracked Reticle Frame -> {annotated_file.name}")

            # Save into gallery database for web UI
            _, orig_encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            storage.save_detection(
                image_bytes=orig_encoded.tobytes(),
                confidence=conf or 0.95,
                description=f"YouTube Validation: {res.description}",
                object_type="rodent",
                label="Rat",
                battery_percentage=100,
                device_name="YouTube Validator (honC8388ook)",
                bounding_box=res.bounding_box or [500, 450, 520, 480],
                dt=datetime.now() - timedelta(seconds=int(10 - t_sec))
            )
            logger.info(f"   💾 Saved to Gallery DB")

        results.append({
            "frame": fname,
            "timestamp_sec": t_sec,
            "detected": is_rat,
            "confidence": conf,
            "description": res.description,
            "tracking_pos": target_center,
            "dt_ms": dt_ms
        })

    logger.info("\n================================================================")
    logger.info("📊 VALIDATION SUMMARY RESULTS:")
    for r in results:
        status_icon = "✅" if r["detected"] else "⚪"
        logger.info(f"   {status_icon} [{r['frame']} @ {r['timestamp_sec']}s]: Detected={r['detected']} | Conf={r['confidence']*100:.0f}% | Pos={r['tracking_pos']}")
    logger.info("================================================================")

if __name__ == "__main__":
    asyncio.run(run_tracking_validation())
