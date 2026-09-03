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
logger = logging.getLogger("video_validator")

async def validate_rat_video():
    video_path = Path("data/rat_test_video.mp4")
    if not video_path.exists():
        logger.error(f"Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"📹 Video Loaded: {video_path.name}")
    logger.info(f"   Resolution: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration_sec:.1f}s")

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

    # Sample 1 frame every 1.5 seconds (or key motion moments)
    sample_interval_frames = max(1, int(fps * 1.5))
    frame_indices = list(range(0, total_frames, sample_interval_frames))

    logger.info(f"🔍 Sampling {len(frame_indices)} frames across the video for Gemma vision rat detection & tracking...")

    results = []
    detected_count = 0

    base_time = datetime.now() - timedelta(seconds=int(duration_sec))

    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp_sec = frame_no / fps
        sim_dt = base_time + timedelta(seconds=timestamp_sec)

        # Scale frame so longest edge <= 640px (per latency & accuracy spec)
        fh, fw = frame.shape[:2]
        max_dim = max(fh, fw)
        if max_dim > 640:
            scale = 640.0 / max_dim
            proc_frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame

        # Encode to JPEG bytes
        success, encoded_jpg = cv2.imencode(".jpg", proc_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            continue

        jpg_bytes = encoded_jpg.tobytes()

        logger.info(f"\n--- [Frame {frame_no}/{total_frames} @ {timestamp_sec:.2f}s] ---")
        t0 = time.perf_counter()

        res = await client.analyze_image(
            image_bytes=jpg_bytes,
            polygon=None,
            reference_image_bytes=None,
            target_object="rat"
        )

        infer_ms = (time.perf_counter() - t0) * 1000

        is_rat = res.is_detected or res.is_rat_detected or (res.object_type == "rodent")
        conf = res.confidence

        logger.info(f"   AI Verdict: {'🐀 RAT DETECTED' if is_rat else '⚪ No Rat'} ({conf*100:.1f}%) in {infer_ms:.0f}ms")
        logger.info(f"   Subject Type: {res.subject_type} | Label: {res.label}")
        logger.info(f"   Description: {res.description}")
        if res.bounding_box:
            logger.info(f"   Bounding Box: {res.bounding_box}")
            # Center target calculation
            box = res.bounding_box
            ymin = (box[0] / 1000.0) * height if box[0] > 1 else box[0] * height
            xmin = (box[1] / 1000.0) * width if box[1] > 1 else box[1] * width
            ymax = (box[2] / 1000.0) * height if box[2] > 1 else box[2] * height
            xmax = (box[3] / 1000.0) * width if box[3] > 1 else box[3] * width
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            logger.info(f"   🎯 Center Target Position: ({cx:.1f}px, {cy:.1f}px)")

        if is_rat:
            detected_count += 1
            # Save into gallery so it displays in chronological carousel
            saved_record = storage.save_detection(
                image_bytes=jpg_bytes,
                confidence=conf,
                description=f"Video Validation ({timestamp_sec:.1f}s): {res.description}",
                object_type="rodent",
                label="Rat",
                battery_percentage=98,
                device_name="YouTube Video Validator",
                bounding_box=res.bounding_box,
                dt=sim_dt
            )
            logger.info(f"   💾 Saved detection record #{saved_record.id} to gallery database")

        results.append({
            "frame": frame_no,
            "timestamp_sec": round(timestamp_sec, 2),
            "detected": is_rat,
            "confidence": round(conf, 3),
            "label": res.label,
            "description": res.description,
            "bounding_box": res.bounding_box,
            "infer_ms": round(infer_ms, 1)
        })

    cap.release()

    logger.info("\n=======================================================")
    logger.info(f"🎯 VIDEO VALIDATION COMPLETED:")
    logger.info(f"   Total Tested Frames: {len(results)}")
    logger.info(f"   Rat Detections Confirmed: {detected_count}/{len(results)} ({(detected_count/len(results))*100:.1f}%)")
    logger.info("=======================================================")

    with open("data/video_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(validate_rat_video())
