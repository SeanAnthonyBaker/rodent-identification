import cv2
import os
from pathlib import Path

video_path = Path("data/rat_test_video.mp4")
out_dir = Path("data/video_frames")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

print(f"Total Frames: {total_frames}, FPS: {fps}")

# Extract 1 frame per second
for sec in range(int(total_frames / fps) + 1):
    frame_idx = int(sec * fps)
    if frame_idx >= total_frames:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret and frame is not None:
        out_file = out_dir / f"frame_{sec:02d}s.jpg"
        cv2.imwrite(str(out_file), frame)
        print(f"Saved {out_file.name} ({frame.shape[1]}x{frame.shape[0]})")

cap.release()
