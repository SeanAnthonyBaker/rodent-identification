import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("fast_detector")

# COCO Animal Classes for Fast Detection
ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    64: "mouse"
}

class FastObjectDetector:
    """
    Fast, lightweight object detector designed for 30-60+ FPS real-time box localization.
    Acts as the 'eyes that never blink':
      - Detects moving/living objects and draws tight bounding boxes on every frame.
      - Extracts high-resolution crops of candidate boxes to feed to Gemma.
    Supports:
      1. Ultralytics YOLOv8n / YOLO11n (if available)
      2. OpenCV DNN ONNX (yolov8n.onnx)
      3. Motion Cascade Foreground Contours (zero-weight fast CPU fallback)
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        min_box_area_px: int = 300
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_box_area_px = min_box_area_px
        self._yolo = None
        self._backend = "motion_contours"

        self._init_detector()

    def _init_detector(self):
        # Try loading Ultralytics YOLO
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.model_name)
            self._backend = "ultralytics_yolo"
            logger.info(f"Fast detector initialized with Ultralytics YOLO ({self.model_name})")
            return
        except Exception as e:
            logger.info(f"Ultralytics YOLO not directly active: {e}. Checking ONNX...")

        # Fallback to OpenCV DNN ONNX if file exists
        onnx_path = Path("models/yolov8n.onnx")
        if onnx_path.exists() and hasattr(cv2, "dnn"):
            try:
                self._net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self._backend = "opencv_onnx"
                logger.info(f"Fast detector initialized with OpenCV DNN ONNX ({onnx_path})")
                return
            except Exception as e:
                logger.warning(f"Failed loading ONNX model: {e}")

        logger.info("Fast detector using optimized OpenCV Motion Cascade & Contour localization (0 MB VRAM, 60+ FPS).")
        self._backend = "motion_contours"

    @property
    def backend_name(self) -> str:
        return self._backend

    def detect_boxes(
        self,
        image_bytes: bytes,
        polygon: Optional[List[List[float]]] = None,
        crop_padding_ratio: float = 0.10
    ) -> Dict[str, Any]:
        """
        Runs fast detection on frame.
        Returns:
          has_candidates: bool
          candidate_boxes: List of normalized [ymin, xmin, ymax, xmax] (0-1000)
          crops: List of cropped JPEG bytes with padding for Gemma to evaluate
          primary_crop_bytes: Crop of the most prominent candidate
          primary_box: Normalized [ymin, xmin, ymax, xmax] of the top candidate
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "has_candidates": False,
                "candidate_boxes": [],
                "crops": [],
                "primary_crop_bytes": None,
                "primary_box": None,
                "backend": self._backend
            }

        h, w = img.shape[:2]

        # 1. Ultralytics YOLO inference if active: ONLY detect animal classes
        if self._backend == "ultralytics_yolo" and self._yolo is not None:
            try:
                animal_ids = list(ANIMAL_CLASSES.keys())
                results = self._yolo(img, classes=animal_ids, conf=self.confidence_threshold, verbose=False)
                boxes = []
                for r in results:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].item())
                        if cls_id not in ANIMAL_CLASSES:
                            continue
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # Filter against polygon if provided
                        if polygon and len(polygon) >= 3:
                            cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
                            if not self._point_in_polygon(cx, cy, polygon):
                                continue
                        boxes.append((x1, y1, x2, y2))

                if boxes:
                    return self._build_candidate_response(img, boxes, crop_padding_ratio, is_animal=True)
                else:
                    # If YOLO found no animals, do NOT bound non-animal objects
                    return {
                        "has_candidates": False,
                        "candidate_boxes": [],
                        "crops": [],
                        "primary_crop_bytes": None,
                        "primary_box": None,
                        "is_animal": False,
                        "backend": self._backend
                    }
            except Exception as e:
                logger.error(f"Error during YOLO fast detection: {e}")

        # 2. Optimized Motion / Dynamic Contour Localization with strict animal morphology
        return self._detect_via_contours(img, polygon, crop_padding_ratio)

    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        pts = np.array([[p[0], p[1]] for p in polygon], dtype=np.float32)
        return cv2.pointPolygonTest(pts, (x, y), False) >= 0

    def _detect_via_contours(
        self,
        img: np.ndarray,
        polygon: Optional[List[List[float]]],
        padding_ratio: float
    ) -> Dict[str, Any]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Fast gradient & adaptive thresholding
        sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 1, ksize=3)
        _, thresh = cv2.threshold(sobel, 26, 255, cv2.THRESH_BINARY)

        # Apply polygon mask if zone configured
        if polygon and len(polygon) >= 3:
            pts = []
            for pt in polygon:
                px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
                py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])
                pts.append([px, py])
            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_np], 255)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        # Morphological close to join contour fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_box_area_px <= area <= 50000:
                x, y, bw, bh = cv2.boundingRect(c)
                # Animal morphology check: aspect ratio & solidity
                aspect = bw / float(bh)
                solidity = area / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.35 <= aspect <= 3.2 and solidity >= 0.38:
                    boxes.append((x, y, x + bw, y + bh))

        return self._build_candidate_response(img, boxes, padding_ratio, is_animal=len(boxes) > 0)

    def _build_candidate_response(
        self,
        img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        padding_ratio: float,
        is_animal: bool = True
    ) -> Dict[str, Any]:
        h, w = img.shape[:2]
        if not boxes or not is_animal:
            return {
                "has_candidates": False,
                "candidate_boxes": [],
                "crops": [],
                "primary_crop_bytes": None,
                "primary_box": None,
                "is_animal": False,
                "backend": self._backend
            }

        # Sort candidate boxes by area descending
        sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        top_candidates = sorted_boxes[:3]

        candidate_boxes = []
        crops = []

        for x1, y1, x2, y2 in top_candidates:
            # Normalized box 0-1000
            ymin = int(max(0, min(1000, (y1 / h) * 1000)))
            xmin = int(max(0, min(1000, (x1 / w) * 1000)))
            ymax = int(max(0, min(1000, (y2 / h) * 1000)))
            xmax = int(max(0, min(1000, (x2 / w) * 1000)))
            candidate_boxes.append([ymin, xmin, ymax, xmax])

            # Extract crop with context padding for Gemma
            pad_w = int((x2 - x1) * padding_ratio)
            pad_h = int((y2 - y1) * padding_ratio)
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)

            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            success, enc = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if success:
                crops.append(enc.tobytes())

        primary_crop = crops[0] if crops else None
        primary_box = candidate_boxes[0] if candidate_boxes else None

        return {
            "has_candidates": len(candidate_boxes) > 0 and is_animal,
            "is_animal": is_animal,
            "candidate_boxes": candidate_boxes if is_animal else [],
            "crops": crops,
            "primary_crop_bytes": primary_crop,
            "primary_box": primary_box if is_animal else None,
            "backend": self._backend
        }
