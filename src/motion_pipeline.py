import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("motion_pipeline")

class MotionCascadePipeline:
    """Fast CPU/CUDA background subtraction and high-resolution sub-crop extractor."""

    def __init__(
        self,
        reference_path: str = "data/reference_baseline.jpg",
        var_threshold: float = 30.0,
        history: int = 300,
        min_contour_area: int = 250
    ):
        self.reference_path = Path(reference_path)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_contour_area = min_contour_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self._cached_reference_bytes: Optional[bytes] = None
        self._load_reference()

    def _load_reference(self):
        if self.reference_path.exists():
            try:
                self._cached_reference_bytes = self.reference_path.read_bytes()
                logger.info(f"Loaded negative baseline reference image ({len(self._cached_reference_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"Failed loading reference baseline: {e}")

    def save_reference_baseline(self, image_bytes: bytes) -> bool:
        """Saves current clean frame as negative contrast baseline."""
        try:
            self.reference_path.write_bytes(image_bytes)
            self._cached_reference_bytes = image_bytes
            logger.info("Saved new negative baseline reference image.")
            return True
        except Exception as e:
            logger.error(f"Error saving reference baseline: {e}")
            return False

    def get_reference_baseline(self) -> Optional[bytes]:
        return self._cached_reference_bytes

    def get_roi_mask(self, shape: Tuple[int, int], polygon: Optional[List[List[float]]] = None) -> Optional[np.ndarray]:
        """Creates a binary mask for the ROI polygon."""
        if not polygon or len(polygon) < 3:
            return None
        h, w = shape
        pts = []
        for pt in polygon:
            px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
            py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])
            pts.append([px, py])
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_np], 255)
        return mask

    def process_frame(
        self,
        image_bytes: bytes,
        polygon: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """Analyzes frame with background subtractor inside ROI zone and extracts high-res dynamic sub-crops."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "has_motion": False,
                "motion_percent": 0.0,
                "motion_boxes": [],
                "roi_crop_bytes": image_bytes,
                "crop_bbox": None,
                "original_shape": (1280, 720)
            }

        h, w = img.shape[:2]
        roi_mask = self.get_roi_mask((h, w), polygon)

        # 1. Apply background subtraction
        fg_mask = self.bg_subtractor.apply(img)

        # 2. Mask with ROI polygon so changes outside zone are completely zeroed out
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        # 3. Morphological filtering to eliminate isolated sensor noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.dilate(fg_clean, kernel, iterations=2)

        # 4. Measure motion activity
        non_zero = cv2.countNonZero(fg_clean)
        roi_area = cv2.countNonZero(roi_mask) if roi_mask is not None else (h * w)
        motion_percent = (non_zero / max(1, roi_area)) * 100.0

        # 5. Extract bounding boxes around moving contours
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_contour_area:
                x, y, bw, bh = cv2.boundingRect(c)
                motion_boxes.append((x, y, bw, bh))

        has_motion = len(motion_boxes) > 0 or motion_percent > 0.35
        primary_boundary = None
        if motion_boxes:
            sorted_boxes = sorted(motion_boxes, key=lambda b: b[2] * b[3], reverse=True)
            bx, by, bbw, bbh = sorted_boxes[0]
            ymin = int((by / h) * 1000)
            xmin = int((bx / w) * 1000)
            ymax = int(((by + bbh) / h) * 1000)
            xmax = int(((bx + bbw) / w) * 1000)
            primary_boundary = [max(0, min(1000, ymin)), max(0, min(1000, xmin)), max(0, min(1000, ymax)), max(0, min(1000, xmax))]

        crop_bytes, crop_bbox, shape = self.extract_target_zone_crop(image_bytes, polygon)

        return {
            "has_motion": has_motion,
            "has_material_delta": has_motion,
            "motion_percent": round(motion_percent, 2),
            "delta_percent": round(motion_percent, 2),
            "motion_boxes": motion_boxes,
            "object_boundary": primary_boundary,
            "roi_crop_bytes": crop_bytes or image_bytes,
            "focused_crop_bytes": crop_bytes or image_bytes,
            "crop_bbox": crop_bbox,
            "original_shape": shape
        }

    def extract_target_zone_crop(
        self,
        image_bytes: bytes,
        polygon: Optional[List[List[float]]] = None
    ) -> Tuple[Optional[bytes], Optional[List[int]], Tuple[int, int]]:
        """Extracts the high-resolution crop of the target zone polygon."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, (1280, 720)

        h, w = img.shape[:2]
        if not polygon or len(polygon) < 3:
            return image_bytes, [0, 0, h, w], (w, h)

        pts = [(int(p[0] * w) if p[0] <= 1.0 else int(p[0]), int(p[1] * h) if p[1] <= 1.0 else int(p[1])) for p in polygon]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x = max(0, min(xs))
        max_x = min(w, max(xs))
        min_y = max(0, min(ys))
        max_y = min(h, max(ys))

        # Add 4% padding around target zone box
        pad_x = int((max_x - min_x) * 0.04)
        pad_y = int((max_y - min_y) * 0.04)
        crop_x1 = max(0, min_x - pad_x)
        crop_y1 = max(0, min_y - pad_y)
        crop_x2 = min(w, max_x + pad_x)
        crop_y2 = min(h, max_y + pad_y)

        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            zone_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            success, enc = cv2.imencode(".jpg", zone_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if success:
                return enc.tobytes(), [crop_y1, crop_x1, crop_y2, crop_x2], (w, h)

        return image_bytes, [0, 0, h, w], (w, h)

    def compute_zone_delta(
        self,
        current_bytes: bytes,
        polygon: Optional[List[List[float]]] = None,
        delta_threshold: float = 0.40
    ) -> Dict[str, Any]:
        """Compares current frame directly against the locked baseline reference specifically in the target zone."""
        nparr = np.frombuffer(current_bytes, np.uint8)
        curr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if curr_img is None:
            return {
                "has_material_delta": False,
                "delta_percent": 0.0,
                "focused_crop_bytes": current_bytes,
                "crop_bbox": None,
                "original_shape": (1280, 720)
            }

        h, w = curr_img.shape[:2]
        roi_mask = self.get_roi_mask((h, w), polygon)
        ref_bytes = self.get_reference_baseline()

        # If no reference image exists, use MOG2 background subtractor
        if not ref_bytes:
            return self.process_frame(current_bytes, polygon)

        ref_nparr = np.frombuffer(ref_bytes, np.uint8)
        ref_img = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
        if ref_img is None or ref_img.shape != curr_img.shape:
            # Resize ref image if dimensions differ
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (w, h))
            else:
                return self.process_frame(current_bytes, polygon)

        # 1. Convert to grayscale & blur
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (7, 7), 0)
        ref_blur = cv2.GaussianBlur(ref_gray, (7, 7), 0)

        # 2. Absolute pixel difference
        diff = cv2.absdiff(curr_blur, ref_blur)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

        # 3. Mask difference strictly within ROI polygon
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # 4. Morphological clean-up (remove sensor noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh_clean = cv2.dilate(thresh_clean, kernel, iterations=2)

        # 5. Measure delta activity within target zone
        non_zero = cv2.countNonZero(thresh_clean)
        roi_area = cv2.countNonZero(roi_mask) if roi_mask is not None else (h * w)
        delta_percent = (non_zero / max(1, roi_area)) * 100.0

        # 6. Extract dynamic bounding contours
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        delta_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_contour_area:
                x, y, bw, bh = cv2.boundingRect(c)
                delta_boxes.append((x, y, bw, bh))

        has_material_delta = delta_percent >= delta_threshold or len(delta_boxes) > 0
        primary_boundary = None
        if delta_boxes:
            sorted_boxes = sorted(delta_boxes, key=lambda b: b[2] * b[3], reverse=True)
            bx, by, bbw, bbh = sorted_boxes[0]
            ymin = int((by / h) * 1000)
            xmin = int((bx / w) * 1000)
            ymax = int(((by + bbh) / h) * 1000)
            xmax = int(((bx + bbw) / w) * 1000)
            primary_boundary = [max(0, min(1000, ymin)), max(0, min(1000, xmin)), max(0, min(1000, ymax)), max(0, min(1000, xmax))]

        # 7. Extract high-resolution crop of the target zone
        crop_bytes, crop_bbox, shape = self.extract_target_zone_crop(current_bytes, polygon)

        return {
            "has_material_delta": has_material_delta,
            "has_motion": has_material_delta,
            "delta_percent": round(delta_percent, 2),
            "motion_percent": round(delta_percent, 2),
            "delta_boxes": delta_boxes,
            "object_boundary": primary_boundary,
            "focused_crop_bytes": crop_bytes or current_bytes,
            "roi_crop_bytes": crop_bytes or current_bytes,
            "crop_bbox": crop_bbox,
            "original_shape": shape
        }
