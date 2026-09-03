import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger("storage")

class DetectionRecord(BaseModel):
    id: int
    event_id: Optional[str] = None
    frame_index: int = 1
    timestamp: str
    formatted_time: str
    filename: str
    image_url: str
    confidence: float
    description: str
    object_type: str = "rat"
    label: str = "Rat"
    battery_percentage: Optional[int] = None
    device_name: str
    bounding_box: Optional[List[int]] = None
    created_at: float


class StorageManager:
    """Manages SQLite database and image storage for rodent and pheasant detections."""

    def __init__(self, detections_dir: str = "data/detections", db_path: str = "data/detections.db"):
        self.detections_dir = Path(detections_dir)
        self.db_path = Path(db_path)
        self._init_storage()

    def _init_storage(self):
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    frame_index INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    formatted_time TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT,
                    object_type TEXT DEFAULT 'rat',
                    label TEXT DEFAULT 'Rat',
                    battery_percentage INTEGER,
                    device_name TEXT,
                    bounding_box TEXT,
                    created_at REAL NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(created_at)")
            
            # Migration check: Ensure object_type, label, event_id & frame_index columns exist in existing tables
            cursor.execute("PRAGMA table_info(detections)")
            columns = [info[1] for info in cursor.fetchall()]
            if "object_type" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN object_type TEXT DEFAULT 'rat'")
            if "label" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN label TEXT DEFAULT 'Rat'")
            if "event_id" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN event_id TEXT")
            if "frame_index" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN frame_index INTEGER DEFAULT 1")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_id ON detections(event_id)")
            conn.commit()

    def save_detection(
        self,
        image_bytes: bytes,
        confidence: float,
        description: str,
        battery_percentage: Optional[int],
        device_name: str,
        bounding_box: Optional[List[int]] = None,
        object_type: str = "rat",
        label: Optional[str] = None,
        dt: Optional[datetime] = None,
        event_id: Optional[str] = None,
        frame_index: int = 1
    ) -> DetectionRecord:
        """Saves image file to disk and record into database."""
        now = dt or datetime.now()
        timestamp_iso = now.isoformat()
        formatted_time = now.strftime("%b %d, %Y - %I:%M:%S %p")
        time_slug = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        
        # Clean slug for filename
        ot = object_type.lower().strip()
        if ot in ["tree", "trees"]:
            obj_slug = "tree"
            default_label = "Tree"
        elif ot in ["bird", "birds", "pheasant"]:
            obj_slug = "bird"
            default_label = "Bird"
        elif ot in ["horse", "horses", "pony", "equine"]:
            obj_slug = "horse"
            default_label = "Horse"
        elif ot in ["horses_poo", "horse_poo", "horses poo", "poo", "manure"]:
            obj_slug = "horses_poo"
            default_label = "Horses poo"
        else:
            obj_slug = "rat"
            default_label = "Rat"

        filename = f"{obj_slug}_detection_{time_slug}.jpg"
        filepath = self.detections_dir / filename

        # Write image file
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        bbox_json = json.dumps(bounding_box) if bounding_box else None
        created_at = now.timestamp()
        display_label = label or default_label
        session_event_id = event_id or f"evt_{int(created_at)}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    event_id, frame_index, timestamp, formatted_time, filename, confidence,
                    description, object_type, label, battery_percentage, device_name,
                    bounding_box, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_event_id,
                frame_index,
                timestamp_iso,
                formatted_time,
                filename,
                confidence,
                description,
                object_type,
                display_label,
                battery_percentage,
                device_name,
                bbox_json,
                created_at
            ))
            rec_id = cursor.lastrowid
            conn.commit()

        return DetectionRecord(
            id=rec_id,
            event_id=session_event_id,
            frame_index=frame_index,
            timestamp=timestamp_iso,
            formatted_time=formatted_time,
            filename=filename,
            image_url=f"/api/detections/{rec_id}/image",
            confidence=round(confidence, 3),
            description=description,
            object_type=object_type,
            label=display_label,
            battery_percentage=battery_percentage,
            device_name=device_name,
            bounding_box=bounding_box,
            created_at=created_at
        )

    def list_detections(
        self,
        order: str = "asc",
        limit: int = 200,
        offset: int = 0,
        object_type: Optional[str] = None
    ) -> List[DetectionRecord]:
        """Lists detections in chronological order ('asc' for oldest first, 'desc' for newest first), optionally filtered by object type."""
        order_clause = "ASC" if order.lower() == "asc" else "DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if object_type and object_type.lower() not in ["all", "any", ""]:
                ot = object_type.lower()
                if ot in ["rat", "rodent", "mouse"]:
                    cursor.execute(f"""
                        SELECT * FROM detections
                        WHERE LOWER(object_type) IN ('rat', 'rodent', 'mouse') OR LOWER(label) IN ('rat', 'rodent', 'mouse')
                        ORDER BY created_at {order_clause}
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                else:
                    cursor.execute(f"""
                        SELECT * FROM detections
                        WHERE LOWER(object_type) = ? OR LOWER(label) = ?
                        ORDER BY created_at {order_clause}
                        LIMIT ? OFFSET ?
                    """, (ot, ot, limit, offset))
            else:
                cursor.execute(f"""
                    SELECT * FROM detections
                    ORDER BY created_at {order_clause}
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            rows = cursor.fetchall()

        results = []
        for r in rows:
            bbox = json.loads(r["bounding_box"]) if r["bounding_box"] else None
            keys = r.keys()
            obj_type = r["object_type"] if "object_type" in keys and r["object_type"] else "rat"
            obj_label = r["label"] if "label" in keys and r["label"] else ("Pheasant" if obj_type == "pheasant" else "Rat")
            ev_id = r["event_id"] if "event_id" in keys and r["event_id"] else f"evt_{r['id']}"
            fr_idx = r["frame_index"] if "frame_index" in keys and r["frame_index"] else 1
            results.append(DetectionRecord(
                id=r["id"],
                event_id=ev_id,
                frame_index=fr_idx,
                timestamp=r["timestamp"],
                formatted_time=r["formatted_time"],
                filename=r["filename"],
                image_url=f"/api/detections/{r['id']}/image",
                confidence=round(r["confidence"], 3),
                description=r["description"] or "",
                object_type=obj_type,
                label=obj_label,
                battery_percentage=r["battery_percentage"],
                device_name=r["device_name"] or "Ring Camera",
                bounding_box=bbox,
                created_at=r["created_at"]
            ))
        return results

    def get_detection(self, detection_id: int) -> Optional[DetectionRecord]:
        """Retrieves a single detection by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
            r = cursor.fetchone()

        if not r:
            return None

        bbox = json.loads(r["bounding_box"]) if r["bounding_box"] else None
        keys = r.keys()
        obj_type = r["object_type"] if "object_type" in keys and r["object_type"] else "rat"
        obj_label = r["label"] if "label" in keys and r["label"] else ("Pheasant" if obj_type == "pheasant" else "Rat")
        ev_id = r["event_id"] if "event_id" in keys and r["event_id"] else f"evt_{r['id']}"
        fr_idx = r["frame_index"] if "frame_index" in keys and r["frame_index"] else 1
        return DetectionRecord(
            id=r["id"],
            event_id=ev_id,
            frame_index=fr_idx,
            timestamp=r["timestamp"],
            formatted_time=r["formatted_time"],
            filename=r["filename"],
            image_url=f"/api/detections/{r['id']}/image",
            confidence=round(r["confidence"], 3),
            description=r["description"] or "",
            object_type=obj_type,
            label=obj_label,
            battery_percentage=r["battery_percentage"],
            device_name=r["device_name"] or "Ring Camera",
            bounding_box=bbox,
            created_at=r["created_at"]
        )

    def delete_detection(self, detection_id: int) -> bool:
        """Deletes a single detection record and its corresponding image file."""
        record = self.get_detection(detection_id)
        if record:
            filepath = self.detections_dir / record.filename
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    logger.error(f"Error removing image file {filepath}: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        return deleted

    def delete_detections_batch(self, detection_ids: List[int]) -> int:
        """Deletes a list of detection records and their image files."""
        if not detection_ids:
            return 0
        deleted_count = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(detection_ids))
            cursor.execute(f"SELECT id, filename FROM detections WHERE id IN ({placeholders})", detection_ids)
            rows = cursor.fetchall()
            for r in rows:
                filepath = self.detections_dir / r["filename"]
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception as e:
                        logger.error(f"Error removing image file {filepath}: {e}")
            cursor.execute(f"DELETE FROM detections WHERE id IN ({placeholders})", detection_ids)
            conn.commit()
            deleted_count = cursor.rowcount
        return deleted_count

    def delete_event(self, event_id: str) -> int:
        """Deletes all frames belonging to a specific event session."""
        deleted_count = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename FROM detections WHERE event_id = ? OR ('evt_' || id) = ?", (event_id, event_id))
            rows = cursor.fetchall()
            for r in rows:
                filepath = self.detections_dir / r["filename"]
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception as e:
                        logger.error(f"Error removing image file {filepath}: {e}")
            cursor.execute("DELETE FROM detections WHERE event_id = ? OR ('evt_' || id) = ?", (event_id, event_id))
            conn.commit()
            deleted_count = cursor.rowcount
        return deleted_count

    def list_events(
        self,
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
        object_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Groups continuous detections into single sighting events."""
        order_clause = "ASC" if order.lower() == "asc" else "DESC"
        filter_sql = ""
        params: List[Any] = []
        if object_type and object_type.lower() not in ["all", "any", ""]:
            ot = object_type.lower()
            if ot in ["rat", "rodent", "mouse"]:
                filter_sql = "WHERE LOWER(object_type) IN ('rat', 'rodent', 'mouse') OR LOWER(label) IN ('rat', 'rodent', 'mouse')"
            else:
                filter_sql = "WHERE LOWER(object_type) = ? OR LOWER(label) = ?"
                params.extend([ot, ot])

        query = f"""
            SELECT 
                COALESCE(event_id, 'evt_' || id) AS session_id,
                object_type,
                label,
                device_name,
                MIN(created_at) AS start_ts,
                MAX(created_at) AS end_ts,
                MIN(formatted_time) AS formatted_start,
                MAX(formatted_time) AS formatted_end,
                MIN(timestamp) AS start_iso,
                MAX(timestamp) AS end_iso,
                MAX(confidence) AS max_confidence,
                COUNT(*) AS frame_count,
                MIN(id) AS first_frame_id,
                MAX(id) AS latest_frame_id
            FROM detections
            {filter_sql}
            GROUP BY session_id
            ORDER BY start_ts {order_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        events = []
        for r in rows:
            duration = max(0.0, round(r["end_ts"] - r["start_ts"], 1))
            events.append({
                "event_id": r["session_id"],
                "object_type": r["object_type"],
                "label": r["label"],
                "device_name": r["device_name"] or "Ring Camera",
                "start_time": r["start_iso"],
                "end_time": r["end_iso"],
                "formatted_start_time": r["formatted_start"],
                "formatted_end_time": r["formatted_end"],
                "duration_seconds": duration,
                "frame_count": r["frame_count"],
                "confidence_max": round(r["max_confidence"], 3),
                "preview_image_url": f"/api/detections/{r['latest_frame_id']}/image",
                "first_frame_id": r["first_frame_id"],
                "latest_frame_id": r["latest_frame_id"]
            })
        return events

    def get_event_frames(self, event_id: str) -> List[DetectionRecord]:
        """Returns all frames belonging to an event session in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM detections
                WHERE event_id = ? OR ('evt_' || id) = ?
                ORDER BY created_at ASC, frame_index ASC
            """, (event_id, event_id))
            rows = cursor.fetchall()

        frames = []
        for r in rows:
            bbox = json.loads(r["bounding_box"]) if r["bounding_box"] else None
            keys = r.keys()
            obj_type = r["object_type"] if "object_type" in keys and r["object_type"] else "rat"
            obj_label = r["label"] if "label" in keys and r["label"] else "Rat"
            ev_id = r["event_id"] if "event_id" in keys and r["event_id"] else f"evt_{r['id']}"
            fr_idx = r["frame_index"] if "frame_index" in keys and r["frame_index"] else 1
            frames.append(DetectionRecord(
                id=r["id"],
                event_id=ev_id,
                frame_index=fr_idx,
                timestamp=r["timestamp"],
                formatted_time=r["formatted_time"],
                filename=r["filename"],
                image_url=f"/api/detections/{r['id']}/image",
                confidence=round(r["confidence"], 3),
                description=r["description"] or "",
                object_type=obj_type,
                label=obj_label,
                battery_percentage=r["battery_percentage"],
                device_name=r["device_name"] or "Ring Camera",
                bounding_box=bbox,
                created_at=r["created_at"]
            ))
        return frames

    def clear_all_detections(self) -> int:
        """Deletes all detection records and clears the image directory."""
        deleted_files = 0
        if self.detections_dir.exists():
            for p in self.detections_dir.iterdir():
                if p.is_file():
                    try:
                        p.unlink()
                        deleted_files += 1
                    except Exception as e:
                        logger.error(f"Error removing {p}: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()

        logger.info(f"Cleared all detections from database and disk ({deleted_files} files removed).")
        return deleted_files

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on total detections and latest capture."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MAX(created_at) FROM detections")
            count, latest_ts = cursor.fetchone()

        return {
            "total_detections": count or 0,
            "latest_detection_timestamp": datetime.fromtimestamp(latest_ts).isoformat() if latest_ts else None
        }
