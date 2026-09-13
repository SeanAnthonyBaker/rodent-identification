import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
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


class RolandSupabaseClient:
    """PostgREST and Storage HTTP client for Supabase database services running on roland1."""

    def __init__(self, base_url: str = "http://roland1:54321", api_key: Optional[str] = None, timeout: float = 3.0):
        self.base_url = (base_url or "http://roland1:54321").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.rest_url = f"{self.base_url}/rest/v1"
        self.storage_url = f"{self.base_url}/storage/v1"

    @property
    def headers(self) -> Dict[str, str]:
        hdrs = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            hdrs["apikey"] = self.api_key
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    def check_health(self, table: str = "detections") -> bool:
        """Checks if Supabase PostgREST service on roland1 is reachable and healthy."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.rest_url}/{table}?limit=1", headers=self.headers)
                return resp.status_code < 500
        except Exception:
            return False

    def insert(self, table: str, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Inserts a detection row into Supabase table on roland1."""
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "return=representation"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.rest_url}/{table}", headers=hdrs, json=record)
            resp.raise_for_status()
            data = resp.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else (data if isinstance(data, dict) else None)

    def select(
        self,
        table: str,
        order: str = "asc",
        limit: int = 200,
        offset: int = 0,
        object_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries detections from Supabase on roland1 with ordering, limit, and type filter."""
        params: Dict[str, Any] = {
            "select": "*",
            "order": f"created_at.{order.lower()}",
            "limit": limit,
            "offset": offset
        }
        if object_type and object_type.lower() not in ["all", "any", ""]:
            ot = object_type.lower()
            if ot in ["rat", "rodent", "mouse"]:
                params["or"] = "(object_type.ilike.%rat%,object_type.ilike.%rodent%,object_type.ilike.%mouse%,label.ilike.%rat%,label.ilike.%rodent%,label.ilike.%mouse%)"
            else:
                params["or"] = f"(object_type.ilike.%{ot}%,label.ilike.%{ot}%)"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.rest_url}/{table}", headers=self.headers, params=params)
            resp.raise_for_status()
            return resp.json() or []

    def get_by_id(self, table: str, record_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single detection row by ID from Supabase."""
        params = {"id": f"eq.{record_id}", "select": "*"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.rest_url}/{table}", headers=self.headers, params=params)
            resp.raise_for_status()
            rows = resp.json()
            return rows[0] if rows else None

    def delete_by_id(self, table: str, record_id: int) -> bool:
        """Deletes a detection row by ID from Supabase."""
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "return=representation"
        params = {"id": f"eq.{record_id}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(f"{self.rest_url}/{table}", headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
            return len(data) > 0 if isinstance(data, list) else (resp.status_code in [200, 204])

    def delete_batch(self, table: str, record_ids: List[int]) -> int:
        """Deletes multiple detection rows by ID."""
        if not record_ids:
            return 0
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "return=representation"
        id_str = ",".join(str(i) for i in record_ids)
        params = {"id": f"in.({id_str})"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(f"{self.rest_url}/{table}", headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
            return len(data) if isinstance(data, list) else len(record_ids)

    def delete_event(self, table: str, event_id: str) -> int:
        """Deletes all frames belonging to an event session."""
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "return=representation"
        params = {"or": f"(event_id.eq.{event_id},id.eq.{event_id.replace('evt_', '') if event_id.startswith('evt_') and event_id[4:].isdigit() else -1})"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(f"{self.rest_url}/{table}", headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
            return len(data) if isinstance(data, list) else 1

    def get_event_frames(self, table: str, event_id: str) -> List[Dict[str, Any]]:
        """Fetches all frames for an event session from Supabase."""
        clean_id = event_id.replace('evt_', '') if event_id.startswith('evt_') and event_id[4:].isdigit() else -1
        params = {
            "select": "*",
            "or": f"(event_id.eq.{event_id},id.eq.{clean_id})",
            "order": "created_at.asc,frame_index.asc"
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.rest_url}/{table}", headers=self.headers, params=params)
            resp.raise_for_status()
            return resp.json() or []

    def clear_all(self, table: str) -> int:
        """Clears all detection records from Supabase."""
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "return=representation"
        params = {"id": "gt.0"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(f"{self.rest_url}/{table}", headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
            return len(data) if isinstance(data, list) else 0

    def get_stats(self, table: str) -> Dict[str, Any]:
        """Gets count and latest timestamp from Supabase."""
        hdrs = dict(self.headers)
        hdrs["Prefer"] = "count=exact"
        params = {"select": "id,created_at", "order": "created_at.desc", "limit": "1"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.rest_url}/{table}", headers=hdrs, params=params)
            resp.raise_for_status()
            count = 0
            crange = resp.headers.get("Content-Range", "")
            if "/" in crange:
                total_part = crange.split("/")[-1]
                if total_part.isdigit():
                    count = int(total_part)
            rows = resp.json()
            latest_ts = rows[0]["created_at"] if rows and "created_at" in rows[0] else None
            return {
                "total_detections": count or (len(rows) if rows else 0),
                "latest_detection_timestamp": datetime.fromtimestamp(latest_ts).isoformat() if latest_ts else None
            }

    def upload_image(self, bucket: str, filename: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> Optional[str]:
        """Uploads image crop to Supabase Storage bucket on roland1."""
        try:
            url = f"{self.storage_url}/object/{bucket}/{filename}"
            hdrs = {
                "Content-Type": mime_type,
                "x-upsert": "true"
            }
            if self.api_key:
                hdrs["apikey"] = self.api_key
                hdrs["Authorization"] = f"Bearer {self.api_key}"
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=hdrs, content=image_bytes)
                if resp.status_code in [200, 201]:
                    return f"{self.storage_url}/object/public/{bucket}/{filename}"
        except Exception as e:
            logger.debug(f"Supabase Storage bucket upload skipped: {e}")
        return None


class StorageManager:
    """Manages database and image storage for detections using Supabase services on roland1 with SQLite fallback."""

    def __init__(
        self,
        detections_dir: str = "data/detections",
        db_path: str = "data/detections.db",
        backend: str = "auto",
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_table: str = "detections",
        supabase_bucket: str = "detections",
        fallback_to_sqlite: bool = True
    ):
        self.detections_dir = Path(detections_dir)
        self.db_path = Path(db_path)
        self.backend = backend.lower()
        self.supabase_url = (supabase_url or "http://roland1:54321").rstrip("/")
        self.supabase_key = supabase_key or ""
        self.supabase_table = supabase_table
        self.supabase_bucket = supabase_bucket
        self.fallback_to_sqlite = fallback_to_sqlite

        self._init_sqlite_storage()

        self.supabase_client = RolandSupabaseClient(
            base_url=self.supabase_url,
            api_key=self.supabase_key
        )
        self.is_supabase_active = False

        if self.backend in ["supabase", "auto"]:
            if self.supabase_client.check_health(self.supabase_table):
                self.is_supabase_active = True
                logger.info(f"Connected to Supabase Database Services on roland1 ({self.supabase_url}) [table: {self.supabase_table}]")
            else:
                if self.backend == "supabase" and not self.fallback_to_sqlite:
                    raise ConnectionError(f"Could not connect to required Supabase service on roland1 ({self.supabase_url})")
                logger.info(f"Supabase on roland1 ({self.supabase_url}) is offline. Operating in resilient local SQLite fallback mode.")

    def _init_sqlite_storage(self):
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
            
            # Migration check
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

    @property
    def backend_info(self) -> Dict[str, Any]:
        """Provides status and telemetry for storage and database services."""
        return {
            "active_backend": "supabase" if self.is_supabase_active else "sqlite",
            "configured_backend": self.backend,
            "supabase_node": "roland1",
            "supabase_url": self.supabase_url,
            "supabase_active": self.is_supabase_active,
            "fallback_to_sqlite": self.fallback_to_sqlite,
            "sqlite_path": str(self.db_path)
        }

    def _format_record(self, r: Dict[str, Any]) -> DetectionRecord:
        """Helper to convert dictionary or SQLite row to DetectionRecord."""
        bbox = r.get("bounding_box")
        if isinstance(bbox, str) and bbox:
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None
        elif not isinstance(bbox, list):
            bbox = None

        rec_id = int(r.get("id", 0))
        ev_id = r.get("event_id") or f"evt_{rec_id}"
        fr_idx = int(r.get("frame_index") or 1)
        obj_type = r.get("object_type") or "rat"
        obj_label = r.get("label") or ("Pheasant" if obj_type == "pheasant" else "Rat")
        
        return DetectionRecord(
            id=rec_id,
            event_id=ev_id,
            frame_index=fr_idx,
            timestamp=r.get("timestamp", ""),
            formatted_time=r.get("formatted_time", ""),
            filename=r.get("filename", ""),
            image_url=f"/api/detections/{rec_id}/image",
            confidence=round(float(r.get("confidence", 0.0)), 3),
            description=r.get("description") or "",
            object_type=obj_type,
            label=obj_label,
            battery_percentage=r.get("battery_percentage"),
            device_name=r.get("device_name") or "Roland 1 Camera",
            bounding_box=bbox,
            created_at=float(r.get("created_at", 0.0))
        )

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
        """Saves image file to disk and record into Supabase on roland1 (mirrored in SQLite)."""
        now = dt or datetime.now()
        timestamp_iso = now.isoformat()
        formatted_time = now.strftime("%b %d, %Y - %I:%M:%S %p")
        time_slug = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        
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

        # Write image file locally
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        bbox_json = json.dumps(bounding_box) if bounding_box else None
        created_at = now.timestamp()
        display_label = label or default_label
        session_event_id = event_id or f"evt_{int(created_at)}"

        rec_id: Optional[int] = None

        # 1. Supabase insert on roland1
        if self.is_supabase_active:
            try:
                supa_payload = {
                    "event_id": session_event_id,
                    "frame_index": frame_index,
                    "timestamp": timestamp_iso,
                    "formatted_time": formatted_time,
                    "filename": filename,
                    "confidence": round(confidence, 4),
                    "description": description,
                    "object_type": object_type,
                    "label": display_label,
                    "battery_percentage": battery_percentage,
                    "device_name": device_name,
                    "bounding_box": bbox_json,
                    "created_at": created_at
                }
                res = self.supabase_client.insert(self.supabase_table, supa_payload)
                if res and "id" in res:
                    rec_id = int(res["id"])
                    # Optional bucket upload
                    self.supabase_client.upload_image(self.supabase_bucket, filename, image_bytes)
            except Exception as e:
                logger.warning(f"Supabase insert on roland1 failed: {e}. Falling back to SQLite.")

        # 2. SQLite insert (used as primary or dual-persistence cache)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if rec_id is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO detections (
                        id, event_id, frame_index, timestamp, formatted_time, filename, confidence,
                        description, object_type, label, battery_percentage, device_name,
                        bounding_box, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec_id, session_event_id, frame_index, timestamp_iso, formatted_time,
                    filename, confidence, description, object_type, display_label,
                    battery_percentage, device_name, bbox_json, created_at
                ))
            else:
                cursor.execute("""
                    INSERT INTO detections (
                        event_id, frame_index, timestamp, formatted_time, filename, confidence,
                        description, object_type, label, battery_percentage, device_name,
                        bounding_box, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_event_id, frame_index, timestamp_iso, formatted_time,
                    filename, confidence, description, object_type, display_label,
                    battery_percentage, device_name, bbox_json, created_at
                ))
                rec_id = cursor.lastrowid
            conn.commit()

        return DetectionRecord(
            id=rec_id or 1,
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
        """Lists detections chronologically, optionally filtered by object type."""
        if self.is_supabase_active:
            try:
                rows = self.supabase_client.select(
                    table=self.supabase_table,
                    order=order,
                    limit=limit,
                    offset=offset,
                    object_type=object_type
                )
                return [self._format_record(r) for r in rows]
            except Exception as e:
                logger.warning(f"Supabase select on roland1 failed: {e}. Falling back to SQLite.")

        # SQLite fallback
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
            results.append(self._format_record(dict(r)))
        return results

    def get_detection(self, detection_id: int) -> Optional[DetectionRecord]:
        """Retrieves a single detection by ID."""
        if self.is_supabase_active:
            try:
                row = self.supabase_client.get_by_id(self.supabase_table, detection_id)
                if row:
                    return self._format_record(row)
            except Exception as e:
                logger.warning(f"Supabase get_by_id failed: {e}. Falling back to SQLite.")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
            r = cursor.fetchone()

        if not r:
            return None
        return self._format_record(dict(r))

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

        if self.is_supabase_active:
            try:
                self.supabase_client.delete_by_id(self.supabase_table, detection_id)
            except Exception as e:
                logger.warning(f"Supabase delete failed: {e}")

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

        # Remove image files
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

        if self.is_supabase_active:
            try:
                self.supabase_client.delete_batch(self.supabase_table, detection_ids)
            except Exception as e:
                logger.warning(f"Supabase batch delete failed: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(detection_ids))
            cursor.execute(f"DELETE FROM detections WHERE id IN ({placeholders})", detection_ids)
            conn.commit()
            deleted_count = cursor.rowcount
        return deleted_count

    def delete_event(self, event_id: str) -> int:
        """Deletes all frames belonging to a specific event session."""
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

        if self.is_supabase_active:
            try:
                self.supabase_client.delete_event(self.supabase_table, event_id)
            except Exception as e:
                logger.warning(f"Supabase delete_event failed: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
                "device_name": r["device_name"] or "Roland 1 Camera",
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
        if self.is_supabase_active:
            try:
                rows = self.supabase_client.get_event_frames(self.supabase_table, event_id)
                if rows:
                    return [self._format_record(r) for r in rows]
            except Exception as e:
                logger.warning(f"Supabase get_event_frames failed: {e}. Falling back to SQLite.")

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
            frames.append(self._format_record(dict(r)))
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

        if self.is_supabase_active:
            try:
                self.supabase_client.clear_all(self.supabase_table)
            except Exception as e:
                logger.warning(f"Supabase clear_all failed: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()

        logger.info(f"Cleared all detections from database and disk ({deleted_files} files removed).")
        return deleted_files

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on total detections and latest capture."""
        if self.is_supabase_active:
            try:
                return self.supabase_client.get_stats(self.supabase_table)
            except Exception as e:
                logger.warning(f"Supabase get_stats failed: {e}. Falling back to SQLite.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MAX(created_at) FROM detections")
            count, latest_ts = cursor.fetchone()

        return {
            "total_detections": count or 0,
            "latest_detection_timestamp": datetime.fromtimestamp(latest_ts).isoformat() if latest_ts else None
        }
