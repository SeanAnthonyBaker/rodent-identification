import os
from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class NodeSettings(BaseModel):
    role: Literal["detection", "inference", "standalone"] = "detection"
    detection_node: str = "roland1"
    inference_node: str = "roland3"
    supabase_node: str = "roland1"

class RingSettings(BaseModel):
    token_file: str = "ring_token.json"
    device_name: Optional[str] = "Galaxy Tab A11+"
    sample_interval_seconds: int = 5
    active_detection_interval_seconds: int = 2
    mock_if_unavailable: bool = True
    phone_camera_url: Optional[str] = "http://127.0.0.1:8085/video"

class InferenceSettings(BaseModel):
    endpoint_url: str = "http://localhost:11434"
    endpoint_type: Literal["ollama", "openai", "gemini"] = "gemini"
    model_name: str = "gemini-3.7-flash"
    confidence_threshold: float = 0.75
    timeout_seconds: float = 45.0
    detection_polygon: Optional[list] = None
    camera_polygons: dict = Field(default_factory=dict)
    target_object: str = "bird"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.7-flash"

class StorageSettings(BaseModel):
    backend: Literal["supabase", "sqlite", "auto"] = "auto"
    supabase_url: str = "http://roland1:54321"
    supabase_key: Optional[str] = ""
    supabase_table: str = "detections"
    supabase_bucket: str = "detections"
    fallback_to_sqlite: bool = True
    detections_dir: str = "data/detections"
    db_path: str = "data/detections.db"
    max_images_to_keep: int = 1000

class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class AppConfig(BaseSettings):
    node: NodeSettings = Field(default_factory=NodeSettings)
    ring: RingSettings = Field(default_factory=RingSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "AppConfig":
        path = Path(config_path)
        if not path.exists() and not path.is_absolute():
            candidate = Path(__file__).resolve().parent.parent / config_path
            if candidate.exists():
                path = candidate

        data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        # Environment variable overrides
        if "inference" not in data:
            data["inference"] = {}
        if env_inf_url := os.environ.get("RODENT_INFERENCE_ENDPOINT_URL"):
            data["inference"]["endpoint_url"] = env_inf_url
        if env_inf_type := os.environ.get("RODENT_INFERENCE_ENDPOINT_TYPE"):
            data["inference"]["endpoint_type"] = env_inf_type
        if env_model := os.environ.get("RODENT_INFERENCE_MODEL_NAME"):
            data["inference"]["model_name"] = env_model

        if "storage" not in data:
            data["storage"] = {}
        if env_supa_url := (os.environ.get("RODENT_SUPABASE_URL") or os.environ.get("SUPABASE_URL")):
            data["storage"]["supabase_url"] = env_supa_url
        if env_supa_key := (os.environ.get("RODENT_SUPABASE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ACCESS_TOKEN")):
            data["storage"]["supabase_key"] = env_supa_key
        if env_backend := os.environ.get("RODENT_STORAGE_BACKEND"):
            data["storage"]["backend"] = env_backend

        return cls(**data)

config = AppConfig.load()
