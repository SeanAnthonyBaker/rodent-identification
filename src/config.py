from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class RingSettings(BaseModel):
    token_file: str = "ring_token.json"
    device_name: Optional[str] = None
    sample_interval_seconds: int = 60
    active_detection_interval_seconds: int = 2
    mock_if_unavailable: bool = True
    phone_camera_url: Optional[str] = "http://192.168.1.165:8080/video"

class InferenceSettings(BaseModel):
    endpoint_url: str = "http://localhost:11434"
    endpoint_type: Literal["ollama", "openai", "gemini"] = "ollama"
    model_name: str = "tulkah_gemma4_12b:latest"
    confidence_threshold: float = 0.75
    timeout_seconds: float = 45.0
    detection_polygon: Optional[list] = None
    camera_polygons: dict = Field(default_factory=dict)
    target_object: str = "rat"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.7-flash"

class StorageSettings(BaseModel):
    detections_dir: str = "data/detections"
    db_path: str = "data/detections.db"
    max_images_to_keep: int = 1000

class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class AppConfig(BaseSettings):
    ring: RingSettings = Field(default_factory=RingSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "AppConfig":
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return cls(**data)
        return cls()

config = AppConfig.load()
