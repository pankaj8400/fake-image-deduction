import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent

class Settings(BaseSettings):
    """Production application settings."""
    APP_NAME: str = "Deepfake & Fake Image Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model Configuration
    # Supports real, ai_generated, faceswap, photoshop
    MODEL_PATH: str = str(BASE_DIR / "hybrid_deepfake_smoke_test.pth")
    NUM_CLASSES: int = 4
    # Single permanent production mapping:
    # 0: real, 1: photoshop, 2: ai_generated, 3: faceswap
    CLASSES: List[str] = ["real", "photoshop", "ai_generated", "faceswap"]
    IMAGE_SIZE: int = 224

    # Device Selection
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Groq API Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_FALLBACK_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TEMPERATURE: float = 0.2
    GROQ_MAX_TOKENS: int = 150

    # Image upload constraints
    MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        # Check both backend/.env and project/.env
        env_file = (
            str(BASE_DIR / ".env"),
            str(PROJECT_DIR / ".env")
        )
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_resolved_model_path(self) -> Path:
        """Resolve model path accurately across different working directories."""
        p = Path(self.MODEL_PATH)
        if p.is_file():
            return p
        
        # Alternative checks
        candidates = [
            BASE_DIR / "hybrid_deepfake_smoke_test.pth",
            PROJECT_DIR / "backend" / "hybrid_deepfake_smoke_test.pth",
            Path.cwd() / "backend" / "hybrid_deepfake_smoke_test.pth",
            Path.cwd() / "hybrid_deepfake_smoke_test.pth",
        ]
        for c in candidates:
            if c.is_file():
                return c
        
        return p

settings = Settings()
