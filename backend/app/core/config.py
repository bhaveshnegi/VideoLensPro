"""
Enhanced configuration management for production
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Video Analyzer Microservice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Security
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=100, env="RATE_LIMIT_MAX_REQUESTS")
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    
    # MongoDB
    MONGODB_URL: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    MONGODB_MAX_POOL_SIZE: int = Field(default=10, env="MONGODB_MAX_POOL_SIZE")
    MONGODB_TIMEOUT_MS: int = Field(default=5000, env="MONGODB_TIMEOUT_MS")
    
    # ChromaDB (Vector Database)
    CHROMA_HOST: str = Field(default="chromadb", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")
    ANONYMIZED_TELEMETRY: bool = Field(default=False, env="ANONYMIZED_TELEMETRY")
    
    # Redis (for caching and rate limiting)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Video Processing
    FRAME_INTERVAL: int = Field(default=5, env="FRAME_INTERVAL")
    MIN_DURATION_SEC: int = Field(default=15, env="MIN_DURATION_SEC")
    MIN_WORDS: int = Field(default=20, env="MIN_WORDS")
    MAX_PROCESSING_TIME_SEC: int = Field(default=300, env="MAX_PROCESSING_TIME_SEC")
    

    # Video Quality Scoring Configuration
    SCORE_ENABLE_QUALITY_ANALYSIS: bool = Field(default=False, env="SCORE_ENABLE_QUALITY_ANALYSIS")
    SCORE_BASE: float = Field(default=50.0, env="SCORE_BASE")
    
    # Legacy scoring weights (for backward compatibility)
    SCORE_DURATION_WEIGHT: float = Field(default=1.0, env="SCORE_DURATION_WEIGHT")
    SCORE_WORD_COUNT_WEIGHT: float = Field(default=0.5, env="SCORE_WORD_COUNT_WEIGHT")
    
    # Quality metric weights (multiplied by max points for each category)
    SCORE_TECHNICAL_WEIGHT: float = Field(default=1.0, env="SCORE_TECHNICAL_WEIGHT")
    SCORE_VISUAL_WEIGHT: float = Field(default=1.0, env="SCORE_VISUAL_WEIGHT")
    SCORE_CONTENT_WEIGHT: float = Field(default=1.0, env="SCORE_CONTENT_WEIGHT")
    SCORE_AUDIO_WEIGHT: float = Field(default=1.0, env="SCORE_AUDIO_WEIGHT")
    # Models
    YOLO_MODEL_PATH: str = Field(default="models/Yolo/yolov5nu.pt", env="YOLO_MODEL_PATH")
    DEEPFACE_HOME: str = Field(default="models/.deepFace", env="DEEPFACE_HOME")
    FACENET_MODEL_PATH: str = Field(default="models/.deepFace/weights/facenet512_weights.h5", env="FACENET_MODEL_PATH")
    WHISPER_MODEL_PATH: str = Field(default="models/whisper/tiny", env="WHISPER_MODEL_PATH")
    
    # Hugging Face
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    LOG_TO_CONSOLE: bool = Field(default=True, env="LOG_TO_CONSOLE")
    AI_LOG_DIR: str = Field(default="AI_logs", env="AI_LOG_DIR")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Background Tasks
    ENABLE_BACKGROUND_TASKS: bool = Field(default=True, env="ENABLE_BACKGROUND_TASKS")
    TASK_TIMEOUT_SEC: int = Field(default=600, env="TASK_TIMEOUT_SEC")
    
    # Health Checks
    HEALTH_CHECK_TIMEOUT_SEC: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT_SEC")
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f'ENVIRONMENT must be one of {allowed_envs}')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'LOG_LEVEL must be one of {allowed_levels}')
        return v.upper()
    
    @validator('MAX_FILE_SIZE_MB')
    def validate_max_file_size(cls, v):
        if v <= 0 or v > 2000:  # Max 2GB
            raise ValueError('MAX_FILE_SIZE_MB must be between 1 and 2000')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Environment-specific overrides
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False
    settings.LOG_LEVEL = "INFO"
elif settings.ENVIRONMENT == "development":
    settings.DEBUG = True
    settings.LOG_LEVEL = "INFO"