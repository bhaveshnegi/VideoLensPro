"""
Video Hash Configuration

Configuration for video duplicate detection using perceptual hashing.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VideoHashConfig(BaseSettings):
    """Configuration for video hashing"""
    
    HASH_FRAME_INTERVAL_SEC: float = Field(
        default=1.0,
        description="Interval in seconds between sampled frames"
    )
    
    HASH_SIZE: int = Field(
        default=16,
        description="Size of perceptual hash matrix (8, 16, or 32)"
    )
    
    HASH_MAX_FRAMES: int = Field(
        default=60,
        description="Maximum frames to sample for hashing"
    )
    
    HASH_SIMILARITY_THRESHOLD: float = Field(
        default=90.0,
        description="Similarity threshold percentage for duplicate detection"
    )
    
    ENABLE_CONTENT_HASH: bool = Field(
        default=True,
        description="Enable perceptual content-based hashing"
    )
    
    ENABLE_METADATA_HASH: bool = Field(
        default=True,
        description="Enable metadata-based signature"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields from .env
    )


# Global instance
video_hash_config = VideoHashConfig()
