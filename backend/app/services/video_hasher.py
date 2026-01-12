import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import imagehash

from app.core.logging import get_logger

logger = get_logger(__name__)


class VideoHasher:
    """
    Perceptual video hasher that generates content-based hashes.
    This detects duplicate videos even if they have been re-encoded,
    transferred across devices, or have different metadata.
    """
    
    def __init__(
        self,
        frame_interval_sec: float = 1.0,
        hash_size: int = 16,
        max_frames: int = 60
    ):
        """
        Initialize VideoHasher.
        
        Args:
            frame_interval_sec: Interval in seconds between sampled frames
            hash_size: Size of perceptual hash (8, 16, or 32)
            max_frames: Maximum number of frames to sample (to limit processing time)
        """
        self.frame_interval_sec = frame_interval_sec
        self.hash_size = hash_size
        self.max_frames = max_frames
    
    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of raw file bytes.
        Fast but only detects exact file matches.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Hexadecimal hash string
        """
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def compute_content_hash(self, file_path: Path) -> str:
        """
        Compute perceptual content hash based on video frames.
        Detects duplicates even with different encodings or metadata.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Hexadecimal hash string representing video content
        """
        try:
            frame_hashes = self._extract_frame_hashes(file_path)
            
            if not frame_hashes:
                logger.warning(f"No frames extracted for perceptual hashing: {file_path}")
                # Fallback to file hash if frame extraction fails
                return self.compute_file_hash(file_path)
            
            # Combine all frame hashes into a single signature
            combined = "|".join(frame_hashes)
            
            # Hash the combined signature
            content_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            logger.debug(f"Content hash computed from {len(frame_hashes)} frames: {content_hash[:16]}...")
            return content_hash
            
        except Exception as e:
            logger.error(f"Error computing content hash: {e}", exc_info=True)
            # Fallback to file hash on error
            return self.compute_file_hash(file_path)
    
    def compute_metadata_signature(self, file_path: Path) -> str:
        """
        Compute hash based on essential video metadata.
        Uses duration, resolution, and frame count (device-independent properties).
        
        Args:
            file_path: Path to video file
            
        Returns:
            Hexadecimal hash string
        """
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            # Get essential metadata
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Create signature from essential properties
            # Round to handle minor floating point differences
            signature = f"{width}x{height}_{round(fps, 1)}fps_{round(duration, 1)}s_{frame_count}frames"
            
            return hashlib.md5(signature.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Error computing metadata signature: {e}")
            return ""
    
    def compute_all_hashes(self, file_path: Path) -> dict:
        """
        Compute all hash types for a video file.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with file_hash, content_hash, and metadata_hash
        """
        return {
            "video_file_hash": self.compute_file_hash(file_path),
            "video_content_hash": self.compute_content_hash(file_path),
            "video_metadata_hash": self.compute_metadata_signature(file_path),
            "hash_version": "v2"
        }
    
    def _extract_frame_hashes(self, file_path: Path) -> List[str]:
        """
        Extract frames at regular intervals and compute perceptual hashes.
        
        Args:
            file_path: Path to video file
            
        Returns:
            List of hexadecimal hash strings for each frame
        """
        cap = cv2.VideoCapture(str(file_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {file_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            logger.warning(f"Invalid FPS ({fps}) for video: {file_path}")
            cap.release()
            return []
        
        # Calculate frame interval in frame numbers
        frame_interval = int(fps * self.frame_interval_sec)
        if frame_interval <= 0:
            frame_interval = 1
        
        # Sample frames at intervals
        frame_positions = list(range(0, frame_count, frame_interval))
        
        # Limit to max_frames to prevent excessive processing
        if len(frame_positions) > self.max_frames:
            # Sample evenly across the video
            step = len(frame_positions) // self.max_frames
            frame_positions = frame_positions[::step][:self.max_frames]
        
        frame_hashes = []
        
        for frame_pos in frame_positions:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Compute perceptual hash for this frame
            try:
                frame_hash = self._compute_frame_hash(frame)
                frame_hashes.append(frame_hash)
            except Exception as e:
                logger.warning(f"Error hashing frame {frame_pos}: {e}")
                continue
        
        cap.release()
        
        return frame_hashes
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute perceptual hash for a single frame using difference hash (dHash).
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            
        Returns:
            Hexadecimal hash string
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Compute difference hash (dHash)
        # dHash is robust to minor changes and good for video frames
        dhash = imagehash.dhash(pil_image, hash_size=self.hash_size)
        
        return str(dhash)
    
    @staticmethod
    def compute_hamming_distance(hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes.
        Used to determine similarity between videos.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Hamming distance (number of differing characters)
        """
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    @staticmethod
    def compute_similarity_percentage(hash1: str, hash2: str) -> float:
        """
        Compute similarity percentage between two hashes.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Similarity percentage (0-100)
        """
        distance = VideoHasher.compute_hamming_distance(hash1, hash2)
        max_distance = max(len(hash1), len(hash2))
        
        if max_distance == 0:
            return 100.0
        
        similarity = (1 - distance / max_distance) * 100
        return round(similarity, 2)
