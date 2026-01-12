import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import wave
import struct

from app.core.logging import get_logger

logger = get_logger(__name__)


class VideoQualityService:
    """
    Analyzes video quality across multiple dimensions:
    - Technical Quality (resolution, fps, bitrate)
    - Visual Quality (brightness, sharpness, stability)
    - Content Quality (face clarity, product visibility)
    - Audio Quality (volume, clarity, speech rate)
    """
    
    def __init__(self):
        self.logger = logger
    
    def analyze_quality(
        self,
        video_path: Path,
        frames_dir: Path,
        audio_path: Optional[Path],
        metadata: Dict[str, Any],
        face_confidence: float = 0.0,
        product_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis
        
        Args:
            video_path: Path to video file
            frames_dir: Directory containing extracted frames
            audio_path: Path to extracted audio file
            metadata: Video metadata (duration, resolution, fps, etc.)
            face_confidence: Confidence score from face detection
            product_confidence: Confidence score from product detection
            
        Returns:
            Dictionary with quality scores and breakdown
        """
        try:
            technical_score = self._analyze_technical_quality(metadata)
            visual_score = self._analyze_visual_quality(frames_dir, metadata)
            content_score = self._analyze_content_quality(face_confidence, product_confidence)
            audio_score = self._analyze_audio_quality(audio_path, metadata)
            
            return {
                "technical_score": round(technical_score, 2),
                "visual_score": round(visual_score, 2),
                "content_score": round(content_score, 2),
                "audio_score": round(audio_score, 2),
                "total_quality_score": round(
                    technical_score + visual_score + content_score + audio_score, 2
                )
            }
        except Exception as e:
            self.logger.warning(f"Quality analysis failed: {e}, using default scores")
            # Return middle-range fallback scores
            return {
                "technical_score": 7.5,
                "visual_score": 7.5,
                "content_score": 5.0,
                "audio_score": 5.0,
                "total_quality_score": 25.0
            }
    
    def _analyze_technical_quality(self, metadata: Dict[str, Any]) -> float:
        """
        Analyze technical quality metrics (0-15 points)
        - Resolution (0-5)
        - Frame rate (0-5)
        - Bitrate (0-5)
        """
        score = 0.0
        
        # Resolution scoring (0-5 points)
        width = metadata.get("width", 0)
        height = metadata.get("height", 0)
        pixels = width * height
        
        if pixels >= 1920 * 1080:  # 1080p+
            score += 5.0
        elif pixels >= 1280 * 720:  # 720p
            score += 3.5
        elif pixels >= 854 * 480:  # 480p
            score += 2.0
        else:
            score += 0.5
        
        # Frame rate scoring (0-5 points)
        fps = metadata.get("fps", 0)
        if fps >= 30:
            score += 5.0
        elif fps >= 24:
            score += 3.5
        elif fps >= 15:
            score += 2.0
        else:
            score += 0.5
        
        # Bitrate scoring (0-5 points)
        bitrate = metadata.get("bitrate", 0)
        
        # Expected bitrate based on resolution
        if pixels >= 1920 * 1080:
            if bitrate >= 5_000_000:  # 5 Mbps
                score += 5.0
            elif bitrate >= 3_000_000:
                score += 3.0
            else:
                score += 1.0
        elif pixels >= 1280 * 720:
            if bitrate >= 3_000_000:
                score += 5.0
            elif bitrate >= 2_000_000:
                score += 3.5
            else:
                score += 1.5
        else:
            if bitrate >= 1_500_000:
                score += 4.0
            elif bitrate >= 800_000:
                score += 2.5
            else:
                score += 1.0
        
        return min(score, 15.0)
    
    def _analyze_visual_quality(self, frames_dir: Path, metadata: Dict[str, Any]) -> float:
        """
        Analyze visual quality metrics (0-15 points)
        - Brightness/exposure (0-5)
        - Sharpness/blur (0-5)
        - Motion stability (0-5)
        """
        score = 0.0
        
        try:
            frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
            
            if not frame_files:
                return 7.5  # Default middle score if no frames
            
            # Sample frames (max 10 for performance)
            sample_size = min(10, len(frame_files))
            sampled_frames = [frame_files[i] for i in range(0, len(frame_files), len(frame_files) // sample_size)][:sample_size]
            
            brightness_scores = []
            sharpness_scores = []
            prev_frame = None
            motion_scores = []
            
            for frame_path in sampled_frames:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                # Brightness analysis
                brightness_score = self._analyze_brightness(frame)
                brightness_scores.append(brightness_score)
                
                # Sharpness analysis
                sharpness_score = self._analyze_sharpness(frame)
                sharpness_scores.append(sharpness_score)
                
                # Motion stability analysis
                if prev_frame is not None:
                    motion_score = self._analyze_motion_stability(prev_frame, frame)
                    motion_scores.append(motion_score)
                
                prev_frame = frame
            
            # Aggregate scores
            if brightness_scores:
                score += np.mean(brightness_scores)
            if sharpness_scores:
                score += np.mean(sharpness_scores)
            if motion_scores:
                score += np.mean(motion_scores)
            else:
                score += 2.5  # Default if can't measure motion
                
        except Exception as e:
            self.logger.warning(f"Visual quality analysis failed: {e}")
            return 7.5  # Default middle score on failure
        
        return min(score, 15.0)
    
    def _analyze_brightness(self, frame: np.ndarray) -> float:
        """
        Analyze frame brightness (0-5 points)
        Optimal range: 80-180 (on 0-255 scale)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Optimal brightness range
        if 80 <= mean_brightness <= 180:
            return 5.0
        elif 60 <= mean_brightness < 80 or 180 < mean_brightness <= 200:
            return 3.5
        elif 40 <= mean_brightness < 60 or 200 < mean_brightness <= 220:
            return 2.0
        else:
            return 0.5  # Too dark or too bright
    
    def _analyze_sharpness(self, frame: np.ndarray) -> float:
        """
        Analyze frame sharpness using Laplacian variance (0-5 points)
        Higher variance = sharper image
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Thresholds based on typical values
        if laplacian_var > 500:
            return 5.0
        elif laplacian_var > 200:
            return 4.0
        elif laplacian_var > 100:
            return 3.0
        elif laplacian_var > 50:
            return 2.0
        else:
            return 0.5  # Very blurry
    
    def _analyze_motion_stability(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Analyze motion stability between frames (0-5 points)
        Lower motion = more stable
        """
        try:
            # Resize for performance
            prev_small = cv2.resize(prev_frame, (320, 240))
            curr_small = cv2.resize(curr_frame, (320, 240))
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = np.mean(diff)
            
            # Lower difference = more stable
            if mean_diff < 10:
                return 5.0  # Very stable
            elif mean_diff < 20:
                return 4.0
            elif mean_diff < 30:
                return 3.0
            elif mean_diff < 50:
                return 2.0
            else:
                return 1.0  # Shaky/unstable
                
        except Exception as e:
            self.logger.warning(f"Motion stability analysis failed: {e}")
            return 2.5  # Default middle score
    
    def _analyze_content_quality(self, face_confidence: float, product_confidence: float) -> float:
        """
        Analyze content visibility (0-10 points)
        - Face clarity (0-5)
        - Product visibility (0-5)
        """
        score = 0.0
        
        # Face clarity scoring (0-5 points)
        if face_confidence >= 0.9:
            score += 5.0
        elif face_confidence >= 0.7:
            score += 4.0
        elif face_confidence >= 0.5:
            score += 3.0
        elif face_confidence >= 0.3:
            score += 2.0
        else:
            score += 1.0
        
        # Product visibility scoring (0-5 points)
        if product_confidence >= 0.9:
            score += 5.0
        elif product_confidence >= 0.7:
            score += 4.0
        elif product_confidence >= 0.5:
            score += 3.0
        elif product_confidence >= 0.3:
            score += 2.0
        else:
            score += 1.0
        
        return min(score, 10.0)
    
    def _analyze_audio_quality(self, audio_path: Optional[Path], metadata: Dict[str, Any]) -> float:
        """
        Analyze audio quality (0-10 points)
        - Volume levels (0-3)
        - Audio presence and duration (0-4)
        - Speech rate estimate (0-3)
        """
        if not audio_path or not audio_path.exists():
            return 5.0  # Default middle score if no audio
        
        score = 0.0
        
        try:
            # Open WAV file
            with wave.open(str(audio_path), 'rb') as wav_file:
                # Get audio parameters
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                duration = n_frames / sample_rate
                
                # Read audio data
                frames = wav_file.readframes(n_frames)
                
                # Convert to numpy array (16-bit audio)
                if wav_file.getsampwidth() == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                else:
                    # Fallback for other bit depths
                    return 5.0
                
                # Convert to mono if stereo
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                # Volume level analysis (0-3 points)
                rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
                normalized_rms = rms / 32768.0  # Normalize to 0-1 range
                
                if 0.1 <= normalized_rms <= 0.5:
                    score += 3.0  # Good volume
                elif 0.05 <= normalized_rms < 0.1 or 0.5 < normalized_rms <= 0.7:
                    score += 2.0  # Acceptable
                elif normalized_rms > 0.001:
                    score += 1.0  # Too quiet or too loud
                else:
                    score += 0.0  # Essentially silent
                
                # Audio duration quality (0-4 points)
                duration_ratio = duration / metadata.get("duration_sec", duration)
                if duration_ratio > 0.95:
                    score += 4.0  # Full audio coverage
                elif duration_ratio > 0.8:
                    score += 3.0
                elif duration_ratio > 0.5:
                    score += 2.0
                else:
                    score += 1.0
                
                # Speech rate estimate (0-3 points)
                # Simple heuristic: detect energy variations
                # Higher variation suggests active speech
                energy = audio_data.astype(float) ** 2
                window_size = int(sample_rate * 0.1)  # 100ms windows
                
                if len(energy) > window_size:
                    windowed_energy = []
                    for i in range(0, len(energy) - window_size, window_size):
                        windowed_energy.append(np.mean(energy[i:i+window_size]))
                    
                    if windowed_energy:
                        energy_variation = np.std(windowed_energy) / (np.mean(windowed_energy) + 1e-6)
                        
                        if energy_variation > 0.5:
                            score += 3.0  # Good speech dynamics
                        elif energy_variation > 0.3:
                            score += 2.0
                        else:
                            score += 1.0
                    else:
                        score += 1.5  # Default
                else:
                    score += 1.5  # Too short to analyze
                    
        except Exception as e:
            self.logger.warning(f"Audio quality analysis failed: {e}")
            return 5.0  # Default middle score on failure
        
        return min(score, 10.0)
