"""
Input validation utilities for the Video Analyzer Microservice
"""
import os
from typing import List, Optional, Tuple
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class ValidationError(Exception):
    """Custom validation exception"""
    pass

class FileValidator:
    """File validation utilities"""
    
    @staticmethod
    def validate_file_size(file_size: int) -> Tuple[bool, str]:
        """Validate file size"""
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size <= 0:
            return False, "File size must be greater than 0"
        
        if file_size > max_size:
            return False, f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB}MB)"
        
        return True, "Valid file size"
    
    @staticmethod
    def validate_file_type(filename: str, content: bytes = None) -> Tuple[bool, str]:
        """Validate file type based on extension only"""
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv'}
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return False, f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}"
        
        # Note: MIME type validation removed for Windows compatibility
        # In production, consider using a different approach or installing libmagic
        
        return True, "Valid file type"
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """Validate filename"""
        if not filename or len(filename) == 0:
            return False, "Filename cannot be empty"
        
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        
        # Check for dangerous characters
        dangerous_chars = {'..', '/', '\\', ':', '*', '?', '"', '<', '>', '|'}
        if any(char in filename for char in dangerous_chars):
            return False, "Filename contains dangerous characters"
        
        return True, "Valid filename"

class ModelValidator:
    """Model ID validation utilities"""
    
    @staticmethod
    def validate_model_id(model_id: str) -> Tuple[bool, str]:
        """Validate model ID format"""
        if not model_id:
            return False, "Model ID cannot be empty"
        
        if len(model_id) > 100:
            return False, "Model ID too long (max 100 characters)"
        
        if len(model_id) < 3:
            return False, "Model ID too short (min 3 characters)"
        
        # Allow alphanumeric, hyphens, underscores, dots
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.')
        if not all(c in allowed_chars for c in model_id):
            return False, "Model ID contains invalid characters. Only alphanumeric, hyphens, underscores, and dots allowed"
        
        # Check for reserved words
        reserved_words = {'admin', 'api', 'system', 'root', 'test', 'null', 'undefined'}
        if model_id.lower() in reserved_words:
            return False, f"Model ID '{model_id}' is reserved"
        
        return True, "Valid model ID"

class VideoValidator:
    """Video-specific validation utilities"""
    
    @staticmethod
    def validate_video_metadata(metadata: dict) -> Tuple[bool, str]:
        """Validate video metadata"""
        required_fields = ['width', 'height', 'duration_sec']
        
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required metadata field: {field}"
        
        # Validate dimensions
        if metadata['width'] <= 0 or metadata['height'] <= 0:
            return False, "Invalid video dimensions"
        
        if metadata['width'] > 7680 or metadata['height'] > 4320:  # 8K max
            return False, "Video resolution too high (max 8K)"
        
        # Validate duration
        if metadata['duration_sec'] <= 0:
            return False, "Invalid video duration"
        
        if metadata['duration_sec'] > 3600:  # 1 hour max
            return False, "Video too long (max 1 hour)"
        
        return True, "Valid video metadata"

def validate_upload_file(file: UploadFile) -> Tuple[bool, List[str]]:
    """Comprehensive file validation"""
    errors = []
    
    # Validate filename
    is_valid, error_msg = FileValidator.validate_filename(file.filename)
    if not is_valid:
        errors.append(f"Filename validation failed: {error_msg}")
    
    # Validate file size
    if file.size:
        is_valid, error_msg = FileValidator.validate_file_size(file.size)
        if not is_valid:
            errors.append(f"File size validation failed: {error_msg}")
    
    # Validate file type
    is_valid, error_msg = FileValidator.validate_file_type(file.filename)
    if not is_valid:
        errors.append(f"File type validation failed: {error_msg}")
    
    return len(errors) == 0, errors

def validate_model_id(model_id: str) -> Tuple[bool, List[str]]:
    """Comprehensive model ID validation"""
    errors = []
    
    is_valid, error_msg = ModelValidator.validate_model_id(model_id)
    if not is_valid:
        errors.append(f"Model ID validation failed: {error_msg}")
    
    return len(errors) == 0, errors

def validate_request_data(file: UploadFile, model_id: str) -> Tuple[bool, List[str]]:
    """Validate complete request data"""
    errors = []
    
    # Validate file
    file_valid, file_errors = validate_upload_file(file)
    if not file_valid:
        errors.extend(file_errors)
    
    # Validate model ID
    model_valid, model_errors = validate_model_id(model_id)
    if not model_valid:
        errors.extend(model_errors)
    
    return len(errors) == 0, errors

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove dangerous characters
    dangerous_chars = str.maketrans('', '', '..\\/:*?"<>|')
    sanitized = filename.translate(dangerous_chars)
    
    # Limit length
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:200-len(ext)] + ext
    
    return sanitized

def get_file_info(file: UploadFile) -> dict:
    """Extract file information for logging"""
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": file.size,
        "size_mb": round(file.size / (1024 * 1024), 2) if file.size else 0
    }
