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

def validate_request_data(file: UploadFile) -> Tuple[bool, List[str]]:
    """Validate complete request data"""
    errors = []
    
    # Validate file
    file_valid, file_errors = validate_upload_file(file)
    if not file_valid:
        errors.extend(file_errors)
    
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
