"""
Security utilities for the Video Analyzer Microservice
"""
import os
import hashlib
import hmac
import time
from typing import Optional
from fastapi import HTTPException, Header, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Security configuration
SERVICE_API_KEY = os.getenv("VIDEO_ANALYZER_API_KEY")
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "100"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "500")) * 1024 * 1024  # bytes

# In-memory rate limiting (use Redis in production)
rate_limit_storage = {}

security_scheme = HTTPBearer(auto_error=False)

class SecurityError(Exception):
    """Custom security exception"""
    pass

def verify_api_key(api_key: str) -> bool:
    """Verify API key"""
    if not SERVICE_API_KEY:
        logger.warning("SERVICE_API_KEY not configured - security disabled")
        return True
    
    if not api_key:
        return False
    
    return hmac.compare_digest(api_key, SERVICE_API_KEY)

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    current_time = int(time.time())
    window_start = current_time - RATE_LIMIT_WINDOW
    
    # Clean old entries
    if client_ip in rate_limit_storage:
        rate_limit_storage[client_ip] = [
            timestamp for timestamp in rate_limit_storage[client_ip]
            if timestamp > window_start
        ]
    else:
        rate_limit_storage[client_ip] = []
    
    # Check if limit exceeded
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    return True

def validate_file_size(file_size: int) -> bool:
    """Validate file size"""
    return file_size <= MAX_FILE_SIZE

def validate_file_type(filename: str) -> bool:
    """Validate file type"""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in allowed_extensions

def validate_model_id(model_id: str) -> bool:
    """Validate model ID format"""
    if not model_id or len(model_id) > 100:
        return False
    
    # Allow alphanumeric, hyphens, underscores
    return all(c.isalnum() or c in '-_' for c in model_id)

async def verify_service_access(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None)
) -> bool:
    """Verify service access through API key or Bearer token"""
    client_ip = request.client.host
    
    # Check rate limiting
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Check API key (preferred for service-to-service)
    if x_api_key:
        if verify_api_key(x_api_key):
            logger.info(f"API key authentication successful for IP: {client_ip}")
            return True
        else:
            logger.warning(f"Invalid API key from IP: {client_ip}")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid API key"
            )
    
    # Check Bearer token (for API Gateway integration)
    if authorization:
        # In production, validate JWT token here
        # For now, just check if token exists
        if authorization.credentials:
            logger.info(f"Bearer token authentication successful for IP: {client_ip}")
            return True
        else:
            logger.warning(f"Invalid Bearer token from IP: {client_ip}")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid token"
            )
    
    # No authentication provided
    logger.warning(f"No authentication provided from IP: {client_ip}")
    raise HTTPException(
        status_code=401,
        detail="Unauthorized: API key or Bearer token required"
    )

def get_client_info(request: Request) -> dict:
    """Extract client information for logging"""
    return {
        "ip": request.client.host,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "forwarded_for": request.headers.get("x-forwarded-for"),
        "real_ip": request.headers.get("x-real-ip")
    }
