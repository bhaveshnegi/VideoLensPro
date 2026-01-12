"""
Structured logging configuration for production
"""
import logging
import logging.config
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from app.core.config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class RequestContextFilter(logging.Filter):
    """Add request context to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request context if available
        if hasattr(record, 'request_id'):
            record.extra_fields = getattr(record, 'extra_fields', {})
            record.extra_fields['request_id'] = record.request_id
        
        return True

def setup_logging():
    """Setup structured logging configuration"""
    
    # Determine log format
    if settings.LOG_FORMAT == "json":
        formatter_class = JSONFormatter
    else:
        formatter_class = logging.Formatter
        
    # Base logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": formatter_class,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "()": formatter_class,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
            }
        },
        "filters": {
            "request_context": {
                "()": RequestContextFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "default",
                "stream": sys.stdout,
                "filters": ["request_context"]
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": settings.LOG_LEVEL,
                "handlers": [],
                "propagate": False
            },
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": [],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "ERROR",  # Suppress access logs
                "handlers": [],
                "propagate": False
            }
        }
    }
    
    # Add console handler if enabled
    if settings.LOG_TO_CONSOLE:
        for logger_name in logging_config["loggers"]:
            logging_config["loggers"][logger_name]["handlers"].append("console")
    
    # Add file handler if configured
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        
        # If path is a directory, append default filename
        if log_path.is_dir() or settings.LOG_FILE.endswith(('/', '\\')):
            log_path = log_path / "video_analyzer.log"
            
        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": str(log_path),
            "when": "midnight",
            "interval": 1,
            "backupCount": 7,  # Keep 7 days
            "encoding": "utf-8",
            "filters": ["request_context"]
        }
        
        # Add file handler to all loggers
        for logger_name in logging_config["loggers"]:
            logging_config["loggers"][logger_name]["handlers"].append("file")
    
    # Add separate error log file handler
    error_log_path = Path("AI_logs/errors.log")
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging_config["handlers"]["error_file"] = {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "level": "ERROR",  # Only ERROR and above
        "formatter": "detailed",
        "filename": str(error_log_path),
        "when": "midnight",
        "interval": 1,
        "backupCount": 30,  # Keep 30 days of error logs
        "encoding": "utf-8",
        "filters": ["request_context"]
    }
    
    # Add error file handler to all loggers
    for logger_name in logging_config["loggers"]:
        logging_config["loggers"][logger_name]["handlers"].append("error_file")
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set specific logger levels
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("deepface").setLevel(logging.ERROR)
    
    # Silence noisy database and system loggers
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiosignal").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities"""
    
    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)

def log_request(logger: logging.Logger, request_data: Dict[str, Any], request_id: str = None):
    """Log incoming request"""
    extra_fields = {
        "request_id": request_id,
        "event_type": "request",
        "method": request_data.get("method"),
        "url": request_data.get("url"),
        "client_ip": request_data.get("client_ip"),
        "user_agent": request_data.get("user_agent")
    }
    
    logger.info("Incoming request", extra={"extra_fields": extra_fields})

def log_response(logger: logging.Logger, response_data: Dict[str, Any], request_id: str = None):
    """Log outgoing response"""
    extra_fields = {
        "request_id": request_id,
        "event_type": "response",
        "status_code": response_data.get("status_code"),
        "response_time_ms": response_data.get("response_time_ms"),
        "content_length": response_data.get("content_length")
    }
    
    logger.info("Outgoing response", extra={"extra_fields": extra_fields})

def log_error(logger: logging.Logger, error: Exception, request_id: str = None, context: Dict[str, Any] = None):
    """Log error with context"""
    extra_fields = {
        "request_id": request_id,
        "event_type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    logger.error(f"Error occurred: {str(error)}", extra={"extra_fields": extra_fields}, exc_info=True)

def log_performance(logger: logging.Logger, operation: str, duration_ms: float, request_id: str = None, metadata: Dict[str, Any] = None):
    """Log performance metrics"""
    extra_fields = {
        "request_id": request_id,
        "event_type": "performance",
        "operation": operation,
        "duration_ms": duration_ms,
        "metadata": metadata or {}
    }
    
    logger.info(f"Performance: {operation} took {duration_ms:.2f}ms", extra={"extra_fields": extra_fields})
