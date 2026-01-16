import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from app.core.config import settings
from app.core.logging import setup_logging, get_logger, log_request, log_response, log_error
from app.core.metrics import metrics_collector, RequestMetricsMiddleware
from app.core.tasks import start_task_manager, stop_task_manager
from app.api.v1.router import api_router

# Suppress warnings globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", module="ctranslate2")

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    if settings.ENABLE_BACKGROUND_TASKS:
        await start_task_manager()
        logger.info("Background task manager started")
    
    yield
    
    logger.info("Shutting down application")

    if settings.ENABLE_BACKGROUND_TASKS:
        await stop_task_manager()
        logger.info("Background task manager stopped")

    from app.dependencies import close_database_connections
    await close_database_connections()
    
    logger.info("Application shutdown complete")

# FastAPI app init
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready Video Analysis Microservice",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:9000"] if settings.DEBUG else ["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
if settings.ENABLE_METRICS:
    app.add_middleware(RequestMetricsMiddleware, metrics_collector=metrics_collector)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Only log health check endpoints and errors
    path = request.url.path
    is_health_check = path in ["/", "/health", "/health/detailed"]
    
    client_ip = request.client.host if request.client else "unknown"
    
    # Only log health check requests
    if is_health_check:
        log_request(logger, {
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown")
        }, request_id)

    start_time = time.time()
    try:
        response = await call_next(request)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Only log health check responses
        if is_health_check:
            log_response(logger, {
                "status_code": response.status_code,
                "response_time_ms": response_time_ms,
                "content_length": response.headers.get("content-length", 0)
            }, request_id)
        
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        # Always log errors
        log_error(logger, e, request_id, {
            "method": request.method,
            "url": str(request.url),
            "response_time_ms": response_time_ms
        })
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "message": "An unexpected error occurred"
            }
        )

# Routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "healthy"
    }

@app.get("/health")
async def health():
    from app.core.health import get_quick_health_status
    return await get_quick_health_status()

@app.get("/health/detailed")
async def detailed_health():
    from app.core.health import get_health_status
    return await get_health_status()

@app.get("/metrics")
async def metrics():
    if not settings.ENABLE_METRICS:
        return {"error": "Metrics disabled"}
    from app.core.metrics import get_metrics_summary
    return get_metrics_summary()

@app.get("/info")
async def info():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "features": {
            "background_tasks": settings.ENABLE_BACKGROUND_TASKS,
            "metrics": settings.ENABLE_METRICS,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "rate_limiting": True
        }
    }
