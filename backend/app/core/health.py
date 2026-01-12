"""
Comprehensive health check system
"""
import asyncio
import time
import psutil
from typing import Dict, Any, List
from datetime import datetime
from app.core.config import settings
from app.core.logging import get_logger, log_performance
from app.dependencies import health_check_database

logger = get_logger(__name__)

class HealthCheckResult:
    """Health check result"""
    
    def __init__(self, name: str, status: str, details: Dict[str, Any] = None, response_time_ms: float = 0):
        self.name = name
        self.status = status  # "healthy", "unhealthy", "degraded"
        self.details = details or {}
        self.response_time_ms = response_time_ms
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp.isoformat()
        }

class HealthChecker:
    """Comprehensive health checker"""
    
    def __init__(self):
        self.checks: List[callable] = [
            self.check_database,
            self.check_disk_space,
            self.check_memory,
            self.check_cpu,
            self.check_models,
            self.check_temp_directory
        ]
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()
        results = []
        
        # Run checks concurrently
        check_tasks = [self._run_check(check) for check in self.checks]
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        for result in check_results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                results.append(HealthCheckResult(
                    name="unknown",
                    status="unhealthy",
                    details={"error": str(result)}
                ))
            else:
                results.append(result)
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(results)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": round(response_time, 2),
            "checks": [result.to_dict() for result in results],
            "summary": self._generate_summary(results)
        }
    
    async def _run_check(self, check_func) -> HealthCheckResult:
        """Run a single health check"""
        start_time = time.time()
        
        try:
            result = await check_func()
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time
                return result
            else:
                return HealthCheckResult(
                    name=check_func.__name__,
                    status="healthy",
                    details=result,
                    response_time_ms=response_time
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check {check_func.__name__} failed: {e}")
            return HealthCheckResult(
                name=check_func.__name__,
                status="unhealthy",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        try:
            db_result = await health_check_database()
            
            if db_result["status"] == "healthy":
                return HealthCheckResult(
                    name="database",
                    status="healthy",
                    details={
                        "database": db_result["database"],
                        "ping": db_result["ping"],
                        "write_test": db_result["write_test"],
                        "read_test": db_result["read_test"],
                        "connection_pool_size": db_result["connection_pool_size"]
                    }
                )
            else:
                return HealthCheckResult(
                    name="database",
                    status="unhealthy",
                    details={"error": db_result["error"]}
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/')
            
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            # Determine status based on free space
            if free_percent < 5:
                status = "unhealthy"
            elif free_percent < 15:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                details={
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "free_percent": round(free_percent, 2),
                    "threshold_warning": 15,
                    "threshold_critical": 5
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def check_memory(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            used_gb = memory.used / (1024**3)
            used_percent = memory.percent
            
            # Determine status based on memory usage
            if used_percent > 90:
                status = "unhealthy"
            elif used_percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                details={
                    "total_gb": round(total_gb, 2),
                    "available_gb": round(available_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "used_percent": round(used_percent, 2),
                    "threshold_warning": 80,
                    "threshold_critical": 90
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def check_cpu(self) -> HealthCheckResult:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            # Determine status based on CPU usage
            if cpu_percent > 90:
                status = "unhealthy"
            elif cpu_percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            details = {
                "cpu_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "threshold_warning": 80,
                "threshold_critical": 90
            }
            
            if load_avg:
                details["load_avg_1min"] = round(load_avg[0], 2)
                details["load_avg_5min"] = round(load_avg[1], 2)
                details["load_avg_15min"] = round(load_avg[2], 2)
            
            return HealthCheckResult(
                name="cpu",
                status=status,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="cpu",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def check_models(self) -> HealthCheckResult:
        """Check ML model availability"""
        try:
            from pathlib import Path
            
            model_paths = {
                "yolo": Path(settings.YOLO_MODEL_PATH),
                "whisper": Path(settings.WHISPER_MODEL_PATH),
                "deepface": Path(settings.DEEPFACE_HOME)
            }
            
            model_status = {}
            all_healthy = True
            
            for model_name, model_path in model_paths.items():
                if model_path.exists():
                    model_status[model_name] = {
                        "available": True,
                        "path": str(model_path),
                        "size_mb": round(sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**2), 2)
                    }
                else:
                    model_status[model_name] = {
                        "available": False,
                        "path": str(model_path),
                        "error": "Model file/directory not found"
                    }
                    all_healthy = False
            
            status = "healthy" if all_healthy else "degraded"
            
            return HealthCheckResult(
                name="models",
                status=status,
                details=model_status
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="models",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    async def check_temp_directory(self) -> HealthCheckResult:
        """Check temporary directory accessibility"""
        try:
            import tempfile
            import os
            
            # Test temp directory write access
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(b"test")
                tmp_file.flush()
            
            # Check temp directory size
            temp_dir = tempfile.gettempdir()
            temp_usage = psutil.disk_usage(temp_dir)
            temp_free_gb = temp_usage.free / (1024**3)
            
            status = "healthy" if temp_free_gb > 1 else "degraded"
            
            return HealthCheckResult(
                name="temp_directory",
                status=status,
                details={
                    "temp_dir": temp_dir,
                    "free_space_gb": round(temp_free_gb, 2),
                    "writable": True
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="temp_directory",
                status="unhealthy",
                details={"error": str(e)}
            )
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> str:
        """Calculate overall health status"""
        if not results:
            return "unknown"
        
        statuses = [result.status for result in results]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def _generate_summary(self, results: List[HealthCheckResult]) -> Dict[str, int]:
        """Generate health check summary"""
        summary = {"healthy": 0, "degraded": 0, "unhealthy": 0}
        
        for result in results:
            summary[result.status] += 1
        
        return summary

# Global health checker instance
health_checker = HealthChecker()

async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    return await health_checker.run_all_checks()

async def get_quick_health_status() -> Dict[str, Any]:
    """Get quick health status (database only)"""
    try:
        db_result = await health_check_database()
        return {
            "status": "healthy" if db_result["status"] == "healthy" else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_result["status"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
