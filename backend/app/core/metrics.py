"""
Monitoring and metrics collection
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = datetime.utcnow()
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._get_key(name, labels)
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        key = self._get_key(name, labels)
        self.gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram value"""
        key = self._get_key(name, labels)
        self.histograms[key].append({
            "value": value,
            "timestamp": datetime.utcnow(),
            "labels": labels or {}
        })
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration_ms: float):
        """Record HTTP request metrics"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        self.increment_counter("http_requests_total", labels=labels)
        self.observe_histogram("http_request_duration_ms", duration_ms, labels=labels)
        
        # Status code specific metrics
        if 200 <= status_code < 300:
            self.increment_counter("http_requests_success_total", labels={"method": method, "endpoint": endpoint})
        elif 400 <= status_code < 500:
            self.increment_counter("http_requests_client_error_total", labels={"method": method, "endpoint": endpoint})
        elif 500 <= status_code < 600:
            self.increment_counter("http_requests_server_error_total", labels={"method": method, "endpoint": endpoint})
    
    def record_video_processing(self, duration_ms: float, file_size_mb: float, success: bool):
        """Record video processing metrics"""
        labels = {"success": str(success).lower()}
        
        self.increment_counter("video_processing_total", labels=labels)
        self.observe_histogram("video_processing_duration_ms", duration_ms, labels=labels)
        self.observe_histogram("video_file_size_mb", file_size_mb, labels=labels)
        
        if success:
            self.increment_counter("video_processing_success_total")
        else:
            self.increment_counter("video_processing_failure_total")
    
    def record_model_usage(self, model_name: str, duration_ms: float, success: bool):
        """Record ML model usage metrics"""
        labels = {
            "model": model_name,
            "success": str(success).lower()
        }
        
        self.increment_counter("model_usage_total", labels=labels)
        self.observe_histogram("model_inference_duration_ms", duration_ms, labels=labels)
    
    def record_database_operation(self, operation: str, duration_ms: float, success: bool):
        """Record database operation metrics"""
        labels = {
            "operation": operation,
            "success": str(success).lower()
        }
        
        self.increment_counter("database_operations_total", labels=labels)
        self.observe_histogram("database_operation_duration_ms", duration_ms, labels=labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate histogram statistics
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                histogram_values = [v["value"] for v in values]
                histogram_stats[name] = {
                    "count": len(histogram_values),
                    "min": min(histogram_values),
                    "max": max(histogram_values),
                    "avg": sum(histogram_values) / len(histogram_values),
                    "p50": self._percentile(histogram_values, 50),
                    "p95": self._percentile(histogram_values, 95),
                    "p99": self._percentile(histogram_values, 99)
                }
        
        return {
            "uptime_seconds": uptime,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": histogram_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate metric key with labels"""
        if not labels:
            return name
        
        label_pairs = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in label_pairs)
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

class RequestMetricsMiddleware:
    """Middleware to collect request metrics"""
    
    def __init__(self, app, metrics_collector: MetricsCollector = None):
        self.app = app
        self.metrics_collector = metrics_collector or get_metrics_collector()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request_start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration_ms = (time.time() - request_start_time) * 1000
                
                # Record metrics
                self.metrics_collector.record_request(
                    method=scope["method"],
                    endpoint=scope["path"],
                    status_code=message["status"],
                    duration_ms=duration_ms
                )
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# Global metrics collector
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return metrics_collector

def record_request_metrics(method: str, endpoint: str, status_code: int, duration_ms: float):
    """Record request metrics"""
    metrics_collector.record_request(method, endpoint, status_code, duration_ms)

def record_video_processing_metrics(duration_ms: float, file_size_mb: float, success: bool):
    """Record video processing metrics"""
    metrics_collector.record_video_processing(duration_ms, file_size_mb, success)

def record_model_usage_metrics(model_name: str, duration_ms: float, success: bool):
    """Record model usage metrics"""
    metrics_collector.record_model_usage(model_name, duration_ms, success)

def record_database_operation_metrics(operation: str, duration_ms: float, success: bool):
    """Record database operation metrics"""
    metrics_collector.record_database_operation(operation, duration_ms, success)

def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary"""
    return metrics_collector.get_metrics_summary()
