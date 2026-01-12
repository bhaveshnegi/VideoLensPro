import os
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
# For CPU-bound tasks (video processing), usually 2-4 workers per core is good, 
# but since we do heavy ML, we might want fewer workers to avoid OOM.
# Defaulting to 2 workers as a safe start for this microservice.
workers = int(os.getenv("WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("WORKERS_TIMEOUT", "300"))  # 5 minutes for long video processing
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "video-analyzer"

# Daemon mode
daemon = False
