import multiprocessing
import os

# Gunicorn settings - conservative for Render's memory constraints
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", 2))  # Default to 2 workers
worker_class = "sync"  # Use sync worker class for reliability
timeout = 60  # Reduced timeout for quicker error detection
keepalive = 2
max_requests = 100  # Lower value to prevent memory leaks
max_requests_jitter = 20
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Handle long-running requests more conservatively
graceful_timeout = 30

# Avoid preloading the app which can cause issues
preload_app = False  

# Logging
accesslog = "-"
errorlog = "-"
access_log_format = '%({X-Forwarded-For}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Error debugging settings
capture_output = True
enable_stdio_inheritance = True

# Memory optimization
worker_tmp_dir = "/dev/shm"  # Use shared memory

# Worker process recycling
max_requests_notify = True
worker_abort_on_error = False 