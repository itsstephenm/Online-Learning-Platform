import multiprocessing
import os

# Gunicorn settings
bind = "0.0.0.0:" + os.environ.get("PORT", "8000")
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"  # Using gevent for async workers
worker_connections = 1000
timeout = 120  # Increased timeout to 120 seconds
keepalive = 5
max_requests = 500  # Reduced to prevent memory growth
max_requests_jitter = 100  # Increased jitter to prevent simultaneous restarts
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# For handling long-running requests
graceful_timeout = 30

# Memory optimization
worker_tmp_dir = "/dev/shm"  # Use shared memory for temp files
preload_app = True  # Load application code before worker processes are forked

# Worker process recycling
max_requests_notify = True
worker_abort_on_error = False  # Don't crash everything if one worker fails

# Logging
accesslog = "-"
errorlog = "-"
access_log_format = '%({X-Forwarded-For}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"' 