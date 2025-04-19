import multiprocessing
import os

# Gunicorn settings
bind = "0.0.0.0:" + os.environ.get("PORT", "8000")
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"  # Using gevent for async workers
worker_connections = 1000
timeout = 120  # Increased timeout to 120 seconds
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# For handling long-running requests
graceful_timeout = 30 