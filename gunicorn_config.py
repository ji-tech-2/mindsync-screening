"""
Gunicorn Configuration File
Production-ready settings for MindSync Model Flask API
"""

import multiprocessing
import os

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker Processes
# Formula: (2 x $num_cores) + 1
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"  # Use 'gevent' or 'eventlet' for async if needed
worker_connections = 1000
max_requests = 1000  # Restart workers after this many requests (prevents memory leaks)
max_requests_jitter = (
    50  # Adds randomness to max_requests to avoid all workers restarting simultaneously
)
timeout = 120  # Worker timeout in seconds (increase for ML inference)
keepalive = 5  # Seconds to wait for requests on a Keep-Alive connection

# Threading (if using threaded workers)
threads = 1

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process Naming
proc_name = "mindsync-model-flask"


# Server Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("🚀 Starting Gunicorn server")


def on_reload(server):
    """Called to recycle workers during a reload."""
    server.log.info("🔄 Reloading Gunicorn server")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info(f"✅ Server is ready. Listening on: {bind}")


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("⚠️ Worker received INT or QUIT signal")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("❌ Worker received SIGABRT signal")


# SSL (if needed)
# keyfile = None
# certfile = None

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Preload the application
# This saves memory but means code changes require full restart
preload_app = True

# Graceful timeout for workers
graceful_timeout = 30

# Environment variables
raw_env = []

# Paste Deployment
# paste = None

# Statsd monitoring (optional)
# statsd_host = None
# statsd_prefix = ''

# Prometheus monitoring endpoint (if using prometheus-flask-exporter)
# Access at /metrics
