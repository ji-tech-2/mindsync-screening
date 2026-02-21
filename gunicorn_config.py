"""
Gunicorn Configuration File
Production-ready settings for MindSync Model Flask API

MEMORY OPTIMIZATION:
- Reduced workers to avoid OOM (each worker loads full ML model)
- Using threads for concurrency (shared memory)
- Aggressive worker recycling to prevent memory leaks
"""

import multiprocessing
import os
import logging

logger = logging.getLogger(__name__)

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker Processes - OPTIMIZED FOR ML MODEL MEMORY USAGE
# Default: 2 workers with 2 threads each = 4 concurrent requests
# Adjust GUNICORN_WORKERS and GUNICORN_THREADS based on your memory
workers = int(os.getenv("GUNICORN_WORKERS", "4"))  # Conservative default
worker_class = os.getenv(
    "GUNICORN_WORKER_CLASS", "gthread"
)  # Use threads for memory efficiency
worker_connections = 1000

# Threading - Each worker can handle multiple requests via threads (shared memory!)
threads = int(os.getenv("GUNICORN_THREADS", "2"))  # 2 threads per worker

# Worker Lifecycle - Aggressive recycling to prevent memory bloat
max_requests = int(
    os.getenv("GUNICORN_MAX_REQUESTS", "500")
)  # Restart after 500 requests
max_requests_jitter = 50  # Randomize restart to avoid thundering herd
timeout = 120  # Request timeout (ML inference can be slow)
keepalive = 5  # Keep-alive timeout

# Worker Memory Management
worker_tmp_dir = "/dev/shm"  # Use shared memory for temp files (Linux only)

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
    server.log.info(
        f"⚙️  Configuration: {workers} workers × {threads} threads = {workers * threads} concurrent requests"
    )
    server.log.info(
        f"💾 Memory optimization: worker_class={worker_class}, max_requests={max_requests}"
    )


def on_reload(server):
    """Called to recycle workers during a reload."""
    server.log.info("🔄 Reloading Gunicorn server")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info(f"✅ Server is ready. Listening on: {bind}")
    server.log.info(f"📊 Total concurrent capacity: {workers * threads} requests")


def post_fork(server, worker):
    """Called after a worker has been forked."""
    server.log.info(f"👷 Worker {worker.pid} spawned")


def pre_fork(server, worker):
    """Called before a worker is forked."""
    pass


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info(f"⚠️  Worker {worker.pid} received INT or QUIT signal")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.error(
        f"❌ Worker {worker.pid} received SIGABRT signal (likely OOM kill)"
    )


def worker_exit(server, worker):
    """Called when a worker is exiting."""
    server.log.info(f"👋 Worker {worker.pid} exiting")


def pre_request(worker, req):
    """Called before processing each request."""
    # Log memory usage periodically (every 100 requests)
    if hasattr(worker, "request_count"):
        worker.request_count += 1
    else:
        worker.request_count = 1

    if worker.request_count % 100 == 0:
        try:
            import psutil

            process = psutil.Process(worker.pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            worker.log.info(
                f"📊 Worker {worker.pid} memory: {memory_mb:.1f} MB (after {worker.request_count} requests)"
            )
        except ImportError:
            pass  # psutil not available


def post_request(worker, req, environ, resp):
    """Called after processing each request."""
    pass


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
