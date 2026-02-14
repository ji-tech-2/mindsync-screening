"""
Valkey/Redis cache management
"""

import json
import logging
import valkey

logger = logging.getLogger(__name__)

# Global valkey client
valkey_client = None


def init_app(app):
    """Initialize Valkey client with the app."""
    global valkey_client

    valkey_url = app.config.get("VALKEY_URL", "redis://localhost:6379")
    logger.info("Initializing Valkey connection: %s", valkey_url)

    try:
        valkey_client = valkey.from_url(
            valkey_url,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
            health_check_interval=10,
        )
        valkey_client.ping()
        logger.info("Successfully connected to Valkey/Redis cache")
    except Exception as e:
        logger.error("Failed to connect to Valkey: %s", e)
        logger.warning("Application will run without caching")
        valkey_client = None


def store_prediction(prediction_id, prediction_data):
    """Store prediction data with 24-hour expiration."""
    if valkey_client is None:
        logger.debug("Valkey not available, skipping cache store")
        return

    try:
        key = f"prediction:{prediction_id}"
        valkey_client.setex(key, 86400, json.dumps(prediction_data))  # 24 hours
        logger.debug("Stored prediction %s in cache with 24h TTL", prediction_id)
    except Exception as e:
        logger.error("Valkey store error for %s: %s", prediction_id, e)
        # Don't raise - allow operation to continue without cache


def fetch_prediction(prediction_id):
    """Fetch prediction data from Valkey."""
    if valkey_client is None:
        logger.debug("Valkey not available for fetch")
        return None

    try:
        key = f"prediction:{prediction_id}"
        data = valkey_client.get(key)
        if data:
            logger.debug("Cache hit for prediction %s", prediction_id)
            return json.loads(data)
        logger.debug("Cache miss for prediction %s", prediction_id)
        return None
    except Exception as e:
        logger.error("Valkey fetch error for %s: %s", prediction_id, e)
        return None


def is_available():
    """Check if Valkey cache is available."""
    return valkey_client is not None


def update_prediction(prediction_id, update_data):
    """Update existing prediction data."""
    if valkey_client is None:
        logger.debug("Valkey not available for update")
        return

    try:
        key = f"prediction:{prediction_id}"
        existing = valkey_client.get(key)

        if not existing:
            logger.warning("Prediction %s not found in cache for update", prediction_id)
            return

        data = json.loads(existing)
        data.update(update_data)
        valkey_client.setex(key, 86400, json.dumps(data))
        logger.debug("Updated prediction %s in cache", prediction_id)
    except Exception as e:
        logger.error("Valkey update error for %s: %s", prediction_id, e)
