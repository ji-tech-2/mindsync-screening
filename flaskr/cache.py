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
    logger.info(f"Initializing Valkey connection: {valkey_url}")

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
        logger.error(f"Failed to connect to Valkey: {e}")
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
        logger.debug(f"Stored prediction {prediction_id} in cache with 24h TTL")
    except Exception as e:
        logger.error(f"Valkey store error for {prediction_id}: {e}")
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
            logger.debug(f"Cache hit for prediction {prediction_id}")
            return json.loads(data)
        logger.debug(f"Cache miss for prediction {prediction_id}")
        return None
    except Exception as e:
        logger.error(f"Valkey fetch error for {prediction_id}: {e}")
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
            logger.warning(f"Prediction {prediction_id} not found in cache for update")
            return

        data = json.loads(existing)
        data.update(update_data)
        valkey_client.setex(key, 86400, json.dumps(data))
        logger.debug(f"Updated prediction {prediction_id} in cache")
    except Exception as e:
        logger.error(f"Valkey update error for {prediction_id}: {e}")
