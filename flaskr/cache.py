"""
Valkey/Redis cache management
"""

import json
import logging
import socket
import threading
import time
import uuid
import valkey

logger = logging.getLogger(__name__)

# Global valkey client
valkey_client = None

# Stream configuration
HANDOVER_STREAM = "handover_stream"
CONSUMER_GROUP = "screening_group"
# Generate unique consumer name per instance to avoid round-robin in multi-instance deployments
try:
    hostname = socket.gethostname()
except Exception:
    hostname = "unknown"
CONSUMER_NAME = f"worker_{hostname}_{uuid.uuid4().hex[:8]}"

# Worker control
worker_thread = None
worker_running = False


def init_app(app):
    """Initialize Valkey client with the app."""
    global valkey_client, worker_thread, worker_running

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

        # Create consumer group for handover stream
        _create_consumer_group()

        # Start handover worker thread
        if not app.config.get("DB_DISABLED", False):
            logger.info("Starting handover worker thread")
            worker_running = True
            # Use local variable to avoid race conditions
            local_thread = threading.Thread(
                target=_handover_worker, args=(app,), daemon=True
            )
            local_thread.start()
            # Assign to global after thread is started
            worker_thread = local_thread
            logger.info("Handover worker thread started")
        else:
            logger.info("Database disabled, skipping handover worker")

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


def _create_consumer_group():
    """Create consumer group for handover stream."""
    if valkey_client is None:
        logger.warning("Valkey not available, cannot create consumer group")
        return

    try:
        # Try to create the consumer group
        # XGROUP CREATE stream group id MKSTREAM
        valkey_client.xgroup_create(
            name=HANDOVER_STREAM, groupname=CONSUMER_GROUP, id="$", mkstream=True
        )
        logger.info(
            "Created consumer group '%s' for stream '%s'",
            CONSUMER_GROUP,
            HANDOVER_STREAM,
        )
    except valkey.exceptions.ResponseError as e:
        # Group might already exist
        if "BUSYGROUP" in str(e):
            logger.info(
                "Consumer group '%s' already exists for stream '%s'",
                CONSUMER_GROUP,
                HANDOVER_STREAM,
            )
        else:
            logger.error("Error creating consumer group: %s", e)
    except Exception as e:
        logger.error("Unexpected error creating consumer group: %s", e)


def _handover_worker(app):
    """Background worker to process handover messages from the stream."""
    global worker_running

    logger.info("Handover worker started, listening on stream '%s'", HANDOVER_STREAM)

    while worker_running:
        if valkey_client is None:
            logger.warning("Valkey client not available, stopping handover worker")
            break

        try:
            # Read from the stream using consumer group
            # XREADGROUP GROUP group consumer COUNT 1 BLOCK 2000 STREAMS stream >
            messages = valkey_client.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={HANDOVER_STREAM: ">"},
                count=1,
                block=2000,  # Block for 2 seconds
            )

            if not messages:
                continue

            # Process each message
            for stream_name, message_list in messages:
                for message_id, message_data in message_list:
                    logger.info(
                        "Received handover message %s: %s",
                        (
                            message_id.decode()
                            if isinstance(message_id, bytes)
                            else message_id
                        ),
                        message_data,
                    )

                    # Process the handover
                    success = _process_handover(app, message_data)

                    if success:
                        # Acknowledge the message
                        valkey_client.xack(HANDOVER_STREAM, CONSUMER_GROUP, message_id)
                        logger.info(
                            "Handover message %s acknowledged",
                            (
                                message_id.decode()
                                if isinstance(message_id, bytes)
                                else message_id
                            ),
                        )
                    else:
                        logger.error(
                            "Failed to process handover message %s, not acknowledging",
                            (
                                message_id.decode()
                                if isinstance(message_id, bytes)
                                else message_id
                            ),
                        )

        except Exception as e:
            logger.error("Error in handover worker: %s", e, exc_info=True)
            # Sleep a bit before retrying to avoid tight loop on persistent errors
            time.sleep(5)

    logger.info("Handover worker stopped")


def _process_handover(app, message_data):
    """Process a handover message and update the database.

    Args:
        app: Flask application instance
        message_data: Dict containing 'guest_id' and 'user_id'

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Decode message data if bytes
        if isinstance(message_data, dict):
            decoded_data = {}
            for key, value in message_data.items():
                k = key.decode() if isinstance(key, bytes) else key
                v = value.decode() if isinstance(value, bytes) else value
                decoded_data[k] = v
            message_data = decoded_data

        guest_id_str = message_data.get("guest_id")
        user_id_str = message_data.get("user_id")

        if not guest_id_str or not user_id_str:
            logger.error(
                "Invalid handover message: missing guest_id or user_id - %s",
                message_data,
            )
            return False

        # Validate UUIDs
        try:
            guest_uuid = uuid.UUID(guest_id_str)
            user_uuid = uuid.UUID(user_id_str)
        except ValueError as e:
            logger.error("Invalid UUID format in handover message: %s", e)
            return False

        logger.info(
            "Processing handover: guest_id=%s -> user_id=%s", guest_id_str, user_id_str
        )

        # Import db here to avoid circular imports
        from .db import db, Predictions

        # Reuse the same app context for both commit and rollback
        with app.app_context():
            try:
                # Update all predictions for this guest_id
                predictions_updated = (
                    db.session.query(Predictions)
                    .filter(Predictions.guest_id == guest_uuid)
                    .update(
                        {Predictions.user_id: user_uuid, Predictions.guest_id: None},
                        synchronize_session=False,
                    )
                )

                db.session.commit()

                logger.info(
                    "Handover completed: %d predictions transferred from guest_id=%s to user_id=%s",
                    predictions_updated,
                    guest_id_str,
                    user_id_str,
                )

                return True
            except Exception as db_error:
                logger.error(
                    "Database error during handover: %s", db_error, exc_info=True
                )
                db.session.rollback()
                raise

    except Exception as e:
        logger.error("Error processing handover: %s", e, exc_info=True)
        return False


def stop_worker():
    """Stop the handover worker thread gracefully."""
    global worker_running
    worker_running = False
    if worker_thread and worker_thread.is_alive():
        logger.info("Stopping handover worker thread...")
        worker_thread.join(timeout=5)
        logger.info("Handover worker thread stopped")
