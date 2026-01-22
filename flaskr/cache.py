"""
Valkey/Redis cache management
"""
import json
import valkey

# Global valkey client
valkey_client = None

def init_app(app):
    """Initialize Valkey client with the app."""
    global valkey_client
    
    valkey_url = app.config.get('VALKEY_URL', 'redis://localhost:6379')
    
    try:
        print(f"Connecting to Valkey at {valkey_url}...")
        valkey_client = valkey.from_url(
            valkey_url,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
            health_check_interval=10
        )
        valkey_client.ping()
        print("✅ Successfully connected to Valkey")
    except Exception as e:
        print(f"❌ WARNING: Could not connect to Valkey: {e}")
        print("⚠️ Application will run without caching")
        valkey_client = None

def store_prediction(prediction_id, prediction_data):
    """Store prediction data with 24-hour expiration."""
    if valkey_client is None:
        print("⚠️ Valkey not available, skipping cache store")
        return
    
    try:
        key = f"prediction:{prediction_id}"
        valkey_client.setex(
            key,
            86400,  # 24 hours
            json.dumps(prediction_data)
        )
    except Exception as e:
        print(f"Valkey store error: {e}")
        # Don't raise - allow operation to continue without cache

def fetch_prediction(prediction_id):
    """Fetch prediction data from Valkey."""
    if valkey_client is None:
        return None
    
    try:
        key = f"prediction:{prediction_id}"
        data = valkey_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"Valkey fetch error: {e}")
        return None

def update_prediction(prediction_id, update_data):
    """Update existing prediction data."""
    if valkey_client is None:
        print("⚠️ Valkey not available, skipping cache update")
        return
    
    try:
        key = f"prediction:{prediction_id}"
        existing = valkey_client.get(key)
        
        if not existing:
            print(f"⚠️ Prediction {prediction_id} not found in cache")
            return
        
        data = json.loads(existing)
        data.update(update_data)
        valkey_client.setex(key, 86400, json.dumps(data))
    except Exception as e:
        print(f"Valkey update error: {e}")
