"""
Unit tests for cache.py - Valkey/Redis caching functions
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock


class TestCacheInitialization:
    """Test cache initialization."""
    
    @patch('flaskr.cache.valkey')
    def test_successful_connection(self, mock_valkey):
        """Test successful Valkey connection."""
        from flaskr import cache
        
        # Mock successful connection
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_valkey.from_url.return_value = mock_client
        
        # Create mock app
        app = Mock()
        app.config.get.return_value = 'redis://localhost:6379'
        
        cache.init_app(app)
        
        mock_valkey.from_url.assert_called_once()
        mock_client.ping.assert_called_once()
    
    @patch('flaskr.cache.valkey')
    def test_connection_failure(self, mock_valkey):
        """Test handling of connection failure."""
        from flaskr import cache
        
        # Mock connection failure
        mock_valkey.from_url.side_effect = Exception("Connection failed")
        
        app = Mock()
        app.config.get.return_value = 'redis://localhost:6379'
        
        # Should not raise exception
        cache.init_app(app)
        
        # valkey_client should be None after failure
        assert cache.valkey_client is None


class TestStorePrediction:
    """Test store_prediction function."""
    
    def test_store_with_valid_client(self):
        """Test storing prediction with valid client."""
        from flaskr import cache
        
        mock_client = Mock()
        cache.valkey_client = mock_client
        
        prediction_id = "test-id-123"
        prediction_data = {"status": "ready", "result": {"score": 75}}
        
        cache.store_prediction(prediction_id, prediction_data)
        
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == f"prediction:{prediction_id}"
        assert call_args[0][1] == 86400  # 24 hours
        assert json.loads(call_args[0][2]) == prediction_data
    
    def test_store_with_none_client(self):
        """Test storing prediction when client is None."""
        from flaskr import cache
        
        cache.valkey_client = None
        
        # Should not raise exception
        cache.store_prediction("test-id", {"status": "ready"})
    
    def test_store_with_exception(self):
        """Test handling exception during store."""
        from flaskr import cache
        
        mock_client = Mock()
        mock_client.setex.side_effect = Exception("Store failed")
        cache.valkey_client = mock_client
        
        # Should not raise exception
        cache.store_prediction("test-id", {"status": "ready"})


class TestFetchPrediction:
    """Test fetch_prediction function."""
    
    def test_fetch_existing_prediction(self):
        """Test fetching existing prediction."""
        from flaskr import cache
        
        mock_client = Mock()
        prediction_data = {"status": "ready", "result": {"score": 75}}
        mock_client.get.return_value = json.dumps(prediction_data)
        cache.valkey_client = mock_client
        
        result = cache.fetch_prediction("test-id-123")
        
        assert result == prediction_data
        mock_client.get.assert_called_once_with("prediction:test-id-123")
    
    def test_fetch_nonexistent_prediction(self):
        """Test fetching non-existent prediction."""
        from flaskr import cache
        
        mock_client = Mock()
        mock_client.get.return_value = None
        cache.valkey_client = mock_client
        
        result = cache.fetch_prediction("nonexistent-id")
        
        assert result is None
    
    def test_fetch_with_none_client(self):
        """Test fetching when client is None."""
        from flaskr import cache
        
        cache.valkey_client = None
        
        result = cache.fetch_prediction("test-id")
        
        assert result is None
    
    def test_fetch_with_exception(self):
        """Test handling exception during fetch."""
        from flaskr import cache
        
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Fetch failed")
        cache.valkey_client = mock_client
        
        result = cache.fetch_prediction("test-id")
        
        assert result is None


class TestUpdatePrediction:
    """Test update_prediction function."""
    
    def test_update_existing_prediction(self):
        """Test updating existing prediction."""
        from flaskr import cache
        
        mock_client = Mock()
        existing_data = {"status": "processing", "result": None}
        mock_client.get.return_value = json.dumps(existing_data)
        cache.valkey_client = mock_client
        
        update_data = {"status": "ready", "result": {"score": 75}}
        cache.update_prediction("test-id", update_data)
        
        # Should call setex with merged data
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        stored_data = json.loads(call_args[0][2])
        assert stored_data["status"] == "ready"
        assert stored_data["result"] == {"score": 75}
    
    def test_update_nonexistent_prediction(self):
        """Test updating non-existent prediction."""
        from flaskr import cache
        
        mock_client = Mock()
        mock_client.get.return_value = None
        cache.valkey_client = mock_client
        
        # Should handle gracefully
        cache.update_prediction("nonexistent-id", {"status": "ready"})
        
        # setex should not be called
        mock_client.setex.assert_not_called()
    
    def test_update_with_none_client(self):
        """Test updating when client is None."""
        from flaskr import cache
        
        cache.valkey_client = None
        
        # Should not raise exception
        cache.update_prediction("test-id", {"status": "ready"})
    
    def test_update_with_exception(self):
        """Test handling exception during update."""
        from flaskr import cache
        
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Update failed")
        cache.valkey_client = mock_client
        
        # Should not raise exception
        cache.update_prediction("test-id", {"status": "ready"})


class TestIsAvailable:
    """Test is_available function."""
    
    def test_available_when_client_exists(self):
        """Test is_available returns True when client exists."""
        from flaskr import cache
        
        cache.valkey_client = Mock()
        
        assert cache.is_available() is True
    
    def test_not_available_when_client_none(self):
        """Test is_available returns False when client is None."""
        from flaskr import cache
        
        cache.valkey_client = None
        
        assert cache.is_available() is False


class TestCacheIntegration:
    """Integration tests for cache functionality."""
    
    def test_store_and_fetch_cycle(self):
        """Test complete store and fetch cycle."""
        from flaskr import cache
        
        # Setup mock client with storage
        storage = {}
        mock_client = Mock()
        
        def mock_setex(key, ttl, value):
            storage[key] = value
        
        def mock_get(key):
            return storage.get(key)
        
        mock_client.setex = mock_setex
        mock_client.get = mock_get
        cache.valkey_client = mock_client
        
        # Store prediction
        prediction_id = "integration-test-id"
        prediction_data = {"status": "ready", "result": {"score": 85}}
        cache.store_prediction(prediction_id, prediction_data)
        
        # Fetch prediction
        result = cache.fetch_prediction(prediction_id)
        
        assert result == prediction_data
    
    def test_store_update_fetch_cycle(self):
        """Test store, update, and fetch cycle."""
        from flaskr import cache
        
        # Setup mock client with storage
        storage = {}
        mock_client = Mock()
        
        def mock_setex(key, ttl, value):
            storage[key] = value
        
        def mock_get(key):
            return storage.get(key)
        
        mock_client.setex = mock_setex
        mock_client.get = mock_get
        cache.valkey_client = mock_client
        
        # Store initial prediction
        prediction_id = "update-test-id"
        initial_data = {"status": "processing", "result": None}
        cache.store_prediction(prediction_id, initial_data)
        
        # Update prediction
        update_data = {"status": "ready", "result": {"score": 90}}
        cache.update_prediction(prediction_id, update_data)
        
        # Fetch updated prediction
        result = cache.fetch_prediction(prediction_id)
        
        assert result["status"] == "ready"
        assert result["result"] == {"score": 90}
