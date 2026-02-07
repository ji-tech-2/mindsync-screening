"""
Unit tests for caching functionality (weekly critical factors, weekly chart, daily suggestions)
"""
import pytest
import uuid
from datetime import datetime, timedelta
from flaskr import create_app
from flaskr.db import db, Predictions, PredDetails, WeeklyCriticalFactors, WeeklyChartData, DailySuggestions


@pytest.fixture
def app():
    """Create and configure a test app instance."""
    test_config = {
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'DB_DISABLED': False,
        'GEMINI_API_KEY': 'test_api_key_12345',
    }
    
    app = create_app(test_config)
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def sample_user_id():
    """Generate a sample user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_predictions(app, sample_user_id):
    """Create sample predictions for testing."""
    with app.app_context():
        predictions = []
        today = datetime.utcnow()
        
        # Create predictions for the last 7 days
        for i in range(7):
            pred_date = today - timedelta(days=i)
            pred = Predictions(
                user_id=uuid.UUID(sample_user_id),
                pred_date=pred_date,
                screen_time=5.0,
                work_screen=3.0,
                leisure_screen=2.0,
                sleep_hours=7.0,
                sleep_quality=3,
                stress_level=5.0,
                productivity=7.0,
                exercise=30,
                social=4.0,
                pred_score=75.0
            )
            db.session.add(pred)
            db.session.flush()
            
            # Add some prediction details
            detail1 = PredDetails(
                pred_id=pred.pred_id,
                factor_name="Sleep Quality",
                impact_score=2.5,
                factor_type='improvement'
            )
            detail2 = PredDetails(
                pred_id=pred.pred_id,
                factor_name="Exercise Duration",
                impact_score=1.8,
                factor_type='improvement'
            )
            db.session.add(detail1)
            db.session.add(detail2)
            predictions.append(pred)
        
        db.session.commit()
        return predictions


# ===================== #
#   MODEL TESTS         #
# ===================== #

class TestWeeklyCriticalFactorsModel:
    """Test WeeklyCriticalFactors database model."""
    
    def test_create_weekly_critical_factors(self, app, sample_user_id):
        """Test creating a weekly critical factors record."""
        with app.app_context():
            today = datetime.now().date()
            week_start = today - timedelta(days=6)
            
            record = WeeklyCriticalFactors(
                user_id=uuid.UUID(sample_user_id),
                week_start=week_start,
                week_end=today,
                days=7,
                total_predictions=10,
                top_factors=[
                    {"factor_name": "Sleep Quality", "count": 5, "avg_impact_score": 2.5},
                    {"factor_name": "Exercise", "count": 3, "avg_impact_score": 1.8}
                ],
                ai_advice={"description": "Test advice", "factors": {}}
            )
            
            db.session.add(record)
            db.session.commit()
            
            # Retrieve and verify
            retrieved = WeeklyCriticalFactors.query.first()
            assert retrieved is not None
            assert retrieved.user_id == uuid.UUID(sample_user_id)
            assert retrieved.week_start == week_start
            assert retrieved.total_predictions == 10
            assert len(retrieved.top_factors) == 2
    
    def test_weekly_critical_factors_to_dict(self, app, sample_user_id):
        """Test to_dict method of WeeklyCriticalFactors."""
        with app.app_context():
            today = datetime.now().date()
            week_start = today - timedelta(days=6)
            
            record = WeeklyCriticalFactors(
                user_id=uuid.UUID(sample_user_id),
                week_start=week_start,
                week_end=today,
                days=7,
                total_predictions=10,
                top_factors=[{"factor_name": "Sleep", "count": 5, "avg_impact_score": 2.5}],
                ai_advice={"description": "Test"}
            )
            
            result = record.to_dict()
            
            assert "period" in result
            assert result["period"]["days"] == 7
            assert "stats" in result
            assert result["stats"]["total_predictions"] == 10
            assert "top_critical_factors" in result
            assert "advice" in result


class TestWeeklyChartDataModel:
    """Test WeeklyChartData database model."""
    
    def test_create_weekly_chart_data(self, app, sample_user_id):
        """Test creating a weekly chart data record."""
        with app.app_context():
            today = datetime.now().date()
            week_start = today - timedelta(days=6)
            
            chart_data = [
                {"date": str(week_start), "mental_health_index": 75.0, "has_data": True},
                {"date": str(week_start + timedelta(days=1)), "mental_health_index": 80.0, "has_data": True}
            ]
            
            record = WeeklyChartData(
                user_id=uuid.UUID(sample_user_id),
                week_start=week_start,
                week_end=today,
                chart_data=chart_data
            )
            
            db.session.add(record)
            db.session.commit()
            
            retrieved = WeeklyChartData.query.first()
            assert retrieved is not None
            assert retrieved.user_id == uuid.UUID(sample_user_id)
            assert len(retrieved.chart_data) == 2
    
    def test_weekly_chart_data_to_dict(self, app, sample_user_id):
        """Test to_dict method of WeeklyChartData."""
        with app.app_context():
            today = datetime.now().date()
            week_start = today - timedelta(days=6)
            
            chart_data = [{"date": str(today), "mental_health_index": 75.0}]
            
            record = WeeklyChartData(
                user_id=uuid.UUID(sample_user_id),
                week_start=week_start,
                week_end=today,
                chart_data=chart_data
            )
            
            result = record.to_dict()
            
            assert "data" in result
            assert len(result["data"]) == 1


class TestDailySuggestionsModel:
    """Test DailySuggestions database model."""
    
    def test_create_daily_suggestions(self, app, sample_user_id):
        """Test creating a daily suggestions record."""
        with app.app_context():
            today = datetime.now().date()
            
            record = DailySuggestions(
                user_id=uuid.UUID(sample_user_id),
                date=today,
                prediction_count=3,
                top_factors=[
                    {"factor_name": "Sleep", "impact_score": 2.5},
                    {"factor_name": "Exercise", "impact_score": 1.8}
                ],
                ai_advice={"message": "Get more sleep"}
            )
            
            db.session.add(record)
            db.session.commit()
            
            retrieved = DailySuggestions.query.first()
            assert retrieved is not None
            assert retrieved.user_id == uuid.UUID(sample_user_id)
            assert retrieved.date == today
            assert retrieved.prediction_count == 3
            assert len(retrieved.top_factors) == 2
    
    def test_daily_suggestions_to_dict(self, app, sample_user_id):
        """Test to_dict method of DailySuggestions."""
        with app.app_context():
            today = datetime.now().date()
            
            record = DailySuggestions(
                user_id=uuid.UUID(sample_user_id),
                date=today,
                prediction_count=3,
                top_factors=[{"factor_name": "Sleep", "impact_score": 2.5}],
                ai_advice={"message": "Test"}
            )
            
            result = record.to_dict()
            
            assert "date" in result
            assert "user_id" in result
            assert "stats" in result
            assert result["stats"]["predictions_today"] == 3
            assert "areas_of_improvement" in result
            assert "suggestion" in result


# ===================== #
#   ENDPOINT TESTS      #
# ===================== #

class TestWeeklyCriticalFactorsEndpoint:
    """Test /weekly-critical-factors endpoint with caching."""
    
    def test_cache_miss_creates_new_record(self, client, sample_user_id, sample_predictions):
        """Test that cache miss calculates and stores new data."""
        response = client.get(f'/weekly-critical-factors?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data["status"] == "success"
        assert data["cached"] is False
        assert "top_critical_factors" in data
        assert "advice" in data
        
        # Verify record was created in database
        with client.application.app_context():
            cached = WeeklyCriticalFactors.query.filter_by(
                user_id=uuid.UUID(sample_user_id)
            ).first()
            assert cached is not None
    
    def test_cache_hit_returns_cached_data(self, client, sample_user_id, sample_predictions):
        """Test that cache hit returns cached data without recalculation."""
        # First request - cache miss
        response1 = client.get(f'/weekly-critical-factors?user_id={sample_user_id}')
        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1["cached"] is False
        
        # Second request - cache hit
        response2 = client.get(f'/weekly-critical-factors?user_id={sample_user_id}')
        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2["cached"] is True
        
        # Data should be identical
        assert data1["top_critical_factors"] == data2["top_critical_factors"]
    
    def test_invalid_user_id(self, client):
        """Test with invalid user_id format."""
        response = client.get('/weekly-critical-factors?user_id=invalid-uuid')
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_no_data_returns_empty_factors(self, client, sample_user_id):
        """Test endpoint with no predictions returns empty factors."""
        response = client.get(f'/weekly-critical-factors?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert len(data["top_critical_factors"]) == 0


class TestWeeklyChartEndpoint:
    """Test /chart/weekly endpoint with caching."""
    
    def test_cache_miss_creates_new_record(self, client, sample_user_id, sample_predictions):
        """Test that cache miss calculates and stores new data."""
        response = client.get(f'/chart/weekly?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data["status"] == "success"
        assert data["cached"] is False
        assert "data" in data
        assert len(data["data"]) == 7  # 7 days
        
        # Verify record was created in database
        with client.application.app_context():
            cached = WeeklyChartData.query.filter_by(
                user_id=uuid.UUID(sample_user_id)
            ).first()
            assert cached is not None
    
    def test_cache_hit_returns_cached_data(self, client, sample_user_id, sample_predictions):
        """Test that cache hit returns cached data without recalculation."""
        # First request - cache miss
        response1 = client.get(f'/chart/weekly?user_id={sample_user_id}')
        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1["cached"] is False
        
        # Second request - cache hit
        response2 = client.get(f'/chart/weekly?user_id={sample_user_id}')
        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2["cached"] is True
        
        # Data should be identical
        assert data1["data"] == data2["data"]
    
    def test_missing_user_id(self, client):
        """Test endpoint without user_id."""
        response = client.get('/chart/weekly')
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_chart_data_structure(self, client, sample_user_id, sample_predictions):
        """Test that chart data has correct structure."""
        response = client.get(f'/chart/weekly?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        chart_data = data["data"]
        assert len(chart_data) == 7
        
        # Check first day structure
        day = chart_data[0]
        assert "date" in day
        assert "label" in day
        assert "mental_health_index" in day
        assert "sleep_duration" in day
        assert "has_data" in day


class TestDailySuggestionEndpoint:
    """Test /daily-suggestion endpoint with caching."""
    
    def test_cache_miss_creates_new_record(self, client, sample_user_id, sample_predictions):
        """Test that cache miss calculates and stores new data."""
        response = client.get(f'/daily-suggestion?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data["status"] == "success"
        assert data["cached"] is False
        assert "suggestion" in data
        
        # Verify record was created in database
        with client.application.app_context():
            today = datetime.now().date()
            cached = DailySuggestions.query.filter_by(
                user_id=uuid.UUID(sample_user_id),
                date=today
            ).first()
            assert cached is not None
    
    def test_cache_hit_returns_cached_data(self, client, sample_user_id, sample_predictions):
        """Test that cache hit returns cached data without recalculation."""
        # First request - cache miss
        response1 = client.get(f'/daily-suggestion?user_id={sample_user_id}')
        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1["cached"] is False
        
        # Second request - cache hit
        response2 = client.get(f'/daily-suggestion?user_id={sample_user_id}')
        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2["cached"] is True
        
        # Data should be identical
        assert data1["suggestion"] == data2["suggestion"]
    
    def test_missing_user_id(self, client):
        """Test endpoint without user_id."""
        response = client.get('/daily-suggestion')
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "Missing user_id" in data["error"]
    
    def test_invalid_user_id(self, client):
        """Test with invalid user_id format."""
        response = client.get('/daily-suggestion?user_id=not-a-uuid')
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_no_predictions_today(self, client, sample_user_id):
        """Test endpoint when user has no predictions today."""
        response = client.get(f'/daily-suggestion?user_id={sample_user_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        # Should get a message about no check-ins yet
        assert "suggestion" in data


# ===================== #
#   CACHE INVALIDATION  #
# ===================== #

class TestCacheInvalidation:
    """Test that cache correctly handles time-based invalidation."""
    
    def test_different_weeks_create_separate_cache(self, app, sample_user_id):
        """Test that different week ranges create separate cache entries."""
        with app.app_context():
            today = datetime.now().date()
            
            # Create cache for this week
            cache1 = WeeklyCriticalFactors(
                user_id=uuid.UUID(sample_user_id),
                week_start=today - timedelta(days=6),
                week_end=today,
                days=7,
                total_predictions=5,
                top_factors=[],
                ai_advice={}
            )
            
            # Create cache for last week
            cache2 = WeeklyCriticalFactors(
                user_id=uuid.UUID(sample_user_id),
                week_start=today - timedelta(days=13),
                week_end=today - timedelta(days=7),
                days=7,
                total_predictions=3,
                top_factors=[],
                ai_advice={}
            )
            
            db.session.add(cache1)
            db.session.add(cache2)
            db.session.commit()
            
            # Query should return both
            all_caches = WeeklyCriticalFactors.query.filter_by(
                user_id=uuid.UUID(sample_user_id)
            ).all()
            
            assert len(all_caches) == 2
    
    def test_different_days_create_separate_cache(self, app, sample_user_id):
        """Test that different days create separate cache entries."""
        with app.app_context():
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # Create cache for today
            cache1 = DailySuggestions(
                user_id=uuid.UUID(sample_user_id),
                date=today,
                prediction_count=2,
                top_factors=[],
                ai_advice={}
            )
            
            # Create cache for yesterday
            cache2 = DailySuggestions(
                user_id=uuid.UUID(sample_user_id),
                date=yesterday,
                prediction_count=1,
                top_factors=[],
                ai_advice={}
            )
            
            db.session.add(cache1)
            db.session.add(cache2)
            db.session.commit()
            
            # Query should return both
            all_caches = DailySuggestions.query.filter_by(
                user_id=uuid.UUID(sample_user_id)
            ).all()
            
            assert len(all_caches) == 2
            
            # Query for specific day should return only one
            today_cache = DailySuggestions.query.filter_by(
                user_id=uuid.UUID(sample_user_id),
                date=today
            ).first()
            
            assert today_cache is not None
            assert today_cache.prediction_count == 2
