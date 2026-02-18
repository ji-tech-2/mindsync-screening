"""
Unit tests for predict.py endpoints and processing functions
"""

import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from flask import Flask


@pytest.fixture
def app():
    """Create a test Flask app."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["DB_DISABLED"] = False
    app.config["GEMINI_API_KEYS"] = "test-api-key"
    app.config["JWT_PUBLIC_KEY"] = "test-public-key"

    # Initialize extensions
    from flaskr import db as db_module

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db_module.db.init_app(app)

    with app.app_context():
        db_module.db.create_all()

    # Register blueprint
    from flaskr.predict import bp

    app.register_blueprint(bp)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


class TestPredictEndpoint:
    """Test /predict endpoint."""

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.threading.Thread")
    def test_predict_success(self, mock_thread, mock_cache, mock_model, client):
        """Test successful prediction request."""
        mock_model.model = MagicMock()
        mock_cache.is_available.return_value = True

        data = {
            "age": 25,
            "gender": "Male",
            "occupation": "Student",
            "sleep_hours": 7.0,
            "sleep_quality_1_5": 3,
            "stress_level_0_10": 5,
        }

        response = client.post("/predict", json=data)

        assert response.status_code == 202
        json_data = response.get_json()
        assert "prediction_id" in json_data
        assert json_data["status"] == "processing"
        mock_thread.assert_called_once()

    @patch("flaskr.predict.model")
    def test_predict_model_not_loaded(self, mock_model, client):
        """Test prediction when model is not loaded."""
        mock_model.model = None

        response = client.post("/predict", json={})

        assert response.status_code == 500
        json_data = response.get_json()
        assert "error" in json_data

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    def test_predict_no_storage_backend(self, mock_cache, mock_model, client, app):
        """Test prediction when no storage backend is available."""
        mock_model.model = MagicMock()
        mock_cache.is_available.return_value = False

        with app.app_context():
            app.config["DB_DISABLED"] = True

            response = client.post("/predict", json={})

            assert response.status_code == 503
            json_data = response.get_json()
            assert "No storage backend available" in json_data["error"]


class TestGetResultEndpoint:
    """Test /result/<prediction_id> endpoint."""

    @patch("flaskr.predict.cache")
    def test_get_result_processing(self, mock_cache, client):
        """Test getting result while still processing."""
        prediction_id = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = {
            "status": "processing",
            "created_at": datetime.now().isoformat(),
        }

        response = client.get(f"/result/{prediction_id}")

        assert response.status_code == 202
        json_data = response.get_json()
        assert json_data["status"] == "processing"

    @patch("flaskr.predict.cache")
    def test_get_result_ready(self, mock_cache, client):
        """Test getting completed result."""
        prediction_id = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = {
            "status": "ready",
            "result": {
                "prediction_score": 75.0,
                "health_level": "healthy",
            },
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        response = client.get(f"/result/{prediction_id}")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "ready"
        assert "result" in json_data

    def test_get_result_invalid_uuid(self, client):
        """Test with invalid UUID format."""
        response = client.get("/result/invalid-uuid")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "Invalid ID format" in json_data["error"]

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.read_from_db")
    def test_get_result_from_database(self, mock_read_db, mock_cache, client):
        """Test fallback to database when not in cache."""
        prediction_id = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = None
        mock_read_db.return_value = {
            "status": "success",
            "data": {
                "prediction_id": prediction_id,
                "prediction_score": 70.0,
                "prediction_date": datetime.now().isoformat(),
                "details": [],
            },
        }

        response = client.get(f"/result/{prediction_id}")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "ready"


class TestAdviceEndpoint:
    """Test /advice endpoint."""

    @patch("flaskr.predict.ai")
    def test_advice_success(self, mock_ai, client):
        """Test successful advice generation."""
        mock_ai.get_ai_advice.return_value = {
            "description": "Focus on sleep",
            "factors": {},
        }

        data = {
            "prediction_score": 65.0,
            "mental_health_category": "average",
            "wellness_analysis": {
                "areas_for_improvement": [],
                "strengths": [],
            },
        }

        response = client.post("/advice", json=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "success"
        assert "ai_advice" in json_data

    def test_advice_missing_inputs(self, client):
        """Test advice endpoint with missing required inputs."""
        response = client.post("/advice", json={})

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data


class TestStreakEndpoint:
    """Test /streak endpoint with JWT authentication."""

    def test_streak_unauthorized(self, client):
        """Test streak endpoint without JWT returns 401."""
        with patch("flaskr.predict.get_jwt_identity", return_value=None):
            response = client.get("/streak")
        assert response.status_code == 401

    @patch("flaskr.predict.Predictions")
    def test_streak_no_predictions(self, mock_predictions, client, app):
        """Test streak endpoint when no predictions exist."""
        user_id = str(uuid.uuid4())

        with app.app_context():
            mock_predictions.query.filter_by.return_value.all.return_value = []

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/streak")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"
            assert "daily" in json_data["data"]
            assert "weekly" in json_data["data"]
            assert "current_streak" in json_data["data"]
            # Daily should have 7 entries (Mon-Sun)
            assert len(json_data["data"]["daily"]) == 7
            # Weekly should have 7 entries (last 7 weeks)
            assert len(json_data["data"]["weekly"]) == 7
            # All should have has_screening = False
            assert all(not day["has_screening"] for day in json_data["data"]["daily"])
            assert all(
                not week["has_screening"] for week in json_data["data"]["weekly"]
            )
            # Current streaks should be 0
            assert json_data["data"]["current_streak"]["daily"] == 0
            assert json_data["data"]["current_streak"]["weekly"] == 0

    @patch("flaskr.predict.Predictions")
    def test_streak_daily_grace_one_day(self, mock_predictions, client, app):
        """If today not screened but yesterday screened, daily streak should persist."""
        user_id = str(uuid.uuid4())

        pred_yesterday = MagicMock()
        pred_yesterday.pred_date = datetime.now() - timedelta(days=1)

        pred_two_days_ago = MagicMock()
        pred_two_days_ago.pred_date = datetime.now() - timedelta(days=2)

        with app.app_context():
            mock_predictions.query.filter_by.return_value.all.return_value = [
                pred_yesterday,
                pred_two_days_ago,
            ]

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/streak")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"
            assert json_data["data"]["current_streak"]["daily"] == 2

    @patch("flaskr.predict.Predictions")
    def test_streak_weekly_grace_one_week(self, mock_predictions, client, app):
        """If current week empty but previous weeks have check-ins, weekly streak persists."""
        user_id = str(uuid.uuid4())

        pred_last_week = MagicMock()
        pred_last_week.pred_date = datetime.now() - timedelta(days=7)

        pred_two_weeks_ago = MagicMock()
        pred_two_weeks_ago.pred_date = datetime.now() - timedelta(days=14)

        with app.app_context():
            mock_predictions.query.filter_by.return_value.all.return_value = [
                pred_last_week,
                pred_two_weeks_ago,
            ]

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/streak")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"
            assert json_data["data"]["current_streak"]["weekly"] == 2

    def test_streak_db_disabled(self, client, app):
        """Test streak endpoint when database is disabled."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            app.config["DB_DISABLED"] = True

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/streak")

            assert response.status_code == 503


class TestHistoryEndpoint:
    """Test /history endpoint with JWT authentication."""

    def test_history_unauthorized(self, client):
        """Test history without JWT returns 401."""
        with patch("flaskr.predict.get_jwt_identity", return_value=None):
            response = client.get("/history")
        assert response.status_code == 401

    @patch("flaskr.predict.read_from_db")
    def test_history_success(self, mock_read_db, client):
        """Test successful history retrieval."""
        user_id = str(uuid.uuid4())
        mock_read_db.return_value = {
            "status": "success",
            "data": [
                {
                    "prediction_id": str(uuid.uuid4()),
                    "prediction_score": 75.0,
                    "prediction_date": datetime.now().isoformat(),
                    "details": [],
                }
            ],
        }

        with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
            response = client.get("/history")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "success"
        assert "data" in json_data

    @patch("flaskr.predict.read_from_db")
    def test_history_no_predictions(self, mock_read_db, client):
        """Test history when no predictions exist."""
        user_id = str(uuid.uuid4())
        mock_read_db.return_value = {
            "status": "not_found",
        }

        with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
            response = client.get("/history")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["count"] == 0


class TestWeeklyCriticalFactorsEndpoint:
    """Test /weekly-critical-factors endpoint with JWT authentication."""

    def test_weekly_critical_factors_unauthorized(self, client):
        """Test endpoint without JWT returns 401."""
        with patch("flaskr.predict.get_jwt_identity", return_value=None):
            response = client.get("/weekly-critical-factors")
        assert response.status_code == 401

    def test_weekly_critical_factors_db_disabled(self, client, app):
        """Test endpoint when database is disabled."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            app.config["DB_DISABLED"] = True

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/weekly-critical-factors")

            assert response.status_code == 503

    @patch("flaskr.predict.WeeklyCriticalFactors")
    @patch("flaskr.predict.db")
    @patch("flaskr.predict.ai")
    def test_weekly_critical_factors_cached(
        self, mock_ai, mock_db, mock_weekly, client, app
    ):
        """Test returning cached weekly critical factors."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            # Mock cached data
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {
                "top_critical_factors": [],
                "advice": {},
            }
            mock_weekly.query.filter_by.return_value.first.return_value = cached_mock

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/weekly-critical-factors")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["cached"] is True

    @patch("flaskr.predict.WeeklyCriticalFactors")
    @patch("flaskr.predict.db")
    @patch("flaskr.predict.ai")
    @patch("flaskr.predict.Predictions")
    @patch("flaskr.predict.PredDetails")
    def test_weekly_critical_factors_fresh_calculation(
        self,
        mock_pred_details,
        mock_predictions,
        mock_ai,
        mock_db,
        mock_weekly,
        client,
        app,
    ):
        """Test fresh calculation of weekly critical factors."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            # No cached data
            mock_weekly.query.filter_by.return_value.first.return_value = None

            # Configure pred_date to support comparison operators (needed for filter)
            pred_date_mock = MagicMock()
            pred_date_mock.__ge__ = MagicMock(return_value=MagicMock())
            pred_date_mock.__le__ = MagicMock(return_value=MagicMock())
            mock_predictions.pred_date = pred_date_mock

            # Configure user_id for comparison
            user_id_mock = MagicMock()
            user_id_mock.__eq__ = MagicMock(return_value=MagicMock())
            mock_predictions.user_id = user_id_mock

            # Configure pred_id for comparison
            pred_id_mock = MagicMock()
            pred_id_mock.__eq__ = MagicMock(return_value=MagicMock())
            mock_predictions.pred_id = pred_id_mock
            mock_pred_details.pred_id = pred_id_mock

            # Configure impact_score for comparison
            impact_score_mock = MagicMock()
            impact_score_mock.__gt__ = MagicMock(return_value=MagicMock())
            mock_pred_details.impact_score = impact_score_mock

            # Mock query results for the main query
            mock_db.session.query.return_value.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
                MagicMock(factor_name="Sleep", occurrence_count=5, avg_impact_score=2.5)
            ]

            # Mock the total_predictions query with chained filters
            filter_mock = MagicMock()
            filter_mock.filter.return_value = filter_mock  # Allow chaining
            filter_mock.scalar.return_value = 10
            mock_db.session.query.return_value.filter.return_value = filter_mock

            mock_ai.get_weekly_advice.return_value = {
                "description": "Weekly advice",
                "factors": {},
            }

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/weekly-critical-factors?days=7")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"


class TestDailySuggestionEndpoint:
    """Test /daily-suggestion endpoint with JWT authentication."""

    def test_daily_suggestion_unauthorized(self, client):
        """Test endpoint without JWT returns 401."""
        with patch("flaskr.predict.get_jwt_identity", return_value=None):
            response = client.get("/daily-suggestion")
        assert response.status_code == 401

    @patch("flaskr.predict.DailySuggestions")
    def test_daily_suggestion_cached(self, mock_daily, client, app):
        """Test returning cached daily suggestion."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {
                "suggestion": {"message": "Great job!"},
                "areas_of_improvement": [],
            }
            mock_daily.query.filter_by.return_value.first.return_value = cached_mock

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/daily-suggestion")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["cached"] is True


class TestWeeklyChartEndpoint:
    """Test /weekly-chart-data endpoint with JWT authentication."""

    def test_weekly_chart_unauthorized(self, client):
        """Test endpoint without JWT returns 401."""
        with patch("flaskr.predict.get_jwt_identity", return_value=None):
            response = client.get("/weekly-chart-data")
        assert response.status_code == 401

    @patch("flaskr.predict.WeeklyChartData")
    def test_weekly_chart_cached(self, mock_chart, client, app):
        """Test returning cached weekly chart data."""
        user_id = str(uuid.uuid4())
        with app.app_context():
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {"data": []}
            mock_chart.query.filter_by.return_value.first.return_value = cached_mock

            with patch("flaskr.predict.get_jwt_identity", return_value=user_id):
                response = client.get("/weekly-chart-data")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["cached"] is True
