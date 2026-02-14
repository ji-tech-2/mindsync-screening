"""
Unit tests for predict.py endpoints and processing functions
"""

import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from flask import Flask


@pytest.fixture
def app():
    """Create a test Flask app."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["DB_DISABLED"] = False
    app.config["GEMINI_API_KEY"] = "test-api-key"

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
    """Test /streak/<user_id> endpoint."""

    def test_streak_invalid_user_id(self, client):
        """Test streak endpoint with invalid user_id."""
        response = client.get("/streak/invalid-uuid")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "Invalid user_id format" in json_data["error"]

    @patch("flaskr.predict.UserStreaks")
    def test_streak_no_record(self, mock_streaks, client, app):
        """Test streak endpoint when no record exists."""
        user_id = str(uuid.uuid4())

        with app.app_context():
            mock_streaks.query.get.return_value = None

            response = client.get(f"/streak/{user_id}")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"
            assert json_data["data"]["daily"]["current"] == 0

    def test_streak_db_disabled(self, client, app):
        """Test streak endpoint when database is disabled."""
        with app.app_context():
            app.config["DB_DISABLED"] = True

            response = client.get(f"/streak/{str(uuid.uuid4())}")

            assert response.status_code == 503


class TestHistoryEndpoint:
    """Test /history/<user_id> endpoint."""

    def test_history_invalid_user_id(self, client):
        """Test history with invalid user_id."""
        response = client.get("/history/invalid-uuid")

        assert response.status_code == 400

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

        response = client.get(f"/history/{user_id}")

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

        response = client.get(f"/history/{user_id}")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["count"] == 0


class TestWeeklyCriticalFactorsEndpoint:
    """Test /weekly-critical-factors endpoint."""

    def test_weekly_critical_factors_db_disabled(self, client, app):
        """Test endpoint when database is disabled."""
        with app.app_context():
            app.config["DB_DISABLED"] = True

            response = client.get("/weekly-critical-factors")

            assert response.status_code == 503

    def test_weekly_critical_factors_invalid_user_id(self, client):
        """Test with invalid user_id parameter."""
        response = client.get("/weekly-critical-factors?user_id=invalid")

        assert response.status_code == 400

    @patch("flaskr.predict.WeeklyCriticalFactors")
    @patch("flaskr.predict.db")
    @patch("flaskr.predict.ai")
    def test_weekly_critical_factors_cached(
        self, mock_ai, mock_db, mock_weekly, client, app
    ):
        """Test returning cached weekly critical factors."""
        with app.app_context():
            # Mock cached data
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {
                "top_critical_factors": [],
                "advice": {},
            }
            mock_weekly.query.filter_by.return_value.first.return_value = cached_mock

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
        with app.app_context():
            # No cached data
            mock_weekly.query.filter_by.return_value.first.return_value = None

            # Mock query results
            mock_db.session.query.return_value.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
                MagicMock(factor_name="Sleep", occurrence_count=5, avg_impact_score=2.5)
            ]

            mock_db.session.query.return_value.filter.return_value.scalar.return_value = (
                10
            )

            mock_ai.get_weekly_advice.return_value = {
                "description": "Weekly advice",
                "factors": {},
            }

            response = client.get("/weekly-critical-factors?days=7")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["status"] == "success"


class TestDailySuggestionEndpoint:
    """Test /daily-suggestion endpoint."""

    def test_daily_suggestion_missing_user_id(self, client):
        """Test without user_id parameter."""
        response = client.get("/daily-suggestion")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "Missing user_id" in json_data["error"]

    def test_daily_suggestion_invalid_user_id(self, client):
        """Test with invalid user_id."""
        response = client.get("/daily-suggestion?user_id=invalid")

        assert response.status_code == 400

    @patch("flaskr.predict.DailySuggestions")
    def test_daily_suggestion_cached(self, mock_daily, client, app):
        """Test returning cached daily suggestion."""
        with app.app_context():
            user_id = str(uuid.uuid4())
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {
                "suggestion": {"message": "Great job!"},
                "areas_of_improvement": [],
            }
            mock_daily.query.filter_by.return_value.first.return_value = cached_mock

            response = client.get(f"/daily-suggestion?user_id={user_id}")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["cached"] is True


class TestWeeklyChartEndpoint:
    """Test /chart/weekly endpoint."""

    def test_weekly_chart_missing_user_id(self, client):
        """Test without user_id parameter."""
        response = client.get("/chart/weekly")

        assert response.status_code == 400

    def test_weekly_chart_invalid_user_id(self, client):
        """Test with invalid user_id."""
        response = client.get("/chart/weekly?user_id=invalid")

        assert response.status_code == 400

    @patch("flaskr.predict.WeeklyChartData")
    def test_weekly_chart_cached(self, mock_chart, client, app):
        """Test returning cached weekly chart data."""
        with app.app_context():
            user_id = str(uuid.uuid4())
            cached_mock = MagicMock()
            cached_mock.to_dict.return_value = {"data": []}
            mock_chart.query.filter_by.return_value.first.return_value = cached_mock

            response = client.get(f"/chart/weekly?user_id={user_id}")

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["cached"] is True
