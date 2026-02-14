"""
Additional tests for predict.py to improve coverage.
Targets: result partial/error statuses, advice from prediction list,
streak exceptions, history DB disabled/errors, process_prediction branches,
save_to_db streak failure, weekly chart fresh calculation, daily suggestion branches.
"""

import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from flask import Flask


@pytest.fixture
def app():
    """Create a test Flask app with in-memory SQLite."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["DB_DISABLED"] = False
    app.config["GEMINI_API_KEY"] = "test-api-key"

    from flaskr import db as db_module

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db_module.db.init_app(app)

    with app.app_context():
        db_module.db.create_all()

    from flaskr.predict import bp

    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ─────────────────────────────────────────────
#  /result — partial and error statuses
# ─────────────────────────────────────────────


class TestGetResultPartialStatus:
    """Test the partial status branch in /result."""

    @patch("flaskr.predict.cache")
    def test_get_result_partial(self, mock_cache, client):
        pid = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = {
            "status": "partial",
            "result": {"prediction_score": 60.0, "health_level": "average"},
            "created_at": datetime.now().isoformat(),
        }

        response = client.get(f"/result/{pid}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "partial"
        assert "result" in data
        assert "Prediction ready" in data["message"]


class TestGetResultErrorStatus:
    """Test the error status branch in /result."""

    @patch("flaskr.predict.cache")
    def test_get_result_error(self, mock_cache, client):
        pid = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = {
            "status": "error",
            "error": "Model failure",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        response = client.get(f"/result/{pid}")
        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"
        assert data["error"] == "Model failure"


class TestGetResultDBDisabledFallback:
    """Test result endpoint when cache miss and DB disabled."""

    @patch("flaskr.predict.cache")
    def test_get_result_db_disabled(self, mock_cache, client, app):
        pid = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = None

        with app.app_context():
            app.config["DB_DISABLED"] = True
            response = client.get(f"/result/{pid}")

        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "not_found"


class TestGetResultNotFoundAnywhere:
    """Test result not found in cache or DB."""

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.read_from_db")
    def test_get_result_not_found(self, mock_read_db, mock_cache, client):
        pid = str(uuid.uuid4())
        mock_cache.fetch_prediction.return_value = None
        mock_read_db.return_value = {"status": "not_found", "error": "Not found"}

        response = client.get(f"/result/{pid}")
        assert response.status_code == 404


# ─────────────────────────────────────────────
#  /advice — prediction from list
# ─────────────────────────────────────────────


class TestAdvicePredictionFromList:
    """Test advice endpoint extracting score from prediction list."""

    @patch("flaskr.predict.ai")
    def test_advice_with_prediction_list(self, mock_ai, client):
        mock_ai.get_ai_advice.return_value = {"description": "Advice", "factors": {}}

        data = {
            "prediction": [72.5],
            "mental_health_category": "healthy",
            "wellness_analysis": {
                "areas_for_improvement": [],
                "strengths": [],
            },
        }
        response = client.post("/advice", json=data)
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"

    @patch("flaskr.predict.ai")
    def test_advice_with_prediction_scalar(self, mock_ai, client):
        """prediction is a scalar, not a list."""
        mock_ai.get_ai_advice.return_value = {"description": "Advice", "factors": {}}

        data = {
            "prediction": 72.5,
            "mental_health_category": "healthy",
            "wellness_analysis": {"areas_for_improvement": [], "strengths": []},
        }
        response = client.post("/advice", json=data)
        assert response.status_code == 200

    def test_advice_exception(self, client):
        """Trigger exception in advice endpoint."""
        # Sending None as json will cause an error
        response = client.post(
            "/advice", data="not-json", content_type="application/json"
        )
        assert response.status_code == 400


# ─────────────────────────────────────────────
#  /streak — exception handling
# ─────────────────────────────────────────────


class TestStreakException:
    """Test /streak exception handling."""

    @patch("flaskr.predict.UserStreaks")
    def test_streak_db_exception(self, mock_streaks, client, app):
        user_id = str(uuid.uuid4())
        mock_streaks.query.get.side_effect = Exception("DB error")

        with app.app_context():
            response = client.get(f"/streak/{user_id}")

        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"

    @patch("flaskr.predict.UserStreaks")
    def test_streak_existing_record(self, mock_streaks, client, app):
        """Test with an existing streak record."""
        user_id = str(uuid.uuid4())
        mock_record = MagicMock()
        mock_record.to_dict.return_value = {
            "user_id": user_id,
            "daily": {"current": 5, "last_date": "2026-02-13"},
            "weekly": {"current": 2, "last_date": "2026-02-13"},
        }
        mock_streaks.query.get.return_value = mock_record

        with app.app_context():
            response = client.get(f"/streak/{user_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["data"]["daily"]["current"] == 5


# ─────────────────────────────────────────────
#  /history — DB disabled, error branch
# ─────────────────────────────────────────────


class TestHistoryDBDisabled:
    """Test /history when DB is disabled."""

    def test_history_db_disabled(self, client, app):
        uid = str(uuid.uuid4())
        with app.app_context():
            app.config["DB_DISABLED"] = True
            response = client.get(f"/history/{uid}")

        assert response.status_code == 503
        data = response.get_json()
        assert "Database is disabled" in data["message"]


class TestHistoryException:
    """Test /history exception paths."""

    @patch("flaskr.predict.read_from_db")
    def test_history_read_error(self, mock_read, client):
        uid = str(uuid.uuid4())
        mock_read.return_value = {"status": "error", "error": "DB error"}

        response = client.get(f"/history/{uid}")
        assert response.status_code == 400

    @patch("flaskr.predict.read_from_db")
    def test_history_exception(self, mock_read, client):
        uid = str(uuid.uuid4())
        mock_read.side_effect = Exception("Unexpected")

        response = client.get(f"/history/{uid}")
        assert response.status_code == 500


# ─────────────────────────────────────────────
#  process_prediction — error/fallback branches
# ─────────────────────────────────────────────


class TestProcessPredictionBranches:
    """Cover more branches in process_prediction."""

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.model")
    @patch("flaskr.predict.ai")
    def test_process_prediction_ai_advice_none(
        self, mock_ai, mock_model, mock_cache, app
    ):
        """When AI returns None, fallback advice is used."""
        from flaskr.predict import process_prediction

        mock_model.model = MagicMock()
        mock_model.model.predict.return_value = [65.0]
        mock_model.analyze_wellness_factors.return_value = {
            "areas_for_improvement": [],
            "strengths": [],
        }
        mock_model.categorize_mental_health_score.return_value = "average"
        mock_ai.get_ai_advice.return_value = None  # AI fails

        with app.app_context():
            app.config["DB_DISABLED"] = True
            process_prediction(
                str(uuid.uuid4()),
                {"age": 25, "gender": "Male"},
                datetime.now().isoformat(),
                app,
            )

        # Should have updated with "ready" status despite AI failure
        assert mock_cache.update_prediction.called
        last_call = mock_cache.update_prediction.call_args[0][1]
        assert last_call["status"] == "ready"
        assert (
            "AI advice could not be generated"
            in last_call["result"]["advice"]["description"]
        )

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.model")
    def test_process_prediction_model_exception(self, mock_model, mock_cache, app):
        """When model.predict raises, error status is stored."""
        from flaskr.predict import process_prediction

        mock_model.model = MagicMock()
        mock_model.model.predict.side_effect = Exception("Bad input")

        with app.app_context():
            process_prediction(
                str(uuid.uuid4()),
                {"age": 25},
                datetime.now().isoformat(),
                app,
            )

        assert mock_cache.update_prediction.called
        last_call = mock_cache.update_prediction.call_args[0][1]
        assert last_call["status"] == "error"

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.model")
    @patch("flaskr.predict.ai")
    @patch("flaskr.predict.save_to_db")
    def test_process_prediction_db_save_fails(
        self, mock_save, mock_ai, mock_model, mock_cache, app
    ):
        """When DB save raises, prediction still completes."""
        from flaskr.predict import process_prediction

        mock_model.model = MagicMock()
        mock_model.model.predict.return_value = [70.0]
        mock_model.analyze_wellness_factors.return_value = {
            "areas_for_improvement": [],
            "strengths": [],
        }
        mock_model.categorize_mental_health_score.return_value = "healthy"
        mock_ai.get_ai_advice.return_value = {"description": "ok", "factors": {}}
        mock_save.side_effect = Exception("DB down")

        with app.app_context():
            app.config["DB_DISABLED"] = False
            process_prediction(
                str(uuid.uuid4()),
                {"age": 25},
                datetime.now().isoformat(),
                app,
            )

        # Should still update cache with "ready"
        last_call = mock_cache.update_prediction.call_args[0][1]
        assert last_call["status"] == "ready"

    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.model")
    @patch("flaskr.predict.ai")
    def test_process_prediction_with_list_input(
        self, mock_ai, mock_model, mock_cache, app
    ):
        """When json_input is a list instead of dict."""
        from flaskr.predict import process_prediction

        mock_model.model = MagicMock()
        mock_model.model.predict.return_value = [60.0]
        mock_model.analyze_wellness_factors.return_value = {
            "areas_for_improvement": [],
            "strengths": [],
        }
        mock_model.categorize_mental_health_score.return_value = "average"
        mock_ai.get_ai_advice.return_value = {"description": "ok", "factors": {}}

        with app.app_context():
            app.config["DB_DISABLED"] = True
            process_prediction(
                str(uuid.uuid4()),
                [{"age": 25, "gender": "Male"}],
                None,  # test created_at=None fallback
                app,
            )

        last_call = mock_cache.update_prediction.call_args[0][1]
        assert last_call["status"] == "ready"


# ─────────────────────────────────────────────
#  save_to_db — streak failure branch
# ─────────────────────────────────────────────


class TestSaveToDbStreakFailure:
    """Cover streak update failure in save_to_db."""

    @patch("flaskr.predict._update_user_streaks")
    @patch("flaskr.predict._parse_current_date")
    @patch("flaskr.predict.db")
    def test_save_to_db_streak_exception(self, mock_db, mock_parse, mock_update, app):
        from flaskr.predict import save_to_db

        mock_parse.return_value = datetime.now().date()
        mock_update.side_effect = Exception("streak DB error")

        with app.app_context():
            app.config["DB_DISABLED"] = False
            # Should not raise even though streak update fails
            save_to_db(
                str(uuid.uuid4()),
                {
                    "user_id": str(uuid.uuid4()),
                    "work_screen_hours": 4,
                    "leisure_screen_hours": 2,
                    "sleep_hours": 7,
                    "stress_level_0_10": 5,
                    "productivity_0_100": 60,
                    "social_hours_per_week": 10,
                    "sleep_quality_1_5": 3,
                    "exercise_minutes_per_week": 150,
                    "screen_time_hours": 6,
                },
                70.0,
                {"areas_for_improvement": [], "strengths": []},
                {"description": "ok", "factors": {}},
            )


# ─────────────────────────────────────────────
#  /weekly-critical-factors — more branches
# ─────────────────────────────────────────────


class TestWeeklyCriticalFactorsNoBranches:
    """Cover exception branch in weekly-critical-factors."""

    @patch("flaskr.predict.WeeklyCriticalFactors")
    @patch("flaskr.predict.db")
    def test_exception_rollback(self, mock_db, mock_wcf, client, app):
        """Exception triggers rollback."""
        with app.app_context():
            # Raise before any Predictions comparison happens
            mock_wcf.query.filter_by.side_effect = Exception("boom")

            response = client.get("/weekly-critical-factors")

        assert response.status_code == 500
        mock_db.session.rollback.assert_called()


# ─────────────────────────────────────────────
#  /daily-suggestion — more branches
# ─────────────────────────────────────────────


class TestDailySuggestionBranches:
    """Cover additional daily suggestion branches."""

    @patch("flaskr.predict.DailySuggestions")
    @patch("flaskr.predict.db")
    def test_daily_suggestion_exception(self, mock_db, mock_ds, client, app):
        """Exception triggers rollback."""
        with app.app_context():
            uid = str(uuid.uuid4())
            mock_ds.query.filter_by.side_effect = Exception("boom")
            response = client.get(f"/daily-suggestion?user_id={uid}")

        assert response.status_code == 500
        mock_db.session.rollback.assert_called()


class TestDailySuggestionDBDisabled:
    """Test daily-suggestion when DB disabled."""

    def test_db_disabled(self, client, app):
        with app.app_context():
            app.config["DB_DISABLED"] = True
            uid = str(uuid.uuid4())
            response = client.get(f"/daily-suggestion?user_id={uid}")
        assert response.status_code == 503


# ─────────────────────────────────────────────
#  /chart/weekly — fresh calculation and errors
# ─────────────────────────────────────────────


class TestWeeklyChartFresh:
    """Cover chart error/exception paths."""

    @patch("flaskr.predict.WeeklyChartData")
    @patch("flaskr.predict.db")
    def test_weekly_chart_exception(self, mock_db, mock_chart, client, app):
        """Exception in chart endpoint triggers error response."""
        with app.app_context():
            uid = str(uuid.uuid4())
            mock_chart.query.filter_by.side_effect = Exception("boom")
            response = client.get(f"/chart/weekly?user_id={uid}")

        assert response.status_code == 500

    def test_weekly_chart_db_disabled(self, client, app):
        with app.app_context():
            app.config["DB_DISABLED"] = True
            uid = str(uuid.uuid4())
            response = client.get(f"/chart/weekly?user_id={uid}")
        assert response.status_code == 503


# ─────────────────────────────────────────────
#  /predict — exception branch
# ─────────────────────────────────────────────


class TestPredictException:
    """Cover the exception branch in /predict endpoint."""

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    def test_predict_bad_json(self, mock_cache, mock_model, client):
        """When request body causes an exception."""
        mock_model.model = MagicMock()
        mock_cache.is_available.return_value = True

        response = client.post(
            "/predict", data="not-json", content_type="application/json"
        )
        assert response.status_code == 400


# ─────────────────────────────────────────────
#  read_from_db — additional branches
# ─────────────────────────────────────────────


class TestReadFromDbBranches:
    """Cover more branches in read_from_db."""

    def test_read_from_db_db_disabled(self, app):
        from flaskr.predict import read_from_db

        with app.app_context():
            app.config["DB_DISABLED"] = True
            result = read_from_db(prediction_id=str(uuid.uuid4()))
        assert result["status"] == "disabled"

    def test_read_from_db_invalid_prediction_uuid(self, app):
        from flaskr.predict import read_from_db

        with app.app_context():
            result = read_from_db(prediction_id="not-a-uuid")
        assert result["status"] == "bad_request"

    def test_read_from_db_invalid_user_uuid(self, app):
        from flaskr.predict import read_from_db

        with app.app_context():
            result = read_from_db(user_id="not-a-uuid")
        assert result["status"] == "bad_request"

    def test_read_from_db_no_params(self, app):
        from flaskr.predict import read_from_db

        with app.app_context():
            result = read_from_db()
        assert result["status"] == "bad_request"

    @patch("flaskr.predict.db")
    def test_read_from_db_exception(self, mock_db, app):
        from flaskr.predict import read_from_db

        mock_db.select.side_effect = Exception("query failed")

        with app.app_context():
            result = read_from_db(prediction_id=str(uuid.uuid4()))
        assert result["status"] == "error"
