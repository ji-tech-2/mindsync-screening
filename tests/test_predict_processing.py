"""
Unit tests for predict.py processing and database functions
"""

import time
import uuid
import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from flask import Flask


@pytest.fixture
def app_context():
    """Create a test Flask app context."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["DB_DISABLED"] = False
    app.config["GEMINI_API_KEY"] = "test-key"

    from flaskr import db as db_module

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    db_module.db.init_app(app)

    with app.app_context():
        db_module.db.create_all()
        yield app


class TestProcessPrediction:
    """Test process_prediction background task."""

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.ai")
    @patch("flaskr.predict.save_to_db")
    def test_process_prediction_success(
        self, mock_save_db, mock_ai, mock_cache, mock_model, app_context
    ):
        """Test successful prediction processing."""
        from flaskr.predict import process_prediction

        prediction_id = str(uuid.uuid4())
        json_input = {
            "age": 25,
            "gender": "Male",
            "occupation": "Student",
            "sleep_hours": 7.0,
            "sleep_quality_1_5": 3,
            "stress_level_0_10": 5,
            "exercise_minutes_per_week": 150,
            "social_hours_per_week": 10,
            "productivity_0_100": 75,
            "work_screen_hours": 4,
            "leisure_screen_hours": 2,
            "work_mode": "Hybrid",
        }

        # Mock model predictions
        mock_model.model.predict.return_value = [75.5]
        mock_model.analyze_wellness_factors.return_value = {
            "areas_for_improvement": [{"feature": "Sleep", "impact_score": 2.5}],
            "strengths": [{"feature": "Exercise", "impact_score": -1.5}],
        }
        mock_model.categorize_mental_health_score.return_value = "healthy"

        # Mock AI advice
        mock_ai.get_ai_advice.return_value = {
            "description": "You're doing well",
            "factors": {
                "Sleep": {
                    "advices": ["Get 8 hours of sleep"],
                    "references": ["https://sleep.org"],
                }
            },
        }

        test_user_id = str(uuid.uuid4())
        process_prediction(
            prediction_id,
            json_input,
            datetime.now().isoformat(),
            time.time(),
            app_context,
            test_user_id,
            None,
        )

        # Verify partial result was stored
        assert mock_cache.store_prediction.called
        # Verify final result was updated
        assert mock_cache.update_prediction.called
        # Verify database save was attempted
        assert mock_save_db.called

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    @patch("flaskr.predict.ai")
    def test_process_prediction_wellness_analysis_failure(
        self, mock_ai, mock_cache, mock_model, app_context
    ):
        """Test prediction processing when wellness analysis fails."""
        from flaskr.predict import process_prediction

        prediction_id = str(uuid.uuid4())
        json_input = {"age": 25}

        mock_model.model.predict.return_value = [75.5]
        mock_model.analyze_wellness_factors.return_value = None  # Failure
        mock_model.categorize_mental_health_score.return_value = "healthy"
        mock_ai.get_ai_advice.return_value = {
            "description": "AI advice",
            "factors": {},
        }

        test_user_id = str(uuid.uuid4())
        process_prediction(
            prediction_id,
            json_input,
            datetime.now().isoformat(),
            time.time(),
            app_context,
            test_user_id,
            None,
        )

        # Should still complete with fallback empty analysis
        assert mock_cache.update_prediction.called

    @patch("flaskr.predict.model")
    @patch("flaskr.predict.cache")
    def test_process_prediction_exception_handling(
        self, mock_cache, mock_model, app_context
    ):
        """Test error handling in prediction processing."""
        from flaskr.predict import process_prediction

        prediction_id = str(uuid.uuid4())
        json_input = {"age": 25}

        # Make model predict raise an exception
        mock_model.model.predict.side_effect = Exception("Model error")

        test_user_id = str(uuid.uuid4())
        process_prediction(
            prediction_id,
            json_input,
            datetime.now().isoformat(),
            time.time(),
            app_context,
            test_user_id,
            None,
        )

        # Should update cache with error status
        error_call = [
            call
            for call in mock_cache.update_prediction.call_args_list
            if "error" in str(call)
        ]
        assert len(error_call) > 0


class TestSaveDetailRecords:
    """Test _save_detail_records helper function."""

    @patch("flaskr.predict.db")
    @patch("flaskr.predict.PredDetails")
    @patch("flaskr.predict.Advices")
    @patch("flaskr.predict.References")
    def test_save_detail_records_improvement(
        self, mock_refs, mock_advices, mock_details, mock_db
    ):
        """Test saving improvement factor details."""
        from flaskr.predict import _save_detail_records

        pred_id = uuid.uuid4()
        items = [{"feature": "Sleep", "impact_score": 2.5}]
        ai_advice = {
            "factors": {
                "Sleep": {
                    "advices": ["Get more sleep", "Sleep earlier"],
                    "references": ["https://sleep.org"],
                }
            }
        }

        _save_detail_records(pred_id, items, "improvement", ai_advice)

        # Should create detail record
        mock_details.assert_called_once()
        # Should add advice and references
        assert mock_advices.call_count == 2
        assert mock_refs.call_count == 1

    @patch("flaskr.predict.db")
    @patch("flaskr.predict.PredDetails")
    def test_save_detail_records_strengths(self, mock_details, mock_db):
        """Test saving strength factor details (no advice)."""
        from flaskr.predict import _save_detail_records

        pred_id = uuid.uuid4()
        items = [{"feature": "Exercise", "impact_score": -1.5}]
        ai_advice = {"factors": {}}

        _save_detail_records(pred_id, items, "strengths", ai_advice)

        # Should create detail record but no advice/references
        mock_details.assert_called_once()

    def test_save_detail_records_empty_items(self):
        """Test with empty items list."""
        from flaskr.predict import _save_detail_records

        pred_id = uuid.uuid4()

        # Should not raise any errors
        _save_detail_records(pred_id, [], "improvement", {})


class TestUpdateDailyStreak:
    """Test _update_daily_streak helper function."""

    def test_update_daily_streak_first_checkin(self):
        """Test daily streak on first check-in."""
        from flaskr.predict import _update_daily_streak

        streak_record = Mock()
        streak_record.last_daily_date = None
        current_date = date(2026, 2, 14)

        _update_daily_streak(streak_record, current_date)

        assert streak_record.curr_daily_streak == 1
        assert streak_record.last_daily_date == current_date

    def test_update_daily_streak_consecutive_day(self):
        """Test daily streak on consecutive day."""
        from flaskr.predict import _update_daily_streak

        streak_record = Mock()
        streak_record.last_daily_date = date(2026, 2, 13)
        streak_record.curr_daily_streak = 5
        current_date = date(2026, 2, 14)

        _update_daily_streak(streak_record, current_date)

        assert streak_record.curr_daily_streak == 6
        assert streak_record.last_daily_date == current_date

    def test_update_daily_streak_same_day(self):
        """Test daily streak when checking in same day."""
        from flaskr.predict import _update_daily_streak

        streak_record = Mock()
        current_date = date(2026, 2, 14)
        streak_record.last_daily_date = current_date
        streak_record.curr_daily_streak = 5

        _update_daily_streak(streak_record, current_date)

        # Should not change
        assert streak_record.curr_daily_streak == 5

    def test_update_daily_streak_broken(self):
        """Test daily streak reset when broken."""
        from flaskr.predict import _update_daily_streak

        streak_record = Mock()
        streak_record.last_daily_date = date(2026, 2, 10)
        streak_record.curr_daily_streak = 10
        current_date = date(2026, 2, 14)

        _update_daily_streak(streak_record, current_date)

        assert streak_record.curr_daily_streak == 1
        assert streak_record.last_daily_date == current_date


class TestUpdateWeeklyStreak:
    """Test _update_weekly_streak helper function."""

    def test_update_weekly_streak_first_week(self):
        """Test weekly streak on first check-in."""
        from flaskr.predict import _update_weekly_streak

        streak_record = Mock()
        streak_record.last_weekly_date = None
        current_date = date(2026, 2, 14)

        _update_weekly_streak(streak_record, current_date)

        assert streak_record.curr_weekly_streak == 1
        assert streak_record.last_weekly_date == current_date

    def test_update_weekly_streak_same_week(self):
        """Test weekly streak within same week."""
        from flaskr.predict import _update_weekly_streak

        streak_record = Mock()
        streak_record.last_weekly_date = date(2026, 2, 10)  # Monday
        streak_record.curr_weekly_streak = 3
        current_date = date(2026, 2, 14)  # Friday same week

        _update_weekly_streak(streak_record, current_date)

        # Should not increment
        assert streak_record.curr_weekly_streak == 3

    def test_update_weekly_streak_consecutive_week(self):
        """Test weekly streak on consecutive week."""
        from flaskr.predict import _update_weekly_streak

        streak_record = Mock()
        streak_record.last_weekly_date = date(2026, 2, 10)  # Monday week 1
        streak_record.curr_weekly_streak = 3
        current_date = date(2026, 2, 17)  # Monday week 2

        _update_weekly_streak(streak_record, current_date)

        assert streak_record.curr_weekly_streak == 4
        assert streak_record.last_weekly_date == current_date

    def test_update_weekly_streak_broken(self):
        """Test weekly streak reset when broken."""
        from flaskr.predict import _update_weekly_streak

        streak_record = Mock()
        streak_record.last_weekly_date = date(2026, 2, 1)  # 2 weeks ago
        streak_record.curr_weekly_streak = 5
        current_date = date(2026, 2, 14)

        _update_weekly_streak(streak_record, current_date)

        assert streak_record.curr_weekly_streak == 1
        assert streak_record.last_weekly_date == current_date


class TestParseCurrentDate:
    """Test _parse_current_date helper function."""

    def test_parse_valid_date(self):
        """Test parsing valid date string."""
        from flaskr.predict import _parse_current_date

        result = _parse_current_date("2026-02-14")

        assert result == date(2026, 2, 14)

    def test_parse_invalid_date_format(self):
        """Test fallback on invalid date format."""
        from flaskr.predict import _parse_current_date

        result = _parse_current_date("invalid-date")

        # Should return today's date as fallback
        assert isinstance(result, date)

    def test_parse_none_date(self):
        """Test with None input."""
        from flaskr.predict import _parse_current_date

        result = _parse_current_date(None)

        # Should return today's date
        assert isinstance(result, date)


class TestUpdateUserStreaks:
    """Test _update_user_streaks helper function."""

    @patch("flaskr.predict.db")
    @patch("flaskr.predict.UserStreaks")
    def test_update_user_streaks_new_user(self, mock_streaks, mock_db):
        """Test creating new streak record for new user."""
        from flaskr.predict import _update_user_streaks

        user_id = uuid.uuid4()
        current_date = date(2026, 2, 14)

        # Mock no existing record
        mock_db.session.query.return_value.filter.return_value.with_for_update.return_value.one_or_none.return_value = (
            None
        )

        _update_user_streaks(user_id, current_date)

        # Should create new streak record
        mock_streaks.assert_called_once()
        mock_db.session.add.assert_called_once()

    @patch("flaskr.predict.db")
    @patch("flaskr.predict._update_daily_streak")
    @patch("flaskr.predict._update_weekly_streak")
    def test_update_user_streaks_existing_user(self, mock_weekly, mock_daily, mock_db):
        """Test updating existing streak record."""
        from flaskr.predict import _update_user_streaks

        user_id = uuid.uuid4()
        current_date = date(2026, 2, 14)

        # Mock existing record
        existing_record = Mock()
        mock_db.session.query.return_value.filter.return_value.with_for_update.return_value.one_or_none.return_value = (
            existing_record
        )

        _update_user_streaks(user_id, current_date)

        # Should update both streaks
        mock_daily.assert_called_once_with(existing_record, current_date)
        mock_weekly.assert_called_once_with(existing_record, current_date)


class TestSaveToDb:
    """Test save_to_db function."""

    @patch("flaskr.predict.db")
    @patch("flaskr.predict.Predictions")
    @patch("flaskr.predict._save_detail_records")
    @patch("flaskr.predict._update_user_streaks")
    def test_save_to_db_with_user(
        self, mock_streaks, mock_details, mock_predictions, mock_db, app_context
    ):
        """Test saving prediction with user_id."""
        from flaskr.predict import save_to_db

        prediction_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        json_input = {
            "user_id": user_id,
            "age": 25,
            "screen_time_hours": 7,
            "work_screen_hours": 4,
            "leisure_screen_hours": 3,
            "sleep_hours": 7,
            "sleep_quality_1_5": 3,
            "stress_level_0_10": 5,
            "productivity_0_100": 75,
            "social_hours_per_week": 10,
            "exercise_minutes_per_week": 150,
        }
        prediction_score = 75.5
        wellness_analysis = {
            "areas_for_improvement": [{"feature": "Sleep", "impact_score": 2.5}],
            "strengths": [],
        }
        ai_advice = {"description": "Test advice", "factors": {}}

        save_to_db(
            prediction_id,
            json_input,
            prediction_score,
            wellness_analysis,
            ai_advice,
            user_id=user_id,
        )

        # Should create prediction record
        mock_predictions.assert_called_once()
        # Should save detail records
        assert mock_details.call_count >= 2  # areas + strengths
        # Should update streaks
        mock_streaks.assert_called_once()
        # Should commit
        mock_db.session.commit.assert_called_once()

    @patch("flaskr.predict.db")
    @patch("flaskr.predict.Predictions")
    @patch("flaskr.predict._save_detail_records")
    def test_save_to_db_without_user(
        self, mock_details, mock_predictions, mock_db, app_context
    ):
        """Test saving prediction without user_id (guest)."""
        from flaskr.predict import save_to_db

        prediction_id = str(uuid.uuid4())
        json_input = {
            "age": 25,
            "screen_time_hours": 7,
            "work_screen_hours": 4,
            "leisure_screen_hours": 3,
            "sleep_hours": 7,
            "sleep_quality_1_5": 3,
            "stress_level_0_10": 5,
            "productivity_0_100": 75,
            "social_hours_per_week": 10,
            "exercise_minutes_per_week": 150,
        }
        prediction_score = 75.5
        wellness_analysis = {"areas_for_improvement": [], "strengths": []}
        ai_advice = {}

        save_to_db(
            prediction_id, json_input, prediction_score, wellness_analysis, ai_advice
        )

        # Should still create prediction record
        mock_predictions.assert_called_once()

    def test_save_to_db_disabled(self, app_context):
        """Test save_to_db when database is disabled."""
        from flaskr.predict import save_to_db

        app_context.config["DB_DISABLED"] = True

        # Should return early without errors
        save_to_db(str(uuid.uuid4()), {}, 75.0, {}, {})


class TestReadFromDb:
    """Test read_from_db function."""

    @patch("flaskr.predict.db")
    def test_read_from_db_by_prediction_id(self, mock_db, app_context):
        """Test reading prediction by prediction_id."""
        from flaskr.predict import read_from_db

        prediction_id = str(uuid.uuid4())

        # Mock prediction record
        mock_pred = Mock()
        mock_pred.pred_id = uuid.UUID(prediction_id)
        mock_pred.user_id = None
        mock_pred.guest_id = None
        mock_pred.pred_date = datetime.now()
        mock_pred.pred_score = 75.0
        mock_pred.ai_desc = "Test"
        mock_pred.details = []
        mock_pred.screen_time = 7.0
        mock_pred.work_screen = 4.0
        mock_pred.leisure_screen = 3.0
        mock_pred.sleep_hours = 7.0
        mock_pred.sleep_quality = 3
        mock_pred.stress_level = 5.0
        mock_pred.productivity = 75.0
        mock_pred.exercise = 150
        mock_pred.social = 10.0

        mock_db.session.execute.return_value.scalar_one_or_none.return_value = mock_pred
        mock_db.select.return_value.options.return_value.filter.return_value = (
            MagicMock()
        )

        result = read_from_db(prediction_id=prediction_id)

        assert result["status"] == "success"
        assert result["data"]["prediction_score"] == 75.0

    def test_read_from_db_invalid_uuid(self, app_context):
        """Test with invalid UUID format."""
        from flaskr.predict import read_from_db

        result = read_from_db(prediction_id="invalid-uuid")

        assert result["status"] == "bad_request"
        assert "Invalid prediction_id format" in result["error"]

    @patch("flaskr.predict.db")
    def test_read_from_db_not_found(self, mock_db, app_context):
        """Test when prediction is not found."""
        from flaskr.predict import read_from_db

        prediction_id = str(uuid.uuid4())
        mock_db.session.execute.return_value.scalar_one_or_none.return_value = None

        result = read_from_db(prediction_id=prediction_id)

        assert result["status"] == "not_found"

    def test_read_from_db_no_params(self, app_context):
        """Test without prediction_id or user_id."""
        from flaskr.predict import read_from_db

        result = read_from_db()

        assert result["status"] == "bad_request"
        assert "must be provided" in result["error"]

    def test_read_from_db_disabled(self, app_context):
        """Test when database is disabled."""
        from flaskr.predict import read_from_db

        app_context.config["DB_DISABLED"] = True

        result = read_from_db(prediction_id=str(uuid.uuid4()))

        assert result["status"] == "disabled"


class TestFormatDbOutput:
    """Test format_db_output helper function."""

    def test_format_db_output_complete(self):
        """Test formatting complete database output."""
        from flaskr.predict import format_db_output

        data = {
            "prediction_score": 75.0,
            "ai_desc": "You're doing well",
            "details": [
                {
                    "factor_name": "Sleep",
                    "impact_score": 2.5,
                    "factor_type": "improvement",
                    "advices": ["Sleep more"],
                    "references": ["https://sleep.org"],
                },
                {
                    "factor_name": "Exercise",
                    "impact_score": -1.5,
                    "factor_type": "strengths",
                    "advices": [],
                    "references": [],
                },
            ],
        }

        with patch(
            "flaskr.predict.model.categorize_mental_health_score",
            return_value="healthy",
        ):
            result = format_db_output(data)

        assert result["prediction_score"] == 75.0
        assert result["health_level"] == "healthy"
        assert len(result["wellness_analysis"]["areas_for_improvement"]) == 1
        assert len(result["wellness_analysis"]["strengths"]) == 1
        assert "Sleep" in result["advice"]["factors"]

    def test_format_db_output_missing_desc(self):
        """Test formatting when ai_desc is missing."""
        from flaskr.predict import format_db_output

        data = {
            "prediction_score": 65.0,
            "details": [],
        }

        with patch(
            "flaskr.predict.model.categorize_mental_health_score",
            return_value="average",
        ):
            result = format_db_output(data)

        assert "Description not available" in result["advice"]["description"]
