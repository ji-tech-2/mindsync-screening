"""
Shared pytest fixtures for all tests.
Provides JWT mocking utilities for protected endpoints.
"""

import pytest
import uuid
from unittest.mock import patch
from flask import Flask


# ===================== #
#   JWT MOCK FIXTURES   #
# ===================== #


@pytest.fixture
def mock_jwt_user_id():
    """Generate a consistent user ID for JWT mocking."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_jwt_identity(mock_jwt_user_id):
    """
    Mock get_jwt_identity to return a valid user_id.
    Use as a context manager in tests.
    """

    def _mock(user_id=None):
        target_user_id = user_id if user_id else mock_jwt_user_id
        return patch("flaskr.predict.get_jwt_identity", return_value=target_user_id)

    return _mock


@pytest.fixture
def mock_jwt_guest():
    """Mock get_jwt_identity to return None (guest user)."""
    return patch("flaskr.predict.get_jwt_identity", return_value=None)


# ===================== #
#   APP FIXTURES        #
# ===================== #


@pytest.fixture
def base_app():
    """Create a base test Flask app with minimal configuration."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["DB_DISABLED"] = False
    app.config["GEMINI_API_KEYS"] = "test-api-key"
    app.config["JWT_PUBLIC_KEY"] = "test-public-key"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    from flaskr import db as db_module

    db_module.db.init_app(app)

    with app.app_context():
        db_module.db.create_all()

    from flaskr.predict import bp

    app.register_blueprint(bp)

    return app


@pytest.fixture
def base_client(base_app):
    """Create a test client from base_app."""
    return base_app.test_client()


# ===================== #
#   TEST DATA HELPERS   #
# ===================== #


@pytest.fixture
def sample_prediction_input():
    """Sample input data for predictions."""
    return {
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
        "screen_time_hours": 6,
        "work_mode": "Hybrid",
    }


@pytest.fixture
def sample_wellness_analysis():
    """Sample wellness analysis result."""
    return {
        "areas_for_improvement": [
            {"feature": "Sleep Quality", "impact_score": 2.5},
            {"feature": "Screen Time", "impact_score": 1.8},
        ],
        "strengths": [
            {"feature": "Exercise", "impact_score": -1.5},
            {"feature": "Social", "impact_score": -1.2},
        ],
    }


@pytest.fixture
def sample_ai_advice():
    """Sample AI advice response."""
    return {
        "description": "You're doing well overall.",
        "factors": {
            "Sleep Quality": {
                "advices": ["Maintain consistent sleep schedule"],
                "references": ["https://sleep.org"],
            },
        },
    }


# ===================== #
#   PROCESS PREDICTION  #
#   CALL HELPER         #
# ===================== #


def call_process_prediction(
    prediction_id,
    json_input,
    created_at,
    start_ts,
    app,
    user_id=None,
    guest_id=None,
):
    """
    Helper to call process_prediction with all required arguments.
    Use this in tests to ensure correct signature.
    """
    from flaskr.predict import process_prediction

    process_prediction(
        prediction_id,
        json_input,
        created_at,
        start_ts,
        app,
        user_id,
        guest_id,
    )


@pytest.fixture
def process_prediction_caller():
    """
    Fixture that provides the process_prediction caller helper.
    """
    return call_process_prediction
