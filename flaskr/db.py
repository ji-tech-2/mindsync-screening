"""
Database models and configuration
"""

import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import CheckConstraint

db = SQLAlchemy()


def init_app(app):
    """Initialize database with the app."""
    db.init_app(app)

    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables initialized.")
        except Exception as e:
            print(f"❌ Database initialization warning: {e}")


# ===================== #
#    DATABASE MODELS    #
# ===================== #


class Predictions(db.Model):
    """Main predictions table."""

    __tablename__ = "predictions"

    pred_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), nullable=True, index=True)
    guest_id = db.Column(UUID(as_uuid=True), nullable=True, index=True)
    pred_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Input Data
    screen_time = db.Column(db.Float)
    work_screen = db.Column(db.Float)
    leisure_screen = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    sleep_quality = db.Column(db.Integer)
    stress_level = db.Column(db.Float)
    productivity = db.Column(db.Float)
    exercise = db.Column(db.Integer)
    social = db.Column(db.Float)

    # Output Data
    pred_score = db.Column(db.Float)

    # AI Generated Summary
    ai_desc = db.Column(db.Text, nullable=True)

    # Relationships
    details = db.relationship(
        "PredDetails", backref="prediction", lazy=True, cascade="all, delete-orphan"
    )


class PredDetails(db.Model):
    """Prediction details - wellness factors."""

    __tablename__ = "pred_details"

    detail_id = db.Column(db.Integer, primary_key=True)
    pred_id = db.Column(
        UUID(as_uuid=True), db.ForeignKey("predictions.pred_id"), nullable=False
    )
    factor_name = db.Column(db.Text)
    impact_score = db.Column(db.Float)
    factor_type = db.Column(
        db.String(20),
        default="improvement",
        server_default="improvement",
        comment=(
            "'improvement' or 'strengths' - used to separate "
            "areas_for_improvement vs strengths"
        ),
    )

    # Relationships
    advices = db.relationship(
        "Advices", backref="detail", lazy=True, cascade="all, delete-orphan"
    )
    references = db.relationship(
        "References", backref="detail", lazy=True, cascade="all, delete-orphan"
    )


class Advices(db.Model):
    """AI-generated advice for each factor."""

    __tablename__ = "advices"

    advice_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(
        db.Integer, db.ForeignKey("pred_details.detail_id"), nullable=False
    )
    advice_text = db.Column(db.Text)


class References(db.Model):
    """Reference links for each factor."""

    __tablename__ = "references"

    ref_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(
        db.Integer, db.ForeignKey("pred_details.detail_id"), nullable=False
    )
    reference_link = db.Column(db.String(500))


class UserStreaks(db.Model):
    """Gamification streak for users."""

    __tablename__ = "user_streaks"

    user_id = db.Column(UUID(as_uuid=True), primary_key=True)

    # Daily check-in streaks
    curr_daily_streak = db.Column(db.Integer, default=0)
    last_daily_date = db.Column(db.Date, nullable=True)

    # Weekly check-in streaks
    curr_weekly_streak = db.Column(db.Integer, default=0)
    last_weekly_date = db.Column(db.Date, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        CheckConstraint(
            "(curr_daily_streak = 0) OR (last_daily_date IS NOT NULL)",
            name="check_daily_streak_integrity",
        ),
        CheckConstraint(
            "(curr_weekly_streak = 0) OR (last_weekly_date IS NOT NULL)",
            name="check_weekly_streak_integrity",
        ),
    )

    def to_dict(self):
        """Helper to convert object to dict for JSON response."""
        return {
            "user_id": str(self.user_id),
            "daily": {
                "current": self.curr_daily_streak,
                "last_date": (
                    self.last_daily_date.isoformat() if self.last_daily_date else None
                ),
            },
            "weekly": {
                "current": self.curr_weekly_streak,
                "last_date": (
                    self.last_weekly_date.isoformat() if self.last_weekly_date else None
                ),
            },
        }


class WeeklyCriticalFactors(db.Model):
    """Cache for weekly critical factors analysis."""

    __tablename__ = "weekly_critical_factors"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(UUID(as_uuid=True), nullable=True, index=True)
    week_start = db.Column(db.Date, nullable=False, index=True)
    week_end = db.Column(db.Date, nullable=False)
    days = db.Column(db.Integer, default=7)
    total_predictions = db.Column(db.Integer, default=0)

    # Store JSON data
    top_factors = db.Column(
        db.JSON, nullable=False
    )  # List of {factor_name, count, avg_impact_score}
    ai_advice = db.Column(db.JSON, nullable=True)  # AI-generated advice

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "period": {
                "start_date": self.week_start.isoformat(),
                "end_date": self.week_end.isoformat(),
                "days": self.days,
            },
            "stats": {
                "total_predictions": self.total_predictions,
                "user_id": str(self.user_id) if self.user_id else None,
            },
            "top_critical_factors": self.top_factors,
            "advice": self.ai_advice,
        }


class WeeklyChartData(db.Model):
    """Cache for weekly chart data."""

    __tablename__ = "weekly_chart_data"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(UUID(as_uuid=True), nullable=False, index=True)
    week_start = db.Column(db.Date, nullable=False, index=True)
    week_end = db.Column(db.Date, nullable=False)

    # Store JSON data
    chart_data = db.Column(db.JSON, nullable=False)  # List of daily stats

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {"data": self.chart_data}


class DailySuggestions(db.Model):
    """Cache for daily suggestions."""

    __tablename__ = "daily_suggestions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(UUID(as_uuid=True), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    prediction_count = db.Column(db.Integer, default=0)

    # Store JSON data
    top_factors = db.Column(
        db.JSON, nullable=False
    )  # List of {factor_name, impact_score}
    ai_advice = db.Column(db.JSON, nullable=True)  # AI-generated suggestion

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "date": self.date.isoformat(),
            "user_id": str(self.user_id),
            "stats": {"predictions_today": self.prediction_count},
            "areas_of_improvement": self.top_factors,
            "suggestion": self.ai_advice,
        }


# ===================== #
#    HELPER FUNCTIONS   #
# ===================== #


def is_valid_uuid(val):
    """Validate UUID format."""
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False
