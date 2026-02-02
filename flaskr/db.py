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
    __tablename__ = 'predictions'
    
    pred_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), nullable=True)
    guest_id = db.Column(UUID(as_uuid=True), nullable=True)
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
    
    # Relationships
    details = db.relationship('PredDetails', backref='prediction', lazy=True, cascade="all, delete-orphan")

class PredDetails(db.Model):
    """Prediction details - wellness factors."""
    __tablename__ = 'pred_details'
    
    detail_id = db.Column(db.Integer, primary_key=True)
    pred_id = db.Column(UUID(as_uuid=True), db.ForeignKey('predictions.pred_id'), nullable=False)
    factor_name = db.Column(db.Text)
    impact_score = db.Column(db.Float)
    
    # Relationships
    advices = db.relationship('Advices', backref='detail', lazy=True, cascade="all, delete-orphan")
    references = db.relationship('References', backref='detail', lazy=True, cascade="all, delete-orphan")

class Advices(db.Model):
    """AI-generated advice for each factor."""
    __tablename__ = 'advices'
    
    advice_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('pred_details.detail_id'), nullable=False)
    advice_text = db.Column(db.Text)

class References(db.Model):
    """Reference links for each factor."""
    __tablename__ = 'references'
    
    ref_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('pred_details.detail_id'), nullable=False)
    reference_link = db.Column(db.String(500))

class UserStreaks(db.Model):
    """Gamification streak for users."""
    __tablename__ = 'user_streaks'
    
    user_id = db.Column(UUID(as_uuid=True), primary_key=True)

    # Daily check-in streaks
    curr_daily_streak = db.Column(db.Integer, default=0)
    last_daily_date = db.Column(db.Date, nullable=True)

    # Weekly check-in streaks
    curr_weekly_streak = db.Column(db.Integer, default=0)
    last_weekly_date = db.Column(db.Date, nullable=True)

    __table_args__ = (
        CheckConstraint(
            '(curr_daily_streak = 0) OR (last_daily_date IS NOT NULL)',
            name='check_daily_streak_integrity'
        ),
        CheckConstraint(
            '(curr_weekly_streak = 0) OR (last_weekly_date IS NOT NULL)',
            name='check_weekly_streak_integrity'
        ),
    )

    def to_dict(self):
        """Helper to convert object to dict for JSON response."""
        return {
            "user_id": str(self.user_id),
            "daily": {
                "current": self.curr_daily_streak,
                "last_date": self.last_daily_date.isoformat() if self.last_daily_date else None,
            },
            "weekly": {
                "current": self.curr_weekly_streak,
                "last_date": self.last_weekly_date.isoformat() if self.last_weekly_date else None
            }
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
