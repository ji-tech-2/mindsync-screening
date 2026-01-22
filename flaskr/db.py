"""
Database models and configuration
"""
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID

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
    __tablename__ = 'PREDICTIONS'
    
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
    __tablename__ = 'PRED_DETAILS'
    
    detail_id = db.Column(db.Integer, primary_key=True)
    pred_id = db.Column(UUID(as_uuid=True), db.ForeignKey('PREDICTIONS.pred_id'), nullable=False)
    factor_name = db.Column(db.Text)
    impact_score = db.Column(db.Float)
    
    # Relationships
    advices = db.relationship('Advices', backref='detail', lazy=True, cascade="all, delete-orphan")
    references = db.relationship('References', backref='detail', lazy=True, cascade="all, delete-orphan")

class Advices(db.Model):
    """AI-generated advice for each factor."""
    __tablename__ = 'ADVICES'
    
    advice_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('PRED_DETAILS.detail_id'), nullable=False)
    advice_text = db.Column(db.Text)

class References(db.Model):
    """Reference links for each factor."""
    __tablename__ = 'REFERENCES'
    
    ref_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('PRED_DETAILS.detail_id'), nullable=False)
    reference_link = db.Column(db.String(500))

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
