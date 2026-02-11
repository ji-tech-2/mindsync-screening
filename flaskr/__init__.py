"""
MindSync - Mental Health Prediction API
Application Factory Pattern
"""

import os
from flask import Flask
from dotenv import load_dotenv


def create_app(test_config=None):
    """Application factory function."""

    # Load environment variables
    load_dotenv()

    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Detect DB availability (disabled when env not set)
    database_url = os.getenv("DATABASE_URL")
    db_enabled = bool(database_url)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
        VALKEY_URL=os.getenv("VALKEY_URL", "redis://localhost:6379"),
        DB_DISABLED=not db_enabled,
    )

    # Only set SQLAlchemy URI when DB is enabled
    if db_enabled:
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize database (skip if disabled)
    from . import db

    if not app.config.get("DB_DISABLED", False):
        db.init_app(app)
    else:
        print("⚠️ Database disabled (no DATABASE_URL set). Running without DB.")

    # Initialize Valkey
    from . import cache

    cache.init_app(app)

    # Initialize ML model
    from . import model

    model.init_app(app)

    # Register blueprints
    from . import predict

    app.register_blueprint(predict.bp)

    # Simple health check route
    @app.route("/")
    def health_check():
        return {"status": "active", "message": "MindSync Model API is running."}

    return app
