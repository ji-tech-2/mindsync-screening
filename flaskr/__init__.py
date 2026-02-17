"""
MindSync - Mental Health Prediction API
Application Factory Pattern
"""

import os
import logging
from flask import Flask
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app(test_config=None):
    """Application factory function."""
    logger.info("Starting MindSync Model API initialization")

    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded from .env file")

    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    logger.info("Flask app created with instance path: %s", app.instance_path)

    # Detect DB availability (disabled when env not set)
    database_url = os.getenv("DATABASE_URL")
    db_enabled = bool(database_url)
    logger.info("Database enabled: %s", db_enabled)
    if db_enabled:
        logger.info("Database URL configured from environment")

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
        VALKEY_URL=os.getenv("VALKEY_URL", "redis://localhost:6379"),
        DB_DISABLED=not db_enabled,
        JWT_PUBLIC_KEY=os.getenv("JWT_PUBLIC_KEY"),
    )
    logger.info("Flask configuration loaded")

    # Only set SQLAlchemy URI when DB is enabled
    if db_enabled:
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
        logger.info("SQLAlchemy URI configured")

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
        logger.info("Instance config loaded (if exists)")
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
        logger.info("Test configuration applied")

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
        logger.info("Instance folder created/verified: %s", app.instance_path)
    except OSError:
        logger.debug("Instance folder already exists")

    # Initialize database (skip if disabled)
    from . import db

    if not app.config.get("DB_DISABLED", False):
        logger.info("Initializing database connection")
        db.init_app(app)
    else:
        logger.warning("Database disabled (no DATABASE_URL set). Running without DB.")

    # Initialize Valkey
    from . import cache

    logger.info("Initializing Valkey/Redis cache")
    cache.init_app(app)

    # Initialize ML model
    from . import model

    logger.info("Loading ML model and artifacts")
    model.init_app(app)

    # Register blueprints
    from . import predict

    app.register_blueprint(predict.bp)
    logger.info("API blueprints registered")

    # Simple health check route
    @app.route("/")
    def health_check():
        logger.debug("Health check endpoint called")
        return {"status": "active", "message": "MindSync Model API is running."}

    logger.info("MindSync Model API initialization complete")
    return app
