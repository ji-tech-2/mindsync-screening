"""
WSGI Entry Point for Gunicorn
This module provides the WSGI application callable for production deployment.
"""
from flaskr import create_app

# Create the Flask application instance
# Gunicorn will use this 'app' object
app = create_app()
