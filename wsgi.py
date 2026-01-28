"""
Entry point for running the Flask application
"""
import os
from flaskr import create_app

app = create_app()

if __name__ == '__main__':
    # Use waitress for production or Flask dev server for development
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # For production, use waitress
        from waitress import serve
        print("🚀 Starting production server with Waitress...")
        serve(app, host='0.0.0.0', port=5000)
