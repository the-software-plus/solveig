# run.py
from app import create_app
import os
import gunicorn  # Ensure gunicorn is imported

app = create_app()

if __name__ == '__main__':
    # Run with Flask's development server in debug mode
    if os.getenv("FLASK_ENV") == "development":
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # Run with gunicorn in production
        os.system("gunicorn --bind 0.0.0.0:5000 run:app")