# run.py
from app import create_app
import os
import gunicorn  # Ensure gunicorn is imported

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True)