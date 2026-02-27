"""
Cognitive Memory Analytics System
==================================
Entry point for the Flask application.

Dependencies:
- Flask: Web framework
- LLM (Groq API): Quiz generation — REQUIRES INTERNET + API KEY
- pypdf: PDF text extraction
- SQLite: Local database
- python-dotenv: Loads GROQ_API_KEY from .env file
"""

from flask import Flask
from database.db import init_db
from routes.auth_routes import auth_bp
from routes.pdf_routes import pdf_bp
from routes.quiz_routes import quiz_bp
from routes.curve_routes import curve_bp
import os
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

def create_app():
    """Application factory. Initializes DB and registers all blueprints."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

    # Initialize SQLite database and create tables
    with app.app_context():
        init_db()

    # Register route blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(pdf_bp)
    app.register_blueprint(quiz_bp)
    app.register_blueprint(curve_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
