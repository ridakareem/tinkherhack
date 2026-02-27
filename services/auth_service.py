"""
services/auth_service.py
=========================
Handles user registration, login, and password hashing.

SECURITY:
---------
- Passwords are hashed using werkzeug's generate_password_hash (pbkdf2:sha256).
- Plain-text passwords are NEVER stored.
- Session management is handled by Flask's built-in session (cookie-based).

RULES:
------
- Users can only access their own data (enforced in routes).
- No admin roles — all users are equal.
"""

from werkzeug.security import generate_password_hash, check_password_hash
from typing import Optional, Dict


def register_user(db, username: str, password: str) -> Dict:
    """
    Register a new user with a hashed password.

    Args:
        db: SQLite database connection
        username: Desired username (must be unique)
        password: Plain-text password (will be hashed immediately)

    Returns:
        Dict with keys: success (bool), message (str), user_id (int or None)
    """
    if not username or not password:
        return {"success": False, "message": "Username and password are required.", "user_id": None}

    if len(password) < 4:
        return {"success": False, "message": "Password must be at least 4 characters.", "user_id": None}

    # Check if username already exists
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()

    if existing:
        return {"success": False, "message": "Username already taken.", "user_id": None}

    password_hash = generate_password_hash(password)

    cursor = db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash)
    )
    db.commit()

    return {"success": True, "message": "Account created.", "user_id": cursor.lastrowid}


def login_user(db, username: str, password: str) -> Dict:
    """
    Authenticate a user by username and password.

    Args:
        db: SQLite database connection
        username: Username to look up
        password: Plain-text password to verify

    Returns:
        Dict with keys: success (bool), message (str), user_id (int or None), username (str or None)
    """
    if not username or not password:
        return {"success": False, "message": "Username and password are required.", "user_id": None, "username": None}

    row = db.execute(
        "SELECT id, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()

    if not row:
        return {"success": False, "message": "Invalid username or password.", "user_id": None, "username": None}

    if not check_password_hash(row["password_hash"], password):
        return {"success": False, "message": "Invalid username or password.", "user_id": None, "username": None}

    return {
        "success": True,
        "message": "Login successful.",
        "user_id": row["id"],
        "username": username
    }


def get_user_by_id(db, user_id: int) -> Optional[Dict]:
    """
    Retrieve user info by ID.

    Args:
        db: SQLite database connection
        user_id: User's ID

    Returns:
        Dict with id and username, or None if not found
    """
    row = db.execute(
        "SELECT id, username FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    return dict(row) if row else None
