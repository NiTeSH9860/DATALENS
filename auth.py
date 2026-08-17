"""
CSV Insights Tool — Authentication
------------------------------------------
User registration and login, backed by SQLite. Passwords are hashed
with bcrypt — NEVER stored in plaintext, and never compared with a
simple string equality check (bcrypt's own verify function handles
this safely, including protection against timing attacks).

Run: python auth.py (runs a self-test)
Requires: pip install bcrypt
"""

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import bcrypt

DB_PATH = "data/users.db"

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


class AuthError(Exception):
    """Raised for user-facing auth problems (bad input, duplicate
    username, wrong password) — distinct from unexpected system errors."""
    pass


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def _validate_registration_input(username: str, email: str, password: str) -> None:
    if not USERNAME_PATTERN.match(username):
        raise AuthError(
            "Username must be 3-30 characters, letters/numbers/underscores only."
        )
    if not EMAIL_PATTERN.match(email):
        raise AuthError("That doesn't look like a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")


def create_user(username: str, email: str, password: str) -> int:
    """Register a new user. Raises AuthError with a user-facing message
    on any validation failure or duplicate username/email — never leaks
    raw database exceptions to the caller."""
    _validate_registration_input(username, email, password)

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    with get_db() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, password_hash.decode("utf-8"), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise AuthError("That username is already taken.")
            elif "email" in str(e):
                raise AuthError("That email is already registered.")
            raise AuthError("Could not create account.")


def verify_user(username: str, password: str) -> dict:
    """Check credentials. Returns the user record (without the hash) on
    success, raises AuthError on failure. Uses the SAME error message
    whether the username doesn't exist or the password is wrong — never
    reveal which one it was, since that lets an attacker enumerate
    valid usernames."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        raise AuthError("Incorrect username or password.")

    stored_hash = row["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        raise AuthError("Incorrect username or password.")

    return {"id": row["id"], "username": row["username"], "email": row["email"]}


if __name__ == "__main__":
    import os

    # Self-test against a throwaway DB, not the real one
    DB_PATH = "data/test_users.db"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

    print("Test 1: register a valid user")
    user_id = create_user("testuser", "test@example.com", "securepass123")
    print(f"  Created user id={user_id}")

    print("\nTest 2: correct login succeeds")
    user = verify_user("testuser", "securepass123")
    print(f"  Logged in as: {user}")

    print("\nTest 3: wrong password fails")
    try:
        verify_user("testuser", "wrongpassword")
        print("  FAIL: should have raised AuthError")
    except AuthError as e:
        print(f"  Correctly rejected: {e}")

    print("\nTest 4: duplicate username fails")
    try:
        create_user("testuser", "different@example.com", "anotherpass123")
        print("  FAIL: should have raised AuthError")
    except AuthError as e:
        print(f"  Correctly rejected: {e}")

    print("\nTest 5: password too short fails")
    try:
        create_user("newuser", "new@example.com", "short")
        print("  FAIL: should have raised AuthError")
    except AuthError as e:
        print(f"  Correctly rejected: {e}")

    print("\nTest 6: invalid email fails")
    try:
        create_user("newuser2", "not-an-email", "securepass123")
        print("  FAIL: should have raised AuthError")
    except AuthError as e:
        print(f"  Correctly rejected: {e}")

    print("\nTest 7: password is actually hashed in the database, not plaintext")
    with get_db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username = 'testuser'").fetchone()
        stored = row["password_hash"]
        assert "securepass123" not in stored, "FAIL: plaintext password found in DB!"
        assert stored.startswith("$2b$"), "FAIL: doesn't look like a bcrypt hash!"
        print(f"  Stored hash (not plaintext): {stored[:30]}...")

    print("\nAll tests passed.")
    os.remove(DB_PATH)
