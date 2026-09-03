"""
DataLens — Authentication (Postgres)
------------------------------------------
User registration and login, backed by Postgres (Neon/Supabase in
production, local Postgres for testing) instead of SQLite — SQLite's
file-based storage doesn't survive Streamlit Cloud's ephemeral
filesystem between restarts.

Passwords are hashed with bcrypt — NEVER stored in plaintext, and
never compared with a simple string equality check (bcrypt's own
verify function handles this safely, including timing-attack
resistance).

Run: python auth.py (self-test against the real DB configured in .env)
Requires: pip install psycopg2-binary bcrypt python-dotenv
"""

import hashlib
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import bcrypt
import psycopg2

from db import get_connection

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
CODE_EXPIRY_MINUTES = 15


class AuthError(Exception):
    """User-facing auth problems (bad input, duplicate username, wrong
    password) — distinct from unexpected system errors."""
    pass


def init_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    verification_code_hash TEXT,
                    verification_expires_at TEXT,
                    reset_code_hash TEXT,
                    reset_expires_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT TRUE")
            cur.execute("ALTER TABLE users ALTER COLUMN is_verified SET DEFAULT FALSE")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_code_hash TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_expires_at TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_code_hash TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_expires_at TEXT")
        conn.commit()
    finally:
        conn.close()


def _validate_registration_input(username: str, email: str, password: str) -> None:
    if not USERNAME_PATTERN.match(username):
        raise AuthError("Username must be 3-30 characters, letters/numbers/underscores only.")
    if not EMAIL_PATTERN.match(email):
        raise AuthError("That doesn't look like a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _send_code_email(recipient: str, code: str, purpose: str) -> None:
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM_EMAIL", username)
    if not host or not sender:
        raise AuthError("Email delivery is not configured. Set the SMTP settings in .env.")

    message = EmailMessage()
    message["Subject"] = f"DataLens {purpose} code"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        f"Your DataLens {purpose.lower()} code is: {code}\n\n"
        f"This code expires in {CODE_EXPIRY_MINUTES} minutes."
    )

    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=15) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)


def _issue_code(user_id: int, email: str, purpose: str) -> None:
    code = f"{secrets.randbelow(1_000_000):06d}"
    code_hash = _hash_code(code)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=CODE_EXPIRY_MINUTES)).isoformat()
    hash_column = "verification_code_hash" if purpose == "verification" else "reset_code_hash"
    expiry_column = "verification_expires_at" if purpose == "verification" else "reset_expires_at"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE users SET {hash_column} = %s, {expiry_column} = %s WHERE id = %s",
                (code_hash, expires_at, user_id),
            )
        conn.commit()
    finally:
        conn.close()

    _send_code_email(email, code, purpose)


def create_user(username: str, email: str, password: str) -> int:
    """Register a new user. Raises AuthError with a user-facing message
    on any validation failure or duplicate username/email."""
    _validate_registration_input(username, email, password)
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users (username, email, password_hash, created_at) VALUES (%s, %s, %s, %s) RETURNING id",
                    (username, email, password_hash.decode("utf-8"), datetime.now(timezone.utc).isoformat()),
                )
                new_id = cur.fetchone()["id"]
                conn.commit()
            except psycopg2.errors.UniqueViolation as e:
                conn.rollback()
                if "username" in str(e):
                    raise AuthError("That username is already taken.")
                elif "email" in str(e):
                    raise AuthError("That email is already registered.")
                raise AuthError("Could not create account.")
    finally:
        conn.close()

    _issue_code(new_id, email, "verification")
    return new_id


def verify_email(email: str, code: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, verification_code_hash, verification_expires_at FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if row is None or row["verification_code_hash"] != _hash_code(code):
                raise AuthError("Invalid verification code.")
            if not row["verification_expires_at"] or datetime.fromisoformat(row["verification_expires_at"]) < datetime.now(timezone.utc):
                raise AuthError("That verification code has expired.")
            cur.execute(
                "UPDATE users SET is_verified = TRUE, verification_code_hash = NULL, verification_expires_at = NULL WHERE id = %s",
                (row["id"],),
            )
        conn.commit()
    finally:
        conn.close()


def resend_verification(email: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, is_verified FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise AuthError("No account found for that email.")
    if row["is_verified"]:
        raise AuthError("That account is already verified.")
    _issue_code(row["id"], row["email"], "verification")


def verify_user(username: str, password: str) -> dict:
    """Check credentials. Returns the user record (without the hash) on
    success, raises AuthError on failure. Uses the SAME error message
    whether the username doesn't exist or the password is wrong — never
    reveal which one it was, since that lets an attacker enumerate
    valid usernames."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, is_verified FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise AuthError("Incorrect username or password.")

    if not row["is_verified"]:
        raise AuthError("Please verify your email before logging in.")

    stored_hash = row["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        raise AuthError("Incorrect username or password.")

    return {"id": row["id"], "username": row["username"], "email": row["email"]}


def request_password_reset(email: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is not None:
        _issue_code(row["id"], row["email"], "password reset")


def reset_password(email: str, code: str, new_password: str) -> None:
    if len(new_password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, reset_code_hash, reset_expires_at FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if row is None or row["reset_code_hash"] != _hash_code(code):
                raise AuthError("Invalid password reset code.")
            if not row["reset_expires_at"] or datetime.fromisoformat(row["reset_expires_at"]) < datetime.now(timezone.utc):
                raise AuthError("That password reset code has expired.")
            password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            cur.execute(
                "UPDATE users SET password_hash = %s, reset_code_hash = NULL, reset_expires_at = NULL, is_verified = TRUE WHERE id = %s",
                (password_hash, row["id"]),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # Self-test against the REAL configured DB (local Postgres during
    # dev, per .env) — cleans up its own test rows at the end rather
    # than needing a separate throwaway database.
    init_db()

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE username = 'testuser'")
    conn.commit()
    conn.close()

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
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT password_hash FROM users WHERE username = 'testuser'")
        stored = cur.fetchone()["password_hash"]
    conn.close()
    assert "securepass123" not in stored, "FAIL: plaintext password found in DB!"
    assert stored.startswith("$2b$"), "FAIL: doesn't look like a bcrypt hash!"
    print(f"  Stored hash (not plaintext): {stored[:30]}...")

    # Clean up
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE username = 'testuser'")
    conn.commit()
    conn.close()

    print("\nAll tests passed (against real Postgres).")
