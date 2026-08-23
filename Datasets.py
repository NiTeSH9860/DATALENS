"""
DataLens — Per-User Dataset Storage (Postgres)
------------------------------------------------------
Stores uploaded datasets per user in Postgres — both metadata AND the
actual CSV content, as a TEXT column. This is a deliberate choice: the
original SQLite version stored CSVs on local disk, which has the same
ephemeral-filesystem problem as SQLite itself on Streamlit Cloud.
Storing CSV content in Postgres avoids needing a THIRD external
service (already using Gemini + Postgres) just for file storage.

Tradeoff, stated honestly: this works well for reasonably-sized CSVs
(a portfolio tool's realistic use case) but doesn't scale to very
large files the way dedicated object storage (S3, Supabase Storage)
would — a genuine limitation, not hidden.

Critical security property, tested explicitly below: a user can NEVER
load or delete another user's dataset by guessing/incrementing an ID
— enforced in the SQL WHERE clause itself.

Run: python datasets.py (self-test against the real DB in .env)
Requires: pip install psycopg2-binary pandas
"""

import io
import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from db import get_connection


class DatasetError(Exception):
    """User-facing dataset problems: not found, not owned by this
    user, storage failure — distinct from unexpected system errors."""
    pass


def init_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    filename TEXT NOT NULL,
                    csv_content TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    profile_json TEXT,
                    uploaded_at TEXT NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_dataset(user_id: int, filename: str, df: pd.DataFrame, profile: dict = None) -> str:
    """Save a dataset for a user. Returns the new dataset's ID."""
    dataset_id = str(uuid.uuid4())
    csv_content = df.to_csv(index=False)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO datasets
                   (id, user_id, filename, csv_content, row_count, column_count, profile_json, uploaded_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    dataset_id, user_id, filename, csv_content,
                    len(df), len(df.columns),
                    json.dumps(profile) if profile else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return dataset_id


def list_datasets(user_id: int) -> list[dict]:
    """List all datasets belonging to a user, newest first. Scoped to
    user_id in the WHERE clause — never returns another user's data."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, filename, row_count, column_count, uploaded_at
                   FROM datasets WHERE user_id = %s ORDER BY uploaded_at DESC""",
                (user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def load_dataset(dataset_id: str, user_id: int) -> tuple[pd.DataFrame, dict]:
    """Load a dataset's data and cached profile. Raises DatasetError if
    the dataset doesn't exist OR belongs to a different user — these
    two cases are deliberately indistinguishable to the caller (same
    error message), same principle as auth.py's login error handling."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT csv_content, profile_json FROM datasets WHERE id = %s AND user_id = %s",
                (dataset_id, user_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise DatasetError("Dataset not found.")

    df = pd.read_csv(io.StringIO(row["csv_content"]))
    profile = json.loads(row["profile_json"]) if row["profile_json"] else None
    return df, profile


def delete_dataset(dataset_id: str, user_id: int) -> None:
    """Delete a dataset. Same ownership enforcement as load_dataset —
    the DELETE's WHERE clause itself scopes to user_id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM datasets WHERE id = %s AND user_id = %s",
                (dataset_id, user_id),
            )
            if cur.fetchone() is None:
                raise DatasetError("Dataset not found.")

            cur.execute("DELETE FROM datasets WHERE id = %s AND user_id = %s", (dataset_id, user_id))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import auth

    auth.init_db()
    init_db()

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE username IN ('dsuser1', 'dsuser2')")
    conn.commit()
    conn.close()

    user1_id = auth.create_user("dsuser1", "dsuser1@example.com", "securepass123")
    user2_id = auth.create_user("dsuser2", "dsuser2@example.com", "securepass123")

    print("Test 1: save and list a dataset for user 1")
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dataset_id = save_dataset(user1_id, "test.csv", df1, profile={"n_rows": 3})
    datasets = list_datasets(user1_id)
    assert len(datasets) == 1
    assert datasets[0]["filename"] == "test.csv"
    print(f"  Saved and listed: {datasets[0]}")

    print("\nTest 2: load it back, data and profile match")
    loaded_df, loaded_profile = load_dataset(dataset_id, user1_id)
    assert loaded_df.equals(df1)
    assert loaded_profile == {"n_rows": 3}
    print("  Data and profile match exactly")

    print("\nTest 3: CRITICAL — user 2 cannot load user 1's dataset by ID")
    try:
        load_dataset(dataset_id, user2_id)
        print("  FAIL: user 2 was able to load user 1's dataset!")
        raise SystemExit(1)
    except DatasetError as e:
        print(f"  Correctly blocked: {e}")

    print("\nTest 4: CRITICAL — user 2 cannot delete user 1's dataset by ID")
    try:
        delete_dataset(dataset_id, user2_id)
        print("  FAIL: user 2 was able to delete user 1's dataset!")
        raise SystemExit(1)
    except DatasetError as e:
        print(f"  Correctly blocked: {e}")
    still_there, _ = load_dataset(dataset_id, user1_id)
    assert still_there.equals(df1)
    print("  Confirmed dataset still intact after blocked delete attempt")

    print("\nTest 5: nonexistent dataset ID raises the same error as wrong-owner")
    try:
        load_dataset("00000000-0000-0000-0000-000000000000", user1_id)
        print("  FAIL: should have raised")
    except DatasetError as e:
        print(f"  Correctly raised: {e}")

    print("\nTest 6: owner CAN delete their own dataset")
    delete_dataset(dataset_id, user1_id)
    assert len(list_datasets(user1_id)) == 0
    print("  Owner successfully deleted their own dataset")

    # Clean up
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE username IN ('dsuser1', 'dsuser2')")
    conn.commit()
    conn.close()

    print("\nAll tests passed (against real Postgres).")
