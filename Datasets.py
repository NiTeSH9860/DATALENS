"""
DataLens — Per-User Dataset Storage
------------------------------------------
Stores uploaded datasets per user: metadata (filename, row/column
counts, cached profile) in SQLite, actual CSV content on disk under a
per-user directory.

Critical security property, tested explicitly below: a user can NEVER
load or delete another user's dataset by guessing/incrementing an ID.
Every read/delete operation requires the requesting user_id to match
the dataset's owner — enforced in the SQL WHERE clause itself, not as
an afterthought check in Python (so there's no path that accidentally
skips it).

Run: python datasets.py (self-tests against a throwaway DB)
Requires: pip install pandas
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "data/users.db"  # same DB as auth.py — one users.db, multiple tables
UPLOADS_DIR = "data/user_uploads"


class DatasetError(Exception):
    """User-facing dataset problems: not found, not owned by this user,
    storage failure — distinct from unexpected system errors."""
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
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                profile_json TEXT,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def save_dataset(user_id: int, filename: str, df: pd.DataFrame, profile: dict = None) -> str:
    """Save a dataset for a user. Returns the new dataset's ID.
    The CSV is written to a per-user subdirectory, named by a random
    UUID (not the original filename) — avoids path-traversal risk from
    a malicious filename and avoids collisions between users' files."""
    dataset_id = str(uuid.uuid4())

    user_dir = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, f"{dataset_id}.csv")

    df.to_csv(file_path, index=False)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO datasets
               (id, user_id, filename, file_path, row_count, column_count, profile_json, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id, user_id, filename, file_path,
                len(df), len(df.columns),
                json.dumps(profile) if profile else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    return dataset_id


def list_datasets(user_id: int) -> list[dict]:
    """List all datasets belonging to a user, newest first. Scoped to
    user_id in the WHERE clause — never returns another user's data."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, filename, row_count, column_count, uploaded_at
               FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_dataset(dataset_id: str, user_id: int) -> tuple[pd.DataFrame, dict]:
    """Load a dataset's data and cached profile. Raises DatasetError if
    the dataset doesn't exist OR belongs to a different user — these
    two cases are deliberately indistinguishable to the caller (same
    error message), so a user probing for other people's dataset IDs
    can't tell 'wrong ID' from 'not yours' — same principle as auth.py's
    login error handling."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
            (dataset_id, user_id),
        ).fetchone()

    if row is None:
        raise DatasetError("Dataset not found.")

    if not os.path.exists(row["file_path"]):
        raise DatasetError("Dataset file is missing from storage.")

    df = pd.read_csv(row["file_path"])
    profile = json.loads(row["profile_json"]) if row["profile_json"] else None
    return df, profile


def delete_dataset(dataset_id: str, user_id: int) -> None:
    """Delete a dataset's DB record and its file. Same ownership
    enforcement as load_dataset — the DELETE's WHERE clause itself
    scopes to user_id, so there's no code path where this could delete
    a dataset it doesn't own even if called with a wrong/guessed ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT file_path FROM datasets WHERE id = ? AND user_id = ?",
            (dataset_id, user_id),
        ).fetchone()

        if row is None:
            raise DatasetError("Dataset not found.")

        conn.execute("DELETE FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id))
        conn.commit()

    if os.path.exists(row["file_path"]):
        os.remove(row["file_path"])


if __name__ == "__main__":
    import shutil

    # Self-test against throwaway paths, not the real DB/uploads dir
    DB_PATH = "data/test_users.db"
    UPLOADS_DIR = "data/test_uploads"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(UPLOADS_DIR):
        shutil.rmtree(UPLOADS_DIR)

    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL,
                file_path TEXT NOT NULL, row_count INTEGER NOT NULL, column_count INTEGER NOT NULL,
                profile_json TEXT, uploaded_at TEXT NOT NULL
            )
        """)
        conn.commit()
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    print("Test 1: save and list a dataset for user 1")
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dataset_id = save_dataset(1, "test.csv", df1, profile={"n_rows": 3})
    datasets = list_datasets(1)
    assert len(datasets) == 1
    assert datasets[0]["filename"] == "test.csv"
    print(f"  Saved and listed: {datasets[0]}")

    print("\nTest 2: load it back, data and profile match")
    loaded_df, loaded_profile = load_dataset(dataset_id, 1)
    assert loaded_df.equals(df1)
    assert loaded_profile == {"n_rows": 3}
    print("  Data and profile match exactly")

    print("\nTest 3: CRITICAL — user 2 cannot load user 1's dataset by ID")
    try:
        load_dataset(dataset_id, 2)
        print("  FAIL: user 2 was able to load user 1's dataset!")
        raise SystemExit(1)
    except DatasetError as e:
        print(f"  Correctly blocked: {e}")

    print("\nTest 4: CRITICAL — user 2 cannot delete user 1's dataset by ID")
    try:
        delete_dataset(dataset_id, 2)
        print("  FAIL: user 2 was able to delete user 1's dataset!")
        raise SystemExit(1)
    except DatasetError as e:
        print(f"  Correctly blocked: {e}")
    # Confirm it's genuinely still there after the blocked delete attempt
    still_there, _ = load_dataset(dataset_id, 1)
    assert still_there.equals(df1)
    print("  Confirmed dataset still intact after blocked delete attempt")

    print("\nTest 5: nonexistent dataset ID raises the same error as wrong-owner")
    try:
        load_dataset("00000000-0000-0000-0000-000000000000", 1)
        print("  FAIL: should have raised")
    except DatasetError as e:
        print(f"  Correctly raised: {e}")

    print("\nTest 6: owner CAN delete their own dataset, and file is removed from disk")
    file_path_check = None
    with get_db() as conn:
        row = conn.execute("SELECT file_path FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        file_path_check = row["file_path"]
    assert os.path.exists(file_path_check)

    delete_dataset(dataset_id, 1)
    assert not os.path.exists(file_path_check), "File should be removed from disk"
    assert len(list_datasets(1)) == 0
    print("  Owner successfully deleted their own dataset, file removed from disk")

    print("\nAll tests passed.")
    os.remove(DB_PATH)
    shutil.rmtree(UPLOADS_DIR)
