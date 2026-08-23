"""
DataLens — Database Connection
----------------------------------
Shared Postgres connection helper, used by auth.py and datasets.py.

Reads the connection string from DATABASE_URL (a standard env var name
used by Neon, Supabase, Render, Railway, and most Postgres hosts) via
.env locally, or Streamlit Cloud secrets when deployed.
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def _get_connection_string() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    try:
        import streamlit as st
        return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    raise EnvironmentError(
        "DATABASE_URL not found. Create a .env file with "
        "DATABASE_URL=postgresql://user:password@host:port/dbname "
        "(or set it in Streamlit Cloud secrets)."
    )


def get_connection():
    return psycopg2.connect(_get_connection_string(), cursor_factory=psycopg2.extras.RealDictCursor)