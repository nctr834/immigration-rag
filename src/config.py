"""Shared config: env loading, DB connection params, and locked model/chunking constants."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # text-embedding-3-small output dimension
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

# pgvector table (LlamaIndex prefixes the actual table with "data_")
PG_TABLE_NAME = "immigration_chunks"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _pg_params() -> dict:
    """Resolve Postgres connection params from DATABASE_URL or PG* env vars."""
    url = os.getenv("DATABASE_URL")
    if url:
        p = urlparse(url)
        return {
            "host": p.hostname or "localhost",
            "port": str(p.port or 5432),
            "database": (p.path or "/immigration_rag").lstrip("/"),
            "user": p.username or "postgres",
            "password": p.password or "",
        }
    return {
        "host": os.getenv("PGHOST", "localhost"),
        "port": os.getenv("PGPORT", "5432"),
        "database": os.getenv("PGDATABASE", "immigration_rag"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", "postgres"),
    }


PG = _pg_params()


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key
