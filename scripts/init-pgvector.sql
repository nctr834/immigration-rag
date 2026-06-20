-- Runs once when the Postgres data volume is first initialized
-- (via /docker-entrypoint-initdb.d). Enables pgvector so ingest can
-- create the embedding column.
CREATE EXTENSION IF NOT EXISTS vector;
