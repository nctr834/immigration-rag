# Dockerfile for the FastAPI app.

FROM python:3.12-slim

# Don't write .pyc files; stream stdout/stderr straight to the logs (Render-friendly).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first so Docker caches the (slow) pip layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the source and the committed text corpus (ingest reads data/*.txt). Only
# the .txt is copied; the raw PDFs are not in the image (or in git).
COPY src/ ./src/
COPY data/*.txt ./data/

# Render injects $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# api.py lives in src/ and imports sibling modules (config, generate, ...) by
# bare name, so run from inside src/ as the working dir for imports to resolve.
WORKDIR /app/src
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
