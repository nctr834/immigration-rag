"""FastAPI surface for the RAG pipeline: POST /query -> grounded, sourced answer."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from config import require_openai_key
from generate import Answer, generate
from ingest import chunk_count, ingest
from retrieve import get_retriever

logger = logging.getLogger("immigration_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate config, populate the DB if empty, and warm the retriever.

    On a fresh database (e.g. a new deploy) the chunk table is empty, so ingest
    the committed data/*.txt once. It's idempotent: if chunks already exist this
    is a no-op, so restarts don't re-embed. Then get_retriever() builds the
    in-memory BM25 index once (lru cache).
    """
    require_openai_key()  # fail fast: no point booting without a key
    try:
        if await run_in_threadpool(chunk_count) == 0:
            logger.info("empty database; running one-time ingest of data/*.txt")
            n = await run_in_threadpool(ingest)
            logger.info("ingested %d chunks", n)
        await run_in_threadpool(get_retriever)
        logger.info("retriever warmed at startup")
    except Exception:
        logger.exception("startup ingest/warm-up failed; queries may be unavailable")
    yield


app = FastAPI(
    title="Immigration RAG",
    description=(
        "Ask questions about USCIS immigration forms; answers are grounded in "
        "the instruction documents, with citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """The request contract: a single natural-language question."""

    question: str = Field(min_length=1, max_length=1000)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare URL to the UI so visitors land on the demo, not a 404."""
    return RedirectResponse(url="/ui")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Render and uptime checks."""
    return {"status": "ok"}


@app.post("/query", response_model=Answer)
def query(req: QueryRequest) -> Answer:
    """Accepts any question. The answer is grounded in the retrieved chunks.

    If the forms do not cover the question, the model returns an "I don't know" answer rather than an error.
    """
    try:
        return generate(req.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("query failed for question=%r", req.question)
        raise HTTPException(status_code=502, detail="answer generation failed") from e


# Mount the Gradio UI at /ui so the REST API and the UI are one service. The UI
# calls generate() in-process, not over HTTP. Optional: if gradio isn't
# installed (e.g. the test/CI env), the API still serves /query and /health.
try:
    import gradio as gr

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))
    from app import demo as _ui_demo

    app = gr.mount_gradio_app(app, _ui_demo, path="/ui")
except ImportError:
    logger.info("gradio not installed; skipping /ui mount")
