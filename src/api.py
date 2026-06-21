"""FastAPI surface for the RAG pipeline: POST /query -> grounded, sourced answer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config import require_openai_key
from generate import Answer, generate
from retrieve import get_retriever

logger = logging.getLogger("immigration_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate config and warm the retriever before the app serves traffic.

    get_retriever() reads every chunk out of pgvector and builds the in-memory
    BM25 index once (lru cache).
    """
    require_openai_key()  # fail fast: no point booting without a key
    try:
        await run_in_threadpool(get_retriever)
        logger.info("retriever warmed at startup")
    except Exception:
        logger.exception("retriever warm-up failed; will build lazily on first query")
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
