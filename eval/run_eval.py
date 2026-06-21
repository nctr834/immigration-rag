"""Score the RAG pipeline against eval_set.json with RAGAS.

Runs the vector-only baseline and the hybrid (vector + BM25) retriever over the
same eval set and prints a comparison table of the four metrics (faithfulness,
answer_relevancy, context_precision, context_recall) with the per-metric delta.
Everything except retrieval mode is held constant (chunking, model, prompt, and
reranking), so the delta is attributable to retrieval.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from typing import cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import EMBED_MODEL, LLM_MODEL, require_openai_key
from generate import generate
from retrieve import RetrievalMode, retrieve

EVAL_SET = os.path.join(os.path.dirname(__file__), "eval_set.json")
MAX_CONCURRENCY = 4  # overridden to 1 by --serial, for measuring the parallel speedup
REQUEST_TIMEOUT = 60.0  # per-request OpenAI timeout; the default is short under load


class EvalItem(BaseModel):
    """One graded question: the schema every eval_set.json entry must satisfy."""

    id: str
    question: str
    ground_truth: str
    source: str
    type: str | None = None  # "exact-token" or "conceptual"; for per-type analysis


def load_eval_set(path: str = EVAL_SET) -> list[EvalItem]:
    """Parse and validate eval_set.json into typed EvalItem records."""
    with open(path) as f:
        return [EvalItem(**item) for item in json.load(f)]


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


async def evaluate_pipeline(
    items: list[EvalItem],
    mode: RetrievalMode,
    rerank: bool,
    concurrency: int = MAX_CONCURRENCY,
) -> dict[str, dict[str, float]]:
    """Score retrieve+generate over every item in one retrieval mode.

    Each item runs its whole pipeline (retrieve -> generate -> grade) as one
    concurrent task, bounded by `concurrency`. retrieve/generate are sync and
    are offloaded to threads (each gets its own pgvector connection, which is
    safe), so they overlap with each other and with grading instead of running
    in a serial loop. concurrency=1 reduces this to a serial run. Returns
    {item_id: {metric_name: score}}.
    """
    require_openai_key()
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=REQUEST_TIMEOUT
    )
    llm = llm_factory(LLM_MODEL, client=client, temperature=0)
    embeddings = cast(
        BaseRagasEmbedding,
        embedding_factory("openai", model=EMBED_MODEL, client=client),
    )

    faithfulness = Faithfulness(llm)
    answer_relevancy = AnswerRelevancy(llm, embeddings)
    context_precision = ContextPrecision(llm)
    context_recall = ContextRecall(llm)

    sem = asyncio.Semaphore(concurrency)

    async def _score(coro_factory, attempts: int = 3):
        """Await a metric call, retrying transient OpenAI errors with backoff."""
        for attempt in range(attempts):
            try:
                return await coro_factory()
            except (APITimeoutError, APIConnectionError, RateLimitError):
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def run_item(item: EvalItem) -> tuple[str, dict[str, float]]:
        async with sem:
            contexts = [
                c.text
                for c in await asyncio.to_thread(
                    retrieve, item.question, mode=mode, rerank=rerank
                )
            ]
            answer = await asyncio.to_thread(
                generate, item.question, mode=mode, rerank=rerank
            )
            response = answer.answer
            scores = await asyncio.gather(
                _score(lambda: faithfulness.ascore(item.question, response, contexts)),
                _score(lambda: answer_relevancy.ascore(item.question, response)),
                _score(
                    lambda: context_precision.ascore(
                        item.question, item.ground_truth, contexts
                    )
                ),
                _score(
                    lambda: context_recall.ascore(
                        item.question, contexts, item.ground_truth
                    )
                ),
            )
            per_item = {
                name: result.value for name, result in zip(METRIC_NAMES, scores)
            }
            print(
                f"[{mode.value}] {item.id}: "
                + "  ".join(f"{n}={v:.2f}" for n, v in per_item.items())
            )
            return item.id, per_item

    results = await asyncio.gather(
        *(run_item(item) for item in items), return_exceptions=True
    )
    await client.close()

    scored, failed = {}, []
    for item, result in zip(items, results):
        if isinstance(result, BaseException):
            failed.append(item.id)
        else:
            item_id, per_item = result
            scored[item_id] = per_item
    if failed:
        print(f"[{mode.value}] skipped (errored after retries): {', '.join(failed)}")
    return scored


async def compare_modes(
    items: list[EvalItem], rerank: bool, concurrency: int = MAX_CONCURRENCY
) -> dict[RetrievalMode, dict[str, dict[str, float]]]:
    """Score the baseline and hybrid modes over the same items, sequentially."""
    return {
        mode: await evaluate_pipeline(items, mode, rerank, concurrency)
        for mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID)
    }


def _run(coro):
    """Run coro, then drain pending tasks before closing the loop.

    ragas spawns its own AsyncOpenAI clients; their httpx connections close on a
    later task. asyncio.run() closes the loop first, so those fire against a dead
    loop and dump tracebacks. Draining them first keeps the loop alive until they
    finish.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        return result
    finally:
        loop.close()
        asyncio.set_event_loop(None)


_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


def _averages(per_item: dict[str, dict[str, float]], ids: list[str]) -> dict[str, float]:
    """Average each metric over the given ids that were actually scored."""
    present = [i for i in ids if i in per_item]
    if not present:
        return {name: float("nan") for name in METRIC_NAMES}
    return {
        name: sum(per_item[i][name] for i in present) / len(present)
        for name in METRIC_NAMES
    }


def _print_table(title: str, base: dict[str, float], hyb: dict[str, float]) -> None:
    print(f"\n{title}")
    print(
        f"{'Metric':<18}{'Baseline (vector)':>20}{'Hybrid (vec+BM25)':>20}{'delta':>10}"
    )
    for name in METRIC_NAMES:
        b, h = base[name], hyb[name]
        print(f"{_LABELS[name]:<18}{b:>20.2f}{h:>20.2f}{h - b:>+10.2f}")


def main() -> None:
    """Score baseline vs hybrid and print the comparison table.

    Flags: --rerank turns the production reranker on (default off, so the table
    isolates retrieval mode); --serial forces concurrency=1, for measuring the
    parallel speedup against the default; --limit N runs only the first N items.
    """
    rerank = "--rerank" in sys.argv
    concurrency = 1 if "--serial" in sys.argv else MAX_CONCURRENCY
    items = load_eval_set()
    if "--limit" in sys.argv:
        n = int(sys.argv[sys.argv.index("--limit") + 1])
        items = items[:n]

    started = time.monotonic()
    results = _run(compare_modes(items, rerank, concurrency))
    elapsed = time.monotonic() - started
    base, hyb = results[RetrievalMode.VECTOR], results[RetrievalMode.HYBRID]

    all_ids = [i.id for i in items]
    label = "rerank ON" if rerank else "rerank OFF"
    _print_table(
        f"Overall ({len(all_ids)} items, {label})",
        _averages(base, all_ids),
        _averages(hyb, all_ids),
    )

    for qtype in ("exact-token", "conceptual"):
        ids = [i.id for i in items if i.type == qtype]
        if ids:
            _print_table(
                f"{qtype} ({len(ids)} items)",
                _averages(base, ids),
                _averages(hyb, ids),
            )

    n_runs = len(items) * 2  # both modes
    print(
        f"\nWall clock: {elapsed:.1f}s for {n_runs} item-runs "
        f"(concurrency={concurrency})"
    )


if __name__ == "__main__":
    main()
