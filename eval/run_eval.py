"""Score the RAG pipeline against eval_set.json with RAGAS.

Prints the four average metric scores: faithfulness, answer_relevancy,
context_precision, context_recall. Run it on the vector-only baseline and again
on the hybrid retriever to compare.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from typing import cast

from openai import AsyncOpenAI
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
from retrieve import retrieve

EVAL_SET = os.path.join(os.path.dirname(__file__), "eval_set.json")
MAX_CONCURRENCY = 4


class EvalItem(BaseModel):
    """One graded question: the schema every eval_set.json entry must satisfy."""

    id: str
    question: str
    ground_truth: str
    source: str


def load_eval_set(path: str = EVAL_SET) -> list[EvalItem]:
    """Parse and validate eval_set.json into typed EvalItem records."""
    with open(path) as f:
        return [EvalItem(**item) for item in json.load(f)]


async def evaluate_pipeline(items: list[EvalItem]) -> dict[str, float]:
    """Run retrieve+generate over every item and return {metric_name: average_score}."""
    require_openai_key()
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm = llm_factory(LLM_MODEL, client=client, temperature=0)
    embeddings = cast(
        BaseRagasEmbedding,
        embedding_factory("openai", model=EMBED_MODEL, client=client),
    )

    faithfulness = Faithfulness(llm)
    answer_relevancy = AnswerRelevancy(llm, embeddings)
    context_precision = ContextPrecision(llm)
    context_recall = ContextRecall(llm)
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )

    answered = []
    for item in items:
        contexts = [c.text for c in retrieve(item.question)]
        response = generate(item.question).answer
        answered.append((item, contexts, response))

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def score_item(
        item: EvalItem, contexts: list[str], response: str
    ) -> dict[str, float]:
        async with sem:
            scores = await asyncio.gather(
                faithfulness.ascore(item.question, response, contexts),
                answer_relevancy.ascore(item.question, response),
                context_precision.ascore(item.question, item.ground_truth, contexts),
                context_recall.ascore(item.question, contexts, item.ground_truth),
            )
            per_item = {
                name: result.value for name, result in zip(metric_names, scores)
            }
            print(
                f"{item.id}: " + "  ".join(f"{n}={v:.2f}" for n, v in per_item.items())
            )
            return per_item

    results = await asyncio.gather(*(score_item(*a) for a in answered))
    await client.close()

    return {name: sum(r[name] for r in results) / len(results) for name in metric_names}


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


def main() -> None:
    """Load the set, score it, and print the four averages as a table."""
    scores = _run(evaluate_pipeline(load_eval_set()))
    print(f"Faithfulness:      {scores['faithfulness']:.2f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.2f}")
    print(f"Context Precision: {scores['context_precision']:.2f}")
    print(f"Context Recall:    {scores['context_recall']:.2f}")


if __name__ == "__main__":
    main()
