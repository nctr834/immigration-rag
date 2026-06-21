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
    llm = llm_factory(LLM_MODEL, client=client)
    embeddings = cast(
        BaseRagasEmbedding,
        embedding_factory("openai", model=EMBED_MODEL, client=client),
    )

    faithfulness = Faithfulness(llm)
    answer_relevancy = AnswerRelevancy(llm, embeddings)
    context_precision = ContextPrecision(llm)
    context_recall = ContextRecall(llm)

    totals = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    for item in items:
        contexts = [c.text for c in retrieve(item.question)]
        response = generate(item.question).answer
        scores = await asyncio.gather(
            faithfulness.ascore(item.question, response, contexts),
            answer_relevancy.ascore(item.question, response),
            context_precision.ascore(item.question, item.ground_truth, contexts),
            context_recall.ascore(item.question, contexts, item.ground_truth),
        )
        for name, result in zip(totals, scores):
            totals[name] += result.value

    await client.close()
    return {name: total / len(items) for name, total in totals.items()}


def main() -> None:
    """Load the set, score it, and print the four averages as a table."""
    scores = asyncio.run(evaluate_pipeline(load_eval_set()))
    print(f"Faithfulness:      {scores['faithfulness']:.2f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.2f}")
    print(f"Context Precision: {scores['context_precision']:.2f}")
    print(f"Context Recall:    {scores['context_recall']:.2f}")


if __name__ == "__main__":
    main()
