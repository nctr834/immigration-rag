"""Turn a question into the top-k source chunks it should be answered from.

Retrieval is two stages. First a vector + BM25 fusion fetches a wide candidate
pool: semantic search over the pgvector embeddings, fused with BM25 keyword
matching so exact tokens like form numbers ("I-864", "I-129F") aren't blurred
away. Then an LLM reranker scores each candidate against the question and keeps
the top_k, which fixes cases where the answering chunk is retrievable but ranks
poorly under fusion (e.g. "waived" in the query vs "exceptions" in the doc).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from config import EMBED_MODEL, LLM_MODEL, RERANK_POOL, TOP_K, require_openai_key
from ingest import build_vector_store
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import BaseNode


def _load_all_nodes() -> list["BaseNode"]:
    """Read every stored chunk out of pgvector, for BM25 to index in memory."""
    # get_nodes() requires a filter; every chunk has a file_name, so "file_name
    # is not <sentinel>" matches all of them.
    all_rows = MetadataFilters(
        filters=[
            MetadataFilter(
                key="file_name", operator=FilterOperator.NE, value="__no_such_file__"
            )
        ]
    )
    return build_vector_store().get_nodes(filters=all_rows)


class RetrievalMode(str, Enum):
    """Which retrievers feed the candidate pool.

    VECTOR is the semantic-only baseline; HYBRID adds BM25 keyword fusion. The
    eval compares them with everything else (chunking, model, prompt, rerank)
    held constant, so any metric delta is attributable to retrieval.
    """

    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus the metadata downstream code cites and evaluates."""

    text: str
    source: str  # source document filename (from the node's metadata)
    score: float | None  # similarity score, or None when a retriever doesn't expose one


@lru_cache(maxsize=None)
def get_retriever(
    mode: RetrievalMode = RetrievalMode.HYBRID, pool: int = RERANK_POOL
) -> "BaseRetriever":
    """Return the retriever that fetches the candidate pool for the given mode.

    Cached per (mode, pool): reading all chunks from pgvector and building the
    BM25 index is expensive and the corpus is static within a process, so it
    runs once instead of on every query. The pool is intentionally wider than
    TOP_K so the reranker has enough candidates to promote a well-matching chunk
    that fusion ranked low.
    """
    require_openai_key()

    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    index = VectorStoreIndex.from_vector_store(
        build_vector_store(), embed_model=embed_model
    )
    vector_retriever = index.as_retriever(similarity_top_k=pool)
    if mode is RetrievalMode.VECTOR:
        return vector_retriever

    nodes = _load_all_nodes()
    if not nodes:
        # BM25 can't build on an empty corpus, and it would raise a cryptic
        # "pass exactly one of index, nodes, or docstore". Fail clearly instead.
        raise RuntimeError(
            "The knowledge base is empty. Run ingest (python ingest.py --extract) "
            "to load the documents."
        )

    return QueryFusionRetriever(
        [
            vector_retriever,
            BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=pool),
        ],
        similarity_top_k=pool,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        use_async=False,
    )


@lru_cache(maxsize=None)
def get_reranker(top_k: int = TOP_K) -> "BaseNodePostprocessor":
    """Return the LLM reranker that trims the candidate pool to top_k."""
    require_openai_key()
    return LLMRerank(top_n=top_k, llm=OpenAI(model=LLM_MODEL, temperature=0))


def retrieve(
    question: str,
    top_k: int = TOP_K,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    rerank: bool = True,
) -> list[RetrievedChunk]:
    """Fetch a candidate pool, optionally rerank it, and return the top_k chunks.

    mode and rerank are the eval's two knobs; production uses the defaults
    (hybrid + rerank).
    """
    candidates = get_retriever(mode).retrieve(question)
    if rerank:
        nodes = get_reranker(top_k).postprocess_nodes(candidates, QueryBundle(question))
    else:
        nodes = candidates[:top_k]
    return [
        RetrievedChunk(
            text=node.get_content(),
            source=node.node.metadata.get("file_name", node.node_id),
            score=node.score,
        )
        for node in nodes
    ]

if __name__ == "__main__":
    for chunk in retrieve("Can the I-864 be waived?"):
        print(f"[{chunk.source}] score={chunk.score}\n{chunk.text[:300]}\n")
