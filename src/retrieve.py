"""Turn a question into the top-k source chunks it should be answered from.

Retrieval is a vector + BM25 fusion: semantic search over the pgvector
embeddings, fused with BM25 keyword matching so exact tokens like form numbers
("I-864", "I-129F") aren't blurred away. get_retriever() builds both halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import EMBED_MODEL, TOP_K, require_openai_key
from ingest import build_vector_store
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

if TYPE_CHECKING:
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


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus the metadata downstream code cites and evaluates."""

    text: str
    source: str  # source document filename (from the node's metadata)
    score: float | None  # similarity score, or None when a retriever doesn't expose one


def get_retriever(top_k: int = TOP_K) -> "BaseRetriever":
    """Return the hybrid vector + BM25 retriever used for all queries."""
    require_openai_key()

    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    index = VectorStoreIndex.from_vector_store(
        build_vector_store(), embed_model=embed_model
    )
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=top_k),
            BM25Retriever.from_defaults(
                nodes=_load_all_nodes(), similarity_top_k=top_k
            ),
        ],
        similarity_top_k=top_k,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        use_async=True,
    )
    return retriever


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Embed the question and return its top_k chunks as RetrievedChunk records."""
    retriever = get_retriever(top_k)
    return [
        RetrievedChunk(
            text=chunk.get_content(),
            source=chunk.node.metadata.get("file_name", chunk.node_id),
            score=chunk.score,
        )
        for chunk in retriever.retrieve(question)
    ]

if __name__ == "__main__":
    for chunk in retrieve("Can the I-864 be waived?"):
        print(f"[{chunk.source}] score={chunk.score}\n{chunk.text[:300]}\n")
