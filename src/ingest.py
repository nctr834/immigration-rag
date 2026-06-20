"""Load USCIS PDFs from data/, chunk + embed them, and store them in pgvector.

  - Chunking:  SentenceSplitter(chunk_size=512, chunk_overlap=50)
  - Embedding: text-embedding-3-small
  - Store:     pgvector, accessed through LlamaIndex
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    PG,
    PG_TABLE_NAME,
    require_openai_key,
)

if TYPE_CHECKING:
    from llama_index.vector_stores.postgres import PGVectorStore


# Draft watermark USCIS stamps on pre-release forms, e.g.
#   DRAFT / Not for / Production / 02/18/2025
# Extraction splits it across lines (sometimes with blank lines between), and
# the case varies ("DRAFT" vs "Draft"). Anchoring on the full three-word block
# means a bare "Draft" in body text (e.g. "16. Draft records") is never touched.
_WATERMARK_RE = re.compile(
    r"(?im)^[ \t]*draft[ \t]*\n+[ \t]*not for[ \t]*\n+[ \t]*production[ \t]*\n+"
    r"(?:[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]*\n+)?"
)

# Repeated page header/footer, e.g. "Form I-485 Instructions  01/20/25 Page 41 of 42".
# Low-signal boilerplate on nearly every page; strip the whole line.
# Note the form number and "Instructions" aren't always adjacent: some titles
# read "Form I-485 Supplement A Instructions", so allow words in between.
_PAGE_HEADER_RE = re.compile(
    r"(?im)^[ \t]*Form\s+[A-Z]-\d+\w*[^\n]*?Instructions[^\n]*?Page\s+\d+\s+of\s+\d+[ \t]*\n?"
)


def _clean_text(text: str) -> str:
    """Strip PDF boilerplate (draft watermark, page headers) before chunking.

    Applied to whole documents, not chunks, so a header sitting on a page
    boundary can't end up split across two chunks.
    """
    text = _WATERMARK_RE.sub("", text)
    text = _PAGE_HEADER_RE.sub("", text)
    # Collapse the long blank-line runs PDF layout leaves behind.
    text = re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", text)
    return text.strip()


def build_vector_store() -> "PGVectorStore":
    """The pgvector-backed store that both ingest (write) and retrieve (read) share."""
    vector_store = PGVectorStore.from_params(
        **PG, table_name=PG_TABLE_NAME, embed_dim=EMBED_DIM
    )
    return vector_store


def ingest() -> int:
    """Index every PDF in data/ into the vector store; return the chunk count written."""
    require_openai_key()

    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Strip boilerplate before chunking so headers at page boundaries don't end
    # up split across chunks. Also drop NUL bytes Postgres text columns reject.
    for doc in documents:
        doc.set_content(_clean_text(doc.get_content().replace("\x00", "")))

    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    nodes = sentence_splitter.get_nodes_from_documents(documents)

    # Drop empty chunks (e.g. image-only PDF pages with no text layer).
    nodes = [n for n in nodes if n.get_content().strip()]

    vector_store = build_vector_store()
    try:
        vector_store.clear()
    except Exception:
        pass
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding(model=EMBED_MODEL),
        show_progress=True,
    )

    return len(nodes)


if __name__ == "__main__":
    print(ingest())
