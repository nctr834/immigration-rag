"""Extract USCIS PDFs to cleaned .txt, then chunk + embed the .txt into pgvector.

Two stages so the cleaned text is the committed, inspectable source of truth:

  - extract(): PDF -> _clean_text -> data/<name>.txt (run when PDFs change)
  - ingest():  data/*.txt -> SentenceSplitter -> embeddings -> pgvector

  - Chunking:  SentenceSplitter(chunk_size=512, chunk_overlap=50)
  - Embedding: text-embedding-3-small
  - Store:     pgvector, accessed through LlamaIndex

The .txt files are tracked in git; the raw PDFs are not (they extract to a
fraction of the size). A .txt with no matching PDF, e.g. a hand-curated source,
is left untouched by extract() and still ingested.
"""

from __future__ import annotations

import glob
import os
import re
import sys
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

# Table-of-contents entries: a title, a run of dotted/underscore leader, then a
# page number, e.g. "What Is the Purpose of Form I-485? _________ 2". Is
# navigation with no semantic signal. Body never ends in a leader+number,
# so this won't touch real content.
_TOC_ENTRY_RE = re.compile(r"(?im)^.*?[._]{5,}[ \t]*\d+[ \t]*$\n?")


def _clean_text(text: str) -> str:
    """Strip PDF boilerplate (draft watermark, page headers, TOC) before chunking.

    Applied to whole documents, not chunks, so a header sitting on a page
    boundary can't end up split across two chunks.
    """
    text = _WATERMARK_RE.sub("", text)
    text = _PAGE_HEADER_RE.sub("", text)
    text = _TOC_ENTRY_RE.sub("", text)
    # Collapse the long blank-line runs PDF layout leaves behind.
    text = re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", text)
    return text.strip()


def build_vector_store() -> "PGVectorStore":
    """The pgvector-backed store that both ingest (write) and retrieve (read) share."""
    vector_store = PGVectorStore.from_params(
        **PG, table_name=PG_TABLE_NAME, embed_dim=EMBED_DIM
    )
    return vector_store


def chunk_count() -> int:
    """How many chunks are stored, or 0 if the table doesn't exist yet.

    A cheap COUNT(*) used to decide whether the DB needs a one-time ingest.
    LlamaIndex prefixes the configured table name with "data_".
    """
    import psycopg2

    conn = psycopg2.connect(
        host=PG["host"],
        port=PG["port"],
        dbname=PG["database"],
        user=PG["user"],
        password=PG["password"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM data_{PG_TABLE_NAME}")
            return cur.fetchone()[0]
    except psycopg2.errors.UndefinedTable:
        return 0
    finally:
        conn.close()


def extract() -> int:
    """Extract every PDF in data/ to a cleaned data/<name>.txt; return file count.

    Cleaning (watermark/header stripping) runs here so the committed .txt is
    exactly what gets ingested. A NUL byte is stripped because Postgres text
    columns reject it downstream.
    """
    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    for pdf in pdfs:
        documents = SimpleDirectoryReader(input_files=[pdf]).load_data()
        text = _clean_text(
            "\n".join(d.get_content().replace("\x00", "") for d in documents)
        )
        out = os.path.splitext(pdf)[0] + ".txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return len(pdfs)


def ingest() -> int:
    """Index every .txt in data/ into the vector store; return the chunk count written."""
    require_openai_key()

    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".txt"]).load_data()

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
    if "--extract" in sys.argv:
        print(f"extracted {extract()} PDF(s)")
    print(ingest())
