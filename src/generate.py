"""Answer a question from its retrieved chunks as a validated, sourced object.

  - LLM: gpt-4o-mini (see config.LLM_MODEL).
  - The model returns a structured answer; citations are filled from the chunks
    that were actually retrieved (file + a quoted passage), so they can't be
    hallucinated and the answer is checkable against the source text.
  - On a malformed or non-validating response, retry exactly once, then raise, so
    the system never returns an unvalidated answer.
"""
from __future__ import annotations

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, ValidationError

from config import LLM_MODEL, require_openai_key
from retrieve import RetrievalMode, RetrievedChunk, retrieve

# Shown with every answer. The corpus is a snapshot of USCIS documents, not a
# live feed, and this is not legal advice; say so in the payload so the output
# never implies otherwise.
DISCLAIMER = (
    "Based on a snapshot of USCIS instruction documents, which may be out of "
    "date. Verify against current USCIS guidance. This is not legal advice."
)

# How much of a chunk to quote in a citation. Long enough to locate the passage
# in the source document, short enough to stay a pointer rather than a dump.
QUOTE_CHARS = 240

SYSTEM = (
    "You are an immigration-forms assistant. Answer the question using ONLY the "
    "provided context chunks. The user's wording may differ from the documents' "
    "(e.g. 'waived' vs 'exceptions' or 'does not need to file'); answer based on "
    "meaning, not exact keyword matches. Only say you don't know if the context "
    "truly lacks the information."
)

PROMPT = PromptTemplate(
    "{system}\n\nContext:\n{context}\n\nQuestion: {question}"
)


class _LLMAnswer(BaseModel):
    """What the LLM must produce: just the prose. Sources are attached by us."""

    answer: str


class Citation(BaseModel):
    """A retrieved source: the document plus a quoted passage to check against."""

    source: str  # source document filename
    quote: str  # an excerpt from the retrieved chunk, so the answer is verifiable


class Answer(BaseModel):
    """The response contract shared with the API: the answer, its citations, a disclaimer."""

    answer: str
    sources: list[Citation]
    disclaimer: str = DISCLAIMER


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """One citation per source document, quoting its highest-ranked chunk.

    chunks arrive in rank order, so the first chunk seen for a source is its
    best-matching passage. The quote is taken verbatim from the retrieved text,
    not from the LLM, so it can't be fabricated.
    """
    citations: list[Citation] = []
    seen: set[str] = set()
    for c in chunks:
        if c.source in seen:
            continue
        seen.add(c.source)
        quote = " ".join(c.text.split())[:QUOTE_CHARS].strip()
        citations.append(Citation(source=c.source, quote=quote))
    return citations


def generate(
    question: str,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    rerank: bool = True,
) -> Answer:
    """Retrieve context, ask the LLM, and return a validated Answer (retry once on failure).

    mode and rerank pass straight through to retrieve so the eval can hold
    generation constant while varying retrieval.
    """
    require_openai_key()

    chunks = retrieve(question, mode=mode, rerank=rerank)
    if not chunks:
        raise ValueError(f"No chunks found for question: {question}")

    sources = _citations(chunks)
    context = _format_context(chunks)
    llm = OpenAI(model=LLM_MODEL, temperature=0)

    attempts = 2  # initial attempt + one retry
    for attempt in range(attempts):
        try:
            result = llm.structured_predict(
                _LLMAnswer,
                PROMPT,
                system=SYSTEM,
                context=context,
                question=question,
            )
            return Answer(answer=result.answer, sources=sources)
        except (ValidationError, ValueError):
            if attempt == attempts - 1:
                raise  # both attempts failed to produce a valid answer

    raise RuntimeError("unreachable: loop must return or raise")  # for the type checker


if __name__ == "__main__":
    print(generate("Can the I-864 be waived?").model_dump_json(indent=2))
