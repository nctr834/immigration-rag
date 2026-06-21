"""Answer a question from its retrieved chunks as a validated, sourced object.

  - LLM: gpt-4o-mini (see config.LLM_MODEL).
  - The model returns a structured answer; sources are filled from the chunks
    that were actually retrieved, so citations can't be hallucinated.
  - On a malformed or non-validating response, retry exactly once, then raise, so
    the system never returns an unvalidated answer.
"""
from __future__ import annotations

from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, ValidationError

from config import LLM_MODEL, require_openai_key
from retrieve import RetrievedChunk, retrieve

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


class Answer(BaseModel):
    """The response contract shared with the API: the prose answer + the docs it used."""

    answer: str
    sources: list[str]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def generate(question: str) -> Answer:
    """Retrieve context, ask the LLM, and return a validated Answer (retry once on failure)."""
    require_openai_key()

    chunks = retrieve(question)
    if not chunks:
        raise ValueError(f"No chunks found for question: {question}")

    sources = list(dict.fromkeys(c.source for c in chunks))
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
