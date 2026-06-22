"""generate(): sources come only from retrieved chunks, plus retry/validation behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate as gen
from generate import Answer, _format_context, _LLMAnswer
from retrieve import RetrievedChunk


def _chunks():
    return [
        RetrievedChunk(
            text="The I-864 has exceptions.", source="i-864instr.txt", score=0.9
        ),
        RetrievedChunk(
            text="K-1 marriage within 90 days.", source="K1_Process_V11.txt", score=0.8
        ),
    ]


def _fake_llm(monkeypatch, answer="ok", raise_times=0):
    """Patch gen.OpenAI with an LLM whose structured_predict raises N times, then returns answer."""
    calls = {"n": 0}

    class FakeLLM:
        def structured_predict(self, *a, **k):
            calls["n"] += 1
            if calls["n"] <= raise_times:
                raise ValidationError.from_exception_data("x", [])
            return _LLMAnswer(answer=answer)

    monkeypatch.setattr(gen, "OpenAI", lambda *a, **k: FakeLLM())
    return calls


def test_sources_come_only_from_retrieved_chunks(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: _chunks())
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    _fake_llm(monkeypatch, answer="Could try to cite made-up-doc.pdf here.")

    answer = gen.generate("Can the I-864 be waived?")

    assert isinstance(answer, Answer)
    assert [c.source for c in answer.sources] == ["i-864instr.txt", "K1_Process_V11.txt"]
    # The made-up filename the LLM mentioned must not leak into the citations.
    assert all("made-up-doc.pdf" != c.source for c in answer.sources)


def test_citation_quotes_are_grounded_in_retrieved_text(monkeypatch):
    chunks = _chunks()
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: chunks)
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    _fake_llm(monkeypatch)

    answer = gen.generate("q")
    for cite in answer.sources:
        chunk_text = next(c.text for c in chunks if c.source == cite.source)
        # quote is a verbatim (whitespace-normalized) prefix of the chunk text
        assert cite.quote in " ".join(chunk_text.split())


def test_answer_carries_disclaimer(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: _chunks())
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    _fake_llm(monkeypatch)

    answer = gen.generate("q")
    assert answer.disclaimer == gen.DISCLAIMER
    assert "not legal advice" in answer.disclaimer.lower()


def test_sources_are_deduplicated_preserving_order(monkeypatch):
    dupes = [
        RetrievedChunk(text="a", source="i-864instr.txt", score=0.9),
        RetrievedChunk(text="b", source="i-864instr.txt", score=0.8),
        RetrievedChunk(text="c", source="K1_Process_V11.txt", score=0.7),
    ]
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: dupes)
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    _fake_llm(monkeypatch)

    answer = gen.generate("q")
    assert [c.source for c in answer.sources] == ["i-864instr.txt", "K1_Process_V11.txt"]


def test_no_chunks_raises(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")

    with pytest.raises(ValueError):
        gen.generate("anything")


def test_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: _chunks())
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    calls = _fake_llm(monkeypatch, answer="recovered", raise_times=1)

    answer = gen.generate("q")
    assert answer.answer == "recovered"
    assert calls["n"] == 2  # first attempt failed, retry succeeded


def test_raises_after_both_attempts_fail(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: _chunks())
    monkeypatch.setattr(gen, "require_openai_key", lambda: "test-key")
    calls = _fake_llm(monkeypatch, raise_times=99)

    with pytest.raises(ValidationError):
        gen.generate("q")
    assert calls["n"] == 2  # exactly initial + one retry, no more


def test_format_context_labels_each_chunk_with_source():
    out = _format_context(_chunks())
    assert "[i-864instr.txt]" in out
    assert "[K1_Process_V11.txt]" in out
    assert "The I-864 has exceptions." in out
