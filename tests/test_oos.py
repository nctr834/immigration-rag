"""Out-of-scope scoring: rate math, and that a no-chunks ValueError counts as a refusal."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_eval
from generate import Answer, Citation


def _write_oos(tmp_path, n):
    items = [{"id": f"oos{i}", "question": f"q{i}", "why_uncovered": "x"} for i in range(n)]
    p = tmp_path / "oos.json"
    p.write_text(json.dumps(items))
    return str(p)


def test_no_chunks_counts_as_refusal(tmp_path, monkeypatch):
    path = _write_oos(tmp_path, 2)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "require_openai_key", lambda: "test-key")

    # generate() raises ValueError (no chunks) -> should be scored as a refusal
    # without ever calling the judge or a real OpenAI client.
    def boom(*a, **k):
        raise ValueError("No chunks found")

    class StubClient:
        async def close(self):
            pass

    monkeypatch.setattr(run_eval, "generate", boom)
    monkeypatch.setattr(run_eval, "AsyncOpenAI", lambda *a, **k: StubClient())

    scores = run_eval._run(run_eval.score_out_of_scope(path=path, concurrency=2))
    assert scores["refusal_rate"] == 1.0
    assert scores["false_answer_rate"] == 0.0


def test_rate_math_mixes_refusals_and_answers(tmp_path, monkeypatch):
    path = _write_oos(tmp_path, 4)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "require_openai_key", lambda: "test-key")
    monkeypatch.setattr(
        run_eval,
        "generate",
        lambda q, *a, **k: Answer(answer=f"ans for {q}", sources=[Citation(source="x", quote="y")]),
    )

    # Fake judge: refuse on even-indexed questions, answer on odd.
    async def fake_judge_factory():
        pass

    class FakeClient:
        def __init__(self, *a, **k):
            self.chat = self
            self.completions = self

        async def create(self, **k):
            q = k["messages"][0]["content"]
            verdict = "REFUSED" if "q0" in q or "q2" in q else "ANSWERED"
            msg = type("M", (), {"content": verdict})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        async def close(self):
            pass

    monkeypatch.setattr(run_eval, "AsyncOpenAI", lambda *a, **k: FakeClient())

    scores = run_eval._run(run_eval.score_out_of_scope(path=path, concurrency=2))
    assert scores["refusal_rate"] == 0.5
    assert scores["false_answer_rate"] == 0.5
