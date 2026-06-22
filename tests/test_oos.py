"""Out-of-scope scoring: rate math, and that a no-chunks ValueError counts as a refusal.

The refusal judge is Claude (AsyncAnthropic), independent of the gpt-4o-mini
generator; these tests mock it so no network/keys are touched.
"""

from __future__ import annotations

import json
import sys
import types
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


def _patch_anthropic(monkeypatch, verdict_fn):
    """Patch anthropic.AsyncAnthropic with a fake whose messages.create returns a verdict."""

    class FakeMessages:
        async def create(self, **k):
            prompt = k["messages"][0]["content"]
            block = types.SimpleNamespace(text=verdict_fn(prompt))
            return types.SimpleNamespace(content=[block])

    class FakeAnthropic:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

        async def close(self):
            pass

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


def test_no_chunks_counts_as_refusal(tmp_path, monkeypatch):
    path = _write_oos(tmp_path, 2)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "require_openai_key", lambda: "test-key")

    # generate() raises ValueError (no chunks) -> scored as a refusal without ever
    # calling the judge.
    def boom(*a, **k):
        raise ValueError("No chunks found")

    monkeypatch.setattr(run_eval, "generate", boom)
    _patch_anthropic(monkeypatch, lambda prompt: "ANSWERED")  # never reached

    scores = run_eval._run(run_eval.score_out_of_scope(path=path, concurrency=2))
    assert scores["refusal_rate"] == 1.0
    assert scores["false_answer_rate"] == 0.0


def test_rate_math_mixes_refusals_and_answers(tmp_path, monkeypatch):
    path = _write_oos(tmp_path, 4)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "require_openai_key", lambda: "test-key")
    monkeypatch.setattr(
        run_eval,
        "generate",
        lambda q, *a, **k: Answer(answer=f"ans for {q}", sources=[Citation(source="x", quote="y")]),
    )
    # Judge refuses q0/q2, answers q1/q3.
    _patch_anthropic(
        monkeypatch,
        lambda prompt: "REFUSED" if ("q0" in prompt or "q2" in prompt) else "ANSWERED",
    )

    scores = run_eval._run(run_eval.score_out_of_scope(path=path, concurrency=2))
    assert scores["refusal_rate"] == 0.5
    assert scores["false_answer_rate"] == 0.5


def test_oos_requires_anthropic_key(tmp_path, monkeypatch):
    path = _write_oos(tmp_path, 1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(run_eval, "require_openai_key", lambda: "test-key")

    try:
        run_eval._run(run_eval.score_out_of_scope(path=path))
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError when ANTHROPIC_API_KEY is unset")
