"""retrieve(): rerank-off slicing, mode passthrough, and source fallback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import retrieve as ret
from retrieve import RetrievalMode, retrieve


class FakeNode:
    """Stands in for a llama_index NodeWithScore: the attributes retrieve() reads."""

    def __init__(self, text, file_name=None, node_id="nid", score=0.5):
        self._text = text
        self.node_id = node_id
        self.score = score
        self.node = type(
            "N", (), {"metadata": {"file_name": file_name} if file_name else {}}
        )()

    def get_content(self):
        return self._text


def _patch_retriever(monkeypatch, nodes, capture=None):
    class FakeRetriever:
        def retrieve(self, question):
            if capture is not None:
                capture["question"] = question
            return nodes

    def fake_get_retriever(mode=RetrievalMode.HYBRID, pool=25):
        if capture is not None:
            capture["mode"] = mode
        return FakeRetriever()

    monkeypatch.setattr(ret, "get_retriever", fake_get_retriever)


def test_rerank_off_slices_to_top_k_without_calling_reranker(monkeypatch):
    nodes = [FakeNode(f"chunk {i}", file_name="f.txt", node_id=str(i)) for i in range(10)]
    _patch_retriever(monkeypatch, nodes)

    # If the reranker is called when rerank=False, this raises and fails the test.
    monkeypatch.setattr(
        ret, "get_reranker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reranker called"))
    )

    result = retrieve("q", top_k=3, rerank=False)
    assert len(result) == 3
    assert [c.text for c in result] == ["chunk 0", "chunk 1", "chunk 2"]


def test_mode_is_passed_through_to_get_retriever(monkeypatch):
    capture = {}
    _patch_retriever(monkeypatch, [FakeNode("x", file_name="f.txt")], capture=capture)

    retrieve("hello", mode=RetrievalMode.VECTOR, rerank=False)
    assert capture["mode"] == RetrievalMode.VECTOR
    assert capture["question"] == "hello"


def test_source_falls_back_to_node_id_when_no_file_name(monkeypatch):
    nodes = [FakeNode("orphan chunk", file_name=None, node_id="abc123")]
    _patch_retriever(monkeypatch, nodes)

    result = retrieve("q", top_k=5, rerank=False)
    assert result[0].source == "abc123"


def test_rerank_on_uses_reranker_output(monkeypatch):
    pool = [FakeNode(f"c{i}", file_name="f.txt", node_id=str(i)) for i in range(10)]
    _patch_retriever(monkeypatch, pool)

    reranked = [FakeNode("best", file_name="f.txt", node_id="best")]

    class FakeReranker:
        def postprocess_nodes(self, nodes, query_bundle):
            return reranked

    monkeypatch.setattr(ret, "get_reranker", lambda *a, **k: FakeReranker())

    result = retrieve("q", top_k=5, rerank=True)
    assert len(result) == 1
    assert result[0].text == "best"
