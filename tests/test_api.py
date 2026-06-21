"""/query and /health, with generate() and the startup warm-up mocked out."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import api
from generate import Answer


@pytest.fixture
def client(monkeypatch):
    # Lifespan warms the retriever (hits pgvector) and checks the key; stub both.
    monkeypatch.setattr(api, "require_openai_key", lambda: "test-key")
    monkeypatch.setattr(api, "get_retriever", lambda *a, **k: None)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_returns_answer_with_sources(client):
    fake = Answer(answer="K nonimmigrants generally do not repeat the exam.", sources=["x.txt"])
    with patch.object(api, "generate", return_value=fake) as mock_gen:
        resp = client.post("/query", json={"question": "carry over?"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": fake.answer, "sources": ["x.txt"]}
    mock_gen.assert_called_once_with("carry over?")


def test_query_maps_value_error_to_422(client):
    with patch.object(api, "generate", side_effect=ValueError("no chunks")):
        resp = client.post("/query", json={"question": "??"})
    assert resp.status_code == 422
    assert "no chunks" in resp.json()["detail"]


def test_query_maps_unexpected_error_to_502(client):
    with patch.object(api, "generate", side_effect=RuntimeError("openai down")):
        resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 502


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length=1
