"""Provider selection and HTTP behavior with retrieval and orchestration intact."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
import conversation_client
from agents.multi_agent_rag import SynthesisAgent


@pytest.fixture
def conversation_app(monkeypatch, tmp_path):
    for name, value in {
        "answer_provider": "vertex_conversation", "google_api_key": None,
        "conversation_project_id": "test-project", "conversation_location": "us-central1",
        "conversation_agent_id": "test-agent", "conversation_environment_id": "published-env",
        "conversation_language_code": "en", "conversation_event_name": "RAG_ANSWER",
        "conversation_timeout_seconds": 30, "conversation_max_payload_bytes": 16384,
        "extracted_images_dir": tmp_path, "context_snippet_max_length": 0,
    }.items():
        monkeypatch.setattr(main.settings, name, value)
    monkeypatch.setattr(main, "_rag_orchestrator", None)
    forbidden_gemini = Mock(side_effect=AssertionError("Developer API must never initialize"))
    monkeypatch.setattr(main, "get_gemini", forbidden_gemini)
    image = tmp_path / "ingest-test" / "diagram.png"
    image.parent.mkdir()
    image.write_bytes(b"test image")

    class Embedder:
        def embed_query(self, query):
            return [query]

    class VectorDB:
        def __init__(self):
            self.filters = []

        def search_by_type(self, vector, content_type, top_k, metadata_filter=None):
            self.filters.append(metadata_filter)
            return [{"score": 0.9, "metadata": {
                "type": content_type, "page": 12,
                "content": f"Manual evidence for {vector[0]}" if content_type == "text" else str(image),
            }}]

    vector_db = VectorDB()
    monkeypatch.setattr(main, "get_embedder", lambda: Embedder())
    monkeypatch.setattr(main, "get_vector_db", lambda: vector_db)
    main.app.dependency_overrides[main.get_api_key] = lambda: "test"
    monkeypatch.setattr(main.limiter, "enabled", False)
    try:
        yield SimpleNamespace(gemini=forbidden_gemini, vector_db=vector_db)
    finally:
        main.app.dependency_overrides.pop(main.get_api_key, None)


def test_conversation_route_preserves_grounding_response_and_filter(conversation_app, monkeypatch):
    calls = []

    class AnswerClient:
        def __init__(self, settings):
            assert settings.google_api_key is None

        def generate_answer(self, question, evidence):
            calls.append((question, evidence))
            return "Clean the brew group. (p. 12)"

    monkeypatch.setattr(conversation_client, "ConversationAnswerClient", AnswerClient)
    response = TestClient(main.app).post("/api/ask", json={
        "query": "clean", "metadata_filter": {"author": "Saeco"},
    })
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Clean the brew group. (p. 12)",
        "context_images": ["ingest-test/diagram.png"],
        "citations": [{"page": 12, "type": "text", "score": 0.9}, {"page": 12, "type": "image", "score": 0.9}],
        "quality_warnings": [],
        "agent_trace": ["retrieval", "context_analysis", "synthesis", "quality_control"],
    }
    assert calls[0][0] == "clean"
    assert "Page 12 [text" in calls[0][1]
    assert "Manual evidence for clean" in calls[0][1]
    assert conversation_app.vector_db.filters == [{"author": "Saeco"}] * 2
    conversation_app.gemini.assert_not_called()


@pytest.mark.parametrize("status,message", [
    (429, "Conversation quota exceeded."), (504, "Conversation timed out."),
    (503, "Conversation is unavailable."), (502, "Conversation returned no answer."),
    (413, "Conversation request is too large."),
])
def test_conversation_errors_keep_safe_status_without_fallback(conversation_app, monkeypatch, status, message):
    class AnswerClient:
        def __init__(self, settings):
            pass

        def generate_answer(self, question, evidence):
            raise conversation_client.ConversationError(message, status)

    monkeypatch.setattr(conversation_client, "ConversationAnswerClient", AnswerClient)
    response = TestClient(main.app).post("/api/ask", json={"query": "clean"})
    assert response.status_code == status
    assert response.json()["detail"].startswith(message)
    assert response.headers["x-correlation-id"] in response.json()["detail"]
    conversation_app.gemini.assert_not_called()


def test_concurrent_conversation_requests_keep_question_and_evidence_isolated(conversation_app, monkeypatch):
    barrier = Barrier(2)
    construction = []
    calls = []
    lock = Lock()

    class AnswerClient:
        def __init__(self, settings):
            construction.append(1)

        def generate_answer(self, question, evidence):
            barrier.wait(timeout=5)
            with lock:
                calls.append((question, evidence))
            return f"Answer for {question} (p. 12)"

    monkeypatch.setattr(conversation_client, "ConversationAnswerClient", AnswerClient)

    def ask(question):
        with TestClient(main.app) as client:
            return client.post("/api/ask", json={"query": question})

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(ask, ["clean", "descale"]))
    assert [response.json()["answer"] for response in responses] == ["Answer for clean (p. 12)", "Answer for descale (p. 12)"]
    assert len(construction) == 1
    assert len(calls) == 2
    for question, evidence in calls:
        assert f"Manual evidence for {question}" in evidence
        other = "descale" if question == "clean" else "clean"
        assert f"Manual evidence for {other}" not in evidence
    conversation_app.gemini.assert_not_called()


def test_synthesis_provider_receives_separate_unmodified_question_and_evidence():
    provider = Mock()
    provider.generate_answer.return_value = "Grounded response"
    result = SynthesisAgent(answer_provider=provider).synthesize(
        "question", SimpleNamespace(text_context="manual evidence"),
    )
    assert result == "Grounded response"
    provider.generate_answer.assert_called_once_with("question", "manual evidence")


def test_health_remains_responsive_while_first_provider_initializes(conversation_app, monkeypatch):
    entered, release = Event(), Event()

    class AnswerClient:
        def __init__(self, settings):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test initialization timed out")

        def generate_answer(self, question, evidence):
            return "Answer (p. 12)"

    monkeypatch.setattr(conversation_client, "ConversationAnswerClient", AnswerClient)
    with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(client.post, "/api/ask", json={"query": "clean"})
        try:
            assert entered.wait(timeout=5)
            health = executor.submit(client.get, "/health")
            assert health.result(timeout=3).json() == {"status": "ok"}
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
