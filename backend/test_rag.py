import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_RAG_TESTS") != "1",
    reason="Live RAG smoke test requires cloud credentials and RUN_LIVE_RAG_TESTS=1.",
)
def test_live_rag_smoke():
    from main import get_rag_orchestrator

    orchestrator = get_rag_orchestrator()

    response = orchestrator.ask("How do I clean the brew group?")

    assert response["answer"].strip()
    assert response["agent_trace"] == [
        "retrieval",
        "context_analysis",
        "synthesis",
        "quality_control",
    ]
