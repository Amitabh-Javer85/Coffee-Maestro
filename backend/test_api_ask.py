import unittest
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

import main


class FakeOrchestrator:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def ask(self, query, metadata_filter=None):
        if self.should_fail:
            raise RuntimeError("model unavailable")
        return {
            "answer": f"Answer for {query}",
            "context_images": ["page_1.png"],
            "citations": [{"page": 1, "type": "text", "score": 0.9}],
            "quality_warnings": [],
            "agent_trace": ["retrieval", "context_analysis", "synthesis", "quality_control"],
        }


class AskRouteTests(unittest.TestCase):
    def setUp(self):
        main.app.dependency_overrides[main.get_api_key] = lambda: "test-key"

    def tearDown(self):
        main.app.dependency_overrides.clear()

    def test_ask_route_preserves_response_contract(self):
        client = TestClient(main.app)

        with patch.object(
            main, "get_rag_orchestrator", return_value=FakeOrchestrator()
        ):
            response = client.post(
                "/api/ask",
                headers={"X-API-Key": "super-secret-key-123"},
                json={"query": "brew group"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Answer for brew group")
        self.assertEqual(response.json()["context_images"], ["page_1.png"])
        self.assertEqual(response.json()["agent_trace"][0], "retrieval")

    def test_ask_route_returns_502_when_orchestrator_fails(self):
        client = TestClient(main.app)

        with patch.object(
            main, "get_rag_orchestrator", return_value=FakeOrchestrator(should_fail=True)
        ):
            response = client.post(
                "/api/ask",
                headers={"X-API-Key": "super-secret-key-123"},
                json={"query": "brew group"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("model unavailable", response.json()["detail"])
        self.assertIn(response.headers["x-correlation-id"], response.json()["detail"])

    def test_ask_route_rejects_missing_api_key(self):
        main.app.dependency_overrides.clear()
        client = TestClient(main.app)

        response = client.post("/api/ask", json={"query": "brew group"})

        self.assertEqual(response.status_code, 403)

    def test_remote_peer_cannot_spoof_loopback_with_host_header(self):
        main.app.dependency_overrides.clear()
        client = TestClient(main.app, base_url="http://localhost", client=("192.0.2.10", 3000))
        response = client.post("/api/ask", json={"query": "fixture question"})
        self.assertEqual(response.status_code, 403)

    def test_reset_requires_confirmation(self):
        client = TestClient(main.app)

        response = client.post(
            "/api/reset",
            headers={"X-API-Key": "super-secret-key-123"},
            json={"confirm": False},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("confirm=true", response.json()["detail"])

    def test_ask_route_forwards_metadata_filter(self):
        client = TestClient(main.app)
        orchestrator = Mock()
        orchestrator.ask.return_value = {"answer": "Filtered answer"}
        with patch.object(main, "get_rag_orchestrator", return_value=orchestrator):
            response = client.post("/api/ask", json={"query": "brew group", "metadata_filter": {"author": "A"}})
        self.assertEqual(response.status_code, 200)
        orchestrator.ask.assert_called_once_with("brew group", metadata_filter={"author": "A"})


if __name__ == "__main__":
    unittest.main()
