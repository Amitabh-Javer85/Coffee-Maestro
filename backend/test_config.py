from pathlib import Path

import pytest

from config import PROJECT_ROOT, resolve_ingest_path, settings, Settings
import main


def test_resolve_ingest_path_uses_default_pdf_when_no_path_is_sent():
    assert resolve_ingest_path(None) == settings.default_pdf_path.resolve()


def test_resolve_ingest_path_accepts_project_relative_pdf():
    assert resolve_ingest_path("../Saeco GranAroma 6850.pdf") == (PROJECT_ROOT / "Saeco GranAroma 6850.pdf").resolve()


def test_resolve_ingest_path_rejects_non_pdf_files():
    with pytest.raises(ValueError, match="Only PDF files"):
        resolve_ingest_path("main.py")


def test_resolve_ingest_path_rejects_paths_outside_workspace():
    outside_path = Path.home() / "outside.pdf"

    with pytest.raises(ValueError, match="inside the project workspace"):
        resolve_ingest_path(str(outside_path))


def test_get_gemini_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(main.settings, "google_api_key", None)
    monkeypatch.setattr(main, "_gemini", None)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        main.get_gemini()


@pytest.mark.parametrize("name,value", [("embedding_dimensions", 0), ("ingest_batch_size", 0), ("context_snippet_max_length", -1), ("text_top_k", -1), ("diagram_capture_pages", -1)])
def test_invalid_numeric_settings_fail_early(monkeypatch, name, value):
    monkeypatch.setattr(Settings, name, value)
    with pytest.raises(ValueError, match=name):
        Settings()


@pytest.mark.parametrize("overlap", [-1, 1500, 1501])
def test_chunk_overlap_must_fit_within_chunk_size(monkeypatch, overlap):
    monkeypatch.setattr(Settings, "chunk_size", 1500)
    monkeypatch.setattr(Settings, "chunk_overlap", overlap)
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings()


def test_minimum_chunk_size_cannot_exceed_chunk_limit(monkeypatch):
    monkeypatch.setattr(Settings, "min_chunk_size", settings.chunk_size + 1)
    with pytest.raises(ValueError, match="min_chunk_size"):
        Settings()


@pytest.fixture
def conversation_settings(monkeypatch):
    values = {
        "answer_provider": "vertex_conversation", "google_api_key": None,
        "conversation_project_id": None, "gcp_project_id": "test-project",
        "conversation_location": "us-central1", "conversation_agent_id": "agent-id",
        "conversation_environment_id": "environment-id", "conversation_language_code": "en",
        "conversation_event_name": "RAG_ANSWER", "conversation_timeout_seconds": "30",
        "conversation_max_payload_bytes": "16384",
    }
    for name, value in values.items():
        monkeypatch.setattr(Settings, name, value)
    return Settings()


def test_conversation_uses_project_fallback_without_gemini_key(conversation_settings):
    conversation_settings.validate_answer_provider()
    assert conversation_settings.conversation_project_id == "test-project"
    assert conversation_settings.conversation_timeout_seconds == 30.0
    assert conversation_settings.conversation_max_payload_bytes == 16384


def test_explicit_conversation_project_wins(conversation_settings):
    conversation_settings.conversation_project_id = "dedicated-project"
    conversation_settings.validate_answer_provider()
    assert conversation_settings.conversation_project_id == "dedicated-project"


@pytest.mark.parametrize("name", ["conversation_project_id", "conversation_location", "conversation_agent_id", "conversation_environment_id", "conversation_language_code", "conversation_event_name"])
def test_conversation_requires_explicit_identifiers(conversation_settings, name):
    setattr(conversation_settings, name, "")
    if name == "conversation_project_id":
        conversation_settings.gcp_project_id = None
    with pytest.raises(ValueError, match=name.upper()):
        conversation_settings.validate_answer_provider()


@pytest.mark.parametrize("name,value", [
    ("conversation_timeout_seconds", "bad"), ("conversation_timeout_seconds", 0),
    ("conversation_timeout_seconds", "nan"), ("conversation_timeout_seconds", "inf"),
    ("conversation_max_payload_bytes", "bad"), ("conversation_max_payload_bytes", 0),
])
def test_conversation_requires_positive_finite_bounds(conversation_settings, name, value):
    setattr(conversation_settings, name, value)
    with pytest.raises(ValueError, match="positive"):
        conversation_settings.validate_answer_provider()


def test_gemini_ignores_unused_conversation_configuration(conversation_settings):
    conversation_settings.answer_provider = "gemini"
    conversation_settings.google_api_key = "test-key"
    conversation_settings.conversation_timeout_seconds = "bad"
    conversation_settings.conversation_location = None
    conversation_settings.validate_answer_provider()


def test_selected_gemini_requires_api_key(conversation_settings):
    conversation_settings.answer_provider = "gemini"
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        conversation_settings.validate_answer_provider()


def test_unknown_answer_provider_fails_explicitly(conversation_settings):
    conversation_settings.answer_provider = "typo"
    with pytest.raises(ValueError, match="ANSWER_PROVIDER"):
        conversation_settings.validate_answer_provider()
