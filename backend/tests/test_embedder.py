"""Embedding contract checks without cloud calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from embedder import MultimodalEmbedder
import embedder


@pytest.fixture
def client():
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.25] * settings.embedding_dimensions)]
    )
    return client


def test_document_embedding_uses_configured_model_dimensions_and_instruction(client):
    MultimodalEmbedder(client=client).embed_text("Document passage")
    request = client.models.embed_content.call_args.kwargs
    assert request == {"model": settings.embedding_model, "contents": "title: none | text: Document passage", "config": {"output_dimensionality": settings.embedding_dimensions}}


def test_client_uses_the_embedding_location(monkeypatch, client):
    factory = Mock(return_value=client)
    monkeypatch.setattr(embedder.genai, "Client", factory)
    monkeypatch.setattr(settings, "embedding_location", "global")
    MultimodalEmbedder()
    assert factory.call_args.kwargs == {"vertexai": True, "project": settings.gcp_project_id, "location": "global"}


def test_query_embedding_uses_search_instruction(client):
    MultimodalEmbedder(client=client).embed_query("How to clean?")
    assert client.models.embed_content.call_args.kwargs["contents"] == "task: search result | query: How to clean?"


def test_image_embedding_passes_file_bytes_with_mime_type(tmp_path, client):
    image = tmp_path / "image.png"
    image.write_bytes(b"image bytes")
    MultimodalEmbedder(client=client).embed_image(image)
    part = client.models.embed_content.call_args.kwargs["contents"]
    assert part.inline_data.data == b"image bytes" and part.inline_data.mime_type == "image/png"


def test_text_embedding_returns_configured_vector_size(client):
    assert len(MultimodalEmbedder(client=client).embed_text("Document")) == settings.embedding_dimensions


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected_before_a_cloud_call(text, client):
    with pytest.raises(ValueError, match="Text must not be empty"):
        MultimodalEmbedder(client=client).embed_text(text)
    client.models.embed_content.assert_not_called()


def test_unknown_task_is_rejected(client):
    with pytest.raises(ValueError, match="Unsupported embedding task"):
        MultimodalEmbedder(client=client).embed_text("Text", task="invalid")


@pytest.mark.parametrize("embeddings", [[], [SimpleNamespace(values=[])], [SimpleNamespace(values=[1.0])]])
def test_missing_or_wrong_sized_vectors_are_rejected(client, embeddings):
    client.models.embed_content.return_value = SimpleNamespace(embeddings=embeddings)
    with pytest.raises(RuntimeError, match="embedding"):
        MultimodalEmbedder(client=client).embed_query("Query")


def test_nonfinite_vectors_are_rejected(client):
    client.models.embed_content.return_value.embeddings[0].values[0] = float("nan")
    with pytest.raises(RuntimeError, match="embedding"):
        MultimodalEmbedder(client=client).embed_query("Query")


def test_cloud_failures_propagate(client):
    client.models.embed_content.side_effect = ConnectionError("Cloud unavailable")
    with pytest.raises(ConnectionError, match="Cloud unavailable"):
        MultimodalEmbedder(client=client).embed_query("Query")


def test_missing_image_file_is_rejected(tmp_path, client):
    with pytest.raises(FileNotFoundError):
        MultimodalEmbedder(client=client).embed_image(tmp_path / "missing.png")


def test_nonstandard_image_encoding_is_converted_to_png(tmp_path, client):
    image = tmp_path / "image.jp2"
    with fitz.open() as document:
        document.new_page(width=10, height=10)
        image.write_bytes(document[0].get_pixmap().tobytes("png"))
    MultimodalEmbedder(client=client).embed_image(image)
    part = client.models.embed_content.call_args.kwargs["contents"]
    assert part.inline_data.mime_type == "image/png" and part.inline_data.data.startswith(b"\x89PNG")
