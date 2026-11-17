from pinecone_client import VectorDBClient, cosine_similarity
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def test_cosine_similarity_scores_identical_vectors_highest():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0


def test_mock_search_returns_nearest_vectors_by_score(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.upsert([
        {"id": "far", "values": [0.0, 1.0], "metadata": {"type": "text"}},
        {"id": "near", "values": [1.0, 0.0], "metadata": {"type": "text"}},
    ])

    result = client.search([1.0, 0.0], top_k=1)

    assert result[0]["id"] == "near"


def test_mock_search_by_type_filters_matches(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.upsert([
        {"id": "image", "values": [1.0, 0.0], "metadata": {"type": "image"}},
        {"id": "text", "values": [1.0, 0.0], "metadata": {"type": "text"}},
    ])

    result = client.search_by_type([1.0, 0.0], content_type="text", top_k=5)

    assert [match["id"] for match in result] == ["text"]


@pytest.mark.parametrize("metadata_filter,expected", [
    ({"author": "A"}, ["one"]),
    ({"page": {"$gte": 3}}, ["two"]),
    ({"author": {"$in": ["B"]}}, ["two"]),
    ({"keywords": {"$in": ["coffee"]}}, ["one"]),
    ({"author": {"$ne": "A"}}, ["two"]),
    ({"author": {"$nin": ["A"]}}, ["two"]),
    ({"unknown": {"$exists": False}}, ["one", "two"]),
    ({"$or": [{"author": "A"}, {"page": 3}]}, ["one", "two"]),
    ({"type": "image"}, []),
])
def test_mock_metadata_filters(monkeypatch, metadata_filter, expected):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.upsert([
        {"id": "one", "values": [1, 0], "metadata": {"type": "text", "author": "A", "page": 1, "keywords": ["coffee"]}},
        {"id": "two", "values": [1, 0], "metadata": {"type": "text", "author": "B", "page": 3}},
        {"id": "image", "values": [1, 0], "metadata": {"type": "image", "author": "A", "page": 1}},
    ])
    assert [match["id"] for match in client.search_by_type([1, 0], "text", metadata_filter=metadata_filter)] == expected


def test_real_query_combines_type_without_allowing_override(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.index = Mock()
    client.index.query.return_value = SimpleNamespace(matches=[])
    client.search_by_type([1, 0], "text", metadata_filter={"type": "image"})
    assert client.index.query.call_args.kwargs["filter"] == {"$and": [{"type": "text"}, {"type": "image"}]}


def test_mock_upserts_replace_existing_id(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.upsert([{"id": "one", "values": [1, 0]}])
    client.upsert([{"id": "one", "values": [0, 1]}])
    assert client.mock_db == [{"id": "one", "values": [0, 1]}]


def test_dimension_mismatch_preserves_existing_index(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.pc = Mock()
    client.pc.list_indexes.return_value = [SimpleNamespace(name=client.index_name)]
    client.pc.describe_index.return_value = SimpleNamespace(dimension=client.dimension + 1)
    with pytest.raises(ValueError, match="new index"):
        client._ensure_index()
    client.pc.delete_index.assert_not_called()


def test_invalid_filters_fail_for_mock_and_real_queries(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    with pytest.raises(ValueError, match="Unsupported"):
        client.search_by_type([1, 0], "text", metadata_filter={"page": {"$bad": 1}})


def test_new_index_uses_configured_dimensions_and_waits_for_readiness(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.pc = Mock()
    client.pc.list_indexes.return_value = []
    client.pc.describe_index.side_effect = [
        SimpleNamespace(dimension=client.dimension, status={"ready": False}),
        SimpleNamespace(dimension=client.dimension, status={"ready": True}),
    ]
    monkeypatch.setattr("pinecone_client.time.sleep", lambda seconds: None)
    client._ensure_index()
    assert client.pc.create_index.call_args.kwargs["dimension"] == client.dimension
    assert client.pc.describe_index.call_count == 2


def test_index_readiness_timeout_does_not_connect(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.pc = Mock()
    client.pc.list_indexes.return_value = [SimpleNamespace(name=client.index_name)]
    client.pc.describe_index.return_value = SimpleNamespace(dimension=client.dimension, status={"ready": False})
    times = iter([0, 121])
    monkeypatch.setattr("pinecone_client.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="not ready"):
        client._ensure_index()
    client.pc.Index.assert_not_called()


@pytest.mark.parametrize("metadata_filter", [
    {"$or": []}, {"$unknown": []}, {"page": {"$gt": "three"}},
    {"page": {"$in": 3}}, {"author": {"$eq": []}},
    {"author": {"$exists": "yes"}}, {"author": []},
])
def test_invalid_filter_shapes_are_rejected_before_network(monkeypatch, metadata_filter):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    client = VectorDBClient()
    client.index = Mock()
    with pytest.raises(ValueError):
        client.search_by_type([1, 0], "text", metadata_filter=metadata_filter)
    client.index.query.assert_not_called()
