from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import main

class FakeEmbedder:
    def embed_text(self, text):
        return [1.0, 0.0]
    def embed_image(self, path):
        return [0.0, 1.0]

def chunk(index=0, content_type='text'):
    return {'type': content_type, 'page': 2, 'content': 'Clean weekly.', 'chunk_index': index,
            'headings': {'h1': 'Maintenance'}, 'section_path': 'Maintenance'}

def test_build_vector_propagates_document_and_heading_metadata():
    vector = main.build_vector(chunk(), FakeEmbedder(), {'source_filename': 'manual.pdf', 'author': 'Author'})
    assert vector['metadata'] == {
        'type': 'text', 'page': 2, 'content': 'Clean weekly.', 'chunk_index': 0,
        'source_filename': 'manual.pdf', 'author': 'Author', 'title': '', 'keywords': '',
        'subject': '', 'creator': '', 'creationDate': '', 'modDate': '',
        'section_path': 'Maintenance', 'h1': 'Maintenance', 'h2': '', 'h3': ''}

def test_vector_ids_are_stable_and_distinguish_sources_types_and_chunks():
    keys = [('a.pdf', 0, 'text'), ('b.pdf', 0, 'text'), ('a.pdf', 1, 'text'), ('a.pdf', 0, 'image')]
    ids = [main._vector_id(*key) for key in keys]
    assert len(set(ids + [main._vector_id(*keys[0])])) == 4

def test_build_vector_routes_images_to_image_embedding():
    assert main.build_vector(chunk(content_type='image'), FakeEmbedder(), {})['values'] == [0.0, 1.0]

def test_build_vector_rejects_empty_embeddings():
    with pytest.raises(ValueError, match='empty vector'):
        main.build_vector(chunk(), Mock(embed_text=Mock(return_value=[])), {})

@pytest.fixture
def ingest_dependencies(monkeypatch, tmp_path):
    import document_processor
    processed = SimpleNamespace(chunks=[chunk(0), chunk(1)], metadata={'source_filename': 'manual.pdf'})
    monkeypatch.setattr(document_processor.DocumentProcessor, 'process', lambda self, path: processed)
    embedder, db = FakeEmbedder(), Mock()
    monkeypatch.setattr(main, 'get_embedder', lambda: embedder)
    monkeypatch.setattr(main, 'get_vector_db', lambda: db)
    monkeypatch.setattr(main.settings, 'ingest_batch_size', 1)
    monkeypatch.setattr(main.settings, 'extracted_images_dir', tmp_path)
    return processed, embedder, db

def test_ingest_clears_then_upserts_configured_batches(ingest_dependencies):
    _, _, db = ingest_dependencies
    main.ingest_pdf(Path('manual.pdf'))
    assert [call[0] for call in db.method_calls] == ['delete_all', 'upsert', 'upsert']

def test_ingest_returns_document_metadata(ingest_dependencies):
    assert main.ingest_pdf(Path('manual.pdf'))['metadata'] == {'source_filename': 'manual.pdf'}

def test_ingest_preserves_existing_data_when_embedding_fails(ingest_dependencies, monkeypatch):
    _, embedder, db = ingest_dependencies
    monkeypatch.setattr(embedder, 'embed_text', Mock(side_effect=[[1.0, 0.0], RuntimeError('quota')]))
    with pytest.raises(RuntimeError, match='quota'):
        main.ingest_pdf(Path('manual.pdf'))
    db.delete_all.assert_not_called()

def test_ingest_preserves_existing_data_when_document_is_empty(ingest_dependencies):
    processed, _, db = ingest_dependencies
    processed.chunks = []
    with pytest.raises(ValueError, match='no chunks'):
        main.ingest_pdf(Path('manual.pdf'))
    db.delete_all.assert_not_called()

def test_ingest_propagates_upsert_failure(ingest_dependencies):
    _, _, db = ingest_dependencies
    db.upsert.side_effect = RuntimeError('network')
    with pytest.raises(RuntimeError, match='network'):
        main.ingest_pdf(Path('manual.pdf'))
