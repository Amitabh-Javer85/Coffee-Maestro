"""Offline writer races, failure recovery, and generation image isolation."""

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from urllib.parse import quote
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import main
from agents.multi_agent_rag import ContextAnalysisAgent, EvidenceChunk


@contextmanager
def running(call, release=None):
    """Capture worker failures and always join; no timing-based race sleeps."""
    result = {}

    def target():
        try:
            result['value'] = call()
        except BaseException as exc:
            result['error'] = exc

    thread = Thread(target=target, daemon=True)
    thread.start()
    try:
        yield result
    finally:
        thread.join(timeout=5)
        overdue = thread.is_alive()
        if overdue and release is not None:
            release.set()
            thread.join(timeout=5)
        assert not thread.is_alive(), 'Mutation worker did not terminate.'
        assert not overdue, 'Competing mutation blocked instead of returning immediately.'


@pytest.fixture
def environment(monkeypatch, tmp_path):
    import document_processor

    state = SimpleNamespace(
        processed=[], embedded=[], outputs=[], stored={}, events=[],
        entered=Event(), release=Event(), pause=None, pause_owner=None, error=None,
    )

    def checkpoint(stage):
        if state.pause == stage and state.pause_owner in (None, get_ident()):
            state.pause_owner = get_ident()
            state.entered.set()
            if not state.release.wait(timeout=5):
                raise TimeoutError(f'Test did not release {stage}.')
        if state.error == stage:
            raise RuntimeError(f'Failure during {stage}')

    class Processor:
        def __init__(self, output_image_dir=None):
            self.output = Path(output_image_dir or main.settings.extracted_images_dir)

        def process(self, filepath):
            name = Path(filepath).stem
            state.processed.append(name)
            self.output.mkdir(parents=True, exist_ok=True)
            image = self.output / 'page_1.png'
            image.write_bytes(name.encode())
            state.outputs.append(image)
            checkpoint('extract')
            return SimpleNamespace(
                metadata={'source_filename': Path(filepath).name},
                chunks=[
                    {'type': 'text', 'page': 1, 'chunk_index': 0, 'content': f'{name} first'},
                    {'type': 'text', 'page': 1, 'chunk_index': 1, 'content': f'{name} second'},
                    {'type': 'image', 'page': 1, 'chunk_index': 2, 'content': str(image)},
                ],
            )

    class Embedder:
        def embed_text(self, content):
            state.embedded.append(content)
            checkpoint('embed')
            return [1.0, 0.0]

        def embed_image(self, path):
            return [0.0, 1.0]

    class Database:
        def delete_all(self):
            state.events.append('delete')
            checkpoint('delete')
            state.stored.clear()

        def upsert(self, vectors):
            state.events.append('upsert')
            for vector in vectors:
                state.stored[vector['id']] = vector
            checkpoint('upsert')

    monkeypatch.setattr(document_processor, 'DocumentProcessor', Processor)
    monkeypatch.setattr(main, 'get_embedder', lambda: Embedder())
    monkeypatch.setattr(main, 'get_vector_db', lambda: Database())
    monkeypatch.setattr(main.settings, 'extracted_images_dir', tmp_path)
    monkeypatch.setattr(main.settings, 'ingest_batch_size', 1)
    return state


@pytest.mark.parametrize('stage', ['extract', 'embed', 'upsert'])
def test_competing_ingest_is_rejected_before_any_work(environment, stage):
    state = environment
    state.pause = stage
    with running(lambda: main.ingest_pdf(Path('A.pdf'))) as first:
        try:
            assert state.entered.wait(timeout=5)
            before = (list(state.processed), list(state.embedded), list(state.events))
            with running(lambda: main.ingest_pdf(Path('B.pdf')), release=state.release) as second:
                pass
            assert isinstance(second.get('error'), main.DatasetBusyError)
            assert (state.processed, state.embedded, state.events) == before
        finally:
            state.pause = None
            state.release.set()
    assert 'error' not in first
    assert {v['metadata']['source_filename'] for v in state.stored.values()} == {'A.pdf'}
    assert len(state.stored) == 3


def test_reset_cannot_clear_an_ingestion(environment):
    state = environment
    state.pause = 'upsert'
    with running(lambda: main.ingest_pdf(Path('A.pdf'))) as first:
        try:
            assert state.entered.wait(timeout=5)
            with pytest.raises(main.DatasetBusyError):
                main.reset_dataset()
        finally:
            state.pause = None
            state.release.set()
    assert 'error' not in first
    assert state.events.count('delete') == 1


@pytest.mark.parametrize('competitor', ['ingest', 'reset'])
def test_reset_excludes_other_dataset_mutations(environment, competitor):
    state = environment
    state.pause = 'delete'
    with running(main.reset_dataset) as first:
        try:
            assert state.entered.wait(timeout=5)
            with pytest.raises(main.DatasetBusyError):
                if competitor == 'ingest':
                    main.ingest_pdf(Path('B.pdf'))
                else:
                    main.reset_dataset()
            assert state.processed == []
            assert state.events == ['delete']
        finally:
            state.pause = None
            state.release.set()
    assert first.get('value') == {'message': 'Index cleared successfully.'}


@pytest.mark.parametrize('stage', ['extract', 'embed', 'delete', 'upsert'])
def test_ingest_failure_releases_the_guard(environment, stage):
    environment.error = stage
    with pytest.raises(RuntimeError, match=stage):
        main.ingest_pdf(Path('A.pdf'))
    environment.error = None
    assert main.ingest_pdf(Path('B.pdf'))['upserted'] == 3


def test_reset_failure_releases_the_guard(environment):
    environment.error = 'delete'
    with pytest.raises(RuntimeError, match='delete'):
        main.reset_dataset()
    environment.error = None
    assert main.ingest_pdf(Path('B.pdf'))['upserted'] == 3


def test_each_ingest_uses_a_separate_image_generation(environment):
    main.ingest_pdf(Path('A.pdf'))
    first = environment.outputs[-1]
    main.ingest_pdf(Path('B.pdf'))
    second = environment.outputs[-1]
    assert first.parent != second.parent
    assert first.read_bytes() == b'A'
    assert second.read_bytes() == b'B'
    assert all(Path(v['metadata']['content']).parent == second.parent
               for v in environment.stored.values() if v['metadata']['type'] == 'image')


@pytest.mark.parametrize('stage', ['extract', 'embed'])
def test_failed_generation_preserves_old_vectors_and_images(environment, stage):
    main.ingest_pdf(Path('A.pdf'))
    original = dict(environment.stored)
    old_image = environment.outputs[-1]
    environment.error = stage
    with pytest.raises(RuntimeError, match=stage):
        main.ingest_pdf(Path('B.pdf'))
    assert environment.stored == original
    assert old_image.read_bytes() == b'A'


def test_generated_and_legacy_image_references_are_served(environment):
    main.ingest_pdf(Path('A.pdf'))
    image = environment.outputs[-1]
    root = main.settings.extracted_images_dir
    legacy = root / 'legacy.png'
    legacy.write_bytes(b'legacy image')
    # The app's mount was constructed at import time. A fresh app mounts the
    # real StaticFiles handler over this fixture's isolated root.
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    app.mount('/images', StaticFiles(directory=root), name='images')
    context = ContextAnalysisAgent().build_context([
        EvidenceChunk('image', 1, str(image)), EvidenceChunk('image', 1, str(legacy)),
    ])
    assert context.context_images == [image.relative_to(root).as_posix(), 'legacy.png']
    client = TestClient(app)
    for reference, expected in zip(context.context_images, [b'A', b'legacy image']):
        response = client.get('/images/' + quote(reference, safe='/'))
        assert response.status_code == 200
        assert response.content == expected


@pytest.fixture
def client(environment, monkeypatch, tmp_path):
    pdf = tmp_path / 'A.pdf'
    pdf.touch()
    monkeypatch.setattr(main, 'resolve_ingest_path', lambda value: pdf)
    main.app.dependency_overrides[main.get_api_key] = lambda: 'test-key'
    try:
        with TestClient(main.app) as client:
            yield client
    finally:
        main.app.dependency_overrides.clear()


@pytest.mark.parametrize('endpoint', ['/api/ingest', '/api/reset'])
def test_busy_api_returns_409_while_health_remains_responsive(client, environment, endpoint):
    environment.pause = 'upsert'
    with running(lambda: main.ingest_pdf(Path('A.pdf'))) as first:
        try:
            assert environment.entered.wait(timeout=5)
            response = client.post(endpoint, json={'confirm': True} if endpoint.endswith('reset') else {})
            assert response.status_code == 409
            assert 'retry' in response.json()['detail'].lower()
            assert client.get('/health').status_code == 200
        finally:
            environment.pause = None
            environment.release.set()
    assert 'error' not in first


def test_reset_database_work_runs_in_a_worker_thread(client, monkeypatch):
    threads = []
    database = Mock()
    database.delete_all.side_effect = lambda: threads.append(get_ident())
    monkeypatch.setattr(main, 'get_vector_db', lambda: database)
    event_loop_thread = client.portal.call(get_ident)
    assert client.post('/api/reset', json={'confirm': True}).status_code == 200
    assert threads and threads[0] != event_loop_thread


def test_reset_api_reports_database_errors_without_losing_the_guard(client, environment):
    environment.error = 'delete'
    assert client.post('/api/reset', json={'confirm': True}).status_code == 502
    environment.error = None
    assert client.post('/api/reset', json={'confirm': True}).status_code == 200


def test_constructed_pdf_ingests_and_serves_generation_images(tmp_path, monkeypatch):
    import pymupdf
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    root = tmp_path / 'images'
    monkeypatch.setattr(main.settings, 'extracted_images_dir', root)
    monkeypatch.setattr(main.settings, 'diagram_capture_pages', 1)
    monkeypatch.setattr(main.settings, 'min_chunk_size', 1)
    stored = []
    database = Mock()
    database.delete_all.side_effect = stored.clear
    database.upsert.side_effect = stored.extend
    embedder = Mock()
    embedder.embed_text.return_value = [1.0, 0.0]
    embedder.embed_image.return_value = [0.0, 1.0]
    monkeypatch.setattr(main, 'get_vector_db', lambda: database)
    monkeypatch.setattr(main, 'get_embedder', lambda: embedder)

    def pdf(name, text):
        filepath = tmp_path / name
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), text, fontsize=18)
            document.save(filepath)
        return filepath

    assert main.ingest_pdf(pdf('A.pdf', 'Original document instructions'))['failed'] == 0
    image_vector = next(vector for vector in stored if vector['metadata']['type'] == 'image')
    image = Path(image_vector['metadata']['content'])
    original_bytes = image.read_bytes()
    original_vectors = list(stored)
    embedder.embed_text.side_effect = RuntimeError('Model unavailable')
    with pytest.raises(RuntimeError, match='Model unavailable'):
        main.ingest_pdf(pdf('B.pdf', 'Replacement document instructions'))
    assert stored == original_vectors
    assert image.read_bytes() == original_bytes
    context = ContextAnalysisAgent().build_context([
        EvidenceChunk.from_match({'metadata': image_vector['metadata']}),
    ])
    app = FastAPI()
    app.mount('/images', StaticFiles(directory=root), name='images')
    response = TestClient(app).get('/images/' + quote(context.context_images[0], safe='/'))
    assert response.status_code == 200
    assert response.content == original_bytes
    assert response.headers['content-type'] == 'image/png'
