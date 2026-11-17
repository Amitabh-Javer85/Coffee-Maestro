import logging
import re
import sys

import pytest
from fastapi.testclient import TestClient

from error_logging import configure_error_logging
import main


@pytest.fixture
def error_log(tmp_path):
    root = logging.getLogger()
    uvicorn = logging.getLogger("uvicorn")
    originals = [h for h in root.handlers if getattr(h, "coffee_error_log", False)]
    for handler in originals:
        root.removeHandler(handler)
        uvicorn.removeHandler(handler)
    old_hook = sys.excepthook
    path = configure_error_logging(tmp_path / "Logs")
    try:
        yield path
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "coffee_error_log", False):
                root.removeHandler(handler)
                uvicorn.removeHandler(handler)
                handler.close()
        for handler in originals:
            root.addHandler(handler)
            uvicorn.addHandler(handler)
        sys.excepthook = old_hook


def test_errors_append_with_entry_and_filename_timestamps(error_log):
    logger = logging.getLogger("backend")
    logger.error("first failure")
    logger.error("second failure")
    logger.info("not an error")
    assert re.fullmatch(r"errors_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}\.log", error_log.name)
    text = error_log.read_text(encoding="utf-8")
    assert "first failure" in text and "second failure" in text
    assert "not an error" not in text
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    assert configure_error_logging(error_log.parent) == error_log


def test_failed_ask_writes_exception_to_error_log(error_log, monkeypatch):
    def unavailable():
        raise RuntimeError("secret-question private-evidence api-key-do-not-log")

    monkeypatch.setattr(main, "get_rag_orchestrator", unavailable)
    main.app.dependency_overrides[main.get_api_key] = lambda: "test"
    try:
        response = TestClient(main.app).post("/api/ask", json={"query": "water hardness"})
    finally:
        main.app.dependency_overrides.pop(main.get_api_key, None)
    assert response.status_code == 502
    text = error_log.read_text(encoding="utf-8")
    assert "ask_failed" in text
    assert "Traceback" in text
    assert "RuntimeError" in text
    assert "provider=" in text
    assert response.headers["x-correlation-id"] in text
    assert "secret-question" not in text
    assert "private-evidence" not in text
    assert "api-key-do-not-log" not in text
    assert "water hardness" not in text
