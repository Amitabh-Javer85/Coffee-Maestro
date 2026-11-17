"""Exercise startup without launching processes, browsers, or cloud calls."""
import importlib.util
from pathlib import Path
import runpy
import subprocess
import json
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_desktop_launcher_owns_the_single_browser_open(monkeypatch):
    # Supply the Windows flag when the process boundary is mocked on Linux.
    monkeypatch.setattr(subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False)
    monkeypatch.delenv("COFFEE_MAESTRO_PYTHON", raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("coffee_launcher", ROOT / "start_app.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with (
        patch.object(launcher.os.path, "exists", return_value=True),
        patch.object(launcher, "kill_port_owner"),
        patch.object(launcher, "wait_for_server", return_value=True),
        patch.object(launcher.subprocess, "Popen") as spawn,
        patch.object(launcher.webbrowser, "open") as browser,
    ):
        launcher.run_app()
    assert spawn.call_args.kwargs["env"]["COFFEE_MAESTRO_OPEN_BROWSER"] == "0"
    browser.assert_called_once_with("http://127.0.0.1:8000")


@pytest.mark.parametrize("source", ["file", "shell", "default"])
def test_launcher_runtime_selection_has_explicit_precedence(monkeypatch, tmp_path, source):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.delenv("COFFEE_MAESTRO_PYTHON", raising=False)
    spec = importlib.util.spec_from_file_location("coffee_launcher_selection", ROOT / "start_app.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    if source != "default":
        (tmp_path / "launcher.runtime.json").write_text(json.dumps({"python_executable": "file-python.exe"}), encoding="utf-8")
    if source == "shell":
        monkeypatch.setenv("COFFEE_MAESTRO_PYTHON", "shell-python.exe")
    expected = {"file": "file-python.exe", "shell": "shell-python.exe", "default": None}[source]
    assert launcher.configured_backend_python(str(tmp_path)) == expected


@pytest.mark.parametrize("content", ['{"python_executable": ""}', '{"python_executable": 42}', '[]', '{broken'])
def test_launcher_rejects_malformed_runtime_selection(monkeypatch, tmp_path, content):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.delenv("COFFEE_MAESTRO_PYTHON", raising=False)
    spec = importlib.util.spec_from_file_location("coffee_launcher_invalid", ROOT / "start_app.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    (tmp_path / "launcher.runtime.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.configured_backend_python(str(tmp_path))


@pytest.mark.parametrize("browser_setting, expected_timers", [("0", 0), ("1", 1)])
def test_backend_respects_browser_ownership(monkeypatch, browser_setting, expected_timers):
    monkeypatch.setenv("COFFEE_MAESTRO_OPEN_BROWSER", browser_setting)
    with patch("threading.Timer") as timer, patch("uvicorn.run") as serve:
        runpy.run_path(str(ROOT / "backend/main.py"), run_name="__main__")
    browser_timers = [
        call for call in timer.call_args_list
        if len(call.args) > 1 and getattr(call.args[1], "__name__", "") == "open_browser"
    ]
    assert len(browser_timers) == expected_timers
    serve.assert_called_once()


@pytest.mark.parametrize("exists", [True, False])
def test_launcher_honors_explicit_interpreter_before_touching_port(monkeypatch, tmp_path, exists):
    # Supply the Windows flag when the process boundary is mocked on Linux.
    monkeypatch.setattr(subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    interpreter = tmp_path / "Python runtime" / "python.exe"
    if exists:
        interpreter.parent.mkdir()
        interpreter.touch()
    monkeypatch.setenv("COFFEE_MAESTRO_PYTHON", str(interpreter))
    spec = importlib.util.spec_from_file_location("coffee_launcher_override", ROOT / "start_app.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with (
        patch.object(launcher.os.path, "exists", return_value=True),
        patch.object(launcher, "kill_port_owner") as kill,
        patch.object(launcher, "wait_for_server", return_value=True),
        patch.object(launcher.subprocess, "Popen") as spawn,
        patch.object(launcher.webbrowser, "open"),
    ):
        launcher.run_app()
    if exists:
        assert spawn.call_args.args[0][0] == str(interpreter)
    else:
        spawn.assert_not_called()
        kill.assert_not_called()
