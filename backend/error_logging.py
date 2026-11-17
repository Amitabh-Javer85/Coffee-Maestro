"""Persistent error logs shared by desktop and backend entry points."""
from datetime import datetime
import logging
from pathlib import Path
import sys


def configure_error_logging(log_dir: Path | None = None) -> Path:
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "coffee_error_log", False):
            return Path(handler.baseFilename)

    if log_dir is None:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
        log_dir = base / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"errors_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.log"
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.coffee_error_log = True
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)

    # Uvicorn's parent logger stops propagation after applying its log config.
    logging.getLogger("uvicorn").addHandler(handler)
    previous_hook = sys.excepthook

    def log_uncaught(exc_type, value, traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("coffee.startup").critical(
                "Uncaught exception", exc_info=(exc_type, value, traceback)
            )
        previous_hook(exc_type, value, traceback)

    sys.excepthook = log_uncaught
    return path
