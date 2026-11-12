"""Periodic browse-image refresh, including deployments from branch kirill."""
import logging
import os
import threading

from . import gaofen

_lock = threading.Lock()
_worker = None
log = logging.getLogger(__name__)


def _loop():
    while True:
        try:
            if gaofen.due():
                gaofen.update()
        except Exception:
            log.exception("Gaofen refresh failed; retaining previous browse images")
        # This runs in a daemon, never on the Streamlit rendering thread.
        threading.Event().wait(3600)


def start_auto_update():
    global _worker
    if os.environ.get("OKNO_GAOFEN_AUTO_REFRESH", "1").lower() in ("0", "false", "off"):
        return None
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="gaofen-browse-update", daemon=True)
            _worker.start()
    return _worker
