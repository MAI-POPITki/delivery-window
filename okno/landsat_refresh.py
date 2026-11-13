"""One non-blocking Landsat collector per Streamlit process, including kirill."""
import logging
import os
import threading

from . import landsat

_lock = threading.Lock()
_worker = None
_stop = threading.Event()
log = logging.getLogger(__name__)


def _loop():
    while not _stop.is_set():
        try:
            if landsat.due():
                landsat.update()
        except Exception:
            log.exception("Landsat update failed; retaining previous observations")
        _stop.wait(3600)


def start_auto_update():
    global _worker
    if os.environ.get("OKNO_LANDSAT_AUTO_REFRESH", "1").lower() in ("0", "false", "off"):
        return None
    with _lock:
        if _worker is None or not _worker.is_alive():
            _stop.clear()
            _worker = threading.Thread(target=_loop, name="landsat-m2-update", daemon=True)
            _worker.start()
    return _worker
