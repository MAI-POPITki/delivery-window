"""One background NASA browse collector per application process."""
import logging
import os
import threading

from . import daily_imagery

_lock = threading.Lock()
_worker = None
_stop = threading.Event()
log = logging.getLogger(__name__)


def _loop():
    while not _stop.is_set():
        try:
            if daily_imagery.due():
                daily_imagery.update()
        except Exception:
            log.exception("NASA daily imagery refresh failed; retaining previous pictures")
        _stop.wait(3600)


def start_auto_update():
    global _worker
    if os.environ.get("OKNO_DAILY_IMAGERY_AUTO_REFRESH", "1").lower() in ("0", "false", "off"):
        return None
    with _lock:
        if _worker is None or not _worker.is_alive():
            _stop.clear()
            _worker = threading.Thread(target=_loop, name="nasa-daily-imagery-update", daemon=True)
            _worker.start()
    return _worker
