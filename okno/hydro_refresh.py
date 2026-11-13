"""One periodic collector per running application process, with a file lock.

GitHub cron only runs a repository's default branch. The app worker therefore
also updates kirill deployments without requiring any change to main.
"""
from datetime import datetime, timedelta, timezone
import logging
import os
import threading

from . import official_hydro as H

INTERVAL = timedelta(hours=6)
_lock = threading.Lock()
_worker = None
_stop = threading.Event()
log = logging.getLogger(__name__)


def due(snapshot, now=None):
    now = datetime.now(timezone.utc) if now is None else H.stamp(now)
    try:
        return now-H.stamp(snapshot["checked_at"]) >= INTERVAL
    except (KeyError, ValueError, TypeError):
        return True


def _loop():
    while not _stop.is_set():
        try:
            if due(H.load_snapshot()):
                from .hydro_ingest import update
                update()
        except Exception:
            # A remote source must never bring down the dashboard or replace
            # the last good data with invented defaults.
            log.exception("Official hydrological refresh failed; keeping last snapshot")
        try:
            from .ykut_phenology import due as phenology_due, update as phenology_update
            if phenology_due():
                phenology_update()
        except Exception:
            log.exception("Yakutia phenology refresh failed; keeping last CSV")
        _stop.wait(3600)


def start_auto_update():
    """Non-blocking; opt out for deterministic offline tests and local research."""
    global _worker
    if os.environ.get("OKNO_HYDRO_AUTO_REFRESH", "1").lower() in ("0", "false", "off"):
        return None
    with _lock:
        if _worker is None or not _worker.is_alive():
            _stop.clear()
            _worker = threading.Thread(target=_loop, name="official-hydro-update", daemon=True)
            _worker.start()
    return _worker
