"""Nonblocking flood refresh, per-source TTLs, locks and retained last good data."""
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import threading

import requests
import pandas as pd

from . import flood as F, flood_m2 as S, flood_observations as O
from .flood_sites import SITES
from .flood_store import atomic_json, read_json, update_lock, version

DIRECTORY=F.DATA/'refresh'
INTERVALS={'levels':timedelta(hours=6),'glofas':timedelta(hours=6),'satellite':timedelta(hours=12)}
_lock=threading.Lock()
_worker=None
_wake=threading.Event()
_points=set()
_requested=set()
log=logging.getLogger(__name__)


def enabled():
    return os.environ.get('OKNO_FLOOD_AUTO_REFRESH',os.environ.get('OKNO_HYDRO_AUTO_REFRESH','1')).lower() not in ('0','false','off')


def kinds(pid):
    if pid not in SITES:raise ValueError('Неизвестный участок')
    return ['levels',*(['glofas'] if pid in F.POINTS else []),'satellite']


def status_path(kind,pid,directory=DIRECTORY):
    if kind not in kinds(pid):raise ValueError('Источник не поддерживается для участка')
    return Path(directory)/f'{kind}-{pid}.json'


def statuses(pid,directory=DIRECTORY):
    return {kind:read_json(status_path(kind,pid,directory)) for kind in kinds(pid)}


def due(kind,status,now):
    # refresh_one checks this after acquiring the kernel lock: a "running"
    # marker then belongs to an interrupted process, not an active writer.
    if status.get('state')=='running':return True
    try:
        checked=datetime.fromisoformat(status['checked_at'])
        if checked.tzinfo is None:return True
    except (KeyError,TypeError,ValueError):return True
    interval=timedelta(minutes=30) if status.get('state')=='error' else INTERVALS[kind]
    return now-checked>=interval


def refresh_glofas(pid,now=None):
    live_clock=now is None
    now=now or datetime.now(timezone.utc)
    point=F.POINTS[pid]
    response=requests.get(F.API,params=dict(latitude=point['lat'],longitude=point['lon'],daily='river_discharge',
                          timezone='GMT',models='seamless_v4',forecast_days=8,ensemble='true'),timeout=(8,45))
    response.raise_for_status()
    received=datetime.now(timezone.utc) if live_clock else now
    payload=response.json()
    frame=F.daily_frame(payload,point)
    members=[c for c in frame if c.startswith('river_discharge_member')]
    days=pd.date_range(received.date()+timedelta(days=1),periods=7)
    complete=frame.reindex(days)[members].dropna(axis=1).columns
    if len(complete)<30:
        raise ValueError('Нет полного ансамбля на семь будущих суток')
    latest=F.DATA/'latest'/f'{pid}.json'
    archive=F.DATA/'runs'/received.date().isoformat()/f'{pid}.json'
    capture=archive.with_name(f"{pid}--{received:%H%M%S%f}.json")
    # Publish the historical capture first; latest is always a complete envelope.
    if not archive.exists():F.save_response(archive,payload,response.url,received.isoformat())
    if not capture.exists():F.save_response(capture,payload,response.url,received.isoformat())
    F.save_response(latest,payload,response.url,received.isoformat())
    return dict(state='ok',fetched_at=received.isoformat(),start=str(days[0].date()),end=str(days[-1].date()),
                members=len(complete),version=F.read_response(latest)['sha256'])


def refresh_satellite(pid,now):
    from scripts.download_flood_m2 import process
    meta=process(pid,(now-timedelta(days=14)).date().isoformat(),now.date().isoformat(),available_by=now.date().isoformat())
    return dict(state='ok',acquired_at=meta['acquired_at'],observed_fraction=meta['statistics']['observed_fraction'],
                version=meta['observation']['item_id'])


def refresh_one(kind,pid,*,force=False,now=None,directory=DIRECTORY):
    live_clock=now is None
    now=now or datetime.now(timezone.utc)
    path=status_path(kind,pid,directory)
    with update_lock(Path(directory)/'locks'/f'{kind}-{pid}.lock') as locked:
        if not locked:return 'already_running'
        previous=read_json(path)
        if not force and not due(kind,previous,now):return 'cached'
        atomic_json(path,{**previous,'state':'running','checked_at':now.isoformat()})
        try:
            source_now=None if live_clock else now
            result=(O.refresh(pid,now=source_now) if kind=='levels' else
                    refresh_glofas(pid,source_now) if kind=='glofas' else refresh_satellite(pid,now))
            completed=datetime.now(timezone.utc) if live_clock else now
            state=result.get('state','ok')
            successful=state=='ok'
            atomic_json(path,dict(kind=kind,pid=pid,state=state,checked_at=completed.isoformat(),
                        last_success_at=completed.isoformat() if successful else previous.get('last_success_at'),
                        result=result))
            return state
        except Exception as exc:
            # Raster clients may embed signed URLs in exceptions; never log them.
            log.warning('Flood refresh %s/%s failed: %s',kind,pid,type(exc).__name__)
            atomic_json(path,{**previous,'kind':kind,'pid':pid,'state':'error',
                              'checked_at':(datetime.now(timezone.utc) if live_clock else now).isoformat(),
                              'error':type(exc).__name__})
            return 'error'


def data_version(pid):
    paths=[*(status_path(kind,pid) for kind in kinds(pid)),O.DATA/f'{pid}.json',
           F.DATA/'latest'/f'{pid}.json',*S.available_runs(pid)]
    return version(paths)


def _loop():
    while True:
        with _lock:
            points=sorted(_points)
            requested=set(_requested)
            _requested.clear()
        for kind in INTERVALS:
            for pid in points:
                if kind in kinds(pid):
                    try:refresh_one(kind,pid,force=(kind,pid) in requested)
                    except Exception as exc:log.warning('Flood worker failed: %s',type(exc).__name__)
        _wake.wait(60)
        _wake.clear()


def start_auto_update(pid):
    global _worker
    kinds(pid)
    if not enabled():return None
    with _lock:
        added=pid not in _points
        _points.add(pid)
        if _worker is None or not _worker.is_alive():
            _worker=threading.Thread(target=_loop,name='flood-update',daemon=True)
            _worker.start()
        if added:_wake.set()
    return _worker


def request_refresh(pid):
    if not enabled():return False
    start_auto_update(pid)
    with _lock:
        # Manual requests share the same locks and a one-minute anti-repeat delay.
        now=datetime.now(timezone.utc)
        for kind,state in statuses(pid).items():
            try:age=now-datetime.fromisoformat(state['checked_at'])
            except (KeyError,ValueError):age=timedelta(days=1)
            if age>=timedelta(minutes=1) and state.get('state')!='running':_requested.add((kind,pid))
    _wake.set()
    return True
