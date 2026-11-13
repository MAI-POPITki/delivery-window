"""Atomic flood-data publication and cross-process update locks."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {} if default is None else default


@contextmanager
def update_lock(path):
    """Kernel lock is released on process exit, including a crashed worker."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    acquired=False
    try:
        try:
            if os.name=='nt':
                import msvcrt
                if path.stat().st_size==0:
                    stream.write(b' ');stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError:
            yield False
            return
        stream.seek(0);stream.truncate()
        stream.write(datetime.now(timezone.utc).isoformat().encode());stream.flush()
        yield acquired
    finally:
        if acquired:
            if os.name=='nt':
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def version(paths):
    result=[]
    for path in paths:
        try:
            stat=Path(path).stat()
            result.append((str(path),stat.st_mtime_ns,stat.st_size))
        except OSError:
            result.append((str(path),0,0))
    return tuple(result)
