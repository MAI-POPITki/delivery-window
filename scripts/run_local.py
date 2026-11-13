"""Run the working ZXC checkout with automatic refresh and an optional browser."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
    if branch.lower() != 'zxc':
        parser.error(f'Expected ZXC, found {branch!r}. Switch your checkout to ZXC first.')
    if importlib.util.find_spec('streamlit') is None:
        parser.error(f'Streamlit is missing. Install requirements.txt using this Python: {sys.executable}')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    url = f'http://localhost:{args.port}'
    print(f'Local preview: {branch} / {commit}\n{url}\nEdits reload automatically. Stop: Ctrl+C.', flush=True)
    env = {**os.environ, 'OKNO_LOCAL_PREVIEW': '1', 'PYTHONUNBUFFERED': '1'}
    command = [sys.executable, '-m', 'streamlit', 'run', str(ROOT/'app/dashboard.py'),
               '--server.address', args.host, '--server.port', str(args.port),
               '--server.runOnSave', 'true', '--server.fileWatcherType', 'poll',
               '--server.headless', 'true', '--browser.gatherUsageStats', 'false']
    process = subprocess.Popen(command, cwd=ROOT, env=env)

    def open_when_ready():
        for _ in range(120):
            if process.poll() is not None:
                return
            try:
                with urlopen(url+'/_stcore/health', timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(url)
                        return
            except OSError:
                pass
            time.sleep(.5)

    if not args.no_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
