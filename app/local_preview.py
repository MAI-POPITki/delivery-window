"""Development status for the opt-in local launcher; no deployment assumptions."""
import os
from pathlib import Path
import subprocess

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def render_local_preview():
    if os.environ.get('OKNO_LOCAL_PREVIEW') != '1':
        return
    try:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(ROOT), *args],
                                           stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
        branch = git('branch', '--show-current')
        commit = git('rev-parse', '--short', 'HEAD')
        changed = bool(git('status', '--porcelain'))
        st.sidebar.caption(f'Локальный просмотр · {branch} · {commit}' +
                           (' · есть изменения' if changed else ''))
    except (OSError, subprocess.SubprocessError):
        st.sidebar.caption('Локальный просмотр · версия Git недоступна')
    st.sidebar.caption('Сохранённые правки кода обновляют страницу автоматически.')
    st.sidebar.button('Перечитать данные', key='local_preview_reload', on_click=st.cache_data.clear)
