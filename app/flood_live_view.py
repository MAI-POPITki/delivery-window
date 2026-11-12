"""Fresh flood observations and transparent update state in all flood screens."""
from datetime import datetime, timezone

import streamlit as st

from okno import flood_refresh as R, flood_observations as O, flood_m2 as S

NAMES={'levels':'Уровни гидропоста','glofas':'Внешний прогноз GloFAS','satellite':'Снимки Sentinel-2'}
STATES={'ok':'данные получены','no_data':'новых измерений источник не выдал',
        'error':'обновление не удалось, сохранены прежние данные','running':'обновление выполняется'}


def human(value):
    if not value:return 'ещё не выполнялось'
    return datetime.fromisoformat(value.replace('Z','+00:00')).strftime('%d.%m.%Y %H:%M UTC')


@st.fragment(run_every='30s')
def watch_changes(pid):
    current=R.data_version(pid)
    key='flood_live_version_'+pid
    previous=st.session_state.get(key,current)
    st.session_state[key]=current
    if previous!=current:st.rerun()


def render_live(pid):
    R.start_auto_update(pid)
    watch_changes(pid)
    with st.expander('Актуальные данные и обновление',expanded=False):
        st.caption('Уровни и GloFAS проверяются раз в 6 часов, снимки — раз в 12 часов, '
                   'пока приложение работает. Новые данные появляются без перезапуска страницы. '
                   'Исторические расчёты ниже сохраняют выбранную дату.')
        if not R.enabled():st.caption('Автоматическая загрузка отключена в настройках этого запуска.')
        if st.button('Обновить данные участка',key='refresh_flood_'+pid,disabled=not R.enabled()):
            R.request_refresh(pid)
            st.info('Проверка источников запрошена. Можно продолжать работу; повторный запрос — не чаще раза в минуту.')
        state=O.summary(pid)
        latest=state['latest']
        if latest:
            date_text=(latest['observed_at'][:10]+' (время не указано)' if latest.get('observation_precision')=='day'
                       else human(latest['observed_at']))
            st.write(f"**Последний уровень: {latest['value_cm']:g} см над нулём поста** · {date_text}")
            st.markdown(f"[Источник измерения]({latest['source_url']})")
            st.caption(f"В последних 31 сутках перед этим замером: {state['days']}/31 дней с измерениями этого источника.")
            if state['age_days']>2:st.warning('Последний замер старше двух суток. Текущее состояние может отличаться.')
        else:st.info('Датированных текущих уровней пока нет. Отсутствие измерения не означает отсутствия паводка.')
        if state['continuous']:
            st.caption('Последний месяц наблюдений этого источника непрерывен. '
                       'Перед использованием в М1 нужно проверить совместимость нуля шкалы с обучающим архивом.')
        missing=[]
        if not state['continuous']:missing.append('непрерывные замеры за месяц')
        if not state['model_ready']:missing.append('проверенная версия модели для текущего года')
        if missing:
            st.caption('Для собственной М1 на сегодня ещё нужны: '+', '.join(missing)+
                       '. Свежий уровень не подменяется архивным прогнозом.')
        rows=[]
        for kind,status in R.statuses(pid).items():
            rows.append({'Данные':NAMES[kind],'Состояние':STATES.get(status.get('state'),'ожидается первая проверка'),
                         'Последняя проверка':human(status.get('checked_at')),
                         'Последняя успешная загрузка':human(status.get('last_success_at'))})
        st.dataframe(rows,hide_index=True,width='stretch',key='flood_refresh_status_'+pid)
        st.caption('Дата проверки не заменяет дату замера или съёмки. GloFAS показывает расход в м³/с, '
                   'гидропост — уровень в см; спутник наблюдает поверхность. Эти величины не взаимозаменяемы.')
