"""Shared opt-in wind loading/unloading inputs for prospective M3 screens."""
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from okno import handling_wind as W

CAPACITY_LABELS={'capacity_before_wind_t':'Мощность до ограничений, т',
                 'wind_known':'Погодные данные для обеих операций полны',
                 'loading_wind_t':'Доступно на погрузке, т','unloading_wind_t':'Доступно на разгрузке, т'}


ROOT=Path(__file__).resolve().parents[1]
CUSTOM='Ввести координаты вручную'


@st.cache_data(ttl=1800,show_spinner=False)
def get_wind(lat,lon):
    return W.fetch(lat,lon)


@st.cache_data(show_spinner=False)
def known_sites():
    """Площадки, координаты которых уже есть в проекте: берега переправы Якутск — Нижний Бестях
    (концы её ледовой линии OSM) и точки путей зимников (гидропосты). Это берег или пост,
    а не адрес конкретного склада; прогноз погоды берётся по сетке модели, поэтому разница
    в несколько километров для ветра обычно несущественна, но точный склад лучше ввести вручную."""
    sites={}
    try:
        for seg in json.loads((ROOT/'data/osm/project_segments.json').read_text(encoding='utf-8')):
            if seg.get('point_id')=='yakutsk_crossing':
                (lon1,lat1),(lon2,lat2)=seg['coordinates'][0],seg['coordinates'][-1]
                sites['Нижний Бестях — берег ледовой переправы (OSM)']=(round(lat1,4),round(lon1,4))
                sites['Якутск — берег ледовой переправы (OSM)']=(round(lat2,4),round(lon2,4))
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        pass
    try:
        from okno.flood import POINTS as FLOOD
        for pid,m in FLOOD.items():
            sites.setdefault(f"{m['river']} — {m['name']} (участок паводков)",(round(m['lat'],4),round(m['lon'],4)))
    except Exception:
        pass
    try:
        from okno import forecast as FC
        from okno.regions import REGIONS, label, points_of
        for region in REGIONS:
            for pid in points_of(region):
                if pid=='yakutsk_crossing' or pid.startswith('lena_') or not FC._weather_path(pid).exists():
                    continue
                meta=FC.point_meta(pid)
                sites.setdefault(label(pid)+' — пост пути',(round(meta['lat'],4),round(meta['lon'],4)))
    except Exception:
        pass
    return sites


def site_for_point(point_id):
    """Название площадки по умолчанию для выбранного пути, если оно есть в справочнике."""
    if not point_id:
        return None
    if point_id=='yakutsk_crossing':
        return 'Нижний Бестях — берег ледовой переправы (OSM)'
    try:
        from okno.flood import POINTS as FLOOD
        if point_id in FLOOD:
            m=FLOOD[point_id]; return f"{m['river']} — {m['name']} (участок паводков)"
    except Exception:
        pass
    from okno.regions import label
    return label(point_id)+' — пост пути'


def _auto_fetch():
    return os.environ.get('OKNO_WIND_AUTO_FETCH','1').lower() not in ('0','false','off')


def render_wind(key,decision,days,arrival_days,default_point=None):
    """None = disabled; tuple of daily calendars = enabled, even if unknown."""
    with st.expander('Погрузка и разгрузка: погода и рабочее время',expanded=False):
        if not st.checkbox('Учитывать погоду и рабочее время',key=key+'_enabled'):return None
        st.caption('Прогноз на высоте 10 м, м/с. Пределы задаются по правилам оборудования/оператора. '
                   'При достижении предела операция исключается. Неизвестные часы не считаются пригодными.')
        as_of=datetime.now(timezone.utc)
        if pd.Timestamp(decision).date()!=as_of.date():
            as_of=(pd.Timestamp(decision,tz='UTC')+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)).to_pydatetime()
        tables=[]
        for kind,label,dates in [('load','Погрузка',days),('unload','Разгрузка',arrival_days)]:
            prefix=key+'_'+kind
            st.markdown('**'+label+'**')
            sites=known_sites()
            options=[CUSTOM]+list(sites)
            default=site_for_point(default_point) if kind=='load' else None
            if prefix+'_site' not in st.session_state and default in sites:
                st.session_state[prefix+'_site']=default
            site=st.selectbox('Площадка',options,key=prefix+'_site',
                              help='Начните вводить название. Координаты берега переправы или поста пути берутся из данных проекта; '
                                   'для точного склада выберите ручной ввод.')
            if site!=CUSTOM:
                st.session_state[prefix+'_lat'],st.session_state[prefix+'_lon']=sites[site]
            a,b=st.columns(2)
            lat=a.number_input('Широта терминала',min_value=-90.,max_value=90.,value=None,key=prefix+'_lat',disabled=site!=CUSTOM)
            lon=b.number_input('Долгота терминала',min_value=-180.,max_value=180.,value=None,key=prefix+'_lon',disabled=site!=CUSTOM)
            tz=a.text_input('Часовой пояс площадки (IANA)',value='Asia/Yakutsk',key=prefix+'_tz')
            source=b.text_input('Основание ограничений оборудования',key=prefix+'_basis')
            speed=a.number_input('Предельная скорость, м/с',min_value=.1,value=None,key=prefix+'_speed')
            gust=b.number_input('Предельные порывы, м/с',min_value=.1,value=None,key=prefix+'_gust')
            st.caption('Пределы ветра — из паспорта крана или погрузчика и инструкции по охране труда при '
                       'грузоподъёмных работах; сервис их не подставляет. Введённые значения сохраняются до конца сессии.')
            duration=a.number_input('Длительность партии, ч',min_value=1,max_value=24,value=1,key=prefix+'_duration')
            batch=b.number_input('Объём партии, т',min_value=.1,value=None,key=prefix+'_batch')
            start=a.number_input('Начало смены, час',min_value=0,max_value=23,value=8,key=prefix+'_start')
            end=b.number_input('Окончание смены, час',min_value=1,max_value=24,value=20,key=prefix+'_end')
            work=a.number_input('Лимит часов операций за сутки',min_value=0.,max_value=24.,value=12.,key=prefix+'_work')
            cold=b.number_input('Минимально допустимая температура, °C (необязательно)',value=None,key=prefix+'_cold')
            rain=a.number_input('Максимально допустимые осадки, мм/ч (необязательно)',min_value=0.,value=None,key=prefix+'_rain')
            st.caption('Пустые поля температуры и осадков отключают эти два ограничения. '
                       'Ноль осадков означает работу только без осадков. Лимит часов — бюджет операций, '
                       'не расчёт графиков труда и отдыха отдельных водителей.')
            signature=f'{lat}|{lon}'
            today=pd.Timestamp(decision).date()==datetime.now(timezone.utc).date()
            have=(st.session_state.get(prefix+'_data') or ('',))[0]==signature
            pressed=st.button('Обновить прогноз' if have else 'Загрузить почасовую погоду',key=prefix+'_fetch',
                              disabled=lat is None or lon is None or not today)
            if lat is not None and lon is not None and today and (pressed or (not have and _auto_fetch())):
                try:
                    with st.spinner('Загрузка почасовой погоды…'):
                        if pressed:get_wind.clear()
                        st.session_state[prefix+'_data']=(signature,get_wind(lat,lon))
                except Exception as exc:st.error('Загрузка погоды не удалась: '+type(exc).__name__)
            if lat is None or lon is None:
                st.caption('Кнопка станет доступна, когда выбрана площадка или введены координаты.')
            if not today:
                st.caption('Дата решения архивная: прогноз погоды на прошлые дни заново не скачать, поэтому кнопка выключена. '
                           'Поставьте дату решения на сегодня или загрузите сохранённый выпуск ниже.')
            loaded=st.session_state.get(prefix+'_data')
            if loaded and loaded[0]==signature:
                st.success('Прогноз погоды загружен автоматически: '+str(loaded[1].get('fetched_at',''))[:16].replace('T',' ')+' UTC, 7 дней по часам.')
            upload=st.file_uploader('Сохранённый выпуск ветра (JSON)',type=['json'],key=prefix+'_upload')
            entry=st.session_state.get(prefix+'_data')
            envelope=None
            if upload is not None:
                try:
                    candidate=json.loads(upload.getvalue())
                    if not isinstance(candidate,dict):raise ValueError('Ожидался JSON-объект выпуска ветра')
                    if candidate.get('latitude')!=lat or candidate.get('longitude')!=lon:
                        raise ValueError('Координаты выпуска не совпадают с терминалом')
                    envelope=candidate
                except (ValueError,TypeError) as exc:st.error(str(exc))
            elif entry and entry[0]==signature:envelope=entry[1]
            table=pd.DataFrame(dict(date=list(dates),known=False,capacity_t=0.))
            if envelope is not None and all(v is not None for v in [lat,lon,speed,gust,batch]) and source.strip():
                try:
                    cutoff=datetime.now(timezone.utc) if pd.Timestamp(decision).date()==datetime.now(timezone.utc).date() else as_of
                    table=W.capacity(envelope,dates,as_of=cutoff,tz=tz,speed_limit=speed,gust_limit=gust,
                                     duration_h=int(duration),batch_t=batch,start_hour=int(start),end_hour=int(end),
                                     min_temperature=cold,max_precipitation=rain,max_operation_hours=work)
                    st.caption('Прогноз получен: '+envelope['fetched_at'])
                    st.download_button('Скачать выпуск ветра',json.dumps(envelope,ensure_ascii=False).encode(),
                                       'wind-'+kind+'.json','application/json',key=prefix+'_download')
                    table.attrs['wind']['basis']=source.strip()
                except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
            else:
                missing=[n for n,v in [('площадка',lat is not None and lon is not None),('прогноз',envelope is not None),
                                       ('пределы ветра и порывов',speed is not None and gust is not None),
                                       ('основание пределов',bool(source.strip())),('объём партии',batch is not None)] if not v]
                st.info('Осталось задать: '+', '.join(missing)+'. Пока мощность неизвестна и равна нулю.')
            st.dataframe(table.rename(columns={'date':'Дата','known':'Полные данные','capacity_t':'Мощность, т',
                'safe_hours':'Пригодных часов','working_hours':'Часов смены','batches':'Партий'}),hide_index=True)
            tables.append(table)
        st.caption('Мощность суток ограничена целыми непрерывными операциями и бюджетом рабочего времени. Это суточный план объёмов, '
                   'а не почасовое расписание отдельных машин. Погрузка и разгрузка — разные площадки; '
                   'допуск на дорогу проверяется отдельно. За горизонтом прогноза мощность неизвестна.')
        return tuple(tables)
