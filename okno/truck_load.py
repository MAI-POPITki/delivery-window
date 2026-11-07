"""Проверка полной массы по записи реестра, без оценки вероятности аварии."""
import math
import pandas as pd
from .external_data import extract_point_id


def route_events(frame, point_id):
    """Гидропост может быть связан с несколькими дорогами: их нельзя смешивать."""
    return frame[frame.comment.map(extract_point_id).eq(point_id)].copy()


def limit_as_of(frame, road, date):
    """Только уже опубликованные фактические события выбранной дороги и сезона.

    Тип open не содержит структурированной массы: из текста она не угадывается.
    Реестр неполон и не заменяет актуальную сводку дорожного ведомства.
    """
    date=pd.Timestamp(date).normalize()
    year=date.year+int(date.month>=9)
    rows=frame[frame.road.eq(road) & frame.hydro_year.astype(int).eq(year)].copy()
    rows['effective']=pd.to_datetime(rows.actual_date, errors='coerce')
    rows['published']=pd.to_datetime(rows.actual_published, errors='coerce')
    rows=rows[rows.effective.le(date)]
    if rows.empty:
        return dict(status='unknown', reason='Нет фактических событий этого сезона на выбранную дату.')
    latest=rows[rows.effective.eq(rows.effective.max())]
    if latest.published.isna().any() or latest.published.gt(date).any():
        return dict(status='unknown', reason='Последнее событие ещё не опубликовано на дату решения или дата публикации неизвестна.')
    if latest.kind.nunique()!=1 or latest.source_actual.nunique()!=1:
        return dict(status='unknown', reason='Несогласованные записи на одну дату: нужна сверка источников.')
    row=latest.iloc[-1]
    result=dict(source=row.source_actual, effective=str(row.effective.date()),
                published=str(row.published.date()), comment=row.comment)
    if row.kind=='close':
        return dict(result,status='closed',reason='В реестре указано закрытие движения. Снижение массы не разрешает проезд.')
    limit={'limit20':20.,'limit30':30.}.get(row.kind)
    if limit is None:
        return dict(result,status='unknown',reason='В записи открытия нет структурированного ограничения полной массы.')
    return dict(result,status='limit',limit_t=limit)


def check_load(tare_t, cargo_t, limit_t):
    """Арифметическое сравнение; не проверяет осевые нагрузки и безопасность льда."""
    if not all(math.isfinite(v) and v>=0 for v in (tare_t,cargo_t,limit_t)) or limit_t==0:
        raise ValueError('Массы должны быть неотрицательными, ограничение — положительным.')
    gross=tare_t+cargo_t
    excess=max(0.,gross-limit_t)
    return dict(gross_t=gross, excess_t=excess, excess_pct=100*excess/limit_t,
                cargo_limit_t=max(0.,limit_t-tare_t), within_mass_limit=gross<=limit_t,
                tare_exceeds_limit=tare_t>limit_t)
