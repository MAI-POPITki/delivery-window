"""Предварительные даты ограничения дороги при заданной связи с гидропостом.

До 60 дней: реальные прошлые сезоны-аналоги, один сценарий на год.
Разброс сценариев не является откалиброванным вероятностным интервалом.
"""
import numpy as np
import pandas as pd


def analog_paths(series, issue, horizon=60, nearest=True):
    issue = pd.Timestamp(issue).normalize()
    if isinstance(horizon,bool) or int(horizon)!=horizon or not 7 <= horizon <= 60:
        raise ValueError('Горизонт должен быть от 7 до 60 суток')
    horizon=int(horizon)
    # Текущий год после issue и любые будущие измерения физически исключены.
    known = series.loc[:issue].copy().sort_index()
    recent_dates = pd.date_range(issue-pd.Timedelta(days=29), issue)
    recent = known.reindex(recent_dates).to_numpy(float)
    if len(recent) != 30 or not np.isfinite(recent).all():
        raise ValueError('Для дат нужны 30 последних ежедневных измерений без пропусков')
    history = known[known.index.year < issue.year]
    candidates = []
    for year in sorted(set(history.index.year)):
        try:
            center = issue.replace(year=year)
        except ValueError:
            center = issue.replace(year=year,day=28)
        best = None
        for shift in (range(-21, 22) if nearest else [0]):
            anchor = center+pd.Timedelta(days=shift)
            dates = pd.date_range(anchor-pd.Timedelta(days=29), anchor+pd.Timedelta(days=horizon))
            values = history.reindex(dates).to_numpy(float)
            if not np.isfinite(values).all():
                continue
            past = values[:30]
            score = float(np.mean(np.abs(past-recent))+2*np.mean(np.abs(np.diff(past)-np.diff(recent))))
            future = values[30:]
            # Уровень совпадает в начале, поправка затухает с масштабом 14 суток.
            path = future+(recent[-1]-past[-1])*np.exp(-np.arange(1, horizon+1)/14)
            row = dict(year=int(year), anchor=anchor, score_cm=score, path=path,
                       source_end=dates[-1])
            if best is None or score < best['score_cm']:
                best = row
        if best is not None:
            candidates.append(best)
    if len(candidates) < 5:
        raise ValueError('Нужно минимум 5 прошлых сезонов с полными рядами для сценариев дат')
    selected = sorted(candidates, key=lambda r:(r['score_cm'], r['year']))[:8] if nearest else candidates
    dates = pd.date_range(issue+pd.Timedelta(days=1), periods=horizon)
    paths = pd.DataFrame({str(r['year']):r['path'] for r in selected}, index=dates)
    metadata = pd.DataFrame([{k:v for k,v in r.items() if k!='path'} for r in selected])
    return paths, metadata


def transitions(values, dates, current, close_cm, reopen_cm, stable_days=3, recovery_days=0, initial_closed=None):
    """Первый период риска: >= close; затем stable дней < reopen и восстановление.

    None означает отсутствие события в пределах прогноза, а не отсутствие риска
    после горизонта. Начальное превышение не является фактом закрытия дороги.
    """
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    if not len(dates) or len(values)!=len(dates) or not np.isfinite(values).all():
        raise ValueError('Нужны конечные уровни на все даты')
    if not dates.equals(pd.date_range(dates[0], periods=len(dates))) or dates.has_duplicates:
        raise ValueError('Даты должны быть последовательными суточными')
    if not np.isfinite([current,close_cm,reopen_cm]).all() or reopen_cm>close_cm:
        raise ValueError('Порог открытия должен быть не выше порога закрытия')
    if (isinstance(stable_days,bool) or int(stable_days)!=stable_days or not 1<=stable_days<=14
            or isinstance(recovery_days,bool) or int(recovery_days)!=recovery_days or not 0<=recovery_days<=30):
        raise ValueError('Некорректная выдержка или срок восстановления')
    issue = dates[0]-pd.Timedelta(days=1)
    already = current>=close_cm
    if initial_closed is not None and not isinstance(initial_closed,(bool,np.bool_)):
        raise ValueError('Текущее состояние должно быть логическим или неизвестным')
    closed=already if initial_closed is None else bool(initial_closed)
    close = issue if closed else None
    run = 0
    for i,(day,level) in enumerate(zip(dates,values)):
        if close is None:
            if level>=close_cm:
                close=day
            continue
        run=run+1 if level<reopen_cm else 0
        if run>=stable_days+recovery_days:
            return dict(close=close, reopen=day, already_above=already, initial_closed=closed,censored=False)
    return dict(close=close, reopen=None, already_above=already,initial_closed=closed,censored=close is not None)


def summarize(paths, current, close_cm, reopen_cm, stable_days=3, recovery_days=0, travel_days=1, initial_closed=None):
    if isinstance(travel_days,bool) or int(travel_days)!=travel_days or not 1<=travel_days<=30:
        raise ValueError('Длительность рейса должна быть от 1 до 30 суток')
    events=pd.DataFrame([dict(scenario=str(c), **transitions(paths[c],paths.index,current,close_cm,reopen_cm,
                                stable_days,recovery_days,initial_closed)) for c in paths])
    if events.empty:
        raise ValueError('Нет сценариев')
    def window(column):
        days=pd.to_datetime(events[column]).dropna().sort_values()
        if days.empty:return None
        return dict(early=days.min().date(), median=days.iloc[len(days)//2].date(), late=days.max().date(), n=len(days))
    close, reopen=window('close'),window('reopen')
    # Рейс должен закончиться строго до самого раннего риска.
    depart = (pd.Timestamp(close['early'])-pd.Timedelta(days=travel_days)).date() if close else None
    issue=(paths.index[0]-pd.Timedelta(days=1)).date()
    return dict(close=close,reopen=reopen,events=events,scenarios=len(events),
                censored=int(events.censored.sum()),already_above=bool(events.already_above.all()),
                initial_closed=bool(events.initial_closed.all()),
                last_departure=depart if depart and depart>issue else None,
                departure_expired=bool(depart and depart<=issue),through=paths.index[-1].date())
