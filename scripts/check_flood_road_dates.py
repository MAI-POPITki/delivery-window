"""Walk-forward проверка дат пересечения уровня; это не разметка закрытий дорог."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from okno.flood_m1 import ROOT,STATIONS,load_levels
from okno.flood_road_window import analog_paths,summarize,transitions


def main():
    rows=[]; skipped=0
    for pid in STATIONS:
        series=load_levels(pid)
        for year in range(2019,2026):
            past=series[(series.index.year<year)&series.index.month.isin([4,5,6,7])].dropna()
            if not len(past):continue
            threshold=float(past.quantile(.9))
            for md in ('03-15','04-01','04-15'):
                issue=pd.Timestamp(f'{year}-{md}')
                dates=pd.date_range(issue+pd.Timedelta(days=1),periods=60)
                actual=series.reindex(dates)
                current=series.get(issue,np.nan)
                if not np.isfinite(current) or actual.isna().any():
                    skipped+=1;continue
                truth=transitions(actual,dates,current,threshold,threshold)
                try:
                    predictions={}
                    for name,nearest in [('analogs',True),('calendar',False)]:
                        paths,_=analog_paths(series,issue,nearest=nearest)
                        predictions[name]=summarize(paths,current,threshold,threshold)
                except ValueError:
                    skipped+=1;continue
                for name,result in predictions.items():
                    for event in ('close','reopen'):
                        # Исключаем тривиальное закрытие уже превышенного уровня.
                        if event=='close' and truth['already_above']:continue
                        estimate=result[event]
                        predicted=(pd.Timestamp(estimate['median']) if estimate and estimate['n']>result['scenarios']/2 else pd.NaT)
                        target=truth[event]
                        error=abs((predicted-target).days) if pd.notna(predicted) and target is not None else np.nan
                        covered=(estimate['early']<=target.date()<=estimate['late']) if target is not None and estimate else False
                        rows.append(dict(pid=pid,issue=issue.date(),method=name,event=event,threshold_cm=threshold,
                            actual=target,predicted=predicted,error_days=error,covered=covered,
                            missed=target is not None and pd.isna(predicted),false_alarm=target is None and pd.notna(predicted)))
    d=pd.DataFrame(rows)
    d.to_csv(ROOT/'reports/flood_road_dates.csv',index=False)
    lines=['# Проверка предварительных дат ограничения проезда','',
        '**Проверяется пересечение уровня q90, не фактические приказы о закрытии дорог.** '
        'Порог q90 вычисляется только по апрелю–июлю прошлых лет. '
        'Фиксированные даты расчёта: 15 марта, 1 и 15 апреля, 2019–2025; горизонт 60 суток. '
        'Аналоги и календарный соперник получают одинаковые известные измерения. '
        'Дата выдаётся при наличии события более чем в половине сценариев. '
        'Ошибка даты считается только для совпавших событий, пропуски и ложные события показаны отдельно. '
        'Начальное превышение исключено из оценки даты закрытия. '
        'Повторное использование ранее изученного архива не является новым независимым тестом.','',
        '| Событие | Метод | Выпусков | Совпало | MAE даты, сут. | Пропущено | Ложных | Покрытие разбросом реальных событий |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (event,method),g in d.groupby(['event','method']):
        actual_count=g.actual.notna().sum()
        coverage=g.loc[g.actual.notna(),'covered'].mean()
        lines.append(f"| {event} | {method} | {len(g)} | {g.error_days.notna().sum()} | {g.error_days.mean():.1f} | "
                     f"{g.missed.sum()} / {actual_count} | {g.false_alarm.sum()} | {coverage:.0%} |")
    lines+=['',f'Пропущено выпусков из-за неполного ряда / недостатка аналогов: {skipped}.',
        '', '**В основной интерфейс поставлен calendar, а не analogs:** подбор похожих сезонов '
        'на этой проверке дал худшую ошибку дат и больше пропусков. Выбор сделан после просмотра '
        'этой таблицы, поэтому она не является независимым подтверждением выбранного варианта. '
        'Сезонный ориентир также даёт пропуски и ложные события; это предварительная версия, не доказанный дорожный прогноз.',
        '', 'Разброс — min–max дат событий выбранных сезонов, не калиброванный вероятностный интервал. '
        'Сценарии без закрытия/открытия не исчезают из знаменателя. '
        'Отсутствие открытия за 60 суток не превращается в выдуманную дату.','',
        '## Как рассчитываются даты','',
        'Для каждого прошлого года ищется ближайший по последним 30 дням сезон в ±21 дне от календарной даты. '
        'Расстояние — MAE уровня + 2×MAE суточных изменений. Выбираются до 8 разных лет (минимум 5). '
        'Будущая траектория — измерения того сезона плюс разность текущих уровней, затухающая как exp(−lead/14). '
        'Параметры фиксированы для этой первой версии, не подбирались по приведённой таблице. '
        'Соперник использует все полные прошлые сезоны на той же календарной дате с такой же поправкой. '
        'Краткосрочный CatBoost М1 показан отдельно и не выдаётся за 60-дневную модель.','',
        'Закрытие моделируется первым достижением заданного дорожного порога; открытие — после заданного числа '
        'суток строго ниже второго порога плюс заданное время восстановления (тоже ниже порога). '
        'Три дня и нулевой ремонт — настройки сценария, не факты о дороге. '
        'Если дата близка к концу горизонта, выдержка может не завершиться.','',
        '## Граница доказательств','',
        'Пока нет набора независимых фактических закрытий/открытий с привязанными порогами конкретных дорог. '
        'Поэтому нельзя заявлять MAE этих дат как точность прогноза дорожных приказов. '
        'В интерфейсе нужны название дороги и источник порогов; автоматической подстановки опасного уровня реки нет. '
        'М2 наблюдает воду и не используется для выдумывания будущей даты открытия. '
        'М3 пока сохраняет строгую проверку 1–7 суток; 60-дневные аналоги не дают разрешения на рейс.']
    (ROOT/'reports/flood_road_dates.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(d.groupby(['event','method']).agg(n=('error_days','size'),matched=('error_days','count'),mae=('error_days','mean'),missed=('missed','sum'),false_alarm=('false_alarm','sum')).to_string())


if __name__=='__main__':main()
