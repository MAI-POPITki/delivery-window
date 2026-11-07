"""Проверка пространственного переноса сохранённой М1 на пост Коми без переобучения."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from okno import forecast, m1
from okno.features import build_m1_dataset
from okno.paths import MODELS, REPORTS
from okno.regions import all_gauges


def main():
    assert 70378 not in all_gauges(), 'Проверяемый пост не должен попадать в обучение'
    rows=[]
    for kind in ('open', 'close'):
        data=build_m1_dataset(kind, codes=[70378], include_sat=False)
        pred=m1.predict(forecast._model(kind), data, kind)
        for lead,(lo,hi) in m1.LEAD_BUCKETS.items():
            selected=data[data.lead_days.between(lo,hi)]
            if selected.empty:
                continue
            rows.append(dict(event=kind, lead_days=lead, rows=len(selected),
                seasons=selected.hydro_year.nunique(),
                calendar_mae_days=(selected.target_day-selected.cal_target).abs().mean(),
                model_mae_days=(selected.target_day-pred.loc[selected.index,'q50']).abs().mean(),
                model_sha256=hashlib.sha256((MODELS/f'm1_{kind}.cbm').read_bytes()).hexdigest()))
    table=pd.DataFrame(rows)
    table.to_csv(REPORTS/'komi_coverage.csv',index=False)
    lines=['# Коми: пространственная проверка существующей модели', '',
        'Пост 70378: Мезень — Макарьб. Источник замеров: Shiklomanov et al.,',
        'Arctic Data Center, DOI 10.18739/A22Z12Q02; погода ERA5 уже есть в репозитории.',
        'Пост исключён из списка обучения. Модели и данные не изменяются.', '',
        '| Событие | Горизонт, сут | Строк | Сезонов | MAE календаря | MAE М1 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        name='Открытие по толщине' if row['event']=='open' else 'Начало таяния'
        lines.append(f"| {name} | {row['lead_days']} | {row['rows']} | {row['seasons']} | {row['calendar_mae_days']:.2f} | {row['model_mae_days']:.2f} |")
    lines += ['', 'Это пространственный перенос готовой общей модели. Годы других регионов могут',
        'пересекаться с проверочными: это не проверка прогноза только по прошлым сезонам.',
        'Горизонты отсчитаны от фактического события, не от фиксированной даты решения.',
        'У открытия всего три проверочных сезона на горизонте 14 суток; оно отключено.',
        'Прежняя модель, обученная только на Якутии, здесь была хуже календаря.',
        'Более низкая средняя ошибка нынешней модели не доказывает статистическую значимость,',
        'качество интервалов, безопасность проезда или экономию. Регион не включён в интерфейс: общий протокол допуска не пройден.',
        'Один гидропост не означает наличие подтверждённого транспортного объекта.']
    (REPORTS/'komi_coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Проверка завершена: reports/komi_coverage.csv и .md')


if __name__=='__main__':
    main()
