"""Reproducible rolling evaluation of the one-region low-water beta."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from okno import low_water as L


def score(g):
    y=g.event.to_numpy(); p=g.probability.to_numpy();warn=p>=.5
    return dict(n=len(g),events=int(y.sum()),mae_cm=float(abs(g.predicted_cm-g.actual_cm).mean()),
        persistence_mae_cm=float(abs(g.persistence_cm-g.actual_cm).mean()),
        calendar_mae_cm=float(abs(g.calendar_cm-g.actual_cm).mean()),
        low_days_mae_cm=float(abs(g.loc[g.event,'predicted_cm']-g.loc[g.event,'actual_cm']).mean()) if y.any() else None,
        coverage80=float(g.actual_cm.between(g.lower_cm,g.upper_cm).mean()),
        brier=float(np.mean((p-y)**2)),
        persistence_brier=float(np.mean((g.persistence_probability-y)**2)),
        calendar_brier=float(np.mean((g.calendar_probability-y)**2)),
        climatology_brier=float(np.mean((g.climatology_probability-y)**2)),
        tp=int(sum(warn & y)),fn=int(sum(~warn & y)),fp=int(sum(warn & ~y)),tn=int(sum(~warn & ~y)))


def main():
    s=L.M.load_levels(L.PID); table=L.M.dataset(s);parts=[]
    for year in range(2019,2024):
        model,meta,cal,limit=L.bundle(year)
        r=table[(table.issue.dt.year==year)&L.season(table.issue)&L.season(table.target_date)].copy()
        p=L.predict_rows(s.loc[:f'{year}-12-31'],r,year,model,cal,limit)
        p['actual_cm']=r.target;p['threshold_cm']=limit;p['event']=p.actual_cm.le(limit)
        p['issue_cm']=r.h0;p['year']=year
        parts.append(p)
    raw=pd.concat(parts,ignore_index=True)
    R=L.M.ROOT/'reports';raw.to_csv(R/'low_water_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary={str(h):score(g) for h,g in raw.groupby('lead')}
    byyear={f'{y}:{h}':score(g) for (y,h),g in raw.groupby(['year','lead'])}
    # Demonstration: latest year, current level above threshold, subsequent crossing.
    choices=raw[(raw.year==2023)&raw.issue_cm.gt(raw.threshold_cm)&raw.event&raw.probability.ge(.5)]
    demo=choices.sort_values(['lead','issue']).iloc[0]
    data=dict(version=L.VERSION,pid=L.PID,months=list(L.MONTHS),years=list(range(2019,2024)),
        threshold='q20 June–September, years before calibration year',rows=len(raw),
        scores=summary,by_year=byyear,demo=dict(issue=str(demo.issue.date()),lead=int(demo.lead)))
    (R/'low_water_metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# Маловодье: бета для Лены у Якутска','',
        'Пост 3045, АИС ГМВО через архив «Уровень-Спутник». Единицы — см относительно нуля поста. '
        'Июнь–сентябрь 2019–2023; дата решения и цель обе внутри этих месяцев. '
        'Последний пригодный замер архива: '+str(s.last_valid_index().date())+'.',
        '', '## Протокол', '',
        'Для каждого оценочного года существующий CatBoost обучен до года калибровки. '
        'На предыдущем лете выбирается метод с минимальной MAE отдельно для горизонтов 1–7: '
        'сохранение уровня, календарная медиана, линейный тренд за 7/14 суток, CatBoost. '
        'Порог q20 — нижние 20% летних уровней только лет до калибровки. '
        'Календарь использует тот же ранний архив. Остатки предыдущего лета задают центральный '
        '80%-й интервал и сглаженную эмпирическую вероятность H ≤ порог. '
        'Это отдельная сезонная калибровка, веса действующего М1 не менялись.', '',
        'Оценочный год исключён из обучения, выбора метода, порога и калибровки. '
        'Архив ранее изучался в проекте: проверка ретроспективная, не независимый новый сезон. '
        'Пять сезонов не равны числу независимых прогнозов. Соседние дни коррелируют. '
        'Горизонт определяется календарными сутками; событие — уровень на выбранную дату, не хотя бы раз за неделю.', '',
        '## Все допустимые даты', '',
        '| Сутки | N | Низких дней | MAE, см | Сохранение, см | Календарь, см | Покрытие 80% | Brier | Сохранение | Календарь | Частота | TP/FN/FP/TN при 50% |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for h,r in summary.items():
        lines.append(f"| {h} | {r['n']} | {r['events']} | {r['mae_cm']:.2f} | {r['persistence_mae_cm']:.2f} | {r['calendar_mae_cm']:.2f} | {r['coverage80']:.1%} | {r['brier']:.4f} | {r['persistence_brier']:.4f} | {r['calendar_brier']:.4f} | {r['climatology_brier']:.4f} | {r['tp']}/{r['fn']}/{r['fp']}/{r['tn']} |")
    lines+=['','## По сезонам (горизонты 1 и 7 суток)','','| Год | Сутки | N | Событий | MAE / сохранение, см | Brier / сохранение | Покрытие |','|---:|---:|---:|---:|---|---|---|']
    for key,r in byyear.items():
        y,h=key.split(':')
        if h in ('1','7'):lines.append(f"| {y} | {h} | {r['n']} | {r['events']} | {r['mae_cm']:.2f} / {r['persistence_mae_cm']:.2f} | {r['brier']:.4f} / {r['persistence_brier']:.4f} | {r['coverage80']:.1%} |")
    lines+=['','## Практическое применение и пределы','','Оператор видит ожидаемый уровень, диапазон, вероятность снижения ниже отметки и даты, '
        'на которые стоит проверить условия загрузки или отправки. Порог оператора задаётся отдельно; '
        'таблица качества относится к статистическому q20, не к любому введённому порогу. '
        'Порог предупреждения 50% — настройка демонстрации, не экономически обоснованный норматив.', '',
        'Уровень поста не равен глубине фарватера. Без совместной истории глубин, осадки и запаса под килем '
        'бета не выдаёт разрешение на проход, допустимую загрузку, закрытие навигации или обещание экономии. '
        'Низкая вода не обязательно означает официальное опасное гидрологическое явление. '
        'При отсутствии событий в отдельном сезоне полноту обнаружения определить нельзя.', '',
        'Вероятности и интервалы не гарантированы в новом сезоне: фактическое покрытие показано выше. '
        'На отдельных сезонах вероятностный прогноз проигрывает сохранению уровня. '
        'Для оперативного запуска нужны непрерывные свежие 31 сутки того же поста/нуля/источника '
        'и новый пакет предыдущих сезонов. Свежие данные в отдельном паводковом хранилище '
        'не подставляются вместо отсутствующих лет обучения и калибровки.', '',
        f"Архивный пример перехода ниже порога: решение {demo.issue:%d.%m.%Y}, горизонт {int(demo.lead)} сут. "
        f"Прогноз {demo.predicted_cm:.1f} см, вероятность {demo.probability:.1%}, порог {demo.threshold_cm:.1f} см, факт {demo.actual_cm:.1f} см. "
        'Это иллюстрация, а не отдельная оценка качества.', '',
        'Воспроизведение: `python scripts/check_low_water.py`. Сырые прогнозы — '
        '`reports/low_water_predictions.csv.gz`; все сезонные метрики — `reports/low_water_metrics.json`.']
    (R/'low_water_beta.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scores=summary,demo=data['demo']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
