"""Reproduce the causal A3 report: python src/corridor_simulator.py."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from okno.corridor import CorridorConfig,POLICY_LABELS,compare_policies,run_sensitivity_check
from okno.corridor_data import case_inputs,verified_years
from okno.report import md_table


def calculate_case(year,cfg=None):
    cfg=cfg or CorridorConfig()
    c=case_inputs(year)
    rs=compare_policies(cfg,c['ac'],c['ao'],c['cc'],c['co'],c['qc'],c['qo'],c['forecasts'])
    table=pd.DataFrame([dict(hydro_year=year,policy=p,ml_available=c['model_available'],**s)
                        for p,(s,_) in rs.items()])
    return c,rs,table


def run_reference_case(cfg=None):
    """Explicit artificial example; no CatBoost output is invented."""
    _,rs,table=calculate_case('synthetic',cfg)
    return rs['calendar'][0],rs['ml'][0],table


def run_verified_backtest(cfg=None):
    return pd.concat([calculate_case(y,cfg)[2] for y in verified_years()],ignore_index=True)


def main():
    reports=ROOT/'reports'
    cfg=CorridorConfig()
    c,rs,table=calculate_case(2024,cfg)
    historical=run_verified_backtest(cfg)
    historical.to_csv(reports/'corridor_backtest_verified.csv',index=False)
    daily=pd.concat([d.assign(policy=p) for p,(_,d) in rs.items()],ignore_index=True)
    daily.to_csv(reports/'corridor_daily_2024.csv',index=False,float_format='%.8f')
    sensitivity=run_sensitivity_check(cfg,c['ac'],c['ao'],c['cc'],c['co'],
                                      forecasts=c['forecasts'],q20_close=c['qc'],q20_open=c['qo'])
    sensitivity.to_csv(reports/'corridor_sensitivity.csv',index=False)
    view=table[['policy','total_cost_rub','served_tons','terminal_backlog_tons','days_with_deficit',
                'shortage_ton_days','cost_holding_yak_rub']].copy()
    view.policy=view.policy.map(POLICY_LABELS)
    view.columns=['Способ','Всего, ₽','Получил потребитель, т','Не исполнено к концу, т',
                  'Дни с ожиданием','Ожидание, т·сут','Склад Якутска, ₽']
    s=table.set_index('policy')
    diff=s.loc['ml','total_cost_rub']-s.loc['calendar','total_cost_rub']
    pct=100*diff/s.loc['calendar','total_cost_rub']
    files=['okno/corridor.py','okno/corridor_data.py','src/model_window.py',
           'reports/window_forecasts_a3.csv','data/rosstat_prices/yakutsk_mchs_crossing_dates.csv']
    manifest=dict(config=asdict(cfg),case={k:v for k,v in c.items() if k!='forecasts'},
                  model_scope='physical satellite breakup; road conversion estimated on earlier verified years',
                  economic_scope='operator demand, tariffs, allocated capacities, warehouse and opening stock are assumptions',
                  sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
    (reports/'corridor_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=['# А3: сквозной коридор с последовательными решениями','',
        '**Прежние оценки «−31% на 24 реальных сезонах» отозваны.** '
        'В них поздний прогноз использовался для ранних отправок, а дорожные окна заменялись спутниковой аномалией и 27 сутками.','',
        '## Что теперь сравнивается','',
        'Одинаковая эвристика управления запасом для календарной медианы, календарного q20 и ML q20. '
        'Отдельный контроль — отсутствие предзавоза. Все получают текущий статус переправы при наступлении события. '
        'Будущие фактические даты не попадают в политику. ML использует только выпуски не позднее текущего дня.','',
        'Груз отправляется из хаба, находится в пути 6 суток и только затем попадает в Нижний Бестях. '
        'Суточная отправка по ж/д ограничена 120 т; штатная переправа — 60 т/сут, резерв — 8 т/сут. '
        'Это выделенные мощности условного оператора, не пропускная способность всей переправы. '
        'Основной склад ограничен 650 т; дополнительный склад по умолчанию отсутствует. '
        'Невыполненный спрос переносится как очередь заказов, штраф начисляется на каждый день ожидания.','',
        '## Подтверждённый период 2024 года, условная экономика','',
        'Начало: 1 марта, горизонт 120 суток. Первый полностью закрытый день — 16 апреля '
        '(закрытие вечером 15 апреля); открытие паромов — 24 мая: **38 суток**, без подмены на 27. '
        'Спрос 25 т/сут, всего 3 000 т. Начальный запас 225 т одинаков для всех стратегий. '
        'Тарифы: ж/д 8 500 ₽/т, штатная переправа 1 800 ₽/т, резерв 35 000 ₽/т, '
        'хранение на правом берегу 450 ₽/т·сут, в Якутске 220 ₽/т·сут, ожидание 12 000 ₽/т·сут. '
        '**Все эти экономические величины — допущения**, не подтверждённые тарифы.','',md_table(view),'',
        f'Разница ML к календарной медиане: **{diff/1e6:+.3f} млн ₽ ({pct:+.2f}%)**. '
        'Положительное значение означает удорожание. Выигрыш модели не предполагается заранее.','',
        'Результат отражает сочетание преобразования спутника в дорожные даты, эвристики запасов и условных цен. '
        'Это не подтверждённая экономия или убыток реального перевозчика.','',
        '## Доступность и качество данных','',
        f'Полные проверенные операционные окна есть для {verified_years()}. '
        'В 2020–2022 не подтверждены точные даты закрытия: эти сезоны не участвуют в операционном сравнении. '
        'Даты открытия исправлены по публикациям: 2019 — 25 мая, 2020 — 22 мая, '
        '2021 — 26 мая, 2022 — 30 мая, 2023 — 26 мая, 2024 — 24 мая. '
        'При недостатке прошлых пар дорожных/спутниковых дат ML явно заменяется календарным q20 '
        'и не считается самостоятельным подтверждением модели.','',
        'Модель физического вскрытия обучается с расширением только по прошлым годам. '
        'Никаких будущих целей для первых трёх лет. Основная проверка — на фиксированное 10 марта, '
        'а не на выбранный задним числом горизонт до факта. Калибровка использует прошлые ошибки. '
        'Погода — позднейший ERA5 с номинальной задержкой пять суток; исторические версии '
        'оперативных данных и реальные выпуски прогнозов не восстановлены. Спутниковые метки — '
        'ретроспективная обработка архива; их историческая доступность не подтверждена. '
        'Это исследовательская реконструкция, не доказательство реально выпущенных в 2024 году предупреждений.','',
        'Перевод в дорожные даты оценивается только по более ранним подтверждённым фазам. '
        'Связь слабая и выборка мала: [диагностика](operational_shift_validation.csv). '
        'Физическое вскрытие не определяет несущую способность льда и разрешение движения.','',
        '## Устойчивость','',
        f'Проверена {len(sensitivity)} различная конфигурация: одна база и ±30% ненулевых параметров. '
        f'В {int(sensitivity.effect_positive.sum())} ML дешевле календарной медианы. '
        'Это сетка чувствительности, не вероятность будущей экономии. Начальный запас, '
        'время ж/д, спрос и мощности варьируются вместе с тарифами по одному параметру за раз.','',
        '## Воспроизведение','',
        '```bash','python src/model_window.py','python src/validate_data.py','python src/corridor_simulator.py',
        'python -m unittest discover -s tests -v','```','',
        '- [Подтверждённые сезоны и все стратегии](corridor_backtest_verified.csv)',
        '- [Полный дневной журнал 2024](corridor_daily_2024.csv)',
        '- [Чувствительность](corridor_sensitivity.csv)',
        '- [Параметры и контрольные суммы](corridor_manifest.json)',
        '- [Физическая модель и её ограничения](window_model_report.md)',
        '- [Проверки с честными WARN](../docs/validation_report.md)','',
        'Для проверки реальной экономической выгоды нужны отправки и потребление оператора, '
        'доступные мощности, договорные цены, реальные моменты публикации данных и ограничения массы по дням. '
        'Агрегированный сезонный объём 400 000 т из отдельного эксперимента не подменяет поток одного склада.']
    for url in c['sources']:
        lines.append(f'- Источник события: {url}')
    (reports/'corridor_before_after.md').write_text('\n'.join(lines)+'\n')
    print(view.round(2).to_string(index=False))
    print(f'ML minus calendar: {diff:,.0f} RUB ({pct:+.2f}%), sensitivity variants {len(sensitivity)}')


if __name__=='__main__':
    main()
