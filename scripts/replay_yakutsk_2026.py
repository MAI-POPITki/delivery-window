"""Reproduce the scenario report: python scripts/replay_yakutsk_2026.py.

Prepare inputs first with prepare_yakutsk_replay.py (lags 5, 0 and 7).
No model is retrained or tuned on this scenario's outcome.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from okno.yakutsk_replay import ACTUAL, ANNOUNCED, RULE_LABELS, Scenario, capacity_table, simulate_all

PREFIX = "yakutsk_spring_2026"
INPUTS = ROOT / "data/scenarios"
REPORTS = ROOT / "reports"
SOURCES = {
    "announced": "https://ysia.ru/gruzopodemnost-perepravy-yakutsk-nizhnij-bestyah-snizyat-s-1-aprelya/",
    "actual_30t": "https://gtrksakha.ru/amp/2026/03/27/dosrochno-snizhena-gruzopoduemnost-na-pereprave-yakutsk-nizhnij-bestyah/",
    "actual_20t": "https://yakutiamedia.ru/news/2456829/",
    "actual_closure": "https://hotu.su/ru/news/20260416/30731.html",
    "satellite": "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc",
    "weather": "https://open-meteo.com/en/docs/historical-weather-api",
}


def load_inputs(lag=5):
    path = INPUTS if lag == 5 else INPUTS / f"lag{lag}"
    return (pd.read_csv(path / f"{PREFIX}_forecasts.csv"),
            pd.read_csv(path / f"{PREFIX}_observations.csv"))


def comparison(scenario=Scenario(), lag=5, announced_schedule=ANNOUNCED):
    f, o = load_inputs(lag)
    results = simulate_all(f, o, scenario, announced_schedule=announced_schedule)
    frame = pd.DataFrame([r.summary for r in results.values()])
    calendar_cost = frame.loc[frame.rule == 'calendar', 'cost_rub'].iloc[0]
    frame['saving_vs_calendar_rub'] = calendar_cost-frame.cost_rub
    frame['saving_vs_calendar_pct'] = frame.saving_vs_calendar_rub/calendar_cost*100
    return results, frame


def sensitivity():
    base = Scenario()
    variants = [('base', 'Исходные параметры', base, 5)]
    for q in [2500, 3500, 4500, 5500, 6000, 6500, 7000]:
        variants.append((f'Q={q}', 'Объём заказа', replace(base,Q_t=q),5))
    for field in ['c_road','c_extra','c_backup','c_hold_week']:
        for factor in [.7,1.3]:
            variants.append((f'{field}x{factor}', 'Расходы ±30%', replace(base,**{field:getattr(base,field)*factor}),5))
    for field, vals in [('base_trucks',[8,12]),('extra_trucks',[0,10]),('tare_t',[10,14]),
                        ('max_payload_t',[24,32]),('trips_per_workday',[.7,1.3]),
                        ('close_margin_days',[14,28,35]),('climatology_halfwidth_days',[5,14]),
                        ('n_scenarios',[500,4000])]:
        for value in vals:
            variants.append((f'{field}={value}', 'Парк, оборот и модель', replace(base,**{field:value}),5))
    variants.append(('six_workdays','Рабочая неделя',replace(base,weekdays=(0,1,2,3,4,5)),5))
    for lag in [0,7]:
        variants.append((f'weather_lag={lag}', 'Задержка погоды',base,lag))
    rows=[]
    for name, group, config, lag in variants:
        _, frame = comparison(config,lag)
        frame['variant'], frame['group'], frame['weather_lag_days'] = name, group, lag
        for field in ['base_trucks','extra_trucks','tare_t','max_payload_t','trips_per_workday',
                      'close_margin_days','c_road','c_extra','c_backup','c_hold_week']:
            frame[field]=getattr(config,field)
        rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def fmt(value, places=2):
    return f'{value:,.{places}f}'.replace(',', ' ')


def md_table(frame):
    out=frame.copy().fillna('—')
    for col in out:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col]=out[col].map(fmt)
    return '| '+' | '.join(map(str,out.columns))+' |\n| '+' | '.join(['---']*len(out.columns))+' |\n'+''.join(
        '| '+' | '.join(map(str,row))+' |\n' for row in out.itertuples(index=False,name=None))


def main():
    REPORTS.mkdir(exist_ok=True)
    volume_assumption = json.loads((INPUTS/f'{PREFIX}_volume.json').read_text(encoding='utf-8'))
    season_total = float(volume_assumption['season_total_t'])
    fleet_rows = []
    for name, weekdays in [('Пять рабочих дней', (0,1,2,3,4)),
                           ('Шесть рабочих дней', (0,1,2,3,4,5))]:
        per_truck = capacity_table(replace(Scenario(), base_trucks=1, extra_trucks=0,
                                           weekdays=weekdays)).base_capacity_t.sum()
        fleet_rows.append({'График': name, 'Одна машина за 16.03–15.04, т': per_truck,
                           'Машин для всего сезонного объёма за этот период': math.ceil(season_total/per_truck)})
    results, summary = comparison()
    daily = pd.concat([r.schedule for r in results.values()],ignore_index=True)
    decisions = pd.concat([r.decisions for r in results.values()],ignore_index=True)
    sens = sensitivity()
    boundary_rows = []
    for q in (5000, 6500):
        _, frame = comparison(replace(Scenario(), Q_t=q),
                              announced_schedule=replace(ANNOUNCED, last_open_date='2026-04-14'))
        frame['announced_last_open_date'] = '2026-04-14'
        boundary_rows.append(frame)
    boundary = pd.concat(boundary_rows, ignore_index=True)
    for name, frame in [('summary',summary),('daily',daily),('decisions',decisions),('sensitivity',sens),
                        ('calendar_boundary', boundary)]:
        frame.to_csv(REPORTS/f'{PREFIX}_{name}.csv',index=False,float_format='%.8f')
    files = [INPUTS/f'{PREFIX}_forecasts.csv',INPUTS/f'{PREFIX}_observations.csv',
             INPUTS/f'{PREFIX}_volume.json',
             ROOT/'okno/yakutsk_replay.py',ROOT/'scripts/prepare_yakutsk_replay.py']
    manifest={'scenario':asdict(Scenario()),'announced_schedule':asdict(ANNOUNCED),
              'actual_schedule_evaluation_only':asdict(ACTUAL),'sources':SOURCES,
              'season_volume_assumption': volume_assumption,
              'parameter_status':'Order, fleet, turnaround and economics are assumptions authorized by user; no operator savings claimed.',
              'input_and_code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (INPUTS/f'{PREFIX}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    s=summary.set_index('rule')
    extra_cost=s.loc['forecast_m2','cost_rub']-s.loc['calendar','cost_rub']
    percent=extra_cost/s.loc['calendar','cost_rub']*100
    days=(pd.Timestamp(s.loc['calendar','completion_date'])-pd.Timestamp(s.loc['forecast_m2','completion_date'])).days
    table=summary[['rule','delivered_t','backup_t','extra_t','hold_rub','cost_rub','cost_rub_per_t','completion_date']].copy()
    table['rule']=table.rule.map(RULE_LABELS)
    table.columns=['Способ','По зимнику, т','Резерв, т','Доп. техникой, т','Хранение, ₽','Всего, ₽','₽/т','Заказ доставлен']
    volume=sens[sens['group'].isin(['Исходные параметры','Объём заказа'])].copy()
    volume=volume.pivot(index='Q_t',columns='rule',values=['cost_rub','backup_t'])
    vol=pd.DataFrame({'Заказ, т':volume.index,
        'Календарь, ₽':volume['cost_rub']['calendar'].values,
        'М1+М2, ₽':volume['cost_rub']['forecast_m2'].values,
        'Экономия М1+М2, ₽':(volume['cost_rub']['calendar']-volume['cost_rub']['forecast_m2']).values,
        'Резерв календаря, т':volume['backup_t']['calendar'].values,
        'Резерв М1+М2, т':volume['backup_t']['forecast_m2'].values})
    fc_sens=sens[sens.rule=='forecast_m2']
    positive=int((fc_sens.saving_vs_calendar_rub>.01).sum())
    heavy = sens[sens.variant == 'Q=6500'].set_index('rule')
    boundary_table = boundary[boundary.rule.isin(['calendar', 'forecast_m2'])][[
        'Q_t', 'rule', 'backup_t', 'cost_rub', 'saving_vs_calendar_pct']].copy()
    boundary_table['rule'] = boundary_table.rule.map(RULE_LABELS)
    boundary_table.columns = ['Заказ, т', 'Способ', 'Резерв, т', 'Всего, ₽', 'Экономия к календарю, %']
    f,obs=load_inputs()
    q_start=f.iloc[0]
    scene_table=obs[obs.date.isin(['2026-03-18','2026-04-11'])][['date','available_date','p_ice']].copy()
    scene_table.columns=['Дата съёмки UTC','Первое доступное решение UTC','P(лёд)']
    lines=[
        '# Переправа Нижний Бестях → Якутск: сценарная проверка весны 2026', '',
        f'Общий сезонный объём {fmt(season_total, 0)} т задан пользователем как допущение. '
        'Экономическое сравнение ниже относится к прежней тестовой партии 5 000 т. '
        'Реальный остаток на 16 марта неизвестен.', '',
        f'**При согласованных параметрах наш метод дороже адаптивного календаря на {extra_cost:,.2f} ₽ ({percent:.2f}%). '
        f'Оба доставляют все 5 000 т; наш план завершает завоз на {days} суток раньше.**'.replace(',',' '), '',
        'Это один сценарный ретроспективный эксперимент. Ограничения движения реальные; заказ, парк, оборот и расходы — '
        'допущения. Результат не является подтверждённой экономией или убытком реального оператора. Параметры не подбирались '
        'после расчёта для получения выигрыша.', '',
        '## Сезонный объём и остаток', '',
        f'**{fmt(season_total, 0)} т на направлении Якутск — Нижний Бестях за сезон 2025/26** '
        '— пользовательское допущение, не подтверждённая статистика. Неизвестно, сколько перевезено '
        'до 16 марта и как объём распределён между направлениями. Поэтому общий объём не подменяет '
        'остаток заказа, а результаты партии 5 000 т не масштабируются автоматически на весь сезон.', '',
        'Если условно весь сезонный объём ещё предстоит вывезти с 16 марта по 15 апреля, '
        'при одном рейсе в рабочий день и полезной загрузке 28/18/8 т необходимый парк составит:', '',
        md_table(pd.DataFrame(fleet_rows)), '',
        'Это минимум по объёму при полной загрузке, без очередей, ремонта и ограничений смен; '
        'он не подтверждает фактическую пропускную способность переправы. Десять машин за этот период '
        'дают 4 440 т при пятидневке или 5 160 т при шестидневке. Продление срока после закрытия '
        '15 апреля не добавляет перевозок по льду. Для плана на весь сезон нужны его начало, '
        'изменения допустимой массы за зиму и фактическая производительность парка.', '',
        '## Входные данные', '',
        '- Начало: 16.03.2026; весь остаток 5 000 т доступен к перевозке с этого дня.',
        '- Основной парк: 10 машин; дополнительный: до 5. Тара: 12 т; технический максимум груза: 28 т.',
        '- Один завершённый рейс за рабочий день на машину; рабочие дни — понедельник–пятница. '
        'Доставка и возврат в пределах того же дня — допущение. Модель распределяет тонны, а не целочисленные рейсы.',
        '- Обычная доставка: 4 000 ₽/т; надбавка дополнительной техники: 1 500 ₽/т; '
        'резерв: 20 000 ₽/т; хранение: 100 ₽/т за семь календарных дней.',
        '- Запас нужен к 16.04.2026. Резерв приходит 26.05.2026; доступность и вместимость резерва — допущения. '
        '20 000 ₽/т включают резервную доставку и связанные издержки; обычная перевозка за эти же тонны второй раз не начисляется.',
        '- Хранение начисляется с фактического дня доставки до 16 апреля. Денежные суммы номинальные; '
        'раздельные НДС, фиксированная аренда машин, расход топлива и кредитные платежи не моделируются.', '',
        '| Событие | Объявлено в начале марта | Факт для оценки |',
        '| --- | --- | --- |',
        '| Полная масса до 30 т | 1 апреля | 27 марта |',
        '| Полная масса до 20 т | 8 апреля | 8 апреля |',
        '| Последний день движения | 15 апреля | 15 апреля, до вечернего закрытия |', '',
        'В предварительной публикации указано закрытие с 15 апреля без точного часа. '
        'Для расчёта 15 апреля включено в предварительное окно как последний день движения — '
        'это условие эксперимента, а не подтверждённое время из предварительного объявления. '
        'Фактическое закрытие вечером 15 апреля подтверждено отдельно.', '',
        'При таре 12 т лимиты 40/30/20 т дают полезную загрузку 28/18/8 т. '
        'Без дополнительных машин фактическая вместимость составляет **4 440 т**, с ними — **6 660 т**. '
        'Предварительный график обещал основному парку 4 740 т. Три рабочих дня с досрочным ограничением отнимают 300 т. '
        'Поэтому для заказа 5 000 т минимум 560 т требуют дополнительных машин.', '',
        '## Сравниваемые методы', '',
        '1. **Календарь с пересчётом:** использует опубликованные даты ограничений; распределяет остаток пропорционально '
        'оставшейся базовой вместимости, заранее нанимает дополнительную технику для расчётного дефицита. '
        'После нового ограничения пересчитывает план, а не продолжает ехать с прежней загрузкой.',
        '2. **Планировщик + календарь ±9 суток:** тот же выбор по ожидаемым расходам, но вместо М1 — '
        'сценарный разброс дат вокруг официального календаря. Девять суток — допущение, не оценка исторической частоты.',
        '3. **М1 + планировщик:** каждый рабочий день пересчитывает ожидаемые расходы и доступные объёмы по прогнозу М1.',
        '4. **М1 + спутник + планировщик:** дополнительно отсекает противоречащие свежему М2-наблюдению сценарии физического вскрытия.',
        '5. **Максимально быстрый завоз:** ежедневно использует основной и дополнительный парк до выполнения заказа.', '',
        'Все способы имеют одинаковый заказ, расходы, парк, рабочие дни и опорную дату потребления. '
        'Текущий лимит проверяется при въезде на дорогу и известен всем в день перевозки; '
        'знание будущих фактических ограничений ни одному плану не передаётся.', '',
        '## Результаты исходного сценария', '', md_table(table), '',
        f'В прогнозном плане дополнительной техникой перевезено {s.loc["forecast_m2","extra_t"]:,.0f} т '
        f'против {s.loc["calendar","extra_t"]:,.0f} т по календарю. '
        f'Разница надбавки за дополнительные машины: {s.loc["forecast_m2","extra_rub"]-s.loc["calendar","extra_rub"]:,.2f} ₽; '
        f'разница хранения: {s.loc["forecast_m2","hold_rub"]-s.loc["calendar","hold_rub"]:,.2f} ₽.'.replace(',',' '), '',
        'При текущих допущениях преимущество раннего завершения не компенсирует эти расходы. '
        'Прогнозный план совпал с максимально быстрым завозом. Одна и та же нулевая величина недовоза '
        'не означает одинаковую экономическую эффективность.', '',
        '## Спутник и временная изоляция', '', md_table(scene_table), '',
        'Наблюдения Sentinel-1 реальны. Обзорные изображения получены через официальный rendered_preview сервиса '
        'Planetary Computer. Это целые сцены с условными цветами, не вырезки 3×3 км над переправой. '
        'Исходные VV/VH-файлы недоступны из среды (HTTP 403); оценки М2 перепроверены текущим классификатором '
        'по сохранённым радарным признакам в репозитории.', '',
        '![Обзор радарной сцены 18 марта](../docs/replay_yakutsk_2026/20260318_scene_preview.png)', '',
        '![Обзор радарной сцены 11 апреля](../docs/replay_yakutsk_2026/20260411_scene_preview.png)', '',
        'Источник изображений: ESA/Copernicus Sentinel‑1; обработка RTC — Catalyst; каталог и обзоры — Microsoft Planetary Computer.', '',
        'Решение принимается в 00:00 UTC. Для снимка принят запас публикации 24 часа после конца съёмки; '
        'поэтому сцена 18 марта вечером используется только с 20 марта, а сцена 11 апреля — с 13 апреля. '
        'Точные исторические моменты первой публикации неизвестны. Срок свежести — 14 суток; '
        'январские снимки к началу опыта уже устарели.', '',
        '**Добавление М2 не изменило график и расходы в исходном сценарии.** '
        'Свежие снимки подтверждают наличие льда, но не исключают сценарии вскрытия из прогноза. '
        'Даже 11 апреля, после снижения допустимой массы до 20 т, классификатор всё ещё уверенно видит лёд. '
        'Это не измерение прочности или разрешённого веса. Из эффекта слияния удалён шум повторной оценки квантилей Монте-Карло.', '',
        'М1 и М2 обучены на целевых данных не позднее 2024 года. Файлы моделей зафиксированы; '
        'М1 запускается с физически обрезанной погодой. Базовый лаг погоды — пять суток; дополнительно проверены 0 и 7. '
        'ERA5 — позднейший реанализ, а не неизменённый архив оперативных наблюдений. Исторические версии погоды '
        'и реально опубликованные в 2026 году прогнозы не восстановлены; это ограничивает силу ретроспективного вывода.', '',
        '## Главная неопределённость модели', '',
        'М1 предсказывает **физическое вскрытие**, а не дату разрешения конкретной массы. '
        'В исходном расчёте последний день проезда = вскрытие −21 сутки; переходы к 30/20 т '
        'помещаются за 14/7 суток до этого дня, как в предварительном графике. '
        'Это явная гипотеза, а не калибровка по факту 2026 года. Запасы 14/21/28/35 суток проверены без переобучения. '
        'Нормативные ограничения определяются управляющим дорогой, а не классификатором М2.', '',
        '## Чувствительность', '', md_table(vol), '',
        f'Полная таблица содержит {fc_sens.shape[0]} вариантов параметров и все пять способов. '
        f'В {positive} из этих вариантов М1+М2 дешевле календаря. '
        'Это счёт по выбранной сетке, **не вероятность успеха в реальной эксплуатации**.', '',
        'Проверены объёмы 2 500–7 000 т, каждый вид расходов ±30%, основной и дополнительный парк, тара, '
        'полезная загрузка, оборот машин, шестидневная неделя, запас до закрытия, разброс календаря, '
        'лаг погоды и число сценариев Монте-Карло. Меняется один параметр за раз; фактический график ограничений фиксирован. '
        'Объёмы выше 6 660 т при базовом парке физически требуют резерва при любом способе.', '',
        'При повышенной нагрузке ранний завоз может уменьшить резервную доставку; точные условия и величины '
        'приведены в таблице. Этот эффект следует отделять от пользы именно М1: контроль «везти сразу» '
        'и календарь с разбросом позволяют увидеть вклад простого изменения стратегии.', '',
        f'**Пример с заказом 6 500 т:** адаптивный календарь перевозит по льду '
        f'{fmt(heavy.loc["calendar", "delivered_t"])} т, отправляя в резерв '
        f'{fmt(heavy.loc["calendar", "backup_t"])} т; расходы — '
        f'{fmt(heavy.loc["calendar", "cost_rub"])} ₽. '
        f'М1+М2 доставляет по льду все 6 500 т за {fmt(heavy.loc["forecast_m2", "cost_rub"])} ₽, '
        f'экономия — {fmt(heavy.loc["forecast_m2", "saving_vs_calendar_rub"])} ₽ '
        f'({heavy.loc["forecast_m2", "saving_vs_calendar_pct"]:.2f}%). '
        + '**Однако календарь ±9 суток и максимально быстрый завоз дают те же расходы и объём. '
        'Это преимущество более ранних отправок, а не доказанное преимущество машинного обучения.**', '',
        '### Как трактовать предварительное закрытие 15 апреля', '',
        'Отдельно повторён расчёт с более строгим прочтением предварительного объявления: '
        'последний плановый день — 14 апреля. Фактические ограничения, парк, затраты и прогнозы не менялись. '
        'После 14 апреля текущий разрешённый проезд по-прежнему известен всем при въезде. '
        'Для 5 000 т календарь остаётся дешевле. При 6 500 т ранние отправки по-прежнему уменьшают резерв, '
        'однако денежный эффект меньше. Контроли «календарь ±9 суток» и «везти сразу» опять дают тот же '
        'результат, что М1+М2.', '', md_table(boundary_table), '',
        '## Что требуется для проверки на практике', '',
        'Этот опыт пока не доказывает преимущество М1/М2 над простыми правилами отправки. '
        'Для такой проверки нужны фактический заказ, дневные отправки, оборот и стоимость машин хотя бы одного '
        'перевозчика, а также архив прогнозов, выпущенных до решений. Отдельно нужно связать прогноз '
        'вскрытия с допустимой полной массой и проверить эту связь на нескольких сезонах. '
        'Сравнение следует сохранять с адаптивным календарём и правилом «везти сразу», учитывая фиксированную '
        'аренду, целые рейсы и реальную потребность в запасе.', '',
        '## Воспроизведение и проверка', '',
        '```bash',
        'python scripts/prepare_yakutsk_replay.py',
        'python scripts/prepare_yakutsk_replay.py --weather-lag-days 0 --output-dir data/scenarios/lag0',
        'python scripts/prepare_yakutsk_replay.py --weather-lag-days 7 --output-dir data/scenarios/lag7',
        'python scripts/replay_yakutsk_2026.py',
        'python -m unittest discover -s tests -v',
        '```', '',
        '- [Итоги по методам](yakutsk_spring_2026_summary.csv)',
        '- [Каждая дневная отправка и её расходы](yakutsk_spring_2026_daily.csv)',
        '- [Журнал доступных прогнозов и снимков](yakutsk_spring_2026_decisions.csv)',
        '- [Полная чувствительность](yakutsk_spring_2026_sensitivity.csv)',
        '- [Проверка трактовки последнего планового дня](yakutsk_spring_2026_calendar_boundary.csv)',
        '- [Параметры, источники и контрольные суммы](../data/scenarios/yakutsk_spring_2026.json)',
        '- [Допущение о сезонном объёме](../data/scenarios/yakutsk_spring_2026_volume.json)',
        '- [Происхождение погодных и спутниковых данных](../data/scenarios/yakutsk_spring_2026_provenance.json)', '',
        '## Источники фактических ограничений', '',
    ]
    lines.extend(f'- [{key}]({url})' for key,url in SOURCES.items())
    (REPORTS/f'{PREFIX}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[['rule','delivered_t','backup_t','cost_rub','saving_vs_calendar_pct']].round(3).to_string(index=False))
    print(f'Report: {REPORTS/PREFIX}.md; sensitivity variants: {len(fc_sens)}')


if __name__ == '__main__':
    main()
