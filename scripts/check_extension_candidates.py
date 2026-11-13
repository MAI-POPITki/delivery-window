"""Compare evidence for implemented transfers, without equating unit tests to accuracy."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from okno import waterbodies as W,handling_wind as H
from okno.truck_load import check_load
from scripts.download_waterbody import read_scene


def weather(gust=False,cold=False,rain=False,missing=False,budget=24):
    now=datetime(2026,11,13,tzinfo=timezone.utc);times=pd.date_range(now,periods=72,freq='h')
    hourly=dict(time=times.strftime('%Y-%m-%dT%H:%M').tolist(),wind_speed_10m=[1.]*72,wind_gusts_10m=[2.]*72,
        temperature_2m=[-20.]*72,precipitation=[0.]*72)
    if gust:hourly['wind_gusts_10m'][33]=15.
    if cold:hourly['temperature_2m'][33]=-51.
    if rain:hourly['precipitation'][33]=.2
    if missing:hourly['wind_speed_10m'][33]=None
    payload=dict(timezone='GMT',utc_offset_seconds=0,hourly=hourly,
        hourly_units=dict(wind_speed_10m='m/s',wind_gusts_10m='m/s',temperature_2m='°C',precipitation='mm'))
    e=dict(payload=payload,source=H.API,kind='weather_forecast',fetched_at=now.isoformat(),
        sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return float(H.capacity(e,[pd.Timestamp('2026-11-14').date()],as_of=now,tz='UTC',speed_limit=10,gust_limit=15,
        duration_h=2,batch_t=10,start_hour=8,end_hour=12,min_temperature=-50,max_precipitation=0,max_operation_hours=budget).capacity_t.iloc[0])


def main():
    root=W.M.ROOT;R=root/'reports';meta,mask,stats,arrays=W.compare_sample()
    # Recompute the actual saved masks from spectral crops (not from output masks).
    source=[]
    for i,scene in enumerate(meta['scenes']):
        item=json.loads((W.DATA/'scenes'/scene['item_id']/'stac.json').read_text())
        bands,scl=read_scene(item,meta['grid'],W.DATA/'scenes'/scene['item_id'])
        valid,water=W.M.classify(bands,scl)
        np.testing.assert_array_equal(valid,arrays[f'valid_{i}']);np.testing.assert_array_equal(water,arrays[f'water_{i}'])
        source.append((bands,scl))
    sensitivity=[]
    for shift in (-.05,0.,.05):
        # Fixed algorithm, diagnostic perturbation only; no threshold tuned to get desired change.
        from unittest.mock import patch
        with patch.dict(W.M.PARAMS,{'mndwi_min':.1+shift}):
            a=[W.M.classify(bands,scl) for bands,scl in source]
        _,s=W.compare_masks(*a[0],*a[1],arrays['aoi'],W.pixel_area_km2(meta['grid']))
        sensitivity.append(dict(mndwi_min=.1+shift,delta_common_km2=s['delta_common_km2'],common_fraction=s['common_fraction']))
    Image.fromarray(W.image(mask,arrays['aoi'])).save(R/'waterbody_change.png')
    low=json.loads((R/'low_water_metrics.json').read_text())
    road=[]
    for name,tare,cargo,limit,excess,empty in [('24 т против 20 т',12,12,20,4,False),
        ('Ровно 20 т',10,10,20,0,False),('Пустая машина тяжелее лимита',24,0,20,4,True)]:
        result=check_load(tare,cargo,limit)
        assert result['excess_t']==excess and result['tare_exceeds_limit']==empty
        road.append(dict(case=name,**result))
    handling=[]
    for name,kwargs,expected in [('Без помех',{},20),('Порыв на пределе',{'gust':True},10),
        ('Мороз ниже сценарного предела',{'cold':True},10),('Осадки выше сценарного предела',{'rain':True},10),
        ('Бюджет часов меньше длительности партии',{'budget':1},0),('Неизвестный час',{'missing':True},0)]:
        actual=weather(**kwargs);assert actual==expected
        handling.append(dict(case=name,expected_capacity_t=expected,actual_capacity_t=actual))
    output=dict(waterbody=stats,mndwi_sensitivity=sensitivity,road_mass=road,handling=handling)
    (R/'extension_candidates.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    wb=['# Водоёмы: эксперимент беты','',f"Объект: {meta['name']}, {meta['region']}. Sentinel‑2 L2A через Earth Search/AWS, "
        f"{meta['scenes'][0]['acquired_at'][:10]} и {meta['scenes'][1]['acquired_at'][:10]}.",
        '', '## Что измерено','',
        f"Фиксированная область: {stats['aoi_km2']:.4f} км². Геометрия OSM плюс 8 пикселей буфера. "
        'Это контрольная область вокруг водоёма, не независимая историческая разметка берегов.',
        f"Совместно видно {stats['common_fraction']:.2%}, или {stats['common_km2']:.4f} км². "
        f"На этой части вода: {stats['before_common_km2']:.4f} → {stats['after_common_km2']:.4f} км². "
        f"Разность {stats['delta_common_km2']:+.4f} км² — всего {round(stats['delta_common_km2']/W.pixel_area_km2(meta['grid']))} пикселя.",
        f"Простое вычитание наблюдаемых площадей разных дат дало бы "
        f"{stats['after_observed_water_km2']-stats['before_observed_water_km2']:+.4f} км²: "
        'оно смешивает изменение воды и различие видимости. В интерфейсе используется общая видимая область.',
        f"Границы изменения всей области только из-за неизвестных пикселей: "
        f"{stats['full_area_delta_bounds_km2'][0]:+.4f}…{stats['full_area_delta_bounds_km2'][1]:+.4f} км². "
        'Они включают ноль: знак изменения всей области не установлен. Спектральные ошибки в эти границы не включены.',
        '', '## Проверки','',
        'Обе маски повторно рассчитаны из сохранённых спектральных вырезок, совпадение точное. '
        'SHA-256 проверяется для вырезок, каталога, изображений и масок. Сетка единая UTM, '
        '20×20 м, площадь пикселя 0,0004 км². Порог совместной видимости для вывода — 80% (политика беты).',
        '', '| MNDWI > | Изменение на общей области, км² | Совместная видимость |','|---:|---:|---:|']
    for s in sensitivity:wb.append(f"| {s['mndwi_min']:.2f} | {s['delta_common_km2']:+.4f} | {s['common_fraction']:.2%} |")
    wb+=['','Изменение порога может менять и общий набор допустимых пикселей. Это диагностика чувствительности, '
        'не оценка точности и не доверительный интервал. Порог после просмотра не подбирался.', '',
        '## Оценка результата','',
        '**Расчётная бета работает, выраженного изменения площади на этой паре снимков не подтверждено.** '
        'Два пикселя нельзя рекламировать как доказанное движение берега. Независимых датированных контуров '
        'или полевых наблюдений нет; IoU и ошибка площади для этого водоёма не измерены. '
        'Согласие с SCL не является независимой проверкой: SCL используется в рабочем правиле.', '',
        'Для пилота: независимые контуры нескольких дат, разные сезоны и облачность, чувствительность '
        'прибрежных смешанных пикселей. Для объёма и глубины нужны батиметрия и уровни; по площади они не выводятся.', '',
        '## Источники и воспроизведение','',
        'Earth Search: https://earth-search.aws.element84.com/v1 · Contains modified Copernicus Sentinel data 2021.',
        'AOI: '+meta['aoi']['source_url']+' · © OpenStreetMap contributors, ODbL.',
        'Использованы исходные продукты baseline 03.00/03.01 с scale из STAC и нулевым offset. '
        'Продукты baseline ≥04 пока отклоняются: нужна независимая проверка поправок у источника. '
        'Рабочая паводковая М2 не менялась.',
        '`python scripts/download_waterbody.py` — источник и вырезки, '
        '`python scripts/check_extension_candidates.py` — повторный расчёт и отчёты.']
    (R/'waterbody_beta.md').write_text('\n'.join(wb)+'\n')
    s1,s7=low['scores']['1'],low['scores']['7']
    lines=['# Перенос модулей: что удалось и что пока не доказано','',
        'Состояние на 13.11.2026. Программные тесты, архивные метрики и проверка реальной пользы '
        'отдельны: число пройденных тестов не является точностью модели.', '',
        '| Возможность | Проверка | Результат | Готовность |','|---|---|---|---|',
        f"| Низкая вода: Лена, пост 3045 | июнь–сентябрь 2019–2023, 1–7 суток, оценочный год исключён | за сутки MAE {s1['mae_cm']:.2f} против {s1['persistence_mae_cm']:.2f} см на {s1['n']} прогнозах; за неделю {s7['mae_cm']:.2f} против {s7['persistence_mae_cm']:.2f} см | наиболее убедительный перенос прогноза на архиве; не оперативный допуск |",
        '| Площадь водоёма | два реальных снимка, общая видимость, повторный расчёт и чувствительность | расчёт работает; изменение 2 пикселя, знак для всей области неизвестен; независимой точности нет | бета наблюдения; сильное изменение на примере не подтверждено |',
        '| Масса грузовика | 3 сценария с известным ответом + существующие тесты датированных дорожных ограничений | 12+12 т против 20 т: снять минимум 4 т; пустая машина выше лимита не допускается снижением груза | расчёт при известных входах; профиль не подтверждает популярность машины или безопасность дороги |',
        '| Погода и смены для погрузки/разгрузки М3 | 6 сценариев с заданной погодой, лимитами и ожидаемой мощностью | ограничения уменьшают мощность; неизвестный час не разрешает отправку | функциональность проверена; нет план-факта перевозчика и доказанной экономии |',
        '| Ветер как дополнительный признак М2 | ранее выполненный past-only эксперимент: 666 сцен, 4 сезона | среднее +0,28 п.п. balanced accuracy; интервал −0,56…+0,99 п.п. | устойчивое улучшение не доказано; рабочая модель не заменена |',
        '| Снабжение стройплощадок и сезонная судовая навигация | отдельная бета и предметные данные отсутствуют | результат не измерен | идеи, не реализованные возможности |', '',
        '## Воспроизводимые автомобильные сценарии','',
        'Массы, погода и пороги ниже — явные тестовые условия, не статистика автопарков, '
        'не заводские нормативы и не ограничения для любого грузовика. '
        'В UI добавлена подпись «Расчёт по типовому профилю техники». Новые марки/профили не выдумывались.', '',
        '| Сценарий | Ожидаемая мощность, т | Получено, т |','|---|---:|---:|']
    for r in handling:lines.append(f"| {r['case']} | {r['expected_capacity_t']:g} | {r['actual_capacity_t']:g} |")
    lines+=['','В этих сценариях: смена 08–12 UTC, партия 10 т за 2 часа, ветер <10 м/с, '
        'порывы <15 м/с, температура не ниже −50 °C, без осадков. Это сценарные параметры оператора. '
        'Графики труда и отдыха конкретных водителей, тормозной путь, осевые нагрузки '
        'и устойчивость автопоезда этими сценариями не проверены.', '',
        '## Приоритет','',
        'Для презентации сильнее всего подтверждён архивный прогноз низкой воды одного поста. '
        'Автомобильные ограничения М3 понятны и проверяемы, но требуют реальных норм оборудования и рейсов '
        'для оценки пользы. Водоёмы можно показывать как перенос спутникового наблюдения, '
        'без обещания точности площади или доказанного обмеления. Ветер в М2 пока не даёт устойчивого выигрыша.', '',
        'Подробности: [низкая вода](low_water_beta.md), [водоём](waterbody_beta.md), '
        '[ветер в М2](m2_wind_experiment.md). Сценарные результаты: `extension_candidates.json`.']
    (R/'extension_candidates.md').write_text('\n'.join(lines)+'\n')
    print('Reports ready: low water / waterbody / truck mass / handling; no field-validation claim.')


if __name__=='__main__':main()
