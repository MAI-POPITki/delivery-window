"""Archived open-water area comparison for one reservoir."""
import json

import numpy as np
from PIL import Image
import streamlit as st

from okno import waterbodies as W


def render_waterbody():
    st.title('Водоёмы · бета')
    try:
        meta,mask,s,arrays=W.compare_sample()
    except (ValueError,KeyError,OSError) as exc:
        st.error('Пример недоступен: '+str(exc))
        return
    first,last=meta['scenes'][0],meta['scenes'][1]
    st.write(f"**{meta['name']}** · {meta['region']}")
    st.info(f"Архивные наблюдения Sentinel‑2: {first['acquired_at'][:10]} и {last['acquired_at'][:10]}. "
            'Это сравнение снимков, не прогноз будущего состояния.')
    st.caption('Считаем наблюдаемую воду в фиксированной области вокруг водоёма. '
               'Изменение сравнивается только там, где поверхность видна на обе даты. '
               'Площадь не определяет объём воды, глубину или возможность проезда.')
    a,b,c,d=st.columns(4)
    a.metric('Вода раньше · общая видимая область',f"{s['before_common_km2']:.3f} км²")
    b.metric('Вода позже · та же область',f"{s['after_common_km2']:.3f} км²")
    c.metric('Изменение на общей области',f"{s['delta_common_km2']:+.3f} км²" if s['comparable'] else 'Недостаточно видимости')
    d.metric('Видно на обе даты',f"{s['common_fraction']:.1%}")
    st.caption(f"Разность соответствует {abs(s['delta_common_km2'])/W.pixel_area_km2(meta['grid']):.0f} пикселям. "
               'Малое изменение не подтверждает движение берега без независимой проверки.')
    if not s['comparable']:
        st.warning('На обе даты видно меньше 80% контрольной области. Вывод об изменении площади не формируем.')
    if s['full_area_direction']=='unknown':
        st.warning('Неизвестные пиксели не позволяют установить знак изменения всей области. '
                   'Показанная разница относится только к совместно видимой части.')
    view,quality,source=st.tabs(['Снимки и изменение','Что проверено','Источники'])
    with view:
        l,r=st.columns(2)
        for i,col in enumerate([l,r]):
            with Image.open(W.DATA/meta['scenes'][i]['rgb']) as im:rgb=np.array(im)
            # Outline the exact fixed AOI; outside is context, not measured area.
            from scipy.ndimage import binary_erosion
            edge=arrays['aoi'] & ~binary_erosion(arrays['aoi'])
            rgb[edge]=[255,205,40]
            col.image(rgb,caption=meta['scenes'][i]['acquired_at'][:10]+' · RGB, жёлтым — контрольная область',width='stretch')
        st.image(W.image(mask,arrays['aoi']),caption='Изменение наблюдаемой водной поверхности · сетка 20 м',width='stretch')
        st.caption('Синий — вода на обе даты; оранжевый — вода → наблюдаемая суша; '
                   'бирюзовый — наблюдаемая суша → вода; серый — неизвестно хотя бы на одну дату; '
                   'светлый — суша на обе даты. Белое вне области в расчёт не входит.')
        st.write(f"Вода появилась на **{s['gain_km2']:.3f} км²**, перестала наблюдаться на **{s['loss_km2']:.3f} км²** "
                 'совместно видимой области. Это результат спектральной классификации, не подтверждение причины изменения.')
        st.download_button('Скачать результат и источники',json.dumps(dict(metadata=meta,comparison=s),ensure_ascii=False,indent=2).encode(),
                           'waterbody-area.json','application/json')
    with quality:
        st.write(f"Контрольная область: {s['aoi_km2']:.3f} км². Совместно видимая: {s['common_km2']:.3f} км².")
        st.write(f"Видимость раньше: {s['before_observed_fraction']:.1%}; позже: {s['after_observed_fraction']:.1%}.")
        lo,hi=s['full_area_delta_bounds_km2']
        st.write(f'Границы изменения для всей области с учётом неизвестных пикселей: **{lo:+.3f}…{hi:+.3f} км²**.')
        st.caption('Это границы только из-за невидимой поверхности, а не доверительный интервал. '
                   'Ошибки спектрального определения воды в них не учтены.')
        st.write('Проверяются единая сетка, площадь пикселя в метрической проекции, даты, '
                 'целостность файлов и исключение облаков, теней, снега и льда. '
                 'Независимых контуров этого водоёма на эти даты пока нет: локальная точность площади не установлена.')
        report=W.M.ROOT/'reports/waterbody_beta.md'
        if report.exists():st.download_button('Отчёт эксперимента',report.read_bytes(),report.name,'text/markdown')
    with source:
        st.markdown('**Sentinel‑2 L2A:** [Earth Search / Element 84](https://earth-search.aws.element84.com/v1). '
                    'Contains modified Copernicus Sentinel data 2021.')
        for scene in meta['scenes']:
            st.markdown(f"[{scene['item_id']}]({scene['source_url']}) · съёмка {scene['acquired_at']}, "
                        f"baseline {scene['processing_baseline']}, продукт {scene['product_generated_at']}.")
        st.markdown(f"**Область:** [OpenStreetMap]({meta['aoi']['source_url']}), © OpenStreetMap contributors, ODbL. "
                    +meta['aoi']['note'])
        st.write('Повторно используется рабочее спектральное правило паводкового М2 (MNDWI, NDVI и ближний ИК). '
                 'Новая модель не обучалась. Область OSM служит границей анализа, а не независимой разметкой точности.')
