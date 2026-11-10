from okno import map_style as _flood_map_style
"""Одна пользовательская задача: наблюдение → прогноз → план доставки."""
import base64
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno.flood_assessment import assess
from okno.flood_routes import load_routes
from okno.flood import POINTS


def render_assessment(pid,decision,forecast,threshold,probabilistic):
    try:result=assess(pid,decision,forecast,threshold,probabilistic)
    except (ValueError,OSError,KeyError) as exc:
        st.error('Не удалось проверить согласованность источников. Общий вывод не сформирован.');return
    st.subheader(f"{POINTS[pid]['name']}: от обстановки к плану")
    st.caption(f"Решение на {result['decision']:%d.%m.%Y} · прогноз на {result['target']:%d.%m.%Y}. "
               'Все использованные наблюдения получены съёмкой не позже даты решения; архив загружен позднее.')
    a,b=st.columns(2)
    a.markdown('**Что ожидается · М1**')
    a.write(f"Уровень {result['predicted_cm']:.0f} см над нулём поста.")
    if result['probability'] is not None:
        a.write(f"Оценка превышения {threshold:.0f} см: **{result['probability']:.1%}**.")
    else:a.write('Вероятность превышения не рассчитана: выберите событие в настройках прогноза.')
    a.caption('Процент относится к уровню на посту, не к затоплению дороги.')
    b.markdown('**Что наблюдается · М2**')
    b.write(result['satellite_status'])
    sat=result['satellite']
    if sat:
        meta=sat['meta'];stats=meta['statistics']
        b.write(f"Снимок {pd.Timestamp(meta['acquired_at']):%d.%m.%Y}; возраст на решение {sat['age_days']} сут.; "
                f"видно {stats['observed_fraction']:.0%}.")
        if stats['observed_fraction']>0:
            b.caption(f"Видимая вода {stats['water_km2']:.2f} км²; новая относительно летней базы {stats['new_water_km2']:.2f} км². "
                      'При плохом наблюдении малая площадь не означает отсутствия паводка.')
        else:
            b.caption('Вся территория неизвестна. Площадь воды не оценена, нулевой паводок не заявляется.')
    st.info('**Результат для доставки · М3:** прогноз и спутник показаны в одном контексте. '
            'Для допустимого плана ниже нужны маршрут и подтверждённые условия проезда. '
            'Общий шанс проезда пока не откалиброван и не вычисляется перемножением процентов М1 и М2.')
    if sat:
        lon,lat=sat['meta']['station']['geometry']['coordinates']
        roads,_=load_routes(pid);xs=[];ys=[]
        for r in roads:
            xs.extend([p[0] for p in r['coordinates']]+[None]);ys.extend([p[1] for p in r['coordinates']]+[None])
        fig=go.Figure(go.Scattermap(lon=xs,lat=ys,mode='lines',line=dict(color='#475569',width=2),hoverinfo='skip'))
        fig.add_trace(go.Scattermap(lon=[lon],lat=[lat],mode='markers',marker=dict(size=13,color='#111827'),
            text=[f"М1 на {result['target']}: {result['predicted_cm']:.0f} см"],hoverinfo='text'))
        source='data:image/png;base64,'+base64.b64encode((sat['path'].parent/'overlay.png').read_bytes()).decode()
        fig.update_layout(map=dict(style=_flood_map_style.FLOOD_LOCAL_MAP_STYLE,center=dict(lon=lon,lat=lat),zoom=11,
            layers=[dict(sourcetype='image',source=source,coordinates=sat['meta']['overlay_coordinates'],below='traces',opacity=.8)]),
            height=400,margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
        st.plotly_chart(fig,width='stretch')
        st.caption('Карта — наблюдение на дату снимка, не прогноз контура затопления. '
                   'Красный: новая вода; синий: остальная вода; зелёный: суша; серый: неизвестно. '
                   'Мосты и глубина не учтены. © Copernicus Sentinel / OpenStreetMap contributors.')
    from app.flood_m3_view import render_m3
    render_m3(pid,decision,forecast)
