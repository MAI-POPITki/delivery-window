"""Карта и понятная сводка внешнего прогноза высокой воды."""
from datetime import datetime
import math

import plotly.graph_objects as go
import streamlit as st

from okno.flood import ROOT, SOURCE, load_points


def human_date(value):
    return datetime.fromisoformat(value).strftime("%d.%m.%Y")


def number(value):
    return f"{value:,.0f}".replace(",", " ")


def make_map(rows, selected=None):
    fig = go.Figure()
    for available, group in rows.groupby(rows.fraction.notna()):
        marker = dict(size=[23 if pid == selected else 15 for pid in group.pid], color="#8a929e")
        text = [f"{r.river} · {r.name}<br>Нет актуального прогноза" for r in group.itertuples()]
        if available:
            marker.update(color=group.fraction * 100,
                          colorscale=[[0, "#facc15"], [.3, "#fb923c"], [1, "#b91c1c"]], cmin=0, cmax=100,
                          showscale=True, colorbar=dict(title="Превышение<br>порога, %", thickness=12))
            text = [f"{r.river} · {r.name}<br>Выше порога: {r.fraction:.0%} вариантов"
                    f"<br>{human_date(r.start)} — {human_date(r.end)}"
                    for r in group.itertuples()]
        labels = [r.name if len(rows) <= 2 or r.pid == selected else "" for r in group.itertuples()]
        fig.add_trace(go.Scattermap(lat=group.lat, lon=group.lon, mode="markers+text",
                                   text=labels, textposition="top right", hovertext=text,
                                   textfont=dict(color="#172033", size=13),
                                   hoverinfo="text", customdata=group[["pid"]], marker=marker))
    lat_span = float(rows.lat.max() - rows.lat.min())
    lon_span = float(rows.lon.max() - rows.lon.min())
    zoom = max(2., min(7., math.log2(180 / max(lat_span * 2, lon_span, 1)) + .4))
    fig.update_layout(map=dict(style="carto-voyager", center=dict(lat=float((rows.lat.min() + rows.lat.max()) / 2),
                      lon=float((rows.lon.min() + rows.lon.max()) / 2)), zoom=zoom), height=470,
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def render_flood():
    st.title("Паводки")
    st.caption("Прогноз высокой воды на ближайшие 7 дней · Якутия, Тюменская, Курганская и Оренбургская области")
    st.warning("Экспериментальный прогноз. Он оценивает высокий расход воды в реке; "
               "затопление территорий и ледовые заторы пока не прогнозируются. "
               "Для решений о безопасности следуйте сообщениям МЧС и гидрометслужбы.")
    rows, details = load_points()
    region = st.sidebar.selectbox("Регион паводков", ["Все четыре региона", *dict.fromkeys(rows.region)])
    shown = rows if region == "Все четыре региона" else rows[rows.region == region]
    indexed = rows.set_index("pid")
    ids = shown.pid.tolist()
    if st.session_state.get("flood_point") not in ids:
        st.session_state["flood_point"] = ids[0]
    pid = st.selectbox("Река и населённый пункт", ids, key="flood_point",
                       format_func=lambda p: f"{indexed.loc[p, 'river']} · {indexed.loc[p, 'name']}")
    st.plotly_chart(make_map(shown, pid), width="stretch",
                    config={"scrollZoom": True, "displaylogo": False}, key="flood_map")
    st.caption("Цвет — доля вариантов прогноза с высокой водой: от светлого (0%) до красного (100%). "
               "Серый — нет актуальных данных. Выбранная точка крупнее; остальные названия видны при наведении. "
               "Низкая оценка не исключает затопления.")
    if pid not in details:
        st.info("Прогноз для этой точки пока недоступен. " + str(indexed.loc[pid, "status"]))
    else:
        d = details[pid]
        st.subheader(f"{indexed.loc[pid, 'river']} · {indexed.loc[pid, 'name']}")
        st.write(f"**{human_date(d['start'])} — {human_date(d['end'])}**")
        a, b, c = st.columns(3)
        exceeded = round(d["fraction"] * d["members"])
        a.metric("Варианты с высокой водой", f"{exceeded} из {d['members']}",
                 help="Число сценариев, в которых исторический порог превышен хотя бы раз за семь дней. Это не вероятность затопления.")
        b.metric("Ожидаемый максимум расхода", f"{number(d['peak'])} м³/с",
                 help="Наибольшее значение медианного прогноза. Расход — объём воды, проходящий за секунду.")
        c.metric("Дата ожидаемого максимума", human_date(d["peak_date"]),
                 help="Дата максимума в пределах этих семи дней, а не дата пика всего половодья.")
        curve = d["curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q90, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q10, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(37,99,235,.15)", name="Диапазон большинства вариантов",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q50, name="Ожидаемый расход",
                                 line=dict(color="#2563eb", width=3),
                                 hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.0f} м³/с<extra></extra>"))
        if st.checkbox("Показать порог высокой воды на графике", key="flood_show_threshold"):
            fig.add_hline(y=d["threshold"], line_dash="dash", line_color="#b45309",
                          annotation_text="Порог высокой воды")
        fig.update_layout(yaxis_title="Расход воды, м³/с", xaxis=dict(tickformat="%d.%m"),
                          height=330, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15),
                          hovermode="x unified")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(f"Порог высокой воды: {number(d['threshold'])} м³/с. "
                   "Заливка охватывает средние 80% вариантов модели и не гарантирует попадание фактического расхода.")
        fetched = datetime.fromisoformat(d["fetched_at"]).strftime("%d.%m.%Y в %H:%M UTC")
        st.caption(f"Данные получены {fetched}. Даты прогноза указаны по UTC.")
        export = curve.rename(columns={"q10": "Нижняя граница, м³/с", "q50": "Медиана, м³/с",
                                       "q90": "Верхняя граница, м³/с"})
        st.download_button("Скачать прогноз (CSV)", export.to_csv(index_label="Дата UTC").encode("utf-8-sig"),
                           file_name=f"{pid}_flood.csv", mime="text/csv")
    with st.expander("Как читать прогноз и насколько ему можно доверять"):
        st.markdown("Модель рассчитывает несколько вариантов движения воды по речной сети. "
                    "Мы считаем, в скольких из них расход превысит исторический порог хотя бы раз "
                    "за семь суток. **Это не вероятность затопления населённого пункта.** "
                    "Модельные точки имеют разрешение около 5 км и не заменяют замеры гидропостов.")
        st.markdown("**Проверка точности ещё не завершена.** В историческом ряду 2024 года порог был превышен "
                    "у Ишима, Кургана и Оренбурга, но не у Орска, несмотря на известное наводнение. "
                    "Модуль может пропустить опасную ситуацию. Учёт заторов и управления водохранилищами "
                    "требует дополнительной проверки.")
        st.markdown(f"Источник: [GloFAS / Open-Meteo]({SOURCE}), данные Copernicus CEMS, CC BY 4.0. "
                    "Это расчётные расходы, а не измеренные уровни воды. Порог — 80-й процентиль "
                    "годовых максимумов по 24 полным годам 1997–2020. Пропуски не заполняются. "
                    "Время инициализации расчёта поставщик не сообщает; старые или неполные данные скрываются.")
        for label, filename in (("Результаты проверки данных", "flood_validation.md"),
                                ("Проверки программы", "flood_implementation_checks.md")):
            path = ROOT / "reports" / filename
            if path.exists():
                st.download_button(label, path.read_bytes(), file_name=filename, mime="text/markdown")


if __name__ == "__main__":
    render_flood()
