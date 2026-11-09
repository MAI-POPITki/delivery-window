"""Карта высокой воды; не использует прогнозы зимников или выдуманные уровни."""
import plotly.graph_objects as go
import streamlit as st

from okno.flood import ROOT, SOURCE, load_points


def make_map(rows):
    fig = go.Figure()
    for available, group in rows.groupby(rows.fraction.notna()):
        label = "Доля сценариев высокой воды" if available else "Нет актуального расчёта"
        marker = dict(size=17, color="#999999")
        text = [f"{r.river} — {r.name}<br>{r.status}" for r in group.itertuples()]
        if available:
            marker.update(color=group.fraction * 100, colorscale="YlOrRd", cmin=0, cmax=100,
                          showscale=True, colorbar=dict(title="Сценарии, %"))
            text = [f"{r.river} — {r.name}<br>Выше порога: {r.fraction:.0%} сценариев"
                    f"<br>{r.start} — {r.end}<br>Порог: {r.threshold:,.0f} м³/с"
                    for r in group.itertuples()]
        fig.add_trace(go.Scattermap(lat=group.lat, lon=group.lon, mode="markers+text",
                                   text=group.name, textposition="top right", hovertext=text,
                                   hoverinfo="text", customdata=group[["pid"]], marker=marker, name=label))
    fig.update_layout(map=dict(style="carto-voyager", center=dict(lat=59, lon=91), zoom=2.3),
                      height=540, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def render_flood():
    st.title("Паводки · прогноз высокой воды")
    st.caption("Экспериментальный модуль · GloFAS v4 / Open-Meteo · модельный расход, м³/с")
    st.info("Карта показывает долю сценариев превышения исторического порога расхода за ближайшие 7 суток. "
            "Это не вероятность затопления домов: уровни воды, ледовые заторы и границы затопления здесь не рассчитываются.")
    st.warning("Качество предупреждений ещё не подтверждено. В диагностике 2024 года порог расхода "
               "не был превышен у Орска, несмотря на известное наводнение. "
               "Этот показатель нельзя использовать как единственное основание для решений о безопасности.")
    rows, details = load_points()
    region = st.sidebar.selectbox("Регион паводков", ["Все четыре региона", *dict.fromkeys(rows.region)])
    shown = rows if region == "Все четыре региона" else rows[rows.region == region]
    st.plotly_chart(make_map(shown), width="stretch", config={"scrollZoom": True, "displaylogo": False}, key="flood_map")
    st.caption("Жёлтый → красный: 0–100% сценариев. Серый: расчёт отсутствует, повреждён или загрузка старше 48 часов. "
               "Точки обозначают речные ячейки модели (~5 км), а не гидропосты. Низкое значение не означает отсутствие угрозы.")
    pid = st.selectbox("Точка на реке", shown.pid.tolist(),
                       format_func=lambda p: f"{rows.set_index('pid').loc[p, 'river']} — {rows.set_index('pid').loc[p, 'name']}")
    if pid not in details:
        st.warning("Нет актуального расчёта: " + str(rows.set_index("pid").loc[pid, "status"]))
    else:
        d = details[pid]
        a, b, c = st.columns(3)
        a.metric("Сценарии выше порога за 7 суток", f"{d['fraction']:.0%}")
        b.metric("Порог высокой воды", f"{d['threshold']:,.0f} м³/с")
        c.metric("Пик медианного сценария", f"{d['peak']:,.0f} м³/с", help="Дата пика: " + d["peak_date"])
        curve = d["curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q90, line=dict(width=0), name="q90"))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q10, fill="tonexty", line=dict(width=0), name="q10–q90 ансамбля"))
        fig.add_trace(go.Scatter(x=curve.index, y=curve.q50, name="Медиана"))
        fig.add_hline(y=d["threshold"], line_dash="dash", annotation_text="Исторический порог")
        fig.update_layout(yaxis_title="Модельный расход, м³/с", xaxis_title="Дата UTC")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Горизонт {d['start']} — {d['end']} · {d['members']} полных сценариев · "
                   f"архив 1991–2020, {d['years']} полных лет. Загрузка: {d['fetched_at']}. "
                   "Время инициализации расчёта поставщик API не сообщает.")
        st.download_button("Скачать прогноз точки (CSV)", curve.to_csv(index_label="date").encode("utf-8-sig"),
                           file_name=f"{pid}_flood.csv", mime="text/csv")
    with st.expander("Данные, метод и проверки"):
        st.markdown(f"Источник: [Open-Meteo Flood API / GloFAS]({SOURCE}). "
                    "Модельные данные Copernicus CEMS, предоставленные через Open-Meteo (CC BY 4.0). "
                    "Порог — q80 годовых максимумов за 1991–2020; это исследовательское правило, "
                    "а не официальная опасная отметка. Доля сценариев считается по максимуму каждого "
                    "полного члена ансамбля за весь горизонт, а не по максимуму дневных вероятностей.")
        st.markdown("**Качество предупреждений о затоплении пока не подтверждено.** "
                    "Оперативные прогнозы сохраняются отдельно для последующей проверки; реанализ "
                    "не используется как якобы выпущенный в прошлом прогноз. "
                    "Водохранилища и ледовые заторы требуют отдельной проверки.")
        report = ROOT / "reports" / "flood_validation.md"
        if report.exists():
            st.markdown(report.read_text(encoding="utf-8"))
        st.markdown("Полный план проверки: `docs/FLOOD_VALIDATION_PLAN.md`; исходные ответы и SHA-256: `data/flood/`.")


if __name__ == "__main__":
    render_flood()
