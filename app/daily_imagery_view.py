"""Compare MODIS and VIIRS daily browse images at the same point and date."""
from datetime import datetime, timezone

import streamlit as st

from okno import daily_imagery as D
from okno.s1_points import LENA_NAMES


def watch_updates():
    from okno.daily_imagery_refresh import start_auto_update
    start_auto_update()
    _watch()


@st.fragment(run_every="60s")
def _watch():
    path = D.DIRECTORY/"snapshot.json"
    version = path.stat().st_mtime_ns if path.exists() else 0
    previous = st.session_state.get("daily_imagery_version", version)
    st.session_state["daily_imagery_version"] = version
    if previous != version:
        st.rerun()


def _picture(row, point, source, day):
    st.write(f"**{D.SOURCES[source]['name']} · {day}**")
    if not row or row.get("status") in ("not_published", "source_unavailable"):
        st.info("За эту дату изображение ещё не опубликовано или источник недоступен. Выберите другой день или спутник.")
        return
    if row.get("status") == "no_data":
        st.info("В этом окне нет изображения за выбранный день: пропуск съёмки, тёмная область или полярная ночь. Выберите другую дату или спутник.")
        return
    if row.get("status") == "partial":
        st.warning("В мозаике есть пустые или очень тёмные участки; часть окна может быть не покрыта съёмкой.")
    path = D.preview_path(point, source, day)
    if path.exists():
        st.image(str(path), caption=f"Окно около 80×80 км; в центре — {LENA_NAMES[point]}.", width="stretch")
        st.download_button("Скачать PNG", path.read_bytes(), f"{point}_{source}_{day}.png", "image/png",
                           key=f"daily_download_{source}")
    else:
        st.caption("Снимок восстанавливается в кэше. Его можно открыть у источника.")
    st.link_button("Открыть изображение NASA", row["url"])
    st.caption(f"{D.SOURCES[source]['resolution']}. Получено проектом: {row['fetched_at']}.")


@st.fragment(run_every="60s")
def render_daily_imagery(selected_point=None):
    with st.expander("MODIS / VIIRS — ежедневные снимки NASA"):
        st.caption("Цветные суточные мозаики NASA GIBS для визуального обзора. "
                   "Облака могут скрывать реку; здесь нет автоматической оценки льда или облачности.")
        snapshot = D.load_snapshot()
        records = snapshot.get("records", [])
        if not records:
            st.info("Первая загрузка ежедневных снимков выполняется в фоне. Галерея обновится автоматически.")
            return
        keys = list(LENA_NAMES)
        point = st.selectbox("Точка ежедневного обзора", keys, format_func=LENA_NAMES.get,
                             index=keys.index(selected_point) if selected_point in keys else keys.index("yakutsk_crossing"),
                             key="daily_imagery_point")
        rows = [r for r in records if r["point"] == point]
        dates = sorted({r["date"] for r in rows}, reverse=True)
        if not dates:
            st.info("Для этой точки изображения ещё загружаются.")
            return
        # Start at the latest day with actual local imagery, while keeping empty
        # dates selectable. Never silently substitute an older day for a request.
        available = [d for d in dates if any(r["date"] == d and r.get("status") in ("image", "partial") for r in rows)]
        selected = st.session_state.get("daily_imagery_date")
        if selected not in dates:
            st.session_state["daily_imagery_date"] = available[0] if available else dates[0]
        day = st.selectbox("Дата суточной мозаики (UTC)", dates, key="daily_imagery_date")
        age = (datetime.now(timezone.utc).date()-datetime.fromisoformat(day).date()).days
        if age > 2:
            st.warning(f"Показан архивный обзор {day}, возраст {age} суток.")
        if snapshot.get("errors"):
            st.warning("Часть источников не удалось обновить. Сохранённые кадры показаны со своими датами.")
        left, right = st.columns(2)
        for column, label, sources in ((left, "Спутник MODIS", ["modis_terra", "modis_aqua"]),
                                       (right, "Спутник VIIRS", ["viirs_noaa21", "viirs_noaa20"])):
            with column:
                source = st.selectbox(label, sources, format_func=lambda key: D.SOURCES[key]["name"],
                                      key="daily_"+sources[0])
                row = next((r for r in rows if r["source"] == source and r["date"] == day), None)
                _picture(row, point, source, day)
        st.caption("Проверка новых публикаций — раз в 6 часов при работающем сервере. "
                   "Спутники снимают ежедневно, но пригодный обзор конкретной реки бывает не каждый день. "
                   "Дата относится к суточной мозаике; точное время пролёта здесь не задано.")
        st.caption(f"Источник: NASA GIBS / ESDIS. Последняя проверка: {snapshot.get('checked_at', 'не указана')}. "
                   "Численные прогнозы М1 и М2 используют свои проверенные данные.")
        st.markdown("[NASA Worldview](https://worldview.earthdata.nasa.gov/) · "
                    "[О сервисе NASA GIBS](https://www.earthdata.nasa.gov/data/tools/gibs)")
