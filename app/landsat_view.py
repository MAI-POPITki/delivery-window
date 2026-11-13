"""True-colour Landsat windows, optical M2 results and explicit quality flags."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from okno import landsat as L, m2_landsat as M
from okno.s1_points import LENA_NAMES


def watch_updates():
    from okno.landsat_refresh import start_auto_update
    start_auto_update()
    _watch()


@st.fragment(run_every="60s")
def _watch():
    path = L.DIRECTORY/"snapshot.json"
    version = path.stat().st_mtime_ns if path.exists() else 0
    previous = st.session_state.get("landsat_version", version)
    st.session_state["landsat_version"] = version
    if version != previous:
        st.rerun()


@st.fragment(run_every="60s")
def render_landsat(selected_point=None):
    with st.expander("Landsat 8/9 — снимки и оптическая М2", expanded=False):
        st.caption("USGS Collection 2 Level-2 через Microsoft Planetary Computer. "
                   "Отдельная обученная ветка М2 оценивает лёд и воду по спектральным каналам на русле. "
                   "Облака и тени исключаются. Результат не определяет грузоподъёмность переправы.")
        meta = M.metadata()
        if not meta.get("enabled"):
            st.info("Оптическая модель пока не прошла проверку качества. Уверенная оценка не выдаётся.")
            return
        validation = meta["validation"]
        st.write(f"Проверка на последующих сезонах: **{validation['accuracy']:.1%}** верных ответов "
                 f"на **{validation['n']} вырезках**, {validation['seasons']} сезона. "
                 "Разметка — по историческим спутниковым датам ледовых фаз; переходные периоды исключены.")
        path = L.DIRECTORY/"predictions.csv"
        if not path.exists():
            st.info("Снимки загружаются. Результаты появятся автоматически.")
            return
        predictions = pd.read_csv(path)
        usable = predictions[predictions.quality.eq("usable") & predictions.p_ice.notna()]
        latest = usable.sort_values("observed_at").groupby("point").tail(1)
        now = datetime.now(timezone.utc)
        table = []
        for key, name in LENA_NAMES.items():
            candidates = latest[latest.point == key]
            if candidates.empty:
                table.append({"Точка": name, "Снимок": "нет пригодного", "Источник": "—", "Состояние": "нет оценки"})
                continue
            row = candidates.iloc[0]
            age = (now.date()-datetime.fromisoformat(row.date).date()).days
            table.append({"Точка": name, "Снимок": row.date, "Источник": row.source,
                          "Состояние": "лёд" if row.p_ice >= 0.8 else "вода" if row.p_ice <= 0.2 else "неясно",
                          "P(лёд)": round(row.p_ice, 3), "Пригодных пикселей": f"{row.valid_fraction:.0%}", "Возраст, сут": age})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        keys = list(LENA_NAMES)
        key = st.selectbox("Точка для снимка Landsat", keys, format_func=LENA_NAMES.get,
                           index=keys.index(selected_point) if selected_point in keys else keys.index("yakutsk_crossing"),
                           key="landsat_point")
        candidates = latest[latest.point == key]
        if candidates.empty:
            st.info("На этом участке пока нет пригодного снимка: облака, тени, полярная ночь или нет данных.")
            return
        row = candidates.iloc[0]
        age = (now.date()-datetime.fromisoformat(row.date).date()).days
        if age > 14:
            st.warning(f"Снимку {age} суток. Он сохранён для просмотра; для текущей сверки М1 требуется снимок не старше 14 суток.")
        st.write(f"**{LENA_NAMES[key]} · {row.source} · {row.date} · P(лёд) = {row.p_ice:.2f}**")
        chip = L.chip_path(key, row.item_id)
        preview = L.DIRECTORY/"previews"/key/f"{row.item_id}.png"
        if chip.exists() and not preview.exists():
            L.make_preview(key, row.item_id, mask=M.load_masks().get(key))
        if preview.exists():
            st.image(str(preview), caption="Истинные цвета Landsat, участок около 6×6 км. Красный контур — используемая маска русла.", width=600)
        else:
            st.caption("Локальный кэш снимка восстанавливается фоновым загрузчиком.")
        st.caption(f"Съёмка: {row.observed_at}. Оценка появилась у проекта: {row.available_at}. "
                   "Эти данные не подставляются в расчёты на более ранние даты.")
        st.markdown(f"[Метаданные исходной сцены]({L.STAC}/collections/{L.COLLECTION}/items/{row.item_id}) · "
                    "[USGS: данные Landsat](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)")
        st.caption("Проверка новых снимков — раз в 6 часов при работающем сервере. "
                   "Новые кадры появляются по мере съёмки и публикации; зимой оптическая съёмка ограничена освещением.")
