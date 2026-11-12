"""Visual-review gallery; never produces an ice label or a forecast feature."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from okno import gaofen
from okno.s1_points import LENA_NAMES


@st.fragment(run_every="60s")
def render_gaofen(selected_point=None):
    with st.expander("Гаофэнь-3 — текущие обзорные снимки (автообновление)"):
        st.caption("Дополнительные радарные снимки для визуальной проверки реки. "
                   "Они доступны для просмотра и не участвуют в прогнозах и исторических расчётах. "
                   "Численный результат М2 по-прежнему рассчитывается по Sentinel-1.")
        snapshot = gaofen.load_snapshot()
        rows = snapshot.get("points", {})
        if not rows:
            st.info("Первая загрузка Гаофэнь-3 ещё выполняется. Этот блок обновится автоматически.")
            return
        table = []
        now = datetime.now(timezone.utc)
        for key, name in LENA_NAMES.items():
            row = rows.get(key, {})
            scene = row.get("scene", {})
            acquired = scene.get("acquired_date")
            age = (now.date()-datetime.fromisoformat(acquired).date()).days if acquired else None
            table.append({"Точка М2": name, "Дата съёмки": acquired or "нет",
                          "Возраст, сут": age, "Сцен с полным покрытием": row.get("covered_count", 0),
                          "Загрузка": "получен" if row.get("status") == "ready" else "источник недоступен" if row.get("status") == "source_unavailable" else "нет полного покрытия"})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        keys = list(LENA_NAMES)
        key = st.selectbox("Обзорный снимок для точки", keys, format_func=LENA_NAMES.get,
                           index=keys.index(selected_point) if selected_point in keys else keys.index("yakutsk_crossing"),
                           key="gaofen_browse_point")
        row = rows.get(key, {})
        scene = row.get("scene")
        if not scene:
            st.info("Обзорный снимок с полным покрытием этой точки пока не получен.")
        else:
            if row.get("status") == "source_unavailable":
                st.warning("Обновить источник не удалось. Ниже — ранее полученный снимок.")
            elif row.get("status") != "ready":
                st.info("Новых снимков с полным покрытием нет. Ниже — ранее полученный снимок.")
            acquired = scene["acquired_date"]
            age = (now.date()-datetime.fromisoformat(acquired).date()).days
            if age > 14:
                st.warning(f"Снимку {age} суток: он не показывает текущее состояние реки.")
            st.write(f"**{LENA_NAMES[key]} · {acquired} · {scene['polarization_code']}**")
            path = gaofen.preview_path(scene)
            if path.is_file():
                st.image(str(path), caption="Полная радарная сцена, в которую входит точка М2. Источник: NSOAS.",
                         width="stretch")
            else:
                st.caption("Изображение восстанавливается в локальном кэше; его можно открыть у источника.")
            st.link_button("Открыть обзорный снимок в NSOAS", scene["browse_url"])
            st.caption(f"Продукт: {scene['product_name']}. Часовой пояс времени съёмки каталог не указывает, показана дата источника.")
        checked = snapshot.get("checked_at", "")
        st.caption(f"Проверка каталога: {checked}. Источники проверяются раз в 6 часов, пока сервер приложения работает. "
                   "Новые кадры появляются по мере публикации спутниковой съёмки.")
        st.markdown("[Каталог и источник снимков — NSOAS](https://osdds.nsoas.org.cn/HighResolutionSatellite)")
