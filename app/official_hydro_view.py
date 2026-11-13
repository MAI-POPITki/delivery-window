"""Measured post data and river-reach context for all supported regions."""
from datetime import datetime, timezone
import re

import pandas as pd
import streamlit as st

from okno import official_hydro as H
from okno.regions import REGIONS, label


def watch_updates():
    from okno.hydro_refresh import start_auto_update
    start_auto_update()
    _watch_snapshot()


@st.fragment(run_every="60s")
def _watch_snapshot():
    version = H._version(H.SNAPSHOT)
    previous = st.session_state.get("official_hydro_version", version)
    st.session_state["official_hydro_version"] = version
    if previous != version:
        st.rerun()


def _date(value):
    return H.stamp(value).strftime("%d.%m.%Y %H:%M UTC") if pd.notna(value) and value else "не указана"


def render_point(point_id, issue_date):
    snapshot = H.load_snapshot()
    with st.expander("Официальные наблюдения и сводки", expanded=True):
        if not snapshot:
            st.info("Официальные источники проверяются. Измерений пока нет.")
            return
        now = datetime.now(timezone.utc)
        st.caption("Последние полученные сведения. Точные замеры на посту и описание участка реки показаны отдельно.")
        rows = H.observations(point_id, as_of=now, snapshot=snapshot)
        latest = {}
        for row in rows:
            latest.setdefault(row["metric"], row)
        if latest:
            for metric, row in latest.items():
                age = (now-H.stamp(row["observed_at"])).days
                state = "свежий замер" if age <= H.MAX_AGE[metric] else f"архивный замер, {age} сут назад"
                observed_label = (f"{H.stamp(row['observed_at']):%d.%m.%Y} (время не указано)"
                                  if row.get("observation_precision") == "day" else _date(row['observed_at']))
                st.write(f"**{H.METRICS[metric]}: {row['value']:g}** · {observed_label} · {state}")
                if row["scope"] == "reference_post":
                    st.caption("Гидропост «Лена — Якутск» служит ориентиром для переправы. "
                               "Это измерение на посту, а не обследование ледовой дороги.")
                if row.get("note"):
                    st.caption(row["note"])
                if H.stamp(row['available_at']) > H.stamp(issue_date):
                    st.caption(f"На дату решения {issue_date:%d.%m.%Y} этих сведений ещё не было у проекта; "
                               "в прогноз на эту дату они не подставляются.")
                source_name = "Якутское УГМС" if row.get("source") == "yakutia_bulletin" else "Росгидромет / «Метео ДВ»"
                st.markdown(f"[Источник измерения — {source_name}]({row['source_url']})")
        else:
            st.info("Точных датированных замеров для этой точки пока нет. "
                    "Прогноз использует имеющиеся погодные и исторические данные.")
        undated = [r for r in H.observations(point_id, snapshot=snapshot)
                   if r["quality"] == "undated" and H.stamp(r["available_at"]) <= now]
        # A source may return the same undated value again after another value.
        # Order its latest card by retrieval, keeping first availability intact.
        undated.sort(key=lambda r: r.get("fetched_at", r["available_at"]), reverse=True)
        shown = set()
        for row in undated:
            if row["metric"] in shown:
                continue
            shown.add(row["metric"])
            st.caption(f"В карточке источника: {H.METRICS[row['metric']].lower()} — {row['value']:g}. "
                       "Дата замера не указана; значение не используется как текущее.")
        reports = H.bulletins_for(point_id, as_of=now, snapshot=snapshot)
        for report in reports[:2]:
            st.markdown(f"**{report['title']}** · публикация {_date(report['published_at']).split(' ')[0]}")
            st.caption("Сводка по реке или региону; не точный замер в выбранной точке.")
            text = report["text"]
            match = re.search(r"толщин[аыу] льда|на (?:реке|реках)|уровн[ияе] вод|ледостав|ледоход", text, re.I)
            begin = max(0, match.start()-60) if match else 0
            excerpt = text[begin:begin+900]
            st.text(("…" if begin else "") + excerpt + ("…" if begin+900 < len(text) else ""))
            st.markdown(f"[Официальная сводка]({report['source_url']})")


def render_overview(region):
    st.subheader("Официальные гидрологические данные")
    snapshot = H.load_snapshot()
    st.caption("Источники автоматически проверяются раз в 6 часов, пока сервер приложения работает. "
               "При сбое сохраняются последние полученные сведения.")
    if not snapshot:
        st.info("Первая загрузка официальных данных ещё не завершена.")
        return
    st.write(f"Последняя проверка: **{_date(snapshot.get('checked_at'))}**. "
             f"Последнее получение данных: {_date(snapshot.get('last_success_at'))}.")
    coverage = pd.DataFrame(H.coverage(snapshot=snapshot))
    table = coverage[["name", "points", "measured_points", "fresh_points", "bulletins", "last_observation"]].copy()
    table["last_observation"] = table["last_observation"].map(lambda x: _date(x).split(" ")[0] if pd.notna(x) and x else "нет")
    table.columns = ["Регион", "Точек в проекте", "Точек с замерами", "Со свежими замерами",
                     "Официальных сводок", "Последний замер"]
    st.dataframe(table, hide_index=True, width="stretch", key="official_hydro_coverage")
    st.caption("Спутниковые точки не становятся гидропостами. Диапазоны по участкам рек "
               "и значения без даты не добавляются к точным замерам и целевым датам обучения.")
    points = H.registry()
    measurements = snapshot.get("observations", [])
    point_table = []
    for point_id, point in points.items():
        if point["region"] != region:
            continue
        rows = [r for r in measurements if r["point_id"] == point_id and r.get("observed_at")]
        point_table.append({"Точка": label(point_id),
                            "Последний датированный замер": _date(max((r["observed_at"] for r in rows), default=None)),
                            "Измерений": len(rows)})
    with st.expander(f"Покрытие точек: {REGIONS[region]['name']}"):
        st.dataframe(pd.DataFrame(point_table), hide_index=True, width="stretch", key="official_hydro_points")
    with st.expander("Архив официальных сводок"):
        reports = [r for r in H.bulletins_for(snapshot=snapshot) if r["region"] == region]
        if reports:
            choices = {}
            for report in reports:
                display = f"{H.stamp(report['published_at']):%d.%m.%Y} — {report['title']}"
                choices.setdefault(display, report)
            chosen = st.selectbox("Сводка", list(choices), key=f"official_bulletin_{region}")
            report = choices[chosen]
            st.text(report["text"])
            st.markdown(f"[Источник]({report['source_url']})")
        else:
            st.info("Сводок для этого региона пока нет.")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Скачать официальные наблюдения CSV", pd.DataFrame(measurements).to_csv(index=False).encode("utf-8-sig"),
                           "official_hydro_observations.csv", "text/csv", key="official_observations_csv")
    with c2:
        st.download_button("Скачать сводки CSV", pd.DataFrame(snapshot.get("bulletins", [])).to_csv(index=False).encode("utf-8-sig"),
                           "official_hydro_bulletins.csv", "text/csv", key="official_bulletins_csv")
    if any(r.get("status") == "error" for r in snapshot.get("sources", [])):
        st.caption("Часть источников при последней проверке не ответила. Ранее полученные сведения сохранены.")
    st.caption("Источники: Росгидромет, УГМС, НИЦ «Планета». Материалы Северного УГМС — "
               "для некоммерческого исследовательского использования с указанием источника.")
