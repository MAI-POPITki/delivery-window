"""Готовые настройки для воспроизводимого разбора заказа, не фактические результаты."""

import datetime as dt

import streamlit as st


NIZHNEYANSK_BATAGAY = {
    "region": "yakutia",
    "hydro_year": 2024,
    "decision_date": dt.date(2024, 3, 22),
    "point_id": "gauge_3414",
    "horizon": 14,
    "close_margin_days": 21,
    "weather_close_enabled_v2": False,
    "Q_t": 45.0,
    "K_t_week": 28.0,
    "K_extra_t_week": 18.7,
    "c_road": 22447.50,
    "c_extra": 4489.50,
    "c_exp": 500000.0,
    "c_hold_week": 149.80,
}


def load_nizhneyansk_batagay():
    st.session_state.update(NIZHNEYANSK_BATAGAY)
