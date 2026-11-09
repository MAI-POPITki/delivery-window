"""Компактная проверка загрузки в существующем планировщике."""
import streamlit as st
from okno.external_data import load_road_events
from okno.truck_load import route_events, limit_as_of, check_load


def render_truck_load(point_id, decision_date):
    with st.expander('Проверить загрузку машины по ограничениям движения'):
        st.caption(f'Точка и дата из левой панели: {decision_date:%d.%m.%Y}. '
                   'Проверка не меняет недельный план М3: он считает тонны, а не отдельные автомобили.')
        frame=route_events(load_road_events(), point_id)
        if frame.empty:
            st.info('Для выбранной точки нет привязанных дорог в реестре. Ограничение неизвестно.')
            return
        available=frame.groupby('hydro_year').actual_date.agg(['min','max']).reset_index()
        available.columns=['Сезон (год окончания)','Первая запись','Последняя запись']
        st.caption('Доступные периоды реестра для этой точки:')
        st.dataframe(available, hide_index=True, width='stretch')
        original_date=decision_date
        decision_date=st.date_input('Дата проверки ограничения',decision_date,
            key=f'truck_date_{point_id}_{original_date.isoformat()}')
        if decision_date!=original_date:
            st.warning('Проверяется другая дата. Это отдельная проверка ограничения; дата и расчёт М3 не изменены. '
                       'Историческое ограничение не переносится на сегодняшний день.')
        season=decision_date.year+int(decision_date.month>=9)
        roads=sorted(frame.loc[frame.hydro_year.eq(season),'road'].unique())
        if not roads:
            st.info(f'Для сезона {season-1}/{str(season)[2:]} записей нет. '
                    'Выберите дату внутри доступных периодов выше для исторической проверки. '
                    'Для текущей поездки требуется актуальная сводка дорожного ведомства: прошлогодние ограничения не применяются автоматически.')
            return
        road=st.selectbox('Дорога из реестра',roads,index=None,
                          placeholder='Выберите дорогу — гидропост не определяет её однозначно',key='truck_road')
        if road is None:
            st.info('Выберите конкретную дорогу: к одной точке модели привязаны разные направления.')
            return
        a,b=st.columns(2)
        tare=a.number_input('Масса пустой машины с оборудованием, т',0.,200.,12.,.1,key='truck_tare')
        cargo=b.number_input('Масса груза, т',0.,200.,10.,.1,key='truck_cargo')
        st.write(f'Полная масса: **{tare+cargo:.1f} т**.')
        result=limit_as_of(frame,road,decision_date)
        if 'source' in result:
            st.markdown(f"[Источник записи]({result['source']}) · событие {result['effective']} · публикация {result['published']}")
            st.caption(result['comment'])
        if result['status']=='closed':
            st.error(result['reason'])
        elif result['status']=='unknown':
            st.info(result['reason'])
        else:
            check=check_load(tare,cargo,result['limit_t'])
            st.write(f"Ограничение по последней записи: **{result['limit_t']:g} т полной массы**.")
            if check['tare_exceeds_limit']:
                st.error('Даже пустая машина превышает указанное ограничение. Уменьшение груза не решает проблему.')
            elif check['excess_t']>1e-9:
                st.error(f"Превышение {check['excess_t']:.1f} т ({check['excess_pct']:.1f}% от ограничения). "
                         f"Чтобы уложиться по полной массе, уменьшите груз минимум на {check['excess_t']:.1f} т.")
            else:
                st.info('По полной массе превышения нет. Это не разрешение проезда.')
            st.write(f"Остаток под груз по этому ограничению: **{check['cargo_limit_t']:.1f} т**. "
                     'Грузоподъёмность самого автомобиля может быть ниже.')
        st.caption('Используется последняя доступная запись неполного реестра, а не подтверждение текущего режима дороги. '
                   'Открытие без указанной массы не считается разрешением для любой машины. '
                   'Проверьте актуальные ограничения, осевые нагрузки и условия движения у дорожного ведомства. '
                   'Оценка толщины льда М1 не подменяет ограничение; процент превышения не является процентом риска.')
