"""A3 presentation; factual dates are separate from operator assumptions."""
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from okno.corridor import CorridorConfig,POLICY_LABELS,compare_policies,run_sensitivity_check,weekly_dispatch_comparison
from okno.corridor_data import case_inputs,verified_years

ROOT=Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def calculate(config,case,version,sensitivity=False):
    cfg=CorridorConfig(**config)
    c=case_inputs(case)
    results=compare_policies(cfg,c['ac'],c['ao'],c['cc'],c['co'],c['qc'],c['qo'],c['forecasts'])
    sens=run_sensitivity_check(cfg,c['ac'],c['ao'],c['cc'],c['co'],forecasts=c['forecasts'],
                               q20_close=c['qc'],q20_open=c['qo']) if sensitivity else None
    return c,results,sens


def render_weekly_dispatches(results,case,names):
    st.subheader('План отправок по неделям')
    st.caption(f"Существующий способ — {names['ml']}; экономический — {names['economic_ml']}. "
               'Оба используют одинаковые прогнозы. Неделя — 7 дней с начала расчёта; последняя может быть короче. '
               'Здесь собраны последовательные решения по мере обновления прогноза в течение сезона.')
    stages={'hub_dispatched_t':'Отправка из хаба',
            'rail_arrived_nb_t':'Прибытие в Нижний Бестях',
            'shipped_total_t':'Перевозка через Лену',
            'served_t':'Выдача потребителю'}
    stage=st.selectbox('Этап перевозки для недельного сравнения',list(stages),
                        format_func=stages.get,key='corr_weekly_stage')
    weekly=weekly_dispatch_comparison(results,case['start_date'])
    old=weekly[weekly.policy.eq('ml')].set_index('week')
    economic=weekly[weekly.policy.eq('economic_ml')].set_index('week')
    if case['start_date'] is None:
        periods='Дни '+(old.day_start+1).astype(str)+' — '+(old.day_end+1).astype(str)
    else:
        periods=(pd.to_datetime(old.week_start_date).dt.strftime('%d.%m.%Y')+' — '+
                 pd.to_datetime(old.week_end_date).dt.strftime('%d.%m.%Y'))
    view=pd.DataFrame({'Неделя':old.index,'Период':periods,
        'Существующий способ, т':old[stage],'Экономический способ, т':economic[stage],
        'Разница, т':economic[stage]-old[stage]}).reset_index(drop=True)
    st.dataframe(view.round(2),hide_index=True,width='stretch',key='corr_weekly',column_config={
        col:st.column_config.NumberColumn(format='localized') for col in view if col!='Период'})
    chart=go.Figure()
    for col in ('Существующий способ, т','Экономический способ, т'):
        chart.add_bar(x=view.Неделя,y=view[col],name=col,customdata=view.Период,
                      hovertemplate='Неделя %{x}<br>%{customdata}<br>%{y:,.2f} т<extra>%{fullData.name}</extra>')
    chart.update_layout(barmode='group',title=f'{stages[stage]} по неделям',
                        xaxis_title='Неделя с начала расчёта',yaxis_title='т',height=360)
    st.plotly_chart(chart,width='stretch',key='corr_weekly_chart')
    st.caption('Разница = экономический − существующий. Меньшая отправка за одну неделю может означать перенос '
               'на другую; выполнение спроса и общие расходы показаны в итоговом сравнении. '
               'Прибытие по ж/д учитывает время в пути. Перевозка через Лену включает штатный и резервный каналы.')
    st.download_button('Скачать недельные отправки',weekly.to_csv(index=False).encode('utf-8-sig'),
                        'corridor-weekly.csv','text/csv',key='corr_weekly_download')


def render_corridor():
    st.subheader("Сквозной логистический коридор из 4 звеньев (Кейс А3)")
    st.caption("Хаб → железная дорога → Нижний Бестях → переправа → склад Якутска. "
               "Сезон коридора выбирается ниже. "
               "Старые правила предзавоза сохранены. Экономический план выбирает суточную загрузку "
               "ж/д и переправы по минимуму затрат при известных на день решения данных.")
    options=sorted(verified_years(),reverse=True)+['synthetic']
    case=st.selectbox("Сценарий / подтверждённый сезон",options,key='corr_scenario',
                      format_func=lambda y:'Искусственный сценарий (не CatBoost)' if y=='synthetic' else f'{y}: реальные даты переправы')
    with st.expander("Параметры и тарифы 4 звеньев маршрута",expanded=False):
        st.caption("Все тарифы, спрос, выделенные мощности и склады — допущения; подтверждённых цен оператора нет.")
        k1,k2,k3=st.columns(3)
        vals=dict(daily_demand_tons=k1.slider('Суточная потребность d, т/сут',10.,50.,25.,1.,key='corr_d'),
            crossing_cap_normal_tons=k1.slider('Штатная пропускная способность, т/сут',35.,120.,60.,5.,key='corr_qmax'),
            crossing_cap_alt_tons=k1.slider('Лимит резерва в распутицу, т/сут',0.,20.,8.,1.,key='corr_qalt'),
            yak_warehouse_cap_tons=k1.slider('Ёмкость основного склада, т',300.,1200.,650.,50.,key='corr_vmax'),
            rail_cost_per_ton=k2.slider('1. Тариф ж/д, ₽/т',4000.,15000.,8500.,500.,key='corr_crail'),
            nb_holding_cost_per_ton_day=k2.slider('2. Хранение в Н. Бестяхе, ₽/т·сут',150.,1000.,450.,25.,key='corr_hnb'),
            crossing_cost_normal=k2.slider('3а. Штатная переправа, ₽/т',800.,4000.,1800.,100.,key='corr_cnorm'),
            crossing_cost_alt=k2.slider('3б. Резервная переправа, ₽/т',15000.,60000.,35000.,1000.,key='corr_calt'),
            yak_holding_cost_per_ton_day=k3.slider('4а. Хранение в Якутске, ₽/т·сут',100.,500.,220.,10.,key='corr_hyak'),
            yak_overflow_cost_per_ton_day=k3.slider('4б. Хранение на доп. складе, ₽/т·сут',300.,1200.,600.,25.,key='corr_hover'),
            shortage_penalty_per_ton_day=k3.slider('Штраф ожидания, ₽/т·сут',0.,25000.,12000.,500.,key='corr_pdef'))
        ras=k3.slider('Распутица только для искусственного сценария, сут',15,60,38,key='corr_rasp_len',disabled=case!='synthetic')
        vals['rail_lead_time_days']=k1.number_input('Ход по железной дороге, сут',0,20,6,key='corr_rail_days')
        vals['rail_cap_tons']=k2.number_input('Лимит отправки по ж/д, т/сут',1.,1000.,120.,key='corr_rail_cap')
        vals['yak_overflow_capacity_tons']=k3.number_input('Доступный дополнительный склад, т',0.,5000.,0.,key='corr_overflow_cap')
    cfg=CorridorConfig(**vals)
    if min(cfg.crossing_cap_normal_tons,cfg.rail_cap_tons)<=cfg.daily_demand_tons:
        st.warning('Пропускная способность ж/д или переправы не превышает суточное потребление: резерва под предзавоз нет.')
    version=tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in [ROOT/'okno/corridor.py',ROOT/'okno/economic_planner.py',ROOT/'okno/corridor_data.py',
        ROOT/'data/rosstat_prices/yakutsk_mchs_crossing_dates.csv',ROOT/'reports/window_forecasts_a3.csv'])
    want_sens=st.checkbox('Рассчитать чувствительность ±30%',key='corr_sensitivity')
    if case=='synthetic':
        c=case_inputs(case,ras)
        results=compare_policies(cfg,c['ac'],c['ao'],c['cc'],c['co'],c['qc'],c['qo'],c['forecasts'])
        sens=run_sensitivity_check(cfg,c['ac'],c['ao'],c['cc'],c['co'],forecasts=c['forecasts'],q20_close=c['qc'],q20_open=c['qo']) if want_sens else None
    else:
        c,results,sens=calculate(asdict(cfg),case,version,want_sens)
    names=dict(POLICY_LABELS)
    if case=='synthetic':
        names['ml']='Предзавоз: ручное предупреждение'
        names['economic_ml']='Экономический: ручное предупреждение'
    st.info(c['note'])
    if not c['model_available']:
        st.warning('ML для этого сезона заменён календарным q20. Его результат нельзя считать выигрышем модели.')
    if case!='synthetic':
        start=pd.Timestamp(c['start_date'])
        close=start+pd.Timedelta(days=c['ac']);op=start+pd.Timedelta(days=c['ao'])
        st.write(f"Подтверждённый разрыв: **{close:%d.%m.%Y} — {op:%d.%m.%Y}**, {c['ao']-c['ac']} суток. "
                 "Это факты для оценки: до наступления события план их не получает.")
        for i,url in enumerate(c['sources']):
            st.markdown(f"[Источник {'закрытия' if i==0 else 'открытия'}]({url})")
    summary=pd.DataFrame([dict(policy=p,**r[0]) for p,r in results.items()]).set_index('policy')
    if st.session_state.get('corr_focus') not in results:
        st.session_state['corr_focus']='economic_ml'
    focus=st.selectbox('Подход для подробного сравнения',list(results),index=list(results).index('economic_ml'),
                       format_func=names.get,key='corr_focus')
    before,after=summary.loc['calendar'],summary.loc[focus]
    st.write(f'**Выбран: {names[focus]}**')
    if focus.startswith('economic_'):
        st.caption('Экономический план минимизирует перевозку, хранение и стоимость ожидания до конца общего горизонта. '
                   'Он обязан выполнить спрос к сроку, если это физически возможно; иначе сначала минимизирует недовоз. '
                   'Исполняется только решение на сегодня. Ошибка прогноза может ухудшить фактический результат; '
                   'это не гарантия минимальных расходов в реальности. Страховой запас старых правил в эту цель не входит.')
    if not after.terminal_service_complete:
        st.warning(f'К концу расчёта не выполнено {after.terminal_backlog_tons:,.1f} т заказов. Сравнивайте расходы вместе с объёмом и ожиданием.')
    m1,m2,m3,m4=st.columns(4)
    m1.metric('Расходы с оценкой довоза остатка',f'{after.total_cost_rub/1e6:.2f} млн ₽',
        delta=f'{(after.total_cost_rub-before.total_cost_rub)/1e6:+.2f} млн ₽ ({100*(after.total_cost_rub/before.total_cost_rub-1):+.2f}%) к медиане',delta_color='inverse')
    m2.metric('Разница со старым предзавозом ML',f"{(after.total_cost_rub-summary.loc['ml','total_cost_rub'])/1e6:+.2f} млн ₽")
    m3.metric('Дни с невыполненными заказами',f'{after.days_with_deficit:.0f}',delta=f'{after.days_with_deficit-before.days_with_deficit:+.0f}',delta_color='inverse')
    m4.metric('Хранение в Якутске',f'{after.cost_holding_yak_rub/1e6:.2f} млн ₽',delta=f'{(after.cost_holding_yak_rub-before.cost_holding_yak_rub)/1e6:+.2f} млн ₽',delta_color='off')
    view=summary[['total_cost_rub','cash_cost_rub','terminal_commitment_rub','served_tons','terminal_backlog_tons','shortage_ton_days','tons_stuck_nb_ton_days']].copy()
    view.index=view.index.map(names)
    view.index.name='Способ'
    view.columns=['Расходы с довозом, ₽','Расходы за период, ₽','Оценка довоза остатка, ₽','Получено потребителем, т','Не выполнено к концу, т','Ожидание, т·сут','Хранение Н.Бестях, т·сут']
    st.dataframe(view.round(2),width='stretch',key='corr_summary',column_config={
        col:st.column_config.NumberColumn(format='localized') for col in view.columns})
    st.caption(f'Потребность за {cfg.horizon_days} суток: {cfg.daily_demand_tons*cfg.horizon_days:,.0f} т. '
               f'Начальный запас у всех: {after.initial_yak_tons:,.0f} т; отправки прибывают через {cfg.rail_lead_time_days} сут. '
               f'Дополнительное место на складе: {cfg.yak_overflow_capacity_tons:g} т. '
               'Неисполненные заказы сохраняются; штраф считается за каждый день их ожидания.')
    st.caption('Чтобы уменьшение запаса к концу периода не выдавало себя за экономию, у всех способов одинаково '
               'оценён будущий довоз оставшегося заказанного груза: ж/д для остатка в хабе и штатная переправа '
               'для груза в хабе, в пути и в Нижнем Бестяхе. Это расчётная оценка без будущего хранения, '
               'а не выполненные рейсы. Расходы за период показаны отдельно.')
    render_weekly_dispatches(results,c,names)
    chart=go.Figure()
    for col,title in [('cost_rail_rub','Ж/д'),('cost_holding_nb_rub','Хранение Н.Бестях'),
        ('cost_crossing_normal_rub','Штатная переправа'),('cost_crossing_alt_rub','Резерв'),
        ('cost_holding_yak_rub','Склад Якутска'),('cost_shortage_rub','Ожидание'),
        ('terminal_commitment_rub','Оценка довоза остатка')]:
        chart.add_bar(x=[names[p] for p in summary.index],y=summary[col]/1e6,name=title)
    chart.update_layout(barmode='stack',title='Все звенья и издержки ожидания',yaxis_title='млн ₽',height=430)
    st.plotly_chart(chart,width='stretch')
    chart=go.Figure()
    for p in ('calendar','ml','economic_ml'):
        daily=results[p][1]
        chart.add_scatter(x=daily.day,y=daily.inv_yak_t,name=names[p])
    chart.add_scatter(x=results[focus][1].day,y=results[focus][1].inv_nb_t,name='Правый берег: выбранный подход',line=dict(dash='dot'))
    chart.add_vrect(x0=c['ac'],x1=c['ao'],fillcolor='red',opacity=.08,line_width=0)
    chart.add_hline(y=cfg.yak_warehouse_cap_tons+cfg.yak_overflow_capacity_tons,line_dash='dash',annotation_text='всё доступное место')
    chart.update_layout(title='Запасы и фактическая распутица',height=400,xaxis_title='день от начала расчёта',yaxis_title='т')
    st.plotly_chart(chart,width='stretch')
    with st.expander('Проверить даты решений, отправки и грузы в пути'):
        st.caption('Журнал выбранного подхода. optimal означает минимум по известному тогда прогнозу; '
                   'optimal_with_unavoidable_shortfall — выполнить весь спрос при таком прогнозе невозможно.')
        st.dataframe(results[focus][1],hide_index=True,width='stretch')
        st.dataframe(summary[['terminal_hub_tons','terminal_in_transit_tons','terminal_nb_tons','terminal_yak_tons','terminal_backlog_tons','mass_balance_error_tons']],width='stretch')
    if sens is not None:
        st.caption(f'Проверено {len(sens)} различных конфигураций; знак эффекта не задан заранее. Это не вероятность успеха.')
        st.dataframe(sens,width='stretch',hide_index=True)
    fcpath=ROOT/'reports/window_forecasts_a3.csv'
    if fcpath.exists() and case!='synthetic':
        f=pd.read_csv(fcpath);f=f[f.hydro_year==case]
        chart=go.Figure()
        chart.add_scatter(x=list(f.issue_date)+list(f.issue_date[::-1]),y=list(f.q90)+list(f.q10[::-1]),fill='toself',line=dict(width=0),name='Физическое вскрытие q10–q90')
        chart.add_scatter(x=f.issue_date,y=f.q20,name='q20 физического события')
        chart.add_scatter(x=f.issue_date,y=f.q50,name='q50 физического события')
        chart.update_layout(title='Прогнозы выпускаются последовательно; это не разрешение проезда',yaxis_title='день от 1 сентября',height=350)
        st.plotly_chart(chart,width='stretch')
    out=pd.concat([daily.assign(policy=name) for name,(_,daily) in results.items()],ignore_index=True)
    st.download_button('Скачать дневной журнал коридора',out.to_csv(index=False).encode('utf-8-sig'),'corridor-daily.csv','text/csv',key='corr_download')
