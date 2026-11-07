"""A3 validation with explicit WARN for missing sources and limited evidence."""
from pathlib import Path
import sys
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from okno.corridor_data import load_events,past_shifts,verified_years
from okno.corridor import CorridorConfig,compare_policies
from okno.calib import calibrate_all
from okno.report import md_table


def calibrate_satellite_shift():
    """Past-only diagnostic conversion; estimated dates never become facts."""
    sat=pd.read_csv(ROOT/'data/processed/yakutsk_sat_seasons.csv')
    events=load_events()
    rows=[]
    for s in sat.itertuples():
        shifts=past_shifts(int(s.hydro_year),events,sat,min_years=1)
        e=events[events.hydro_year==s.hydro_year]
        e=e.iloc[0] if len(e) else None
        for phase,col in [('close','ice_road_close_date'),('open','ferry_open_date')]:
            if e is None or not e[f'verified_{phase}'] or shifts[phase] is None:
                continue
            estimate=pd.Timestamp(s.sat_breakup_date)+pd.Timedelta(days=shifts[phase])
            rows.append(dict(hydro_year=s.hydro_year,phase=phase,actual_date=e[col],
                converted_satellite_date=estimate.date().isoformat(),
                error_days=abs((estimate-pd.Timestamp(e[col])).total_seconds()/86400),
                fit_years=','.join(map(str,shifts[f'{phase}_years'])),
                status='diagnostic conversion using observed satellite event; not a forecast'))
    validation=pd.DataFrame(rows)
    validation.to_csv(ROOT/'reports/operational_shift_validation.csv',index=False)
    preview=sat.merge(events,on='hydro_year',how='left')
    preview['source_note']='Only verified dated events are facts. No synthetic 24-year road-window reconstruction.'
    preview.to_csv(ROOT/'data/processed/yakutsk_sat_calibrated.csv',index=False)
    return validation,dict(close_mae=validation[validation.phase.eq('close')].error_days.mean(),
                           open_mae=validation[validation.phase.eq('open')].error_days.mean())


def run_validation():
    checks=[]
    def record(code,status,fact):
        checks.append(dict(Код=code,Статус=status,Факт=fact))
        print(f'[{status}] {code}: {fact}',flush=True)
    gaps=dups=nans=0
    paths=list((ROOT/'data/weather').glob('*.csv'))+list((ROOT/'data/weather/gauges').glob('*.csv'))
    for p in paths:
        w=pd.read_csv(p,usecols=['date','temperature_2m_mean'],parse_dates=['date'])
        gaps+=len(pd.date_range(w.date.min(),w.date.max()))-w.date.nunique()
        dups+=int(w.date.duplicated().sum());nans+=int(w.temperature_2m_mean.isna().sum())
    record('V1_WEATHER','PASS' if gaps==dups==nans==0 and len(paths)>0 else 'FAIL',
           f'{len(paths)} рядов; пропуски {gaps}, дубли {dups}, пустые температуры {nans}. Это реанализ, не архив оперативных выпусков.')
    seasons=pd.read_csv(ROOT/'data/processed/hydro_ice_seasons.csv')
    record('V2_HYDRO','PASS' if (~seasons.valid).mean()<.05 else 'FAIL',
           f'{len(seasons)} сезонов; доля брака {(~seasons.valid).mean():.1%}')
    pooled,per,_=calibrate_all()
    record('V3_STEFAN','PASS' if pooled and pooled['r2']>.5 else 'FAIL',
           f"Подбор по {pooled['n']} замерам: alpha={pooled['alpha']:.2f}, R²={pooled['r2']:.2f}; это ошибка подбора, не независимый тест.")
    events=load_events()
    record('V4_OPERATIONAL_SOURCES','WARN' if not events.verified_close.all() else 'PASS',
           f'Полные проверенные окна: {verified_years()}. Неизвестные закрытия исключены; URL и дата публикации указаны у каждой подтверждённой фазы.')
    f=pd.read_csv(ROOT/'reports/window_forecasts_a3.csv')
    clock=((f.train_max_year<f.hydro_year).all()
           and (f.calibration_max_year.dropna()<f.loc[f.calibration_max_year.notna(),'hydro_year']).all()
           and (pd.to_datetime(f.weather_cutoff_date)<=pd.to_datetime(f.issue_date)-pd.Timedelta(days=5)).all())
    record('V5_FORECAST_CLOCK','PASS' if clock else 'FAIL',
           f'{f.hydro_year.nunique()} сезонов: обучение и калибровка только на прошлых целях, номинальная задержка ERA5 5 суток.')
    shift,stats=calibrate_satellite_shift()
    record('V6_OPERATIONAL_CONVERSION','WARN',
           f"Проверка преобразования по предыдущим годам: MAE закрытия {stats['close_mae']:.2f} сут, открытия {stats['open_mae']:.2f} сут. Малой выборки недостаточно для подтверждения рабочих сроков.")
    tariffs=pd.read_csv(ROOT/'data/rosstat_prices/yakutsk_tariffs_and_prices.csv')
    public=tariffs.source_type.str.contains('публичный тариф|статистический ориентир|норматив',case=False,na=False)
    falsely_tagged=public & ~tariffs.source_url.astype('string').fillna('').str.startswith('https://')
    record('V7_ECONOMIC_SOURCES','FAIL' if falsely_tagged.any() else 'WARN',
           'Тарифы и цены — допущения. Операторские документы/коммерческие предложения не подтверждены. Маркировка не заменяет проверку источника.')
    res=compare_policies(CorridorConfig(),46,84,45,86)
    good=all(abs(s['mass_balance_error_tons'])<1e-6 and (d.inv_yak_t<=650+1e-6).all()
             and abs(d.hub_dispatched_t.iloc[:-6].sum()-d.rail_arrived_nb_t.sum())<1e-6
             and abs(s['cash_cost_rub']-d[[col for col in d if col.startswith('cost_')]].to_numpy().sum())<1e-5
             and abs(s['total_cost_rub']-s['cash_cost_rub']-s['terminal_commitment_rub'])<1e-5
             for s,d in res.values())
    record('V8_FLOW_LEDGER','PASS' if good else 'FAIL',
           'Пять подходов: баланс груза, задержка ж/д, жёсткая ёмкость склада и сумма затрат с оценкой довоза. Остатки и заказы в ожидании показываются отдельно.')
    return pd.DataFrame(checks),dict(shifts=shift,shift_stats=stats)


def write_validation_report(summary,meta):
    n_fail=int(summary.Статус.eq('FAIL').sum());n_warn=int(summary.Статус.eq('WARN').sum())
    status='FAIL' if n_fail else 'WARN — расчётные ограничения остаются' if n_warn else 'PASS'
    out=ROOT/'docs/validation_report.md'
    text=['# Проверки А3: источники, время и ограничения','',
          f"Дата: {dt.datetime.now(ZoneInfo('Europe/Moscow')):%d.%m.%Y}. Статус: **{status}**.",'',
          'Прежняя формулировка «7/7 PASS — достоверность подтверждена» отозвана. '
          'Проверки формата и баланса не доказывают реальность тарифов или точность дорожных прогнозов.','',
          md_table(summary),'','## Независимость преобразования спутника в дорожные даты','',
          'Для каждой строки ниже используются только более ранние опубликованные фазы. '
          'Подставляется наблюдённый спутниковый факт, поэтому это диагностика связи событий, '
          'а не проверка ML-прогноза. Неизвестные фазы не восстановлены как фактические.','',
          md_table(meta['shifts']),'',
          'Источники событий: [реестр с URL](../data/rosstat_prices/yakutsk_mchs_crossing_dates.csv). '
          'Дата закрытия — первый полностью закрытый день; при вечернем закрытии используется следующий день. '
          'Для 2024 года: вечером 15 апреля / первый закрытый день 16 апреля; паромы — 24 мая, разрыв 38 суток.','',
          'Параметры спроса, выделенной мощности, ж/д плеча, начального запаса и складов остаются сценарными. '
          'Достоверная экономика перевозчика требует его отправок, договоров, реальных мощностей и издержек.']
    out.write_text('\n'.join(text)+'\n')
    summary.to_csv(ROOT/'reports/validation_summary_a3.csv',index=False)
    return out


def main():
    summary,meta=run_validation()
    write_validation_report(summary,meta)
    return int(summary.Статус.eq('FAIL').any())


if __name__=='__main__':
    raise SystemExit(main())
