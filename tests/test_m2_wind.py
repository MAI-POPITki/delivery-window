"""Wind must preserve units, acquisition timing and season isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from okno import m2_wind as W
from scripts.experiment_m2_wind import folds, metrics, paired_interval, load_inputs, ROOT


def payload():
    return dict(timezone='GMT',utc_offset_seconds=0,latitude=62.,longitude=129.75,
                hourly_units={'wind_speed_10m':'m/s','wind_direction_10m':'°'},
                hourly=dict(time=[f'2020-05-01T{h:02d}:00' for h in range(8)],
                            wind_speed_10m=list(range(1,9)),wind_direction_10m=[90]*8))


class WindInputTests(unittest.TestCase):
    def test_acquisition_uses_first_timestamp_in_utc(self):
        item='S1A_IW_GRDH_1SDV_20200501T053501_20200501T053526_123456_rtc'
        self.assertEqual(W.acquisition_time(item),pd.Timestamp('2020-05-01T05:35:01Z'))
        with self.assertRaises(ValueError):W.acquisition_time('2020-05-01')

    def test_features_use_only_preceding_hour_and_full_past_window(self):
        hourly=W.parse_hourly(payload())
        result=W.wind_features(hourly,'2020-05-01T05:35:00Z')
        self.assertEqual(result['wind_speed_ms'],6)
        self.assertEqual(result['wind_mean6_ms'],3.5)
        self.assertEqual(result['wind_max6_ms'],6)
        self.assertAlmostEqual(result['wind_u_ms'],-6)
        self.assertAlmostEqual(result['wind_v_ms'],0)
        hourly.loc[pd.Timestamp('2020-05-01T06:00Z'):,'speed']=1000
        self.assertEqual(W.wind_features(hourly,'2020-05-01T05:35:00Z'),result)
        self.assertEqual(W.wind_features(hourly,'2020-05-01T05:00:00Z'),result)

    def test_missing_hours_and_unknown_values_are_not_filled(self):
        hourly=W.parse_hourly(payload())
        unknown=hourly.copy()
        unknown.loc[unknown.index[2],:]=np.nan
        for frame in [hourly.drop(hourly.index[2]),unknown]:
            with self.assertRaises(ValueError):W.wind_features(frame,'2020-05-01T05:35:00Z')
        with self.assertRaises(ValueError):W.wind_features(hourly,'2020-05-01T05:00:00')

    def test_wrong_units_duplicate_times_and_negative_speed_rejected(self):
        value=payload();value['hourly_units']['wind_speed_10m']='km/h'
        with self.assertRaises(ValueError):W.parse_hourly(value)
        value=payload();value['hourly']['time'][1]=value['hourly']['time'][0]
        with self.assertRaises(ValueError):W.parse_hourly(value)
        value=payload();value['hourly']['wind_speed_10m'][0]=-1
        with self.assertRaises(ValueError):W.parse_hourly(value)

    def test_frozen_source_and_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wind.json.gz'
            url=W.API+'?'+urlencode(dict(models='era5',timezone='UTC',wind_speed_unit='ms'))
            W.save_response(path,json.dumps(payload()).encode(),url,dict(lat=62.,lon=129.82))
            frame,_=W.load_response(path)
            self.assertEqual(len(frame),8)
            meta=path.with_suffix('.manifest.json')
            data=json.loads(meta.read_text());data['url']=data['url'].replace('models=era5','models=best_match')
            meta.write_text(json.dumps(data))
            with self.assertRaises(ValueError):W.load_response(path)
            data['url']=url;data['sha256']='0'*64;meta.write_text(json.dumps(data))
            with self.assertRaises(ValueError):W.load_response(path)


class WindEvaluationTests(unittest.TestCase):
    def test_forward_folds_cannot_train_on_current_or_future_season(self):
        frame=pd.DataFrame([dict(hydro_year=y,y=label) for y in range(2016,2023) for label in (0,1)])
        for year,train,test in folds(frame,'past_only'):
            self.assertTrue((frame.loc[train,'hydro_year']<year).all())
            self.assertTrue((frame.loc[test,'hydro_year']==year).all())
            self.assertFalse((train&test).any())
        for year,train,test in folds(frame,'leave_year_out'):
            self.assertFalse((frame.loc[train,'hydro_year']==year).any())

    def test_water_as_ice_is_counted_as_a_missed_water_observation(self):
        frame=pd.DataFrame(dict(y=[0,0,1,1],p_ice=[.9,.1,.9,.1]))
        result=metrics(frame)
        self.assertEqual(result['water_as_ice'],1)
        self.assertEqual(result['ice_as_water'],1)
        self.assertEqual(result['balanced_accuracy'],.5)

    def test_paired_season_interval_uses_same_test_rows(self):
        rows=[]
        for year in range(2019,2023):
            for label in [0,1]:
                for variant in ['radar','radar_wind']:
                    rows.append(dict(scheme='past_only',seed=42,variant=variant,hydro_year=year,
                                     point='p',item_id=f'{year}-{label}',y=label,p_ice=float(label)))
        data=pd.DataFrame(rows)
        interval=paired_interval(data,iterations=100)
        self.assertEqual(interval['clusters'],4)
        self.assertEqual(interval['low'],0)
        self.assertEqual(interval['high'],0)
        with self.assertRaises(ValueError):paired_interval(data.iloc[1:],iterations=100)


class SavedWindExperimentTests(unittest.TestCase):
    def test_saved_sources_features_and_predictions_have_same_coverage(self):
        data,coverage=load_inputs()
        self.assertEqual(coverage,dict(labelled=1126,used=1126,excluded=0))
        predictions=pd.read_csv(ROOT/'reports/m2_wind_predictions.csv.gz')
        for (scheme,variant,seed),g in predictions.groupby(['scheme','variant','seed']):
            expected=data[data.hydro_year.between(2019,2022) if scheme=='past_only'
                          else data.hydro_year.between(2017,2022)]
            self.assertEqual(set(zip(g.point,g.item_id)),set(zip(expected.point,expected.item_id)))
            self.assertFalse(g.duplicated(['point','item_id']).any())
            if scheme=='past_only':
                self.assertTrue((g.train_max_year<g.hydro_year).all())
            self.assertTrue(g.p_ice.between(0,1).all())

    def test_saved_scores_recompute_from_predictions(self):
        predictions=pd.read_csv(ROOT/'reports/m2_wind_predictions.csv.gz')
        scores=pd.read_csv(ROOT/'reports/m2_wind_metrics.csv')
        for row in scores.itertuples():
            g=predictions[(predictions.scheme==row.scheme)&(predictions.variant==row.variant)&(predictions.seed==row.seed)]
            recomputed=metrics(g)
            for name in ['balanced_accuracy','accuracy','f1_water','brier','water_as_ice','ice_as_water']:
                self.assertAlmostEqual(recomputed[name],getattr(row,name),places=12)


if __name__=='__main__':unittest.main()
