"""New source identity, measurement units and strictly dated snow features."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import numpy as np
import pandas as pd
from okno import snow
from okno.northeast import STATIONS,load_observations
from okno.regions import REGIONS,region_of
from okno.ice import to_season_day
from scripts import download_snow_data as download

ROOT=Path(__file__).resolve().parents[1]


class NortheastDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gauges,cls.observations=load_observations()

    def test_real_stations_and_units_are_preserved(self):
        self.assertEqual(set(self.gauges.gauge_code),set(STATIONS))
        self.assertEqual(len(self.observations),20450)
        row=self.observations[self.observations.gauge_code.eq(1018)&self.observations.date.eq('2018-04-30')].iloc[0]
        self.assertAlmostEqual(row.ice_cm,97.)
        self.assertAlmostEqual(row.snow_on_ice_cm,24.)
        self.assertFalse(self.observations.duplicated(['gauge_code','date']).any())

    def test_conflicting_dates_and_interpolated_ice_are_not_invented_measurements(self):
        conflict=self.observations[self.observations.gauge_code.eq(1508)&self.observations.date.eq('1998-11-10')]
        self.assertTrue(conflict.empty)
        interpolated=self.observations[self.observations.gauge_code.eq(1578)&self.observations.date.eq('2013-01-10')].iloc[0]
        self.assertTrue(interpolated.author_interpolated)
        self.assertTrue(np.isnan(interpolated.ice_cm))
        self.assertTrue(np.isfinite(interpolated.ice_cm_raw))

    def test_new_source_does_not_create_unobserved_freezeup_or_breakup_dates(self):
        seasons=pd.read_csv(ROOT/'data/processed/hydro_ice_seasons.csv')
        new=seasons[seasons.gauge_code.isin(STATIONS)&seasons.source_type.ne('бюллетень ЯУГМС')]
        self.assertTrue(new[['first_ice_date','freezeup_date','melt_start_date','drift_start_date','ice_end_date']].isna().all().all())
        self.assertGreater(new.open_20t_day.notna().sum(),500)
        old=seasons[seasons.gauge_code.eq(3414)&seasons.source_type.ne('бюллетень ЯУГМС')]
        self.assertEqual(old.hydro_year.max(),2012)

    def test_correct_regions_and_existing_gauges_remain_available(self):
        counts=self.gauges.region_key.value_counts().to_dict()
        self.assertEqual(counts,dict(chukotka=9,yakutia=6,magadan=4))
        self.assertEqual(region_of('gauge_1596'),'chukotka')
        self.assertIn(3414,REGIONS['yakutia']['gauges'])
        for code,(_,region,_) in STATIONS.items():
            self.assertIn(code,REGIONS[region]['gauges'])
            self.assertEqual(region_of(f'gauge_{code}'),region)


class SnowCausalityTests(unittest.TestCase):
    def clear_caches(self):
        snow._ground.cache_clear();snow._measurements.cache_clear();snow._season.cache_clear()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name);(self.root/'snow_depth').mkdir();(self.root/'processed').mkdir()
        self.data_patch=patch.object(snow,'DATA',self.root);self.data_patch.start()
        self.processed_patch=patch.object(snow,'PROCESSED',self.root/'processed');self.processed_patch.start()
        self.ground=pd.DataFrame(dict(date=pd.date_range('2017-11-01',periods=20),snow_depth_cm=np.arange(1,21,dtype=float)))
        self.obs=pd.DataFrame(dict(gauge_code=[1018]*3,hydro_year=[2018]*3,
            date=['2017-11-01','2017-11-09','2017-11-15'],snow_on_ice_cm=[15.,22.,100.]))
        self.save()

    def save(self):
        self.ground.to_csv(self.root/'snow_depth/gauge_1018.csv',index=False)
        self.obs.to_csv(self.root/'processed/hydro_snow_on_ice.csv',index=False)
        self.clear_caches()

    def tearDown(self):
        self.clear_caches();self.processed_patch.stop();self.data_patch.stop();self.temp.cleanup()

    def test_ground_lag_and_river_measurements_are_separate(self):
        d=int(to_season_day('2017-11-10',2018));f=snow.features('gauge_1018',2018,d)
        self.assertEqual(f['snow_ground_cm'],5.)  # 5 November, not 10 November
        self.assertEqual(f['snow_on_ice_cm'],22.)
        self.assertEqual(f['snow_on_ice_age_days'],1.)
        self.assertEqual(f['snow_on_ice_observed'],1.)

    def test_future_or_not_yet_published_snow_cannot_change_a_forecast_input(self):
        d=int(to_season_day('2017-11-10',2018));before=snow.features('gauge_1018',2018,d)
        self.ground.loc[self.ground.date>pd.Timestamp('2017-11-05'),'snow_depth_cm']=999
        self.obs.loc[2,'snow_on_ice_cm']=888
        self.save()
        pd.testing.assert_series_equal(pd.Series(before),pd.Series(snow.features('gauge_1018',2018,d)))

    def test_zero_snow_is_valid_but_a_stale_or_previous_season_measurement_is_not(self):
        self.ground.snow_depth_cm=0.
        self.obs.date=['2017-09-01','2017-09-02','2017-09-03'];self.save()
        d=int(to_season_day('2017-11-10',2018));f=snow.features('gauge_1018',2018,d)
        self.assertEqual(f['snow_ground_cm'],0.)
        self.assertTrue(np.isnan(f['snow_on_ice_cm']))
        other=snow.features('gauge_1018',2019,d)
        self.assertTrue(np.isnan(other['snow_ground_cm']))
        self.assertEqual(other['snow_on_ice_observed'],0.)

    def test_missing_files_do_not_turn_into_zero_depth(self):
        f=snow.features('gauge_12345',2018,70)
        self.assertTrue(np.isnan(f['snow_ground_cm']))
        self.assertTrue(np.isnan(f['snow_on_ice_cm']))

    def test_future_trained_models_and_unvalidated_lena_models_are_not_used(self):
        metadata=dict(models={'open':[dict(train_max_year=2018)]})
        with patch.object(snow,'_metadata',return_value=metadata),patch.object(snow,'_model',side_effect=AssertionError('Future model loaded')):
            self.assertNotIn('quantiles',snow.predict('gauge_1018',2018,70,'open',{}))
            self.assertNotIn('quantiles',snow.predict('lena_test',2027,200,'close',{}))


class SnowResponseTests(unittest.TestCase):
    def payload(self):
        return {'daily':{'time':['2018-01-01','2018-01-02'],
                         'snow_depth_mean':[.2,.0]},'daily_units':{'snow_depth_mean':'m'}}

    def test_wrong_units_missing_dates_and_negative_depth_are_rejected(self):
        valid=download.normalise_daily(self.payload(),'2018-01-01','2018-01-02')
        np.testing.assert_array_equal(valid.snow_depth_mean.to_numpy()*100,[20.,0.])
        units=self.payload();units['daily_units']['snow_depth_mean']='cm'
        dates=self.payload();dates['daily']['time']=['2018-01-01','2018-01-01']
        negative=self.payload();negative['daily']['snow_depth_mean']=[-.1,0.]
        missing=self.payload();missing['daily']['snow_depth_mean']=[None,0.]
        for bad in (units,dates,negative,missing):
            with self.subTest(payload=bad),self.assertRaises(ValueError):
                download.normalise_daily(bad,'2018-01-01','2018-01-02')

    def test_daily_weather_preserves_model_and_timezone_of_new_gauges(self):
        from scripts import download_data as weather
        response=Mock(status_code=200)
        response.json.return_value={'daily':{'time':['2018-01-01'],'temperature_2m_mean':[-20.]}}
        with patch.object(weather.requests,'get',return_value=response) as request:
            weather._get_weather_chunk({'id':'gauge_1018','lat':65.7,'lon':150.9,
                'archive_model':'era5_seamless','archive_timezone':'UTC'},'2018-01-01','2018-01-01')
            self.assertEqual(request.call_args.kwargs['params']['models'],'era5_seamless')
            self.assertEqual(request.call_args.kwargs['params']['timezone'],'UTC')
            weather._get_weather_chunk({'id':'gauge_3414','lat':67.5,'lon':133.4},'2018-01-01','2018-01-01')
            self.assertNotIn('models',request.call_args.kwargs['params'])
            self.assertEqual(request.call_args.kwargs['params']['timezone'],'auto')


class SnowPredictionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ('reference.cbm','full.cbm','reference_base.cbm','full_base.cbm'):
            (self.root/name).touch()
        self.info=dict(issue_ranges={'open':[30,114]},features={'open':['cal_target','snow_ground_cm']},
            baseline_features={'open':['cal_target']},models={'open':[
                dict(file='reference.cbm',baseline_file='reference_base.cbm',train_max_year=2009),
                dict(file='full.cbm',baseline_file='full_base.cbm',train_max_year=2018)]},
            validation={'open':{'groups':{'new_gauges':dict(last_year=2018,improvement_pct=5.,
                                                         validated_points=['gauge_1018'])}}})
        self.fake=Mock();self.fake.predict.return_value=np.array([[-5.,0.,5.]])
        self.patches=[patch.object(snow,'MODELS',self.root),patch.object(snow,'_metadata',return_value=self.info),
            patch.object(snow,'features',return_value={'snow_ground_cm':10.}),
            patch.object(snow,'_model',return_value=self.fake)]
        self.mocks=[p.start() for p in self.patches]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def predict(self,year=2019,point='gauge_1018',issue=72,**kwargs):
        return snow.predict(point,year,issue,'open',{'cal_target':100.},**kwargs)

    def test_historical_and_current_forecasts_choose_only_preceding_training(self):
        old=self.predict(year=2018);new=self.predict(year=2019)
        self.assertEqual(old['train_max_year'],2009)
        self.assertEqual(old['model_file'],'reference.cbm')
        self.assertEqual(new['train_max_year'],2018)
        self.assertEqual(new['model_file'],'full.cbm')
        self.assertNotIn('quantiles',self.predict(year=2009))

    def test_stale_snow_outside_season_and_unvalidated_points_keep_the_baseline(self):
        self.assertNotIn('quantiles',self.predict(issue=140))
        self.assertNotIn('quantiles',self.predict(point='gauge_1500'))
        with patch.object(snow,'features',return_value={'snow_ground_cm':np.nan,'snow_on_ice_cm':np.nan}):
            self.assertNotIn('quantiles',self.predict())
            result=self.predict(use_snow=False)
            self.assertEqual(result['model_file'],'full_base.cbm')

    def test_unhelpful_snow_is_not_applied_to_future_seasons(self):
        self.info['validation']['open']['groups']['new_gauges']['improvement_pct']=-2.
        self.assertNotIn('quantiles',self.predict(year=2019))
        # Earlier research forecasts cannot select a policy from later test results.
        self.assertIn('quantiles',self.predict(year=2018))

    def test_overall_improvement_does_not_authorise_an_unchecked_source_group(self):
        self.info['validation']['open']=dict(last_year=2018,improvement_pct=50.,groups={})
        self.assertNotIn('quantiles',self.predict())
        self.assertIn('quantiles',self.predict(use_snow=False))

    def test_current_forecast_checks_accuracy_without_unavailable_river_measurements(self):
        group=self.info['validation']['open']['groups']['new_gauges']
        group['without_river']=dict(last_year=2018,improvement_pct=-10.,validated_points=['gauge_1018'])
        self.assertNotIn('quantiles',self.predict())
        with patch.object(snow,'features',return_value={'snow_ground_cm':10.,'snow_on_ice_observed':1.}):
            self.assertIn('quantiles',self.predict())


if __name__=='__main__':
    unittest.main()
