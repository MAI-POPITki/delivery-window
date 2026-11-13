"""Refresh must retain good data, preserve decision time, and stay off the UI thread."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import requests
import numpy as np
from streamlit.testing.v1 import AppTest

from okno import flood as F, flood_refresh as R, flood_observations as O
from okno.flood_store import atomic_json, read_json, update_lock

NOW=datetime(2026,11,13,12,tzinfo=timezone.utc)
PID='lena_yakutsk'


def response(payload,url):
    return Mock(content=json.dumps(payload).encode(),url=url,json=Mock(return_value=payload),
                raise_for_status=Mock())


class LiveObservationTests(unittest.TestCase):
    def test_zero_negative_and_dated_levels_survive_without_using_swot(self):
        data=dict(level=[dict(d='2026-11-11',v=-157,source='gmvo'),
                         dict(d='2026-11-12',v=0,source='gmvo'),
                         dict(d='2026-11-14',v=100,source='gmvo'),
                         dict(d='2026-11-12',v=999.999,source='gmvo')],
                  swot=[dict(d='2026-11-13',v=800)])
        rows=O.parse_levels(PID,data,O.LEVEL_API,'a'*64,NOW)
        self.assertEqual([r['value_cm'] for r in rows],[-157.,0.])
        self.assertTrue(all(r['station_code']=='3045' for r in rows))
        self.assertTrue(all(r['observation_precision']=='day' for r in rows))

    def test_meteo_graph_ignores_previous_year_and_checks_station_identity(self):
        data=dict(name='Лена, Якутск',levels=[['2026-11-11 23:00',-157]],
                  prev_y=[['2026-11-11 23:00',900]],agk=[['2026-11-11 23:00',800]])
        rows=O.parse_meteo(PID,data,O.METEO_API,'a'*64,NOW)
        self.assertEqual([r['value_cm'] for r in rows],[-157])
        with self.assertRaises(ValueError):O.parse_meteo(PID,{**data,'name':'Амур, Хабаровск'},O.METEO_API,'a'*64,NOW)

    def test_outage_keeps_snapshot_and_empty_response_keeps_previous_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            payload=dict(level=[dict(d='2026-11-11',v=-157,source='gmvo')])
            with patch.object(O.requests,'request',return_value=response(payload,O.LEVEL_API)):
                # Non-Lena uses a single provider; no dependency on other source order.
                O.refresh('ishim_ishim',directory=folder,now=NOW)
            path=folder/'ishim_ishim.json';original=path.read_bytes()
            with patch.object(O.requests,'request',side_effect=requests.Timeout()):
                with self.assertRaises(RuntimeError):O.refresh('ishim_ishim',directory=folder,now=NOW+timedelta(hours=6))
            self.assertEqual(path.read_bytes(),original)
            with patch.object(O.requests,'request',return_value=response({'level':[]},O.LEVEL_API)):
                result=O.refresh('ishim_ishim',directory=folder,now=NOW+timedelta(hours=6))
            self.assertEqual(result['state'],'no_data')
            self.assertEqual(len(read_json(path)['records']),1)
            self.assertEqual(read_json(path)['last_success_at'],NOW.isoformat())

    def test_retrieval_does_not_backdate_corrections_or_fill_the_month(self):
        row=O.record(PID,'2026-11-11',-157,'meteo_dv',O.METEO_API,'a'*64,NOW)
        again={**row,'available_at':(NOW+timedelta(days=1)).isoformat()}
        self.assertEqual(O.merge([row],[again])[0]['available_at'],NOW.isoformat())
        revision=O.record(PID,'2026-11-11',-156,'meteo_dv',O.METEO_API,'b'*64,NOW+timedelta(days=1))
        self.assertEqual(len(O.merge([row],[revision])),2)
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp)/f'{PID}.json',dict(records=[row]))
            summary=O.summary(PID,directory=tmp,now=NOW)
            self.assertEqual(summary['days'],1)
            self.assertFalse(summary['continuous'])
            self.assertFalse(summary['model_ready'])


class RefreshLifecycleTests(unittest.TestCase):
    def test_abandoned_running_marker_is_retried_after_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=R.status_path('glofas',PID,tmp)
            atomic_json(path,dict(state='running',checked_at=NOW.isoformat()))
            with patch.object(R,'refresh_glofas',return_value=dict(state='ok',version='new')) as fetch:
                self.assertEqual(R.refresh_one('glofas',PID,directory=tmp,now=NOW),'ok')
                fetch.assert_called_once()

    def test_kernel_lock_is_released_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.lock'
            with self.assertRaises(RuntimeError):
                with update_lock(path) as acquired:
                    self.assertTrue(acquired)
                    raise RuntimeError('interrupted')
            with update_lock(path) as acquired:self.assertTrue(acquired)

    def test_ttls_and_short_retry_after_failure(self):
        state=dict(checked_at=NOW.isoformat(),state='ok')
        self.assertFalse(R.due('levels',state,NOW+timedelta(hours=5)))
        self.assertTrue(R.due('levels',state,NOW+timedelta(hours=6)))
        self.assertFalse(R.due('satellite',state,NOW+timedelta(hours=6)))
        self.assertTrue(R.due('levels',{**state,'state':'error'},NOW+timedelta(minutes=30)))

    def test_single_worker_start_does_not_wait_for_network(self):
        with patch.dict('os.environ',{'OKNO_FLOOD_AUTO_REFRESH':'1'}), \
             patch.object(R,'_worker',None),patch.object(R,'_points',set()), \
             patch.object(R.threading,'Thread') as factory:
            worker=factory.return_value;worker.is_alive.return_value=True
            self.assertIs(R.start_auto_update(PID),worker)
            self.assertIs(R.start_auto_update('ishim_ishim'),worker)
            factory.assert_called_once();worker.start.assert_called_once()

    def test_lock_prevents_second_process_and_failure_keeps_last_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=R.status_path('glofas',PID,tmp)
            prior=dict(state='ok',checked_at=(NOW-timedelta(days=1)).isoformat(),
                       last_success_at=(NOW-timedelta(days=1)).isoformat(),result={'version':'old'})
            atomic_json(path,prior)
            lock=Path(tmp)/'locks'/f'glofas-{PID}.lock'
            with update_lock(lock):
                with patch.object(R,'refresh_glofas') as fetch:
                    self.assertEqual(R.refresh_one('glofas',PID,directory=tmp,now=NOW),'already_running')
                    fetch.assert_not_called()
            with patch.object(R,'refresh_glofas',side_effect=requests.Timeout()):
                self.assertEqual(R.refresh_one('glofas',PID,directory=tmp,now=NOW),'error')
            saved=read_json(path)
            self.assertEqual(saved['result'],prior['result'])
            self.assertEqual(saved['last_success_at'],prior['last_success_at'])

    def test_recent_success_does_not_refetch_or_change_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=R.status_path('glofas',PID,tmp)
            atomic_json(path,dict(state='ok',checked_at=NOW.isoformat()))
            with patch.object(R,'refresh_glofas') as fetch:
                self.assertEqual(R.refresh_one('glofas',PID,directory=tmp,now=NOW),'cached')
                fetch.assert_not_called()

    def test_incomplete_glofas_never_replaces_latest(self):
        bad=dict(timezone='GMT',utc_offset_seconds=0,latitude=F.POINTS[PID]['lat'],longitude=F.POINTS[PID]['lon'],
                 daily_units={'river_discharge':'m³/s'},daily={'time':['2026-11-14'],'river_discharge':[10.]})
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);latest=folder/'latest'/f'{PID}.json'
            atomic_json(latest,{'old':'kept'})
            with patch.object(F,'DATA',folder),patch.object(R.requests,'get',return_value=response(bad,F.API)):
                with self.assertRaises(ValueError):R.refresh_glofas(PID,NOW)
            self.assertEqual(read_json(latest),{'old':'kept'})

    def test_dynamic_capture_remains_available_after_latest_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for path,stamp,value in [
                (folder/'runs/2026-11-12'/f'{PID}.json','2026-11-12T06:00:00+00:00',1),
                (folder/'runs/2026-11-12'/f'{PID}--180000000000.json','2026-11-12T18:00:00+00:00',2),
                (folder/'latest'/f'{PID}.json','2026-11-13T06:00:00+00:00',3)]:
                F.save_response(path,dict(value=value),F.API+'?models=seamless_v4',stamp)
            known=F.forecast_as_of(PID,datetime(2026,11,12,20,tzinfo=timezone.utc),folder)
            self.assertEqual(known['payload']['value'],2)

    def test_atomic_snapshot_does_not_leave_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.json'
            atomic_json(path,dict(value=1));atomic_json(path,dict(value=2))
            self.assertEqual(read_json(path),dict(value=2))
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])


class FloodLiveUITests(unittest.TestCase):
    def test_new_satellite_follows_latest_but_does_not_move_archive_date(self):
        from app.flood_m2_view import follow_latest
        state={}
        follow_latest(state,'view',date(2026,11,10),date(2026,11,13))
        self.assertEqual(state['view'],date(2026,11,10))
        follow_latest(state,'view',date(2026,11,12),date(2026,11,13))
        self.assertEqual(state['view'],date(2026,11,12))
        state['view']=date(2024,4,11)
        follow_latest(state,'view',date(2026,11,13),date(2026,11,13))
        self.assertEqual(state['view'],date(2024,4,11))

    def test_disabled_background_still_shows_sources_without_network(self):
        with patch.dict('os.environ',{'OKNO_FLOOD_AUTO_REFRESH':'0'}), \
             patch.object(O.requests,'request') as fetch:
            app=AppTest.from_string("from app.flood_live_view import render_live\nrender_live('lena_yakutsk')").run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(len(app.dataframe),1)
            self.assertTrue(app.button(key='refresh_flood_lena_yakutsk').disabled)
            self.assertTrue(any('уровень' in x.value.lower() for x in app.markdown))
            fetch.assert_not_called()


class SatellitePublicationTests(unittest.TestCase):
    def test_invalid_staged_result_cannot_replace_or_expose_a_run(self):
        from scripts.download_flood_m2 import publish_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);staged=root/'.staging'/'new';staged.mkdir(parents=True)
            final=root/'runs'/'new'
            with patch('scripts.download_flood_m2.M.load_run',side_effect=ValueError('invalid')):
                with self.assertRaises(ValueError):publish_run(staged,final,None,None)
            self.assertFalse(final.exists())

    def test_complete_run_is_published_and_existing_run_is_not_rewritten(self):
        from scripts.download_flood_m2 import publish_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);staged=root/'.staging'/'first';staged.mkdir(parents=True)
            array=np.array([[0,1,2,3]],dtype=np.uint8)
            np.savez_compressed(staged/'mask.npz',mask=array,reference=array)
            atomic_json(staged/'manifest.json',{'version':1})
            final=root/'runs'/'scene'
            with patch('scripts.download_flood_m2.M.load_run',return_value={'version':1}):
                publish_run(staged,final,array,array)
            original=(final/'mask.npz').read_bytes()
            again=root/'.staging'/'again';again.mkdir()
            atomic_json(again/'manifest.json',{'version':2})
            with patch('scripts.download_flood_m2.M.load_run',return_value={'version':1}):
                publish_run(again,final,array,array)
            self.assertEqual((final/'mask.npz').read_bytes(),original)
            self.assertFalse(again.exists())


if __name__=='__main__':unittest.main()
