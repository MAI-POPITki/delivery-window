"""Архив измеренных уровней из публичного сервиса Level-Sputnik / АИС ГМВО."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'data/flood/levels'
MATCH = {'lena_yakutsk':'р.лена - г.якутск','ishim_ishim':'р.ишим - г.ишим',
         'tobol_kurgan':'р.тобол - г.курган','ural_orenburg':'р.урал - г.оренбург',
         'ural_orsk':'р.урал - г.орск'}
BASE='https://level-satellite.ru'


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers['User-Agent']='delivery-window/1.0 (+https://github.com/MAI-POPITki/delivery-window)'
    response=session.post(BASE+'/api/posts_geojson',json={},timeout=40)
    response.raise_for_status()
    stations=response.json()['features']
    for pid,name in MATCH.items():
        matches=[f for f in stations if f['properties']['name'].lower()==name]
        if len(matches)!=1:
            raise ValueError(f'{pid}: ambiguous station')
        station=matches[0]
        code=station['properties']['post_code']
        payload=dict(post_code=code,date_from='2000-01-01',date_to=datetime.now(timezone.utc).date().isoformat())
        r=session.post(BASE+'/api/series',json=payload,timeout=60)
        r.raise_for_status()
        data=r.json()
        summary=session.post(BASE+'/api/post_summary',json={'post_code':code},timeout=40)
        summary.raise_for_status()
        raw=r.content
        (FOLDER/f'{pid}.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        manifest=dict(source=BASE+'/api/series',request=payload,fetched_at=datetime.now(timezone.utc).isoformat(),
                      sha256=hashlib.sha256(raw).hexdigest(),station=station,summary=summary.json(),
                      attribution='Головлев П.П., Иннокентьев А.И., Уровень-Спутник (2026); источник записей gmvo — АИС ГМВО',
                      units='cm above gauge zero; negative values valid')
        (FOLDER/f'{pid}.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        level=pd.DataFrame(data['level'])
        print(pid,code,len(level),level.groupby('source').size().to_dict(),level.d.min(),level.d.max())
        print(level.assign(year=level.d.str[:4]).groupby(['year','source']).size().tail(8).to_dict())


if __name__=='__main__':
    main()
