"""Optional public-data API download; user provides own service key."""
import requests
import pandas as pd
from datetime import date
from model import normalize

ENDPOINT='https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'

def fetch_asos(service_key, start='2006-01-01',end='2025-12-31', progress=None):
    if not service_key.strip(): raise ValueError('공공데이터포털 API 인증키가 필요합니다.')
    chunks=[]; dates=pd.date_range(start,end,freq='YS').tolist()
    if pd.Timestamp(start).year not in [x.year for x in dates]: dates.insert(0,pd.Timestamp(start))
    for i,begin in enumerate(dates):
        year=begin.year
        s=max(pd.Timestamp(start),pd.Timestamp(year,1,1))
        e=min(pd.Timestamp(end),pd.Timestamp(year,12,31))
        if s>e: continue
        params={'ServiceKey':service_key,'pageNo':'1','numOfRows':'400','dataType':'JSON','dataCd':'ASOS','dateCd':'DAY','startDt':s.strftime('%Y%m%d'),'endDt':e.strftime('%Y%m%d'),'stnIds':'143'}
        r=requests.get(ENDPOINT,params=params,timeout=35);r.raise_for_status()
        try: body=r.json()['response']; items=body['body']['items']['item']
        except Exception as ex: raise ValueError(f'{year}년 API 응답 오류: {r.text[:180]}') from ex
        if isinstance(items,dict): items=[items]
        if items: chunks.extend(items)
        if progress: progress((i+1)/len(dates))
    if not chunks: raise ValueError('API에서 관측자료를 받지 못했습니다.')
    raw=pd.DataFrame(chunks)
    return normalize([raw])
