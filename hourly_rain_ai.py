"""Experimental station-level hourly precipitation forecasts from official ASOS HR observations.
Direct 1/3/6 hour models; no radar, no street-level predictions, no invented measurements.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss

URL = 'https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
KST = ZoneInfo('Asia/Seoul')

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_hourly_asos(key, station_id, days=90):
    if not key: raise ValueError('ASOS 인증키가 없습니다.')
    end = datetime.now(KST).replace(minute=0,second=0,microsecond=0)-timedelta(hours=3)
    start = end-timedelta(days=days)
    rows=[]
    # Smaller chunks avoid response limits and allow complete pagination.
    cursor=start
    while cursor<=end:
        stop=min(cursor+timedelta(days=19,hours=23),end)
        page=1
        while True:
            params={'ServiceKey':key,'pageNo':page,'numOfRows':1000,'dataType':'JSON',
                    'dataCd':'ASOS','dateCd':'HR','startDt':cursor.strftime('%Y%m%d'),
                    'startHh':cursor.strftime('%H'),'endDt':stop.strftime('%Y%m%d'),
                    'endHh':stop.strftime('%H'),'stnIds':str(station_id)}
            response=requests.get(URL,params=params,timeout=35)
            if response.status_code in (401,403):
                raise PermissionError('ASOS 시간자료 API 접근이 거부되었습니다(HTTP 401/403). 공공데이터포털에서 ASOS 시간자료 서비스 활용신청 승인, 인증키 유형, 호출 한도를 확인하세요. 인증키는 화면에 표시하지 않습니다.')
            if response.status_code >= 400:
                raise ConnectionError(f'ASOS 시간자료 조회 실패(HTTP {response.status_code}). API 승인 상태와 요청 조건을 확인하세요.')
            try:
                obj=response.json()['response']
                if str(obj['header']['resultCode'])!='00':
                    raise ValueError(str(obj['header'].get('resultMsg','ASOS 시간자료 오류')))
                body=obj['body']; items=(body.get('items') or {}).get('item',[])
                if isinstance(items,dict):items=[items]
                count=int(body.get('totalCount',0))
            except (KeyError,TypeError,AttributeError) as exc:
                raise ValueError('ASOS 시간자료 응답 형식 오류') from exc
            rows.extend(items)
            if page*1000>=count or not items:break
            page+=1
            if page>10:raise ValueError('ASOS 페이지 한도 초과')
        cursor=stop+timedelta(hours=1)
    if not rows:raise ValueError('시간 관측자료가 없습니다.')
    raw=pd.DataFrame(rows)
    required={'tm','rn','ta','hm','ws','pa'}
    if not required.issubset(raw.columns):
        raise ValueError('시간 관측 필드 누락: '+', '.join(sorted(required-set(raw.columns))))
    x=pd.DataFrame({'time':pd.to_datetime(raw.tm,errors='coerce'),
                    **{k:pd.to_numeric(raw[k],errors='coerce') for k in ('rn','ta','hm','ws','pa')}})
    x=x.dropna(subset=['time']).drop_duplicates('time',keep='last').set_index('time').sort_index()
    x=x.asfreq('h')
    x.loc[~x.rn.between(0,250),'rn']=np.nan
    x.loc[~x.ta.between(-45,50),'ta']=np.nan
    x.loc[~x.hm.between(0,100),'hm']=np.nan
    x.loc[~x.ws.between(0,80),'ws']=np.nan
    x.loc[~x.pa.between(850,1100),'pa']=np.nan
    # Missing precipitation is not automatically treated as zero.
    return x

def _features(x):
    out=pd.DataFrame(index=x.index)
    for k in ('rn','ta','hm','ws','pa'):
        for lag in (0,1,2,3,6,12,24):out[f'{k}_lag{lag}']=x[k].shift(lag)
    for n in (3,6,24):out[f'rain_sum{n}']=x.rn.rolling(n,min_periods=n).sum()
    h=x.index.hour.to_numpy(); d=x.index.dayofyear.to_numpy()
    out['hour_sin']=np.sin(2*np.pi*h/24);out['hour_cos']=np.cos(2*np.pi*h/24)
    out['year_sin']=np.sin(2*np.pi*d/365.25);out['year_cos']=np.cos(2*np.pi*d/365.25)
    return out

@st.cache_data(ttl=21600,show_spinner=False)
def forecast_hourly_rain(observations):
    if len(observations)<1100:raise ValueError('시간 관측자료가 1100시간 미만입니다.')
    f=_features(observations)
    latest=observations.dropna(subset=['rn','ta','hm','ws','pa']).index.max()
    if pd.Timestamp.now(tz=KST).tz_localize(None)-latest>pd.Timedelta(hours=9):
        raise ValueError(f'최신 유효 관측이 오래되었습니다: {latest}')
    if not f.loc[latest].notna().all():raise ValueError('최근 24시간 관측에 결측이 있어 예측하지 않습니다.')
    results=[];scores=[]
    for horizon in (1,3,6):
        target=observations.rn.shift(-horizon)
        valid=f.notna().all(axis=1)&target.notna()
        xx=f.loc[valid]; yy=target.loc[valid]
        if len(xx)<850:raise ValueError(f'{horizon}시간 모델의 유효 학습자료가 부족합니다 ({len(xx)}행).')
        # Time-ordered split with 6-hour embargo prevents label leakage at boundary.
        split=int(len(xx)*0.8)
        tr_end=split-6
        if tr_end<500 or len(xx)-split<100:raise ValueError('검증 자료가 부족합니다.')
        trainx=xx.iloc[:tr_end]; testx=xx.iloc[split:]
        trainy=yy.iloc[:tr_end];testy=yy.iloc[split:]
        rain=(trainy>=0.1).astype(int)
        if rain.nunique()<2:raise ValueError('강수/무강수 학습 사례가 부족합니다.')
        classifier=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=12,random_state=42)
        regressor=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=12,random_state=42)
        classifier.fit(trainx,rain);regressor.fit(trainx,np.log1p(trainy))
        probabilities=classifier.predict_proba(testx)[:,1]
        amount=np.expm1(regressor.predict(testx)).clip(0)
        scores.append({'예측시차':f'{horizon}시간 후','검증 표본(시간)':len(testx),
                       '강수확률 Brier(↓)':round(brier_score_loss((testy>=.1).astype(int),probabilities),3),
                       '강수량 MAE(mm)':round(mean_absolute_error(testy,amount),2),
                       '검증 종료':str(testx.index.max())})
        classifier.fit(xx,(yy>=.1).astype(int));regressor.fit(xx,np.log1p(yy))
        latestx=f.loc[[latest]]
        probability=float(classifier.predict_proba(latestx)[0,1]);mm=float(np.expm1(regressor.predict(latestx)[0]))
        results.append({'예측 대상 시각':str(latest+pd.Timedelta(hours=horizon)),
                        '강수 발생 확률(%)':round(probability*100,1),
                        '예상 1시간 강수량(mm)':round(max(0,mm),2),
                        '시차(시간)':horizon})
    return pd.DataFrame(results),pd.DataFrame(scores),latest


def hourly_extremes(observations):
    x=observations[['rn']].dropna().copy()
    if x.empty:return pd.DataFrame()
    x['월']=x.index.to_period('M').astype(str)
    return x.groupby('월').agg(관측시간=('rn','size'),시간강수량_10mm이상=('rn',lambda s:int((s>=10).sum())),
                             시간강수량_20mm이상=('rn',lambda s:int((s>=20).sum())),
                             최대_시간강수량_mm=('rn','max')).reset_index().tail(12)
