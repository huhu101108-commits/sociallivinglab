"""Regional ASOS next-day ML: temporal holdout; no synthetic forecasts."""
from datetime import date,timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, brier_score_loss
from kma_api import ENDPOINT

# Confirmed KMA ASOS station identifiers for regional representative stations.
STATIONS={
 '서울':(108,'서울'),'부산':(159,'부산'),'대구':(143,'대구'),
 '인천':(112,'인천'),'광주':(156,'광주'),'대전':(133,'대전'),
 '울산':(152,'울산'),'수원':(119,'수원'),'춘천':(101,'춘천'),
 '강릉':(105,'강릉'),'청주':(131,'청주'),'전주':(146,'전주'),
 '목포':(165,'목포'),'포항':(138,'포항'),'안동':(136,'안동'),
 '창원':(155,'창원'),'제주':(184,'제주'),'서귀포':(189,'서귀포'),
 '원주':(114,'원주'),'충주':(127,'충주'),'군산':(140,'군산'),
 '여수':(168,'여수'),
}

@st.cache_data(ttl=21600,show_spinner=False)
def fetch_station_observations(key,station_id):
    if not key:raise ValueError('ASOS 인증키가 필요합니다.')
    end=date.today()-timedelta(days=1)
    start=end-timedelta(days=365*3+30)
    rows=[]
    for year in range(start.year,end.year+1):
        a=max(start,date(year,1,1));b=min(end,date(year,12,31))
        params={'ServiceKey':key,'pageNo':'1','numOfRows':'400','dataType':'JSON','dataCd':'ASOS','dateCd':'DAY','startDt':a.strftime('%Y%m%d'),'endDt':b.strftime('%Y%m%d'),'stnIds':str(station_id)}
        response=requests.get(ENDPOINT,params=params,timeout=35)
        response.raise_for_status()
        try:
            payload=response.json()['response']
            if str(payload['header']['resultCode'])!='00':raise ValueError(payload['header'].get('resultMsg','API 오류'))
            body=payload['body'];items=(body.get('items') or {}).get('item',[])
            if isinstance(items,dict):items=[items]
            count=int(body.get('totalCount',len(items)))
            if count>400:raise ValueError('API 연간 조회가 400행을 초과합니다. 페이지 처리가 필요합니다.')
            rows.extend(items)
        except (KeyError,TypeError,AttributeError) as e:raise ValueError('ASOS API 응답 구조 오류') from e
    if not rows:raise ValueError('관측소 자료가 없습니다.')
    raw=pd.DataFrame(rows)
    required={'tm','maxTa','minTa','sumRn'}
    if not required.issubset(raw):raise ValueError('필수 관측 필드가 없습니다: '+', '.join(sorted(required-set(raw.columns))))
    x=pd.DataFrame({'date':pd.to_datetime(raw['tm'],errors='coerce'),'tmax':pd.to_numeric(raw['maxTa'],errors='coerce'),'tmin':pd.to_numeric(raw['minTa'],errors='coerce'),'rain':pd.to_numeric(raw['sumRn'],errors='coerce')})
    x=x.dropna(subset=['date']).drop_duplicates('date',keep='last').set_index('date').sort_index().asfreq('D')
    x.loc[~x.tmax.between(-40,50),'tmax']=np.nan
    x.loc[~x.tmin.between(-45,45),'tmin']=np.nan
    x.loc[~x.rain.between(0,1000),'rain']=np.nan
    # Some ASOS dry days have empty rainfall: only infer zero where temperature observations confirm a complete day.
    dry=x.rain.isna() & x.tmax.notna() & x.tmin.notna()
    x.loc[dry,'rain']=0.0
    return x

def _features(data):
    x=pd.DataFrame(index=data.index)
    for field in ('tmax','tmin','rain'):
        for lag in (1,2,3,7):x[f'{field}_{lag}']=data[field].shift(lag)
        x[f'{field}_avg7']=data[field].shift(1).rolling(7,min_periods=7).mean()
    d=x.index.dayofyear
    x['season_sin']=np.sin(2*np.pi*d/365.25)
    x['season_cos']=np.cos(2*np.pi*d/365.25)
    return x

@st.cache_data(ttl=21600,show_spinner=False)
def train_and_predict(data):
    if len(data)<650:raise ValueError('학습에 필요한 650일 이상의 관측이 없습니다.')
    # A row at date D predicts D using observations through D-1.
    x=_features(data)
    features=list(x.columns)
    valid=x.notna().all(axis=1) & data[['tmax','tmin','rain']].notna().all(axis=1)
    ds=x.loc[valid].copy()
    targets=data.loc[valid,['tmax','tmin','rain']]
    if len(ds)<550:raise ValueError('연속된 유효 관측이 550일 미만입니다.')
    split=int(len(ds)*.8)
    x_train,x_test=ds.iloc[:split],ds.iloc[split:]
    if len(x_test)<80:raise ValueError('검증 기간이 부족합니다.')
    actual=targets.iloc[split:]
    predicted={}; models={}
    for target in ('tmax','tmin'):
        model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=12,l2_regularization=1.0,random_state=42)
        model.fit(x_train,targets.iloc[:split][target])
        predicted[target]=model.predict(x_test)
        model.fit(ds,targets[target]);models[target]=model
    y_train=(targets.iloc[:split]['rain']>=0.1).astype(int)
    if y_train.nunique()<2:raise ValueError('강수 발생/미발생 학습 사례가 모두 필요합니다.')
    classifier=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=12,l2_regularization=1.0,random_state=42)
    classifier.fit(x_train,y_train)
    rain_probs=classifier.predict_proba(x_test)[:,1]
    classifier.fit(ds,(targets['rain']>=0.1).astype(int))
    metrics=pd.DataFrame([
        {'평가 항목':'최고기온 MAE','검증값':round(mean_absolute_error(actual.tmax,predicted['tmax']),2),'단위':'°C'},
        {'평가 항목':'최저기온 MAE','검증값':round(mean_absolute_error(actual.tmin,predicted['tmin']),2),'단위':'°C'},
        {'평가 항목':'강수 발생 Brier score (낮을수록 좋음)','검증값':round(brier_score_loss((actual.rain>=.1).astype(int),rain_probs),3),'단위':'0~1'},
        {'평가 항목':'시간순 검증 표본','검증값':len(x_test),'단위':'일'},
    ])
    holdout=pd.DataFrame({'날짜':actual.index,'실제 최고기온':actual.tmax.values,'AI 최고기온':predicted['tmax'],'실제 최저기온':actual.tmin.values,'AI 최저기온':predicted['tmin'],'실제 강수 발생':(actual.rain>=.1).astype(int).values,'AI 강수확률':rain_probs})
    latest=data.dropna(subset=['tmax','tmin','rain']).index.max()
    # If last observation is stale, no forward-filled pseudo-forecast.
    if (pd.Timestamp(date.today())-latest).days>3:return None,metrics,holdout
    target_day=latest+pd.Timedelta(days=1)
    future=data.reindex(data.index.union(pd.DatetimeIndex([target_day])))
    feat=_features(future).loc[[target_day],features]
    if feat.isna().any(axis=None):return None,metrics,holdout
    pred={'date':target_day.date().isoformat(),'last_observed':latest.date().isoformat(),'tmax':float(models['tmax'].predict(feat)[0]),'tmin':float(models['tmin'].predict(feat)[0]),'rain_prob':float(classifier.predict_proba(feat)[0,1])}
    return pred,metrics,holdout

def climate_periods(data):
    x=data.loc[data.index.month.isin([6,7,8,9]),['rain']].dropna().copy()
    if x.empty:return pd.DataFrame()
    x['연도']=x.index.year
    out=x.groupby('연도').agg(관측일수=('rain','size'),여름철_강수량_mm=('rain','sum'),일강수량_30mm이상_일수=('rain',lambda s:int((s>=30).sum())),일최대강수량_mm=('rain','max')).reset_index()
    return out
