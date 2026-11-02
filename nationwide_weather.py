"""National KMA forecast access and transparent precipitation diagnostics (not trained AI)."""
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from forecast import grid, latest_base, URL
import requests

REGIONS={
'서울':(37.5665,126.9780),'부산':(35.1796,129.0756),'대구':(35.8714,128.6014),
'인천':(37.4563,126.7052),'광주':(35.1595,126.8526),'대전':(36.3504,127.3845),
'울산':(35.5384,129.3114),'세종':(36.4800,127.2890),'수원':(37.2636,127.0286),
'춘천':(37.8813,127.7298),'강릉':(37.7519,128.8761),'청주':(36.6424,127.4890),
'천안':(36.8151,127.1139),'전주':(35.8242,127.1480),'목포':(34.8118,126.3922),
'포항':(36.0190,129.3435),'안동':(36.5684,128.7294),'창원':(35.2279,128.6811),
'진주':(35.1800,128.1076),'제주':(33.4996,126.5312),'서귀포':(33.2541,126.5601),
'원주':(37.3422,127.9202),'충주':(36.9910,127.9259),'군산':(35.9677,126.7369),
'여수':(34.7604,127.6622),'김해':(35.2285,128.8894),'구미':(36.1195,128.3446),
'평택':(36.9921,127.1129),'의정부':(37.7381,127.0337),'고양':(37.6584,126.8320),
'성남':(37.4200,127.1265),'용인':(37.2411,127.1776),'화성':(37.1995,126.8312),
'안산':(37.3219,126.8309),'안양':(37.3943,126.9568),'부천':(37.5034,126.7660),
'남양주':(37.6360,127.2165),'파주':(37.7600,126.7800),'김포':(37.6152,126.7156),
'하남':(37.5393,127.2149),'이천':(37.2720,127.4350),'양평':(37.4914,127.4875),
'속초':(38.2070,128.5918),'동해':(37.5247,129.1143),'태백':(37.1641,128.9856),
'홍천':(37.6972,127.8886),'제천':(37.1326,128.1909),'음성':(36.9403,127.6905),
'서산':(36.7845,126.4503),'당진':(36.8931,126.6283),'보령':(36.3335,126.6127),
'공주':(36.4466,127.1190),'논산':(36.1872,127.0987),'익산':(35.9483,126.9576),
'정읍':(35.5699,126.8560),'남원':(35.4164,127.3904),'순천':(34.9506,127.4872),
'광양':(34.9407,127.6959),'나주':(35.0158,126.7108),'해남':(34.5735,126.5989),
'경주':(35.8562,129.2247),'경산':(35.8251,128.7415),'영천':(35.9733,128.9385),
'상주':(36.4109,128.1591),'김천':(36.1398,128.1136),'문경':(36.5865,128.1866),
'거제':(34.8806,128.6211),'통영':(34.8544,128.4332),'사천':(35.0038,128.0642),
'밀양':(35.5037,128.7466),'양산':(35.3350,129.0373),'울진':(36.9930,129.4004),
'울릉':(37.4845,130.9057),'서귀포동부':(33.2875,126.7400),
}

@st.cache_data(ttl=3600,show_spinner='선택 지역의 공식 예보 조회 중…')
def nationwide_forecast(key,region):
    if region not in REGIONS: raise ValueError('지원하지 않는 지역')
    nx,ny=grid(*REGIONS[region]);base=latest_base()
    params={'authKey':key,'pageNo':1,'numOfRows':2000,'dataType':'JSON',
        'base_date':base.strftime('%Y%m%d'),'base_time':base.strftime('%H%M'),'nx':nx,'ny':ny}
    r=requests.get(URL,params=params,timeout=25);r.raise_for_status()
    payload=r.json();response=payload.get('response',{})
    if str(response.get('header',{}).get('resultCode'))!='00':
        raise ValueError(response.get('header',{}).get('resultMsg','기상청 API 응답 오류'))
    items=response.get('body',{}).get('items',{}).get('item',[])
    if isinstance(items,dict):items=[items]
    if not items:raise ValueError('예보 자료 없음')
    df=pd.DataFrame(items)
    df['date']=pd.to_datetime(df['fcstDate'].astype(str)+df['fcstTime'].astype(str),format='%Y%m%d%H%M')
    p=df.pivot_table(index='date',columns='category',values='fcstValue',aggfunc='first').sort_index()
    return p,base,(nx,ny)

def parse_rain(value):
    if value is None or pd.isna(value):return float('nan')
    t=str(value).strip().replace(',','')
    if t in ('강수없음','없음','0','0.0','0mm','0.0mm'):return 0.0
    # Do not turn a range, inequality or qualitative category into a point estimate.
    if any(mark in t for mark in ('미만','이상','~','-','초과')):return float('nan')
    m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*',t)
    return float(m.group(1)) if m else float('nan')

def rain_summary(frame):
    now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).floor('h')
    frame=frame.loc[frame.index>=now].copy()
    if frame.empty:return pd.DataFrame()
    out=pd.DataFrame({'시각':frame.index.strftime('%m-%d %H:%M')})
    for key,name in [('POP','강수확률(%)'),('REH','상대습도(%)'),('WSD','풍속(m/s)')]:
        out[name]=pd.to_numeric(frame[key],errors='coerce').to_numpy() if key in frame else float('nan')
    out['강수량(mm)']=frame['PCP'].map(parse_rain).to_numpy() if 'PCP' in frame else float('nan')
    out['강수량 원문']=frame['PCP'].astype(str).to_numpy() if 'PCP' in frame else '자료 없음'
    return out

def climate_rain_analysis(path):
    # Existing station data schema varies; only process confidently identified rain columns.
    df=pd.read_csv(path,encoding='utf-8-sig')
    rain_col=next((c for c in ('sumRn','일강수량(mm)','일강수량','rain','precipitation','강수량(mm)') if c in df),None)
    date_col=next((c for c in ('tm','일시','날짜','date') if c in df),None)
    if not rain_col or not date_col:return pd.DataFrame()
    dates=pd.to_datetime(df[date_col],errors='coerce'); rain=pd.to_numeric(df[rain_col],errors='coerce')
    x=pd.DataFrame({'date':dates,'rain':rain}).dropna(subset=['date','rain'])
    x=x[(x.rain>=0)&(x.date.dt.month.isin([6,7,8,9]))].copy()
    x['period']=pd.cut(x.date.dt.year,bins=[1999,2010,2020,2100],labels=['2000~2010','2011~2020','2021~현재'])
    out=x.groupby('period',observed=True).agg(관측일수=('rain','size'),여름철_총강수량_mm=('rain','sum'),일최대강수량_mm=('rain','max'),일강수량_30mm이상_일수=('rain',lambda v:int((v>=30).sum()))).reset_index()
    out=out.rename(columns={'period':'기간'})
    return out
