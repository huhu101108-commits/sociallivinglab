"""Seven-day direct-horizon ASOS research models; genuine time-ordered holdout."""
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, brier_score_loss


def _features(obs):
    # Every feature is available at forecast issue date T, no future observation leakage.
    f=pd.DataFrame(index=obs.index)
    for name in ('tmax','tmin','rain'):
        for lag in (0,1,2,3,7):f[f'{name}_lag{lag}']=obs[name].shift(lag)
        f[f'{name}_avg7']=obs[name].rolling(7,min_periods=7).mean()
        f[f'{name}_avg14']=obs[name].rolling(14,min_periods=14).mean()
    f['sin']=np.sin(2*np.pi*f.index.dayofyear/365.25)
    f['cos']=np.cos(2*np.pi*f.index.dayofyear/365.25)
    return f

def _reg():
    return HistGradientBoostingRegressor(max_iter=75,max_leaf_nodes=10,l2_regularization=2,random_state=42)
def _clf():
    return HistGradientBoostingClassifier(max_iter=75,max_leaf_nodes=10,l2_regularization=2,random_state=42)

@st.cache_data(ttl=21600,show_spinner=False)
def seven_day_models(observations):
    if len(observations)<750:raise ValueError('7일 AI 학습에는 최소 750일 관측이 필요합니다.')
    obs=observations[['tmax','tmin','rain']].sort_index().asfreq('D')
    feats=_features(obs)
    # Restrict forecast to most recent actual complete observation; no synthetic input.
    last=obs.dropna(subset=['tmax','tmin','rain']).index.max()
    if pd.isna(last):raise ValueError('유효한 최신 관측이 없습니다.')
    latest_features=feats.loc[[last]]
    if latest_features.isna().any(axis=None):raise ValueError('최근 14일 관측 결측으로 7일 예측을 생성할 수 없습니다.')
    fresh=(pd.Timestamp(date.today())-last).days<=3
    results=[];metrics=[]
    for horizon in range(1,8):
        target=obs.shift(-horizon)
        eligible=feats.notna().all(axis=1)&target.notna().all(axis=1)
        x=feats.loc[eligible];y=target.loc[eligible]
        if len(x)<550:raise ValueError(f'{horizon}일 후 학습 표본이 부족합니다.')
        split=int(len(x)*.8)
        if len(x)-split<80:raise ValueError('검증 표본이 부족합니다.')
        row={'날짜':(last+pd.Timedelta(days=horizon)).date().isoformat(),'예측일':horizon}
        for field,label in [('tmax','최고기온'),('tmin','최저기온')]:
            model=_reg();model.fit(x.iloc[:split],y[field].iloc[:split])
            val=model.predict(x.iloc[split:]);mae=mean_absolute_error(y[field].iloc[split:],val)
            metrics.append({'예측일':horizon,'평가 항목':f'{label} MAE','검증값':round(float(mae),2),'단위':'°C','검증 표본':len(x)-split})
            if fresh:
                model.fit(x,y[field]);row[label+'(°C)']=round(float(model.predict(latest_features)[0]),1)
        rainy=(y.rain>=0.1).astype(int)
        if rainy.iloc[:split].nunique()==2:
            clf=_clf();clf.fit(x.iloc[:split],rainy.iloc[:split]);probs=clf.predict_proba(x.iloc[split:])[:,1]
            metrics.append({'예측일':horizon,'평가 항목':'강수 발생 Brier','검증값':round(float(brier_score_loss(rainy.iloc[split:],probs)),3),'단위':'0~1','검증 표본':len(x)-split})
            if fresh:
                clf.fit(x,rainy);row['강수확률(%)']=round(float(clf.predict_proba(latest_features)[0,1])*100,1)
        else:
            row['강수확률(%)']=np.nan
        if fresh:results.append(row)
    return pd.DataFrame(results),pd.DataFrame(metrics),last.date().isoformat(),fresh

def daily_rain_extremes(obs):
    """Daily heavy-rain proxy, NOT hourly flash-rain forecast."""
    df=obs[['rain']].dropna().copy()
    df=df[df.index.month.isin([6,7,8,9])]
    if df.empty:return pd.DataFrame()
    df['year']=df.index.year
    return df.groupby('year').agg(관측일수=('rain','size'),여름철_총강수량_mm=('rain','sum'),일강수량_30mm이상=('rain',lambda s:int((s>=30).sum())),일강수량_80mm이상=('rain',lambda s:int((s>=80).sum())),일최대강수량_mm=('rain','max')).reset_index().rename(columns={'year':'연도'})
