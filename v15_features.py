"""v15 user-facing, transparent weather analyses. No fabricated weather observations."""
import numpy as np
import pandas as pd
from nationwide_weather import parse_rain

def closest_region(lat,lon,regions):
    if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError('좌표 범위 오류')
    out=[]
    for name,(a,b) in regions.items():
        dlat=np.radians(a-lat);dlon=np.radians(b-lon)
        h=np.sin(dlat/2)**2+np.cos(np.radians(lat))*np.cos(np.radians(a))*np.sin(dlon/2)**2
        out.append((name,6371*2*np.arcsin(np.sqrt(h))))
    return min(out,key=lambda x:x[1])

def forecast_risk(f):
    now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).floor('h')
    future=f.loc[f.index>=now].head(48)
    if future.empty:future=f.tail(48)
    pop=pd.to_numeric(future['POP'],errors='coerce') if 'POP' in future else pd.Series(dtype=float)
    rain=future['PCP'].map(parse_rain) if 'PCP' in future else pd.Series(dtype=float)
    p=pop.max() if len(pop) else np.nan
    r=rain.max() if len(rain) else np.nan
    # Advisory thresholds, not KMA warning criteria. Unknown amounts stay unknown.
    if pd.isna(p) and pd.isna(r):level=0;label='판단 자료 부족'
    elif (pd.notna(r) and r>=20) or (pd.notna(p) and p>=85):level=3;label='강수 주의 높음'
    elif (pd.notna(r) and r>=10) or (pd.notna(p) and p>=60):level=2;label='강수 주의'
    else:level=1;label='낮음'
    return {'max_pop':p,'max_rain':r,'level':level,'label':label,'hours':len(future)}

def region_risk_frame(rows):
    x=pd.DataFrame(rows)
    colors={0:[130,130,130,210],1:[44,119,180,220],2:[241,151,45,220],3:[211,65,60,235]}
    x['color']=x['수준'].map(colors)
    return x.sort_values(['수준','지역'],ascending=[False,True]).reset_index(drop=True)

def evaluate_alerts(frame,pop_threshold,high_threshold,low_threshold):
    now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).floor('h')
    x=frame.loc[frame.index>=now].head(72)
    msgs=[]
    if x.empty:return ['미래 예보가 없어 기준을 검사할 수 없습니다.']
    for col,limit,greater,label in [('POP',pop_threshold,True,'강수확률'),('TMP',high_threshold,True,'고온'),('TMP',low_threshold,False,'저온')]:
        if col not in x:continue
        vals=pd.to_numeric(x[col],errors='coerce').dropna()
        if vals.empty:continue
        hit=vals[vals>=limit] if greater else vals[vals<=limit]
        if len(hit):msgs.append(f'{label}: {hit.index[0]:%m-%d %H:%M} 기준 도달 (예보값 {hit.iloc[0]:g}, 설정 {limit:g})')
    return msgs

def climate_report(raw):
    if not isinstance(raw,pd.DataFrame):raise ValueError('표 형식 관측자료가 필요합니다.')
    x=raw.copy()
    # Existing fetch_asos produces datetime index and normalized tmax/tmin/rain.
    names={'date':['tm','date','날짜','일시'],'rain':['rain','sumRn','일강수량(mm)','일강수량','강수량(mm)'],'tmax':['tmax','maxTa','최고기온(°C)','최고기온'],'tmin':['tmin','minTa','최저기온(°C)','최저기온']}
    d={}
    for k,aliases in names.items():
        found=next((c for c in aliases if c in x.columns),None)
        if found is not None:d[k]=x[found]
        elif k=='date' and isinstance(x.index,pd.DatetimeIndex):d[k]=pd.Series(x.index,index=x.index)
    if 'date' not in d or 'rain' not in d:raise ValueError('날짜와 일강수량 열이 필요합니다.')
    frame=pd.DataFrame(d).copy()
    frame['date']=pd.to_datetime(frame.date,errors='coerce')
    for k in ('rain','tmax','tmin'):
        if k in frame:frame[k]=pd.to_numeric(frame[k],errors='coerce')
    frame=frame.dropna(subset=['date']).drop_duplicates('date').sort_values('date')
    frame.loc[~frame.rain.between(0,1000),'rain']=np.nan
    frame['year']=frame.date.dt.year
    agg={'관측일수':('rain','count'),'연강수량(mm)':('rain','sum'),'강한비 일수(30mm 이상)':('rain',lambda s:int((s>=30).sum())),'매우 강한비 일수(80mm 이상)':('rain',lambda s:int((s>=80).sum()))}
    if 'tmax' in frame:agg['평균 최고기온(°C)']=('tmax','mean')
    if 'tmin' in frame:agg['평균 최저기온(°C)']=('tmin','mean')
    y=frame.groupby('year').agg(**agg).reset_index().rename(columns={'year':'연도'})
    # Incomplete years should not be used for annual comparisons.
    y=y[y['관측일수']>=330].copy()
    if len(y)<2:raise ValueError('연간 관측일 330일 이상인 연도가 2개 이상 필요합니다.')
    y['연강수량(mm)']=y['연강수량(mm)'].round(1)
    for c in ('평균 최고기온(°C)','평균 최저기온(°C)'):
        if c in y:y[c]=y[c].round(2)
    n=min(10,len(y)//2)
    old=y.iloc[:n];new=y.iloc[-n:]
    summary=[]
    for col in ['연강수량(mm)','강한비 일수(30mm 이상)','매우 강한비 일수(80mm 이상)','평균 최고기온(°C)','평균 최저기온(°C)']:
        if col in y:
            summary.append({'항목':col,'과거 평균':round(old[col].mean(),2),'최근 평균':round(new[col].mean(),2),'차이':round(new[col].mean()-old[col].mean(),2),'비교기간':f'{int(old.연도.min())}~{int(old.연도.max())} / {int(new.연도.min())}~{int(new.연도.max())}'})
    return pd.DataFrame(summary),y

def explain_model(obs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error
    if not {'tmax','tmin','rain'}.issubset(obs.columns):raise ValueError('최고·최저기온·강수량 자료가 필요합니다.')
    x=pd.DataFrame(index=obs.index)
    for c in ('tmax','tmin','rain'):
        for lag in (1,2,3,7):x[f'{c}_lag{lag}']=obs[c].shift(lag)
        x[f'{c}_avg7']=obs[c].shift(1).rolling(7,min_periods=7).mean()
    x['season_sin']=np.sin(2*np.pi*x.index.dayofyear/365.25)
    x['season_cos']=np.cos(2*np.pi*x.index.dayofyear/365.25)
    mask=x.notna().all(axis=1)&obs.tmax.notna()
    x=x.loc[mask];y=obs.loc[mask,'tmax']
    if len(x)<550:raise ValueError('유효 관측일 550일 이상 필요')
    split=int(len(x)*.8)
    model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=12,l2_regularization=1,random_state=42)
    model.fit(x.iloc[:split],y.iloc[:split])
    test=x.iloc[split:];target=y.iloc[split:]
    mae=mean_absolute_error(target,model.predict(test))
    importance=permutation_importance(model,test,target,scoring='neg_mean_absolute_error',n_repeats=3,random_state=42,n_jobs=1)
    result=pd.DataFrame({'변수':x.columns,'오차 증가(°C)':importance.importances_mean.round(3)}).sort_values('오차 증가(°C)',ascending=False)
    return result.head(12),mae
