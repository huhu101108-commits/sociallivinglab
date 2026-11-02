"""Leakage-aware daily maximum-temperature model for Daegu ASOS station 143."""
import io
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE_NAMES=['일시','날짜','date','Date','tm']
TEMP_NAMES=['최고기온(°C)','최고기온(℃)','최고기온','maxTa','tmax']
MIN_NAMES=['최저기온(°C)','최저기온(℃)','최저기온','minTa','tmin']
RAIN_NAMES=['일강수량(mm)','일강수량','sumRn','rain']
HUM_NAMES=['평균 상대습도(%)','평균상대습도(%)','평균 상대습도','avgRhm','humidity']

def parse_csv(content):
    for encoding in ('utf-8-sig','cp949','euc-kr'):
        try: return pd.read_csv(io.BytesIO(content),encoding=encoding,comment='#',low_memory=False)
        except UnicodeDecodeError: pass
    raise ValueError('CSV 인코딩을 읽을 수 없습니다.')

def normalize(frames):
    parsed=[]
    for f in frames:
        f=f.copy().reset_index(drop=True); f.columns=f.columns.astype(str).str.strip()
        def get(names): return next((c for c in names if c in f.columns),None)
        d,t=get(DATE_NAMES),get(TEMP_NAMES)
        if not d or not t: raise ValueError('일시/최고기온 열이 없습니다: '+', '.join(f.columns[:12]))
        if '지점' in f.columns:
            stations=pd.to_numeric(f['지점'],errors='coerce').dropna().unique()
            if len(stations) and any(s!=143 for s in stations): raise ValueError('대구(143) 관측소 자료만 입력하세요.')
        h=get(HUM_NAMES); mn=get(MIN_NAMES); rn=get(RAIN_NAMES)
        parsed.append(pd.DataFrame({'tmin':pd.to_numeric(f[mn],errors='coerce') if mn else np.nan, 'rain':pd.to_numeric(f[rn],errors='coerce') if rn else np.nan, 'date':pd.to_datetime(f[d],errors='coerce'), 'tmax':pd.to_numeric(f[t],errors='coerce'), 'humidity':pd.to_numeric(f[h],errors='coerce') if h else np.nan}))
    data=(pd.concat(parsed,ignore_index=True).reset_index(drop=True).rename_axis(None)
          .dropna(subset=['date']).sort_values(by='date').drop_duplicates(subset=['date'],keep='last').set_index('date'))
    data=data.loc['2006-01-01':].asfreq('D')
    data.loc[~data.tmax.between(-30,50),'tmax']=np.nan
    data.loc[~data.humidity.between(0,100),'humidity']=np.nan
    data.loc[~data.tmin.between(-40,45),'tmin']=np.nan
    data.loc[~data.rain.between(0,1000),'rain']=np.nan
    return data

def features(data):
    t=data.tmax
    x=pd.DataFrame(index=data.index)
    for k in (1,2,3,4,5,7,14,21,28): x[f'tmax_lag{k}']=t.shift(k)
    for w in (3,7,14,28):
        x[f'tmax_mean{w}']=t.shift(1).rolling(w,min_periods=w).mean()
        x[f'tmax_std{w}']=t.shift(1).rolling(w,min_periods=w).std()
    # Recent extremes and warming/cooling trend, using observations strictly before target day.
    for w in (3,7,14):
        x[f'tmax_min{w}']=t.shift(1).rolling(w,min_periods=w).min()
        x[f'tmax_max{w}']=t.shift(1).rolling(w,min_periods=w).max()
    x['trend_3_14']=x['tmax_mean3']-x['tmax_mean14']
    x['trend_7_28']=x['tmax_mean7']-x['tmax_mean28']
    x['day_sin2']=np.sin(4*np.pi*x.index.dayofyear/365.25)
    x['day_cos2']=np.cos(4*np.pi*x.index.dayofyear/365.25)
    x['day_sin']=np.sin(2*np.pi*x.index.dayofyear/365.25)
    x['day_cos']=np.cos(2*np.pi*x.index.dayofyear/365.25)
    x['year']=x.index.year
    # No future humidity is used. Historical humidity is optional.
    if data.humidity.notna().mean()>.85:
        x['humidity_lag1']=data.humidity.shift(1)
        x['humidity_mean7']=data.humidity.shift(1).rolling(7).mean()
    return x

def evaluate(data):
    if len(data)<365*6 or data.tmax.notna().mean()<.90: raise ValueError('최소 6년, 일 최고기온 관측률 90% 이상이 필요합니다.')
    x=features(data); ds=x.assign(target=data.tmax).dropna()
    if len(ds)<365*5: raise ValueError('학습에 사용할 연속 자료가 부족합니다.')
    # Single chronological holdout: avoid fitting six large ensembles on every cold start.
    # The last complete calendar year is held out; then one final model is fitted.
    years=sorted(ds.index.year.unique())
    test_year=years[-2] if len(ds[ds.index.year==years[-1]])<300 else years[-1]
    train=ds[ds.index.year<test_year]
    test=ds[ds.index.year==test_year]
    if len(train)<365*3 or len(test)<200:
        raise ValueError('시간순 검증 자료가 부족합니다.')
    def new_model():
        return HistGradientBoostingRegressor(max_iter=70,max_leaf_nodes=12,
                    max_depth=5,l2_regularization=2,learning_rate=.08,random_state=42,
                    early_stopping=False)
    candidate=new_model()
    candidate.fit(train[x.columns],train.target)
    predictions=candidate.predict(test[x.columns])
    ev=pd.concat([
        pd.DataFrame({'date':test.index,'actual':test.target.to_numpy(),
                      'pred':predictions,'model':'히스토그램 부스팅','year':test_year}),
        pd.DataFrame({'date':test.index,'actual':test.target.to_numpy(),
                      'pred':test.tmax_lag1.to_numpy(),'model':'전날 기온 기준','year':test_year})
    ],ignore_index=True)
    summary=[]
    for name,g in ev.groupby('model'):
        obs=g.actual.to_numpy(); pred=g.pred.to_numpy()
        real_hot=obs>=33; forecast_hot=pred>=33
        tp=(real_hot&forecast_hot).sum(); fp=(~real_hot&forecast_hot).sum(); fn=(real_hot&~forecast_hot).sum()
        summary.append({'모델':name,'MAE(°C)':round(mean_absolute_error(obs,pred),2),
                        'RMSE(°C)':round(np.sqrt(mean_squared_error(obs,pred)),2),
                        '폭염 재현율':round(tp/(tp+fn),3) if tp+fn else np.nan,
                        '폭염 정밀도':round(tp/(tp+fp),3) if tp+fp else np.nan})
    metrics=pd.DataFrame(summary).sort_values('MAE(°C)')
    selected='히스토그램 부스팅'
    model=new_model()
    model.fit(ds[x.columns],ds.target)
    direct_model, direct_columns = train_direct_model(data)
    return {'direct_model':direct_model,'direct_columns':direct_columns,'data':data,'features':x,'dataset':ds,'evaluation':ev,
            'metrics':metrics,'selected':selected,'model':model}

def predict_next(result):
    data=result['data']; future_date=data.index.max()+pd.Timedelta(days=1)
    extension=pd.concat([data,pd.DataFrame(index=[future_date],columns=data.columns)])
    feat=features(extension).iloc[[-1]]
    if feat.isna().any(axis=None): return None
    return future_date,float(result['model'].predict(feat)[0])


def direct_features(data, horizon):
    """At issue date t, use observations no later than t; predict t+h."""
    # features(data) for target t+1 uses observations up to issue date t.
    base = features(data).shift(-1)
    target_dates = data.index + pd.Timedelta(days=horizon)
    base['target_day_sin'] = np.sin(2*np.pi*target_dates.dayofyear/365.25)
    base['target_day_cos'] = np.cos(2*np.pi*target_dates.dayofyear/365.25)
    base['horizon'] = float(horizon)
    return base

def train_direct_model(data):
    """Single pooled direct multi-horizon model; never feeds predictions back as inputs."""
    parts=[]
    for h in range(1,12):
        part=direct_features(data,h)
        part['target']=data['tmax'].shift(-h)
        parts.append(part.dropna())
    train=pd.concat(parts,ignore_index=True)
    if len(train)<365*5*7:
        raise ValueError('직접 다중시점 예측을 학습할 자료가 부족합니다.')
    columns=[c for c in train.columns if c!='target']
    direct=HistGradientBoostingRegressor(max_iter=85,max_leaf_nodes=15,max_depth=5,
             learning_rate=.065,l2_regularization=3,random_state=42,early_stopping=False)
    direct.fit(train[columns],train['target'])
    return direct,columns
