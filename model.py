"""Leakage-aware daily maximum-temperature model for Daegu ASOS station 143."""
import io
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE_NAMES=['일시','날짜','date','Date','tm']
TEMP_NAMES=['최고기온(°C)','최고기온(℃)','최고기온','maxTa','tmax']
HUM_NAMES=['평균 상대습도(%)','평균상대습도(%)','평균 상대습도','avgRhm','humidity']

def parse_csv(content):
    for encoding in ('utf-8-sig','cp949','euc-kr'):
        try: return pd.read_csv(io.BytesIO(content),encoding=encoding,comment='#',low_memory=False)
        except UnicodeDecodeError: pass
    raise ValueError('CSV 인코딩을 읽을 수 없습니다.')

def normalize(frames):
    parsed=[]
    for f in frames:
        f=f.copy(); f.columns=f.columns.astype(str).str.strip()
        def get(names): return next((c for c in names if c in f.columns),None)
        d,t=get(DATE_NAMES),get(TEMP_NAMES)
        if not d or not t: raise ValueError('일시/최고기온 열이 없습니다: '+', '.join(f.columns[:12]))
        if '지점' in f.columns:
            stations=pd.to_numeric(f['지점'],errors='coerce').dropna().unique()
            if len(stations) and any(s!=143 for s in stations): raise ValueError('대구(143) 관측소 자료만 입력하세요.')
        h=get(HUM_NAMES)
        parsed.append(pd.DataFrame({'date':pd.to_datetime(f[d],errors='coerce'), 'tmax':pd.to_numeric(f[t],errors='coerce'), 'humidity':pd.to_numeric(f[h],errors='coerce') if h else np.nan}))
    data=pd.concat(parsed).dropna(subset=['date']).sort_values('date').drop_duplicates('date',keep='last').set_index('date')
    data=data.loc['2006-01-01':].asfreq('D')
    data.loc[~data.tmax.between(-30,50),'tmax']=np.nan
    data.loc[~data.humidity.between(0,100),'humidity']=np.nan
    return data

def features(data):
    t=data.tmax
    x=pd.DataFrame(index=data.index)
    for k in (1,2,3,7,14,28): x[f'tmax_lag{k}']=t.shift(k)
    for w in (3,7,14,28):
        x[f'tmax_mean{w}']=t.shift(1).rolling(w,min_periods=w).mean()
        x[f'tmax_std{w}']=t.shift(1).rolling(w,min_periods=w).std()
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
    return {'data':data,'features':x,'dataset':ds,'evaluation':ev,
            'metrics':metrics,'selected':selected,'model':model}

def predict_next(result):
    data=result['data']; future_date=data.index.max()+pd.Timedelta(days=1)
    extension=pd.concat([data,pd.DataFrame(index=[future_date],columns=data.columns)])
    feat=features(extension).iloc[[-1]]
    if feat.isna().any(axis=None): return None
    return future_date,float(result['model'].predict(feat)[0])
