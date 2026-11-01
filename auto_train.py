"""GitHub Actions daily model selection, using strictly earlier years for fitting."""
import json, os
from pathlib import Path
from datetime import date, timedelta
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from model import direct_features
from kma_api import fetch_asos

ROOT=Path(__file__).parent
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)

def dataset():
    key=os.environ['ASOS_SERVICE_KEY']
    today=date.today()
    end=today-timedelta(days=2)  # completed observation, allow reporting delay
    cached=DATA/'asos_143.csv'
    if cached.exists():
        from model import parse_csv, normalize
        historical=normalize([parse_csv(cached.read_bytes())])
        start=(historical.index.max()+pd.Timedelta(days=1)).date()
        if start<=end:
            recent=fetch_asos(key,start=start.isoformat(),end=end.isoformat())
            combined=pd.concat([historical,recent]); combined=combined[~combined.index.duplicated(keep='last')].asfreq('D')
        else: combined=historical
    else:
        combined=fetch_asos(key,start='2006-01-01',end=end.isoformat())
    if len(combined)<365*15: raise ValueError('과거 관측자료가 부족합니다.')
    combined.reset_index().rename(columns={'index':'date'}).to_csv(cached,index=False,encoding='utf-8-sig')
    return combined

def matrix(data):
    parts=[]
    for h in range(1,12):
        frame=direct_features(data,h)
        frame['target']=data.tmax.shift(-h)
        frame['issue_date']=data.index
        frame['target_date']=data.index+pd.Timedelta(days=h)
        parts.append(frame.dropna())
    return pd.concat(parts,ignore_index=True)

def estimator(iterations,leaves,l2):
    return HistGradientBoostingRegressor(max_iter=iterations,max_leaf_nodes=leaves,max_depth=5,
        learning_rate=.065,l2_regularization=l2,random_state=42,early_stopping=False)



def train_minimum(data):
    """Independently train/validate minimum-temperature AI if observations exist."""
    if 'tmin' not in data or data.tmin.notna().sum()<365*6:
        print('Minimum-temperature data insufficient; skip minimum AI training')
        return
    from model import direct_features
    # Reuse leakage-safe feature engineering, now with tmin history as input.
    cold=data.copy(); cold['tmax']=data['tmin']
    frame=matrix(cold)
    cols=[c for c in frame if c not in ('target','issue_date','target_date')]
    train=frame[frame.target_date<'2024-01-01']
    val=frame[(frame.target_date>='2024-01-01')&(frame.target_date<'2026-01-01')]
    if len(train)<10000 or len(val)<3500:
        print('Minimum-temperature validation data insufficient; skip')
        return
    params=(85,15,3)
    candidate=estimator(*params);candidate.fit(train[cols],train.target)
    score=float(mean_absolute_error(val.target,candidate.predict(val[cols])))
    meta=ART/'min_manifest.json'
    previous=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
    old=previous.get('validation_mae')
    if old is not None and score>=float(old)-0.01:
        print(f'Keep minimum model: candidate={score:.3f}, incumbent={old:.3f}')
        return
    model=estimator(*params);model.fit(frame[cols],frame.target)
    tmp=ART/'min_model.tmp';joblib.dump({'model':model,'columns':cols},tmp)
    tmp.replace(ART/'min_model.joblib')
    meta.write_text(json.dumps({'trained_at':date.today().isoformat(),
        'validation_mae':round(score,4),'validation_period':'2024-01-01 ~ 2025-12-31',
        'training_end':str(data.index.max().date()),'validation_rows':len(val),
        'target':'tmin'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Updated minimum-temperature model',round(score,3))

def main():
    data=dataset(); frame=matrix(data)
    columns=[c for c in frame if c not in ('target','issue_date','target_date')]
    # Never train on any target in the 2024-2025 validation interval.
    train=frame[frame.target_date<'2024-01-01']
    val=frame[(frame.target_date>='2024-01-01')&(frame.target_date<'2026-01-01')]
    if len(train)<10000 or len(val)<3500: raise ValueError('검증 자료가 부족합니다.')
    candidates=[(70,12,2),(85,15,3),(110,12,5)]
    trials=[]
    for it,leaves,l2 in candidates:
        m=estimator(it,leaves,l2); m.fit(train[columns],train.target)
        pred=m.predict(val[columns]); mae=float(mean_absolute_error(val.target,pred))
        trials.append((mae,(it,leaves,l2)))
    trials.sort(key=lambda z:z[0]); best_score,best_params=trials[0]
    manifest_path=ART/'manifest.json'; previous={}
    if manifest_path.exists(): previous=json.loads(manifest_path.read_text(encoding='utf-8'))
    # Small improvements can be noise. Keep incumbent unless >0.01 C improvement.
    incumbent=previous.get('validation_mae')
    if incumbent is not None and best_score>=float(incumbent)-0.01:
        print(f'Keep existing model: candidate={best_score:.3f}, incumbent={incumbent:.3f}')
        train_minimum(data)
        return
    # Refit selected configuration using all observations, for actual deployment.
    model=estimator(*best_params); model.fit(frame[columns],frame.target)
    tmp=ART/'direct_model.tmp'; joblib.dump({'model':model,'columns':columns},tmp)
    tmp.replace(ART/'direct_model.joblib')
    manifest={'trained_at':date.today().isoformat(),'validation_mae':round(best_score,4),
       'validation_period':'2024-01-01 ~ 2025-12-31','training_end':str(data.index.max().date()),
       'parameters':list(best_params),'validation_rows':len(val),'candidate_mae':[round(t[0],4) for t in trials]}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Updated model:',manifest)
    train_minimum(data)

if __name__=='__main__': main()
