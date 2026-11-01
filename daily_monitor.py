"""Record genuine out-of-sample daily forecasts and score them after observations arrive."""
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
from model import normalize, parse_csv
from future7 import predict_seven_days

ROOT=Path(__file__).parent
LOG=ROOT/'data'/'prediction_history.csv'
COLUMNS=['issued_at','target_date','horizon','predicted_tmax','model_trained_at','actual_tmax','absolute_error']

def run():
    obs_path=ROOT/'data'/'asos_143.csv'
    model_path=ROOT/'artifacts'/'direct_model.joblib'
    if not obs_path.exists() or not model_path.exists():
        print('No observations or deployed model yet; skip forecast monitoring')
        return
    obs=normalize([parse_csv(obs_path.read_bytes())])
    log=pd.read_csv(LOG,dtype={'issued_at':str,'target_date':str,'model_trained_at':str}) if LOG.exists() else pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in log: log[col]=np.nan
    # Score only forecasts already issued on earlier days, never backfill predictions.
    actual=obs.tmax.dropna()
    values=actual.to_dict()
    for idx,row in log.iterrows():
        target=pd.Timestamp(row['target_date'])
        if pd.isna(row['actual_tmax']) and target in values:
            a=float(values[target]); p=float(row['predicted_tmax'])
            log.at[idx,'actual_tmax']=a
            log.at[idx,'absolute_error']=abs(a-p)
    today=pd.Timestamp.now(tz='Asia/Seoul').date()
    today_str=today.isoformat()
    if not (log['issued_at'].astype(str).str[:10]==today_str).any():
        saved=joblib.load(model_path)
        manifest_path=ROOT/'artifacts'/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        result={'data':obs,'direct_model':saved['model'],'direct_columns':saved['columns']}
        week=predict_seven_days(result,today=today)
        new=[]
        for _,r in week.iterrows():
            new.append({'issued_at':datetime.now().astimezone().isoformat(),
                'target_date':pd.Timestamp(r['날짜']).date().isoformat(),
                'horizon':int((pd.Timestamp(r['날짜']).date()-today).days)+1,
                'predicted_tmax':round(float(r['AI 예상 최고기온(°C)']),3),
                'model_trained_at':manifest.get('trained_at','unknown'),
                'actual_tmax':np.nan,'absolute_error':np.nan})
        log=pd.concat([log,pd.DataFrame(new)],ignore_index=True)
        print(f'Logged {len(new)} real-time forecasts for {today_str}')
    log=log[COLUMNS].drop_duplicates(['issued_at','target_date'],keep='first')
    LOG.parent.mkdir(exist_ok=True)
    log.to_csv(LOG,index=False,encoding='utf-8-sig')
    scored=log.dropna(subset=['absolute_error'])
    if len(scored):
        print(f'Real-world forecast MAE: {scored.absolute_error.astype(float).mean():.3f} C, {len(scored)} forecasts')
    else:
        print('Waiting for future observations to score predictions')

if __name__=='__main__': run()
