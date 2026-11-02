"""Log genuine daily forecasts and score highest/lowest temperature separately."""
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
from model import normalize, parse_csv
from future7 import predict_seven_days, predict_minimum_seven_days

ROOT=Path(__file__).parent
LOG=ROOT/'data'/'prediction_history.csv'
COLUMNS=['issued_at','target_date','horizon','predicted_tmax','model_trained_at',
         'actual_tmax','absolute_error','predicted_tmin','min_model_trained_at',
         'actual_tmin','min_absolute_error']

def run():
    obs_path=ROOT/'data'/'asos_143.csv'
    model_path=ROOT/'artifacts'/'direct_model.joblib'
    if not obs_path.exists() or not model_path.exists():
        print('No observations or deployed max-temperature model; skip monitoring')
        return
    obs=normalize([parse_csv(obs_path.read_bytes())])
    log=pd.read_csv(LOG,dtype={'issued_at':str,'target_date':str}) if LOG.exists() else pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in log.columns: log[col]=np.nan
    for target_col,pred_col,actual_col,error_col in [
        ('tmax','predicted_tmax','actual_tmax','absolute_error'),
        ('tmin','predicted_tmin','actual_tmin','min_absolute_error')]:
        if target_col not in obs.columns: continue
        values=obs[target_col].dropna().to_dict()
        for idx,row in log.iterrows():
            target=pd.to_datetime(row['target_date'],errors='coerce')
            pred=pd.to_numeric(row[pred_col],errors='coerce')
            if pd.isna(target) or pd.isna(pred) or target not in values: continue
            # Previously issued forecasts only; no retroactive invented forecasts.
            if pd.isna(pd.to_numeric(row[actual_col],errors='coerce')):
                log.at[idx,actual_col]=float(values[target])
            log.at[idx,error_col]=abs(float(values[target])-float(pred))
    today=pd.Timestamp.now(tz='Asia/Seoul').date()
    today_str=today.isoformat()
    if not log['issued_at'].astype(str).str[:10].eq(today_str).any():
        saved=joblib.load(model_path)
        manifest_path=ROOT/'artifacts'/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        result={'data':obs,'direct_model':saved['model'],'direct_columns':saved['columns']}
        week=predict_seven_days(result,today=today)
        min_file=ROOT/'artifacts'/'min_model.joblib'
        min_manifest_path=ROOT/'artifacts'/'min_manifest.json'
        min_manifest=json.loads(min_manifest_path.read_text(encoding='utf-8')) if min_manifest_path.exists() else {}
        lows={}
        if min_file.exists():
            try:
                low_week=predict_minimum_seven_days(obs,min_file,today=today)
                lows={pd.Timestamp(r['날짜']).date():float(r['AI 예상 최저기온(°C)']) for _,r in low_week.iterrows()}
            except (ValueError,KeyError) as exc:
                print(f'Min forecast unavailable: {exc}')
        new=[]
        for _,r in week.iterrows():
            day=pd.Timestamp(r['날짜']).date()
            new.append({'issued_at':datetime.now().astimezone().isoformat(),
                'target_date':day.isoformat(),'horizon':(day-today).days+1,
                'predicted_tmax':round(float(r['AI 예상 최고기온(°C)']),3),
                'model_trained_at':manifest.get('trained_at','unknown'),
                'actual_tmax':np.nan,'absolute_error':np.nan,
                'predicted_tmin':round(lows[day],3) if day in lows else np.nan,
                'min_model_trained_at':min_manifest.get('trained_at','unknown') if day in lows else '',
                'actual_tmin':np.nan,'min_absolute_error':np.nan})
        log=pd.concat([log,pd.DataFrame(new)],ignore_index=True)
        print(f'Logged {len(new)} forecasts; minimum temperature: {len(lows)}')
    log=log[COLUMNS].drop_duplicates(['issued_at','target_date'],keep='first')
    LOG.parent.mkdir(exist_ok=True)
    log.to_csv(LOG,index=False,encoding='utf-8-sig')
    for label,col in [('Maximum','absolute_error'),('Minimum','min_absolute_error')]:
        scores=pd.to_numeric(log[col],errors='coerce').dropna()
        print(f'{label} real-world MAE: {scores.mean():.3f} C over {len(scores)} forecasts' if len(scores) else f'{label}: awaiting scored forecasts')

if __name__=='__main__':run()
