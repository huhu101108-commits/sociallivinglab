"""Direct 7-day ML forecast: predictions are never used as future inputs."""
import pandas as pd
from model import direct_features

def predict_seven_days(result, today=None):
    today=pd.Timestamp(today or pd.Timestamp.now(tz='Asia/Seoul').date()).normalize()
    yesterday=today-pd.Timedelta(days=1)
    historical=result['data'].loc[:yesterday].copy()
    last_actual=historical['tmax'].last_valid_index()
    if last_actual is None:
        raise ValueError('최신 ASOS 최고기온 관측자료가 없습니다.')
    if (yesterday-last_actual).days>4:
        raise ValueError('최신 관측자료가 4일 이상 지연되어 예측을 표시하지 않습니다.')
    model=result['direct_model']
    columns=result['direct_columns']
    dates=pd.date_range(today,periods=7,freq='D')
    rows=[]
    for target in dates:
        horizon=(target-last_actual).days
        if horizon<1 or horizon>11:
            raise ValueError('관측자료가 지연되어 직접 예측 학습 범위를 벗어났습니다. 최신 ASOS 자료를 갱신하세요.')
        # Construct the issue-day row using only historical measurements.
        # Append one empty day solely to create the t+1 lag feature row.
        extended=historical.reindex(historical.index.union(pd.DatetimeIndex([last_actual+pd.Timedelta(days=1)]))).sort_index()
        x=direct_features(extended,horizon).loc[[last_actual]]
        x=x.reindex(columns=columns)
        if x.isna().any(axis=None):
            raise ValueError('최근 28일 관측자료가 불충분하여 예측을 생성할 수 없습니다.')
        prediction=float(model.predict(x)[0])
        if not -30<=prediction<=50:
            raise ValueError('AI 예측값이 비정상적입니다.')
        rows.append({'날짜':target,'AI 예상 최고기온(°C)':prediction,
                     '관측 기반 마지막 날짜':last_actual.date(),
                     '예측 방식':f'관측자료 기반 {horizon}일 직접 예측'})
    return pd.DataFrame(rows)
