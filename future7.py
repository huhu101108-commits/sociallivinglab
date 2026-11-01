"""Seven-day experimental recursive ML forecast, using actual ASOS observations through yesterday."""
import numpy as np
import pandas as pd
from model import features


def predict_seven_days(result, today=None):
    today = pd.Timestamp(today or pd.Timestamp.now(tz='Asia/Seoul').date()).normalize()
    yesterday = today - pd.Timedelta(days=1)
    historical = result['data'].loc[:yesterday].copy()
    if historical.empty or historical['tmax'].last_valid_index() is None:
        raise ValueError('최근 ASOS 최고기온 관측자료가 없습니다.')
    last_actual = historical['tmax'].last_valid_index()
    if (yesterday-last_actual).days > 4:
        raise ValueError('최신 관측자료가 4일 이상 지연되었습니다. 실제 현재 예측으로 표시할 수 없습니다.')
    # The original model optionally trains on historical humidity. No future humidity
    # is available: hold the latest known humidity constant, an explicit approximation.
    last_humidity = historical['humidity'].dropna()
    h = float(last_humidity.iloc[-1]) if len(last_humidity) else np.nan
    target_end = today + pd.Timedelta(days=6)
    for target in pd.date_range(last_actual+pd.Timedelta(days=1),target_end):
        extended = historical.reindex(historical.index.union(pd.DatetimeIndex([target]))).sort_index()
        extended.loc[target,'humidity'] = h
        # For lagged humidity features, fill only days after last observation.
        if target > last_actual + pd.Timedelta(days=1):
            extended.loc[last_actual+pd.Timedelta(days=1):target,'humidity'] = h
        row = features(extended).loc[[target]]
        row = row.reindex(columns=result['features'].columns)
        if row.isna().any(axis=None):
            raise ValueError('최근 28일 관측자료가 불충분하여 7일 예측을 만들 수 없습니다.')
        prediction = float(result['model'].predict(row)[0])
        if not -30 <= prediction <= 50:
            raise ValueError('AI 예측값이 비정상적입니다.')
        historical.loc[target,'tmax'] = prediction
        historical.loc[target,'humidity'] = h
    return pd.DataFrame({'날짜':pd.date_range(today,target_end),
                         'AI 예상 최고기온(°C)':historical.loc[today:target_end,'tmax'].to_numpy(),
                         '관측 기반 마지막 날짜':[last_actual.date()]*7,
                         '예측 방식':['과거 예측을 입력으로 재사용하는 재귀 예측']*7})
