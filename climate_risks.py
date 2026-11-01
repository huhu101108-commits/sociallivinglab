"""Transparent, non-official weather-condition screening from existing observations/forecast."""
import numpy as np
import pandas as pd


def historical_baseline(data, target_dates, column='tmax', window=7):
    """Compare dates to nearby calendar days in prior years, not a formal 30-year normal."""
    if column not in data or data[column].dropna().empty:
        return pd.DataFrame()
    source=data[[column]].dropna().copy()
    source['doy']=source.index.dayofyear
    source['year']=source.index.year
    rows=[]
    for day in pd.to_datetime(target_dates):
        # Prior years only, to avoid future leakage when looking at past dates.
        prev=source[source.year<day.year]
        dist=np.abs(prev.doy.to_numpy()-day.dayofyear)
        dist=np.minimum(dist,366-dist)
        vals=prev.loc[dist<=window,column]
        rows.append({'날짜':day,'과거 중앙값':vals.median() if len(vals)>=40 else np.nan,
                     '상위 90% 기준':vals.quantile(.9) if len(vals)>=40 else np.nan,
                     '하위 10% 기준':vals.quantile(.1) if len(vals)>=40 else np.nan,
                     '비교 관측수':len(vals)})
    return pd.DataFrame(rows)


def weather_risks(forecast):
    """Only forecast categories actually returned by KMA are used."""
    if forecast is None or forecast.empty:
        return [], []
    now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None)
    f=forecast.loc[forecast.index>=now.floor('h')].copy()
    if f.empty: return [], []
    alerts=[]; available=[]
    def num(key):
        if key not in f: return pd.Series(dtype=float)
        available.append(key)
        return pd.to_numeric(f[key],errors='coerce').dropna()
    tmp=num('TMP'); tmx=num('TMX'); tmn=num('TMN'); reh=num('REH'); wsd=num('WSD'); pop=num('POP')
    if not tmx.empty or not tmp.empty:
        peak=max([s.max() for s in (tmx,tmp) if not s.empty])
        if peak>=33: alerts.append(('고온 조건','높음',f'예보 기온 최고 {peak:.1f}°C: 더위 대비가 필요합니다.'))
        elif peak>=30: alerts.append(('고온 조건','주의',f'예보 기온 최고 {peak:.1f}°C: 더운 시간대 활동에 유의하세요.'))
    if not tmn.empty or not tmp.empty:
        low=min([s.min() for s in (tmn,tmp) if not s.empty])
        if low<=-12: alerts.append(('저온 조건','높음',f'예보 기온 최저 {low:.1f}°C: 한랭 노출에 유의하세요.'))
        elif low<=0: alerts.append(('저온 조건','주의',f'예보 기온 최저 {low:.1f}°C: 결빙 및 추위에 대비하세요.'))
    if not wsd.empty and wsd.max()>=9:
        alerts.append(('강한 바람 조건','주의',f'예보 풍속 최대 {wsd.max():.1f}m/s. 순간최대풍속 및 공식 특보와는 다릅니다.'))
    if not reh.empty and reh.min()<=30:
        alerts.append(('낮은 습도 조건','참고',f'예보 상대습도 최저 {reh.min():.0f}%. 산불 위험도나 건조특보 판정은 아닙니다.'))
    if not pop.empty and pop.max()>=70:
        alerts.append(('강수 가능성','참고',f'예보 강수확률 최대 {pop.max():.0f}%. 집중호우의 강도를 뜻하지 않습니다.'))
    if 'PCP' in f:
        available.append('PCP')
        raw=f['PCP'].dropna().astype(str)
        # '30.0mm 이상' -> lower bound 30; '강수없음' -> 0
        amounts=raw.str.extract(r'(\d+(?:\.\d+)?)\s*mm',expand=False).astype(float)
        if amounts.notna().any() and amounts.max()>=20:
            alerts.append(('많은 예보 강수량','주의',f'예보 시간대 강수량 표기 최대 {amounts.max():.1f}mm 이상/해당 구간. 실제 강수·침수 예측은 아닙니다.'))
    if 'SNO' in f:
        available.append('SNO')
        snow=f['SNO'].dropna().astype(str)
        if snow.str.contains(r'\d+(?:\.\d+)?\s*cm',regex=True).any():
            alerts.append(('적설 가능성','참고','예보에 적설량이 표시됩니다. 결빙·교통 상황을 확인하세요.'))
    return alerts,available


def historical_extremes(data):
    if 'tmax' not in data: return pd.DataFrame()
    frame=pd.DataFrame(index=data.index)
    frame['연도']=data.index.year
    frame['33°C 이상 일수']=(data.tmax>=33).where(data.tmax.notna()).astype(float)
    frame['30°C 이상 일수']=(data.tmax>=30).where(data.tmax.notna()).astype(float)
    frame['최고기온 관측일수']=data.tmax.notna().astype(int)
    if 'tmin' in data:
        frame['0°C 미만 일수']=(data.tmin<0).where(data.tmin.notna()).astype(float)
        frame['-12°C 이하 일수']=(data.tmin<=-12).where(data.tmin.notna()).astype(float)
        frame['최저기온 관측일수']=data.tmin.notna().astype(int)
    return frame.groupby('연도').sum(min_count=1).reset_index()
