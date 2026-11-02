"""Align ASOS-based AI hourly precipitation forecasts with KMA village forecasts.
Only compare matching timestamps; do not invent quantitative values from categorical PCP.
"""
import pandas as pd
from nationwide_weather import parse_rain

def align_hourly_forecasts(ai, official):
    columns=['예측 시각','AI 강수확률(%)','기상청 강수확률(%)','AI 1시간 강수량(mm)','기상청 강수량 원문','기상청 정량 강수량(mm)']
    if ai is None or ai.empty or official is None or official.empty:
        return pd.DataFrame(columns=columns)
    official=official.copy()
    official.index=pd.to_datetime(official.index, errors='coerce')
    rows=[]
    for _,a in ai.iterrows():
        ts=pd.to_datetime(a.get('예측 대상 시각'),errors='coerce')
        if pd.isna(ts) or ts not in official.index:continue
        o=official.loc[ts]
        if isinstance(o,pd.DataFrame):o=o.iloc[0]
        pop=pd.to_numeric(o.get('POP'),errors='coerce')
        raw=o.get('PCP')
        rows.append({'예측 시각':ts.strftime('%Y-%m-%d %H:%M'),
            'AI 강수확률(%)':a.get('강수 발생 확률(%)'),
            '기상청 강수확률(%)':pop,
            'AI 1시간 강수량(mm)':a.get('예상 1시간 강수량(mm)'),
            '기상청 강수량 원문':str(raw) if pd.notna(raw) else '자료 없음',
            '기상청 정량 강수량(mm)':parse_rain(raw)})
    return pd.DataFrame(rows,columns=columns)
