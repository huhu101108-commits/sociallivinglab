"""칠성동 AI 폭염 안전지킴이: official forecast + validated research ML."""
from pathlib import Path
import json
import joblib
import requests
from datetime import date, timedelta
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation
from future7 import predict_seven_days, predict_minimum_seven_days
from shelters import parse_shelters, distances_km
from facilities import parse_facilities, CATEGORIES
from safetydata_api import fetch_shelters, normalize_shelters
from cold_shelter_api import fetch_cold_shelters, normalize_cold_shelters
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from model import parse_csv, normalize, evaluate, predict_next, direct_features
from kma_api import fetch_asos
from forecast import fetch_forecast
from nationwide_weather import REGIONS, nationwide_forecast, rain_summary, climate_rain_analysis
from climate_risks import weather_risks, historical_baseline, historical_extremes

st.set_page_config(page_title='칠성동 AI 기후 안전지킴이',page_icon='🌦️',layout='wide')
st.title('🌦️ 칠성동 AI 기후 안전지킴이')
st.caption('기존 칠성동 기온 AI · 전국 지역별 강수예보 · 장기 강수 패턴 비교 · 전국 무더위/한파쉼터')
st.warning('대구 ASOS 관측소 자료와 기상청 예보 격자는 칠성동 개별 주소의 실측값이 아닙니다. 공식 특보·의료 판단을 대체하지 않습니다.')

def secret(name):
    try: return str(st.secrets.get(name,'')).strip()
    except Exception: return ''

@st.cache_data(ttl=3600,show_spinner='기상청 공식 예보를 조회하고 있습니다…')
def cached_forecast(key): return fetch_forecast(key)

@st.cache_data(show_spinner='기상 관측자료를 정리하고 있습니다…')
def cached_csv(content): return normalize([parse_csv(content)])

@st.cache_resource(show_spinner='AI 모델을 준비하고 있습니다…')
def cached_model(df, model_version):
    result = evaluate(df) if not model_version else {'data': df}
    if model_version:
        stored = joblib.load(Path(__file__).parent/'artifacts'/'direct_model.joblib')
        result['direct_model'] = stored['model']
        result['direct_columns'] = stored['columns']
        result['metrics'] = pd.DataFrame()
        result['evaluation'] = pd.DataFrame()
        result['selected'] = '자동 검증 후 배포된 AI 모델'
    return result

@st.cache_data(ttl=86400,show_spinner='기상청 ASOS 20년 관측자료를 수집하고 있습니다…')
def cached_asos(key): return fetch_asos(key,start='2006-01-01',end='2025-12-31')

@st.cache_data(ttl=21600,show_spinner='올해 최신 ASOS 관측자료를 조회하고 있습니다…')
def cached_recent_asos(key,year,end):
    return fetch_asos(key,start=f'{year}-01-01',end=end)

@st.cache_data(show_spinner='과거 20년 AI 비교 그래프를 계산하고 있습니다…')
def historical_model_comparison(data, model_path, model_mtime):
    # Historical predictions from the deployed model are IN-SAMPLE fits, not past forecasts.
    stored=joblib.load(model_path)
    x=direct_features(data, 1)
    cols=stored['columns']
    valid=x[cols].notna().all(axis=1) & data['tmax'].shift(-1).notna()
    if not valid.any():
        return pd.DataFrame(columns=['날짜','실제 최고기온','AI 과거 재현값'])
    pred=stored['model'].predict(x.loc[valid,cols])
    dates=x.index[valid]+pd.Timedelta(days=1)
    actual=data['tmax'].shift(-1).loc[valid].to_numpy()
    return pd.DataFrame({'날짜':dates,'실제 최고기온':actual,'AI 과거 재현값':pred})

def season_focus(forecast):
    """Prioritize the actual forecast temperature, falling back to calendar season."""
    month=pd.Timestamp.now(tz='Asia/Seoul').month
    fallback=('저온', '최저기온', '한파·결빙·강풍·강수') if month in (11,12,1,2) else (('고온','최고기온','폭염·이상고온·강수') if month in (6,7,8) else ('변동','최고·최저기온','큰 일교차·이상기온·강수·강풍'))
    if forecast is None or forecast.empty: return fallback
    now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None)
    f=forecast.loc[forecast.index>=now.floor('h')]
    if f.empty: return fallback
    values=[]
    for col in ('TMP','TMX','TMN'):
        if col in f: values.extend(pd.to_numeric(f[col],errors='coerce').dropna().tolist())
    if not values: return fallback
    low,high=min(values),max(values)
    if low<=5: return ('저온','최저기온','한파·결빙·이상저온·강풍')
    if high>=28: return ('고온','최고기온','폭염·이상고온·건조')
    return ('변동','최고·최저기온','이상기온·일교차·강수·강풍')

tab_weather, tab_temp, tab_safety = st.tabs(['🌧️ 날씨 예측 · 전국', '🌡️ 기온 예측 · 기존 AI', '🗺️ 안전시설 · 행동요령'])
with tab_weather:
    st.header('전국 지역별 강수·집중호우 분석')
    st.caption('기상청 단기예보의 전국 격자별 강수·습도·바람을 분석합니다. AI 강수 모델은 아직 학습·검증되지 않아 예측값으로 표시하지 않습니다.')
    selected_region=st.selectbox('전국 예측 지역',list(REGIONS),index=list(REGIONS).index('대구'),key='national_region')
    national_key=secret('KMA_FORECAST_KEY')
    if national_key:
        try:
            national, issued_n, grid_n=nationwide_forecast(national_key,selected_region)
            st.caption(f'{selected_region} 대표 격자 {grid_n} · 기상청 발표 {issued_n:%Y-%m-%d %H:%M} · 관측소 실측이 아닌 격자 예보')
            result=rain_summary(national)
            if not result.empty:
                c1,c2,c3=st.columns(3)
                c1.metric('예보 강수확률 최대',f"{result['강수확률(%)'].max():.0f}%" if result['강수확률(%)'].notna().any() else '정보 없음')
                c2.metric('예보 시간대 최대 강수량',f"{result['강수량(mm)'].max():.1f}mm" if result['강수량(mm)'].notna().any() else '정량 자료 없음')
                c3.metric('정량 강수 예보 시간대',str(result['강수량(mm)'].notna().sum()))
                st.line_chart(result.set_index('시각')[['강수확률(%)','상대습도(%)']])
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.caption('강수량이 범위형(예: 1mm 미만)이면 정량 강수량에 합산하지 않습니다. 단기예보 제공 기간 이후에는 값을 만들지 않습니다.')
                st.download_button('지역별 강수예보 CSV 다운로드',result.to_csv(index=False).encode('utf-8-sig'),file_name=f'rain_{selected_region}.csv',mime='text/csv')
            else: st.info('해당 지역의 현재 이후 강수예보 자료가 없습니다.')
        except Exception as exc: st.warning(f'전국 날씨 예보 조회 실패: {type(exc).__name__}: {exc}')
    else: st.info('기존 KMA_FORECAST_KEY를 Streamlit Secrets에 설정하면 전국 예보가 표시됩니다.')
    st.subheader('장마·단시간 강수 패턴 비교')
    st.caption('동남아 우기가 국내에 그대로 나타난다는 가정 대신, 실제 관측자료의 여름철 강수 집중도와 극한강수 일수를 비교합니다.')
    rain_file=Path(__file__).parent/'data'/'asos_143.csv'
    if rain_file.exists():
        try:
            comparison=climate_rain_analysis(rain_file)
            if comparison.empty: st.info('기존 관측 CSV에 강수량 열이 없어 강수 추세를 계산할 수 없습니다.')
            else:
                st.dataframe(comparison,use_container_width=True,hide_index=True)
                st.caption('대구 관측소 기준 과거 비교이며, 전국의 장기 기후 추세를 대표하지 않습니다. 장기 추세의 유의성 검증 전에는 증가·감소를 단정하지 않습니다.')
        except Exception as exc: st.warning(f'기후 비교 자료 처리 오류: {exc}')
    else: st.info('관측 CSV가 준비되면 강수 패턴 비교가 표시됩니다.')
    st.divider()
    st.subheader('기존 칠성동 공식 단기예보·위험 감지')

    st.header('1. 기상청 공식 단기예보와 기상 위험 감지')
    key=secret('KMA_FORECAST_KEY')
    forecast=None
    if key:
        try:
            forecast,issued,grid_xy=cached_forecast(key)
            st.caption(f'발표: {issued:%Y-%m-%d %H:%M} KST · 예보 격자 {grid_xy[0]}, {grid_xy[1]} · 최대 1시간 캐시')
            now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None)
            future=forecast[forecast.index>=now.floor('h')]
            if future.empty: future=forecast
            focus, primary, categories=season_focus(forecast)
            st.info(f'**현재 중점: {primary} · {categories}** — 계절과 실제 예보 기온을 반영해 자동 선택했습니다.')
            tmp=pd.to_numeric(future['TMP'],errors='coerce').dropna() if 'TMP' in future else pd.Series(dtype=float)
            tmx=pd.to_numeric(future['TMX'],errors='coerce').dropna() if 'TMX' in future else pd.Series(dtype=float)
            tmn=pd.to_numeric(future['TMN'],errors='coerce').dropna() if 'TMN' in future else pd.Series(dtype=float)
            hum=pd.to_numeric(future['REH'],errors='coerce').dropna() if 'REH' in future else pd.Series(dtype=float)
            high=max([v.max() for v in (tmp,tmx) if len(v)],default=float('nan'))
            low=min([v.min() for v in (tmp,tmn) if len(v)],default=float('nan'))
            a,b=st.columns(2)
            if focus=='저온':
                a.metric('예보 최저기온',f'{low:.1f}°C' if pd.notna(low) else '자료 없음')
                b.metric('예보 최고기온',f'{high:.1f}°C' if pd.notna(high) else '자료 없음')
            elif focus=='고온':
                a.metric('예보 최고기온',f'{high:.1f}°C' if pd.notna(high) else '자료 없음')
                b.metric('예보 최저기온',f'{low:.1f}°C' if pd.notna(low) else '자료 없음')
            else:
                a.metric('예보 최저기온',f'{low:.1f}°C' if pd.notna(low) else '자료 없음')
                b.metric('예보 최고기온',f'{high:.1f}°C' if pd.notna(high) else '자료 없음')
            if len(tmp):
                with st.expander('시간대별 예보 기온 보기'):
                    st.line_chart(tmp.rename('예보 기온(°C)'))
        except Exception:
            st.error('공식 예보를 불러오지 못했습니다. API 승인·인증 설정 또는 네트워크를 확인하세요. 인증키는 표시하지 않습니다.')
    else:
        st.info('운영자가 Streamlit Secrets에 KMA_FORECAST_KEY를 등록하면 공식 예보가 자동 표시됩니다.')
    st.subheader('🌦️ 지금 주의할 기상조건')
    alerts, available_categories = weather_risks(forecast)
    if forecast is None:
        st.info('기상청 예보가 연결되면 현재 조건에 맞는 위험을 표시합니다.')
    else:
        if alerts:
            rank={'높음':0,'주의':1,'참고':2}
            for name,level,description in sorted(alerts,key=lambda item:rank.get(item[1],3))[:3]:
                if level=='높음':st.error(f'**{name}** · {description}')
                elif level=='주의':st.warning(f'**{name}** · {description}')
                else:st.info(f'**{name}** · {description}')
            if len(alerts)>3:
                with st.expander(f'기타 감지 항목 {len(alerts)-3}개'):
                    for name,level,description in alerts[3:]:st.write(f'**{name}** · {description}')
        else:
            st.success('제공된 예보 항목에서 설정 기준을 넘는 기상조건이 없습니다.')
    st.caption('이 결과는 기온·풍속·습도 등의 간단한 기준에 따른 참고 안내이며, 공식 폭염·한파·강풍·호우·건조특보가 아닙니다.')
    st.markdown('[기상청 공식 예보·특보 확인](https://www.weather.go.kr/)')
    
    with st.expander('📅 향후 날짜별 다양한 기상 예측 보기',expanded=False):
        if forecast is not None and not forecast.empty:
            f=forecast.copy()
            f=f[f.index>=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).floor('h')]
            if not f.empty:
                daily=pd.DataFrame(index=f.resample('D').size().index)
                for col,label,agg in [('TMP','기온 최고(°C)','max'),('TMN','최저기온(°C)','min'),
                                      ('TMX','최고기온(°C)','max'),('POP','강수확률 최고(%)','max'),
                                      ('WSD','풍속 최고(m/s)','max'),('REH','습도 최저(%)','min')]:
                    if col in f:
                        vals=pd.to_numeric(f[col],errors='coerce')
                        daily[label]=getattr(vals.resample('D'),agg)()
                if 'PCP' in f:
                    raw=f['PCP'].astype(str).str.extract(r'(\\d+(?:\\.\\d+)?)\\s*mm',expand=False)
                    daily['시간대 강수량 표기 최대(mm)']=pd.to_numeric(raw,errors='coerce').resample('D').max()
                if 'SNO' in f:
                    daily['적설 예보 항목']=f['SNO'].astype(str).resample('D').apply(
                        lambda x:'표기 있음' if x.str.contains(r'\\d+(?:\\.\\d+)?\\s*cm',regex=True).any() else '표기 없음')
                daily.index=daily.index.strftime('%m-%d')
                st.dataframe(daily,use_container_width=True)
                st.caption('기상청 공식 예보 항목만 표시합니다. 강수량 표기는 시간대 최대값이며 일 누적 강수량이 아닙니다.')
            else: st.info('현재 이후의 예보가 없습니다.')
        else: st.info('공식 예보가 연결되면 강수·바람·습도·기온을 날짜별로 볼 수 있습니다.')
with tab_temp:
    st.header('2. AI 7일 기온 전망 · 계절별 이상기온')
    st.caption('과거 기상자료로 학습한 AI가 향후 7일의 기온을 추정합니다. 공식 기상청 예보와는 별개입니다.')
    local=Path(__file__).parent/'data'/'asos_143.csv'
    data=None
    source=None
    if local.exists():
        try:
            data=cached_csv(local.read_bytes()); source='저장소 ASOS CSV'
        except Exception:
            st.warning('저장된 ASOS CSV를 읽지 못했습니다.')
    asos_key=secret('ASOS_SERVICE_KEY')
    if data is None and asos_key:
        try:
            data=cached_asos(asos_key);source='공공데이터포털 ASOS API'
        except Exception:
            st.warning('과거 ASOS 자료 수집에 실패했습니다. 인증키·API 상태를 확인하세요.')
    if data is not None and asos_key:
        try:
            today_kst=pd.Timestamp.now(tz='Asia/Seoul').date()
            yesterday=today_kst-timedelta(days=1)
            # Read each year since the historical dataset ended. Avoid a 2025->2026 gap.
            last_year=data.index.max().year
            recent=[]
            for year in range(last_year+1,yesterday.year+1):
                if year<=yesterday.year:
                    finish=min(yesterday,date(year,12,31))
                    recent.append(cached_recent_asos(asos_key,year,finish.isoformat()).loc[f'{year}-01-01':finish.isoformat()])
            # Refresh the current year even when it was present in the historical CSV.
            if yesterday.year<=last_year:
                recent.append(cached_recent_asos(asos_key,yesterday.year,yesterday.isoformat()).loc[f'{yesterday.year}-01-01':yesterday.isoformat()])
            if recent:
                combined=pd.concat([data]+recent).sort_index()
                data=combined[~combined.index.duplicated(keep='last')].asfreq('D')
                source+=' + 최신 관측자료'
        except Exception:
            st.warning('최신 ASOS 관측자료를 갱신하지 못했습니다. 최신 7일 예측은 제한될 수 있습니다.')
    if data is None:
        st.error('AI 예측용 ASOS 자료가 없습니다. ASOS_SERVICE_KEY 또는 data/asos_143.csv를 확인하세요.')
    else:
        st.caption(f'관측자료: {source} · {data.index.min().date()} ~ {data.index.max().date()}')
        try:
            manifest_path=Path(__file__).parent/'artifacts'/'manifest.json'
            manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
            result=cached_model(data, manifest.get('trained_at') if manifest else None)
            if manifest:
                st.caption(f"AI 모델 갱신: {manifest['trained_at']} · 검증 평균오차 {manifest['validation_mae']:.2f}°C")
            st.subheader('AI 7일 기온 예측')
            try:
                week=predict_seven_days(result)
                st.caption(f'최종 최고기온 실측 날짜: {week.iloc[0]["관측 기반 마지막 날짜"]} · 직접 다중시점 AI 예측 · 단위: °C')
                min_file=Path(__file__).parent/'artifacts'/'min_model.joblib'
                min_meta=Path(__file__).parent/'artifacts'/'min_manifest.json'
                min_ai=False
                min_error=None
                if min_file.exists() and min_meta.exists():
                    try:
                        # Old repository CSVs may contain only maximum temperatures.
                        # Refresh the recent observations rather than silently hiding minimum AI.
                        min_data=data
                        last_min=data['tmin'].last_valid_index() if 'tmin' in data else None
                        yesterday=pd.Timestamp.now(tz='Asia/Seoul').normalize().tz_localize(None)-pd.Timedelta(days=1)
                        if (last_min is None or (yesterday-last_min).days>4) and asos_key:
                            try:
                                fresh=fetch_asos(asos_key,start=(yesterday-pd.Timedelta(days=45)).date().isoformat(),end=yesterday.date().isoformat())
                                min_data=data.copy()
                                for col in ('tmin','tmax','humidity','rain'):
                                    if col in fresh and col in min_data:
                                        common=fresh.index.intersection(min_data.index)
                                        min_data.loc[common,col]=fresh.loc[common,col].combine_first(min_data.loc[common,col])
                            except Exception as refresh_exc:
                                min_error=f'최근 최저기온 자료 재조회 실패: {type(refresh_exc).__name__}'
                        minimum=predict_minimum_seven_days(min_data,min_file)
                        week=week.merge(minimum,on='날짜',how='left',validate='one_to_one')
                        min_ai=True
                    except Exception as exc:
                        min_error=f'{type(exc).__name__}: {exc}'
                focus=season_focus(forecast)[0]
                high_col='AI 예상 최고기온(°C)'
                low_col='AI 예상 최저기온(°C)'
                if min_ai:
                    st.success('최고기온·최저기온 AI 모델 모두 연결됨')
                    view_cols=[low_col,high_col] if focus=='저온' else [high_col,low_col]
                    st.plotly_chart(px.line(week,x='날짜',y=view_cols,markers=True,
                        title='AI 7일 최고·최저기온 예측'),use_container_width=True,key='ai_both_temp_plot')
                    st.dataframe(week[['날짜']+view_cols].style.format({c:'{:.1f}' for c in view_cols}),
                        hide_index=True,use_container_width=True)
                else:
                    st.warning('최저기온 AI 예측을 표시하지 못했습니다. 아래에서 원인을 확인하세요.')
                    with st.expander('최저기온 AI가 표시되지 않는 이유'):
                        if not (min_file.exists() and min_meta.exists()):
                            st.write('GitHub의 artifacts/min_model.joblib 및 min_manifest.json 파일이 없습니다. Actions 자동 학습 실행 기록을 확인하세요.')
                        else:
                            st.write('최저기온 모델 파일은 확인됐지만 예측 계산이 완료되지 않았습니다.')
                            if min_error: st.error(f'상세 원인: {min_error}')
                            last_min=data['tmin'].last_valid_index() if 'tmin' in data else None
                            st.write(f'저장된 최저기온 마지막 관측일: {last_min.date() if last_min is not None else "없음"}')
                        count=int(data['tmin'].notna().sum()) if 'tmin' in data else 0
                        st.write(f'최저기온 관측자료가 있는 날짜: {count:,}일 (학습 최소 조건: 약 2,190일)')
                        st.caption('최저기온 AI가 준비될 때까지 기상청 공식 최저기온 예보를 확인하세요.')
                    st.plotly_chart(px.line(week,x='날짜',y=high_col,markers=True,
                        title='AI 7일 최고기온 예측'),use_container_width=True,key='ai_high_only_plot')
                    st.dataframe(week[['날짜',high_col]].style.format({high_col:'{:.1f}'}),
                        hide_index=True,use_container_width=True)
                if min_ai:
                    with st.expander('AI 모델 검증 성능'):
                        meta=json.loads(min_meta.read_text(encoding='utf-8'))
                        st.write(f'최저기온 모델: 2024~2025 검증 MAE {meta["validation_mae"]:.2f}°C')
                st.download_button('AI 7일 예측 CSV 내려받기',week.to_csv(index=False).encode('utf-8-sig'),'ai_7day.csv','text/csv')
                st.caption('예측일이 멀수록 불확실성이 커질 수 있습니다. 공식 기상특보를 우선 확인하세요.')
    
            except ValueError as exc:
                st.error(str(exc))
            if not result['metrics'].empty:
                st.download_button('AI 검증 지표 CSV 내려받기',result['metrics'].to_csv(index=False).encode('utf-8-sig'),'ai_validation.csv','text/csv')
        except Exception as exc:
            st.error(f'AI 학습/검증 오류: {type(exc).__name__}. 관측자료 누락 및 model.py 버전을 확인하세요.')
    
    with st.expander('📊 20년 기후 분석 · 과거 AI 비교 · 상세 검증', expanded=False):
                observed=data[['tmax']].dropna().reset_index()
                observed.columns=['날짜','실제 최고기온']
                observed['날짜']=pd.to_datetime(observed['날짜'])
                st.caption('2006년 이후 대구 ASOS 143 관측소의 실제 일 최고기온입니다. 날짜 범위를 선택해 확대할 수 있습니다.')
                years=sorted(observed['날짜'].dt.year.unique().tolist())
                chosen_years=st.slider('관측 연도 범위',min_value=years[0],max_value=years[-1],value=(years[0],years[-1]),key='historical_years')
                obs_view=observed[observed['날짜'].dt.year.between(*chosen_years)]
                st.plotly_chart(px.line(obs_view,x='날짜',y='실제 최고기온',title='2006년 이후 실제 최고기온 변화'),use_container_width=True)
                model_file=Path(__file__).parent/'artifacts'/'direct_model.joblib'
                if model_file.exists() and manifest:
                    try:
                        history_fit=historical_model_comparison(data,model_file,model_file.stat().st_mtime_ns)
                        history_fit=history_fit[history_fit['날짜'].dt.year.between(*chosen_years)]
                        if not history_fit.empty:
                            st.plotly_chart(px.line(history_fit,x='날짜',y=['실제 최고기온','AI 과거 재현값'],title='과거 실제 기온 vs AI 재현값 (전체 기간)'),use_container_width=True)
                            st.caption('주의: AI 과거 재현값은 과거 전체 자료로 학습한 현재 모델을 과거 날짜에 다시 적용한 결과입니다. 당시 실제로 발표한 미래 예측이 아니며, 미래 예측 정확도 증거로 사용할 수 없습니다. 실제 미래 예측 정확도는 아래 날짜별 기록에서 확인하세요.')
                    except Exception as graph_exc:
                        st.warning(f'과거 AI 비교 그래프를 표시하지 못했습니다: {type(graph_exc).__name__}')
                else:
                    st.info('저장된 자동 학습 모델이 아직 없어 과거 AI 재현 그래프를 표시할 수 없습니다. 실제 관측 그래프는 계속 볼 수 있습니다.')
                st.subheader('🌡️ 계절별 이상기온 감지 · 과거 관측값 비교')
                st.caption('같은 시기(±7일)의 이전 연도 관측자료 분포와 비교합니다. 30년 기후평년값이나 공식 이상기후 판정이 아닙니다.')
                recent_days=data[['tmax']].dropna().tail(30)
                baseline=historical_baseline(data,recent_days.index)
                if not baseline.empty:
                    baseline['실제 최고기온']=recent_days.tmax.to_numpy()
                    baseline['편차']=baseline['실제 최고기온']-baseline['과거 중앙값']
                    baseline['분류']=np.where(baseline['실제 최고기온']>baseline['상위 90% 기준'],'과거 대비 높은 기온',np.where(baseline['실제 최고기온']<baseline['하위 10% 기준'],'과거 대비 낮은 기온','일반 범위'))
                    st.plotly_chart(px.line(baseline,x='날짜',y=['실제 최고기온','과거 중앙값','상위 90% 기준','하위 10% 기준'],title='최근 30개 관측일의 최고기온과 과거 같은 시기 비교'),use_container_width=True)
                    st.dataframe(baseline[['날짜','실제 최고기온','과거 중앙값','편차','분류']].tail(12).sort_values('날짜',ascending=False),hide_index=True,use_container_width=True)
                st.subheader('📊 연도별 극한기온 발생 일수')
                extremes=historical_extremes(data)
                counts=['33°C 이상 일수','30°C 이상 일수']
                if '최저기온 관측일수' in extremes and extremes['최저기온 관측일수'].sum()>0:
                    counts+=['0°C 미만 일수','-12°C 이하 일수']
                st.plotly_chart(px.line(extremes,x='연도',y=counts,markers=True,title='연도별 기온 기준 초과 일수'),use_container_width=True)
                st.caption('관측일수가 부족한 해의 발생 일수는 실제보다 작을 수 있습니다. 최저기온 자료가 없는 경우 추위 관련 지표는 생략합니다.')
                st.subheader('🔮 AI 7일 최고기온의 과거 대비 이상고온·이상저온 가능성')
                st.caption('AI 최고기온 예측을 과거 같은 시기의 기온 분포와 비교한 참고 분석입니다. 별도의 이상기후 예측 AI는 아닙니다.')
                try:
                    risk_week=predict_seven_days(result)
                    future_baseline=historical_baseline(data,pd.to_datetime(risk_week['날짜']))
                    if not future_baseline.empty:
                        future_baseline['AI 최고기온 예측']=pd.to_numeric(risk_week['AI 예상 최고기온(°C)'],errors='coerce').to_numpy()
                        future_baseline['비교 결과']=np.where(future_baseline['AI 최고기온 예측']>future_baseline['상위 90% 기준'],'과거 대비 고온 가능',np.where(future_baseline['AI 최고기온 예측']<future_baseline['하위 10% 기준'],'과거 대비 저온 가능','일반 범위'))
                        st.dataframe(future_baseline[['날짜','AI 최고기온 예측','과거 중앙값','비교 결과']],hide_index=True,use_container_width=True)
                except (ValueError, KeyError, TypeError) as exc:
                    st.info(f'AI 이상기온 비교 준비 중: {type(exc).__name__}')
                st.subheader('AI 모델 검증 성능')
                st.caption('자동 학습 모델의 MAE는 2024~2025년 자료에 대한 1~11일 직접 예측 통합 검증값입니다. 개별 예측일의 정확도나 미래 성능을 보장하지 않습니다. 기존 모델 모드에서는 아래에 1일 예측 검증값이 표시됩니다.')
                if not result['metrics'].empty:
                    st.dataframe(result['metrics'],hide_index=True,use_container_width=True)
                ev=result['evaluation'];selected=result['selected']
                if not ev.empty:
                    view=ev[ev['model']==selected].copy()
                    st.plotly_chart(px.line(view,x='date',y=['actual','pred'],title=f'{selected} 시간순 검증: 실제값 vs 예측값 (전체 검증기간)'),use_container_width=True)
    
    st.subheader('📅 최고·최저기온 AI 예측 정확도')
    st.caption('매일 실제로 기록한 예측만 나중에 발표된 ASOS 관측값과 비교합니다. 과거 학습값은 정확도 통계에 포함하지 않습니다.')
    log_path=Path(__file__).parent/'data'/'prediction_history.csv'
    if log_path.exists():
        try:
            history=pd.read_csv(log_path,encoding='utf-8-sig')
            if history.empty or not {'issued_at','target_date','horizon'}.issubset(history.columns):
                st.info('예측 기록이 아직 없습니다.')
            else:
                history['target_date']=pd.to_datetime(history['target_date'],errors='coerce')
                history['issued_at']=pd.to_datetime(history['issued_at'],errors='coerce',utc=True)
                for col in ['horizon','predicted_tmax','actual_tmax','predicted_tmin','actual_tmin']:
                    if col not in history:history[col]=float('nan')
                    history[col]=pd.to_numeric(history[col],errors='coerce')
                history=history.dropna(subset=['target_date','issued_at','horizon'])
                history['horizon']=history['horizon'].astype(int)
                horizon=st.selectbox('예측 시점',list(range(1,8)),format_func=lambda n:f'{n}일 전 예측',key='accuracy_horizon')
                kind=st.segmented_control('비교할 기온',options=['최고기온','최저기온'],default='최고기온',key='accuracy_kind')
                pred_col,actual_col=('predicted_tmax','actual_tmax') if kind=='최고기온' else ('predicted_tmin','actual_tmin')
                selected=history[history.horizon.eq(horizon)].sort_values(['target_date','issued_at']).drop_duplicates('target_date',keep='first').sort_values('target_date').copy()
                selected['오차']=abs(selected[pred_col]-selected[actual_col])
                scored=selected.dropna(subset=[pred_col,actual_col])
                a,b,c=st.columns(3)
                a.metric('검증 완료',f'{len(scored)}일')
                b.metric('평균 절대오차',f"{scored['오차'].mean():.2f}°C" if len(scored) else '기록 대기')
                c.metric('예측 기록',f'{selected[pred_col].notna().sum()}일')
                if len(scored):
                    view=scored.set_index('target_date')[[pred_col,actual_col]]
                    view.columns=['AI 예측','실제 관측']
                    st.line_chart(view,y_label='기온 (°C)')
                    st.line_chart(scored.set_index('target_date')['오차'],y_label='절대오차 (°C)')
                else:
                    st.info('아직 해당 기온의 예측과 실제 관측이 모두 기록된 날짜가 없습니다. 최저기온은 업데이트 후 새로 발행한 예측부터 누적됩니다.')
                table=selected[['target_date',pred_col,actual_col,'오차']].copy()
                table.columns=['날짜','AI 예측(°C)','실제 관측(°C)','절대오차(°C)']
                table['날짜']=table['날짜'].dt.strftime('%Y-%m-%d')
                st.dataframe(table.sort_values('날짜',ascending=False),hide_index=True,use_container_width=True)
                st.download_button('선택한 예측 정확도 CSV',table.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'accuracy_{kind}_{horizon}day.csv',mime='text/csv',key='accuracy_download')
                with st.expander('최고·최저기온 전체 오차 요약'):
                    for label,pr,ac in [('최고기온','predicted_tmax','actual_tmax'),('최저기온','predicted_tmin','actual_tmin')]:
                        valid=history.dropna(subset=[pr,ac])
                        st.write(f"{label}: {len(valid)}건 · MAE {abs(valid[pr]-valid[ac]).mean():.2f}°C" if len(valid) else f'{label}: 검증 기록 대기')
        except Exception as exc:
            st.warning(f'예측 기록 표시 오류: {type(exc).__name__}: {exc}')
    else:
        st.info('아직 예측 기록이 없습니다. GitHub Actions 자동 학습 후 기록됩니다.')
with tab_safety:
    st.header('3. 계절 맞춤 행동요령')
    focus,_,_=season_focus(forecast)
    if focus=='저온':
        st.info('추위·결빙에 대비해 방한용품을 준비하고, 빙판길 보행에 주의하세요. 공식 한파특보를 확인하세요.')
    elif focus=='고온':
        st.info('더운 시간대 야외활동을 줄이고, 수분 섭취와 시원한 장소에서의 휴식을 챙기세요. 공식 폭염특보를 확인하세요.')
    else:
        st.info('일교차에 맞는 옷차림을 준비하고, 강수·강풍 예보와 기상청 특보를 확인하세요.')
    st.caption('계절 맞춤 안내는 기상조건에 따른 참고 정보이며 공식 기상특보가 아닙니다.')
@st.cache_data(ttl=86400,show_spinner="행정안전부 전국 무더위쉼터 API를 조회하고 있습니다…")
def cached_safety_shelters(key):
    return normalize_shelters(fetch_shelters(key))

@st.cache_data(ttl=86400,show_spinner='행정안전부 전국 한파쉼터 API를 조회하고 있습니다…')
def cached_cold_shelters(key):
    return normalize_cold_shelters(fetch_cold_shelters(key))

with tab_safety:
    st.header('4. 전국 기상 안전시설 지도')
    st.caption('쉼터 API 처리 코드 v7 · 무더위쉼터 LA/LO · 한파쉼터 LAT/LOT')
    st.caption('무더위쉼터·한파쉼터: 행정안전부 공식 API · 24시간 캐시. 두 API는 별도 인증키가 필요합니다. 실제 운영 여부는 방문 전 확인하세요.')
    if 'geo_enabled' not in st.session_state: st.session_state.geo_enabled=False
    if 'geo_coords' not in st.session_state: st.session_state.geo_coords=None
    c1,c2=st.columns(2)
    with c1:
        if st.button('📍 내 위치 표시',use_container_width=True,key='geo_start'): st.session_state.geo_enabled=True
    with c2:
        if st.button('위치 표시 해제',use_container_width=True,key='geo_stop'):
            st.session_state.geo_enabled=False
            st.session_state.geo_coords=None
    if st.session_state.geo_enabled:
        try:
            loc=streamlit_geolocation()
            if isinstance(loc,dict) and loc.get('latitude') is not None and loc.get('longitude') is not None:
                st.session_state.geo_coords=(float(loc['latitude']),float(loc['longitude']))
        except Exception: st.info('브라우저 위치 권한을 확인하세요.')
    user_lat=user_lon=None
    if st.session_state.geo_enabled and st.session_state.geo_coords:
        user_lat,user_lon=st.session_state.geo_coords
    
    datasets=[]
    safety_key=secret('SAFETYDATA_SERVICE_KEY')
    if safety_key:
        try:
            shelters_api=cached_safety_shelters(safety_key)
            datasets.append(shelters_api)
            st.success(f'행정안전부 API 무더위쉼터 {len(shelters_api):,}곳 조회 완료 (최대 24시간 캐시)')
        except (ValueError, requests.RequestException) as exc:
            st.warning(f'무더위쉼터 API 연결 실패: {exc}')
    else:
        st.info('전국 무더위쉼터를 표시하려면 Streamlit Secrets에 SAFETYDATA_SERVICE_KEY를 설정하세요.')
    cold_key=secret('SAFETYDATA_COLD_SERVICE_KEY')
    if cold_key:
        try:
            cold_api=cached_cold_shelters(cold_key)
            datasets.append(cold_api)
            st.success(f'행정안전부 API 한파쉼터 {len(cold_api):,}곳 조회 완료 (최대 24시간 캐시)')
        except (ValueError, requests.RequestException) as exc:
            st.warning(f'한파쉼터 API 연결 실패: {exc}')
    else:
        st.info('전국 한파쉼터 API를 이용하려면 별도 이용신청 후 Streamlit Secrets에 SAFETYDATA_COLD_SERVICE_KEY를 설정하세요.')
    base=Path(__file__).parent/'data'
    for filename,kind in [('evacuation.csv','민방위 대피시설'),('hospitals.csv','응급의료기관'),('safety_facilities.csv','기타 안전시설')]:
        file=base/filename
        if file.exists():
            try: datasets.append(parse_facilities(file.read_bytes(),kind))
            except ValueError as exc: st.warning(f'{filename}: {exc}')
    with st.expander('공식 전국 시설 CSV 추가 (운영자용)',expanded=False):
        kind=st.selectbox('자료 유형',[c for c in CATEGORIES if c not in ('무더위쉼터','한파쉼터')],key='facility_kind')
        uploaded=st.file_uploader('시설명·위도·경도가 있는 공식 CSV',type=['csv'],key='facility_upload')
        if uploaded:
            try: datasets.append(parse_facilities(uploaded.getvalue(),kind))
            except ValueError as exc: st.warning(str(exc))
        st.caption('전국 데이터는 공공데이터포털 또는 지자체 공식 CSV에서 확보하세요. 시설명·위도·경도 열이 필요하며, 업로드는 현재 세션에만 적용됩니다.')
    if datasets:
        facilities=pd.concat(datasets,ignore_index=True).drop_duplicates(['이름','lat','lon','종류'])
        kinds=sorted(facilities['종류'].dropna().unique().tolist())
        selected_kinds=st.multiselect('지도에 표시할 시설',kinds,default=kinds,key='map_facility_types')
        facilities=facilities[facilities['종류'].isin(selected_kinds)].copy()
    else:
        facilities=pd.DataFrame(columns=['이름','주소','종류','lat','lon'])
        st.info('현재 등록된 시설 데이터가 없습니다. 전국 공식 시설 CSV를 추가하면 지도에 표시됩니다.')
    center_lat,center_lon=(user_lat,user_lon) if user_lat is not None else (35.879,128.602)
    if len(facilities):
        if user_lat is not None:
            facilities['직선거리(km)']=distances_km(facilities,user_lat,user_lon)
            nearby=facilities.nsmallest(20,'직선거리(km)')
            st.dataframe(nearby[['종류','이름','주소','직선거리(km)']],hide_index=True,use_container_width=True)
        else:
            st.caption(f'표시 시설: {len(facilities):,}곳 · 지도 확대/이동으로 확인하세요.')
    colors={'무더위쉼터':[234,100,65,205],'한파쉼터':[42,123,224,205],
            '민방위 대피시설':[127,87,192,205],'응급의료기관':[221,62,87,205],
            '기타 안전시설':[63,153,116,205]}
    layers=[]
    for kind,color in colors.items():
        subset=facilities[facilities['종류']==kind]
        if not subset.empty:
            layers.append(pdk.Layer('ScatterplotLayer',subset,get_position='[lon,lat]',
              get_fill_color=color,get_radius=7,radius_min_pixels=4,radius_max_pixels=9,pickable=True))
    if user_lat is not None:
        layers.append(pdk.Layer('ScatterplotLayer',pd.DataFrame([{'lat':user_lat,'lon':user_lon,'이름':'내 위치','주소':'','종류':'내 위치'}]),
            get_position='[lon,lat]',get_fill_color=[20,85,250,235],get_radius=9,radius_min_pixels=7,pickable=True))
    st.pydeck_chart(pdk.Deck(map_style=None,initial_view_state=pdk.ViewState(
        latitude=center_lat,longitude=center_lon,zoom=11 if user_lat is not None else 6,pitch=0),
        layers=layers,tooltip={'text':'{종류}\\n{이름}\\n{주소}'}),use_container_width=True)
    st.caption('주황: 무더위쉼터 · 파랑: 한파쉼터 · 보라: 대피시설 · 빨강: 응급의료기관 · 초록: 기타 시설 · 진한 파랑: 내 위치')
    st.header('5. 사계절 위험기상 행동요령 (상세)')
    st.caption('일상에서 실천할 수 있는 폭염 대응 방법입니다. 실제 행동은 기상청 공식 특보와 현장 상황을 우선하세요.')
    with st.expander('☀️ 외출 전과 야외활동 중', expanded=True):
        st.markdown('''- 기상청 예보와 폭염특보를 확인하고, 무더운 시간대에는 야외활동을 가급적 줄이세요.
    - 가볍고 통풍이 잘되는 옷을 입고, 모자나 양산을 활용하세요.
    - 목이 마르지 않아도 물을 조금씩 자주 마시세요. 단, 수분 섭취를 제한하라는 의료진의 지시가 있다면 그 지시를 따르세요.
    - 어지럽거나 몸이 불편해지면 즉시 활동을 중단하고 시원한 곳에서 쉬세요.''')
    with st.expander('🏠 실내와 무더위쉼터 이용'):
        st.markdown('''- 냉방기기나 선풍기를 적절히 사용하고 실내 온도를 확인하세요.
    - 실내가 지나치게 덥다면 가까운 무더위쉼터 등 시원한 장소로 이동하세요.
    - 지도에 표시된 쉼터는 방문 전에 운영시간과 이용 가능 여부를 확인하세요.
    - 혼자 지내는 이웃이나 고령자의 안부를 확인하고, 필요하면 주변에 도움을 요청하세요.''')
    with st.expander('🚨 온열질환이 의심될 때'):
        st.markdown('''- 즉시 활동을 멈추고 시원한 곳으로 이동하여 몸을 식히세요.
    - 의식이 또렷하고 삼킬 수 있을 때만 물을 마시도록 하세요.
    - 의식 변화, 쓰러짐 등 심각한 증상이 있으면 즉시 **119**에 신고하세요.
    - 의식이 없거나 삼키기 어려운 사람에게는 억지로 물을 먹이지 마세요.''')
    st.markdown('[기상청 공식 폭염정보](https://www.weather.go.kr/) · [국민재난안전포털](https://www.safekorea.go.kr/)')
    with st.expander('❄️ 한파·결빙 대비'):
        st.markdown('- 외출 시 보온에 신경 쓰고 장시간 추위 노출을 피하세요.\n- 수도관 동파와 빙판길 위험을 확인하세요.\n- 한파특보와 지역 대피·쉼터 안내는 공식 채널에서 확인하세요.')
    with st.expander('🌧️ 호우·강풍·건조 대비'):
        st.markdown('- 호우 예보 시 하천변·지하차도 등 침수 위험 장소를 피하세요.\n- 강풍 예보 시 낙하물과 야외 시설물에 주의하세요.\n- 건조한 날에는 화재 예방 수칙을 지키세요.\n- 침수·산사태·산불의 실제 발생 여부는 이 앱의 기상자료만으로 예측할 수 없습니다.')
    st.caption('자료 출처: 기상청 공식 단기예보, 기상청 ASOS 일자료(연결 시). AI 성능 수치는 실제 자료로 검증된 경우에만 표시됩니다.')
