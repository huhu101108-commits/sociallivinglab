"""칠성동 AI 폭염 안전지킴이: official forecast + validated research ML."""
from pathlib import Path
import json
import joblib
from datetime import date, timedelta
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation
from future7 import predict_seven_days, predict_minimum_seven_days
from shelters import parse_shelters, distances_km
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from model import parse_csv, normalize, evaluate, predict_next, direct_features
from kma_api import fetch_asos
from forecast import fetch_forecast
from climate_risks import weather_risks, historical_baseline, historical_extremes

st.set_page_config(page_title='칠성동 AI 기후 안전지킴이',page_icon='🌦️',layout='wide')
st.title('🌦️ 칠성동 AI 기후 안전지킴이')
st.caption('대구 북구 칠성동 · 사계절 기상 위험 감지 · 2006년 이후 관측자료 분석 · AI 최고기온 예측')
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
            if min_file.exists() and min_meta.exists():
                try:
                    minimum=predict_minimum_seven_days(data,min_file)
                    week=week.merge(minimum,on='날짜',how='left')
                    min_ai=True
                except (ValueError,KeyError) as exc:
                    st.caption(f'AI 최저기온 예측 대기: {exc}')
            focus=season_focus(forecast)[0]
            primary='AI 예상 최저기온(°C)' if focus=='저온' and min_ai else 'AI 예상 최고기온(°C)'
            if focus=='저온' and not min_ai:
                st.caption('최저기온 AI 모델 학습 전입니다. 최고기온 AI 예측을 표시하며 최저기온은 위 기상청 공식 예보를 확인하세요.')
            cols=['날짜',primary]
            if min_ai:
                secondary='AI 예상 최고기온(°C)' if primary=='AI 예상 최저기온(°C)' else 'AI 예상 최저기온(°C)'
                cols.append(secondary)
            st.plotly_chart(px.line(week,x='날짜',y=primary,markers=True,title='현재 기온 조건에 맞춘 AI 7일 예측'),use_container_width=True)
            st.dataframe(week[cols].style.format({c:'{:.1f}' for c in cols if c!='날짜'}),hide_index=True,use_container_width=True)
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

st.subheader('📅 날짜별 AI 예측과 실제 기온 비교')
st.caption('매일 저장한 예측을 나중에 발표된 대구 ASOS 143 관측값과 비교합니다. 관측소 기온은 칠성동 개별 지점의 실측값이 아닙니다.')
log_path=Path(__file__).parent/'data'/'prediction_history.csv'
if log_path.exists():
    try:
        history=pd.read_csv(log_path,encoding='utf-8-sig')
        required={'issued_at','target_date','horizon','predicted_tmax','actual_tmax'}
        if not required.issubset(history.columns):
            st.warning('예측 기록의 필수 열이 누락되었습니다. daily_monitor.py 실행 기록을 확인하세요.')
        elif history.empty:
            st.info('아직 저장된 예측이 없습니다. GitHub Actions의 첫 실행을 기다려 주세요.')
        else:
            history['target_date']=pd.to_datetime(history['target_date'],errors='coerce')
            history['issued_at']=pd.to_datetime(history['issued_at'],errors='coerce',utc=True)
            for c in ['horizon','predicted_tmax','actual_tmax']:
                history[c]=pd.to_numeric(history[c],errors='coerce')
            history=history.dropna(subset=['target_date','issued_at','horizon','predicted_tmax']).copy()
            history['horizon']=history['horizon'].astype(int)
            history['absolute_error']=(history['predicted_tmax']-history['actual_tmax']).abs()
            horizon=st.selectbox('몇 일 전에 예측한 결과를 볼까요?',list(range(1,8)),format_func=lambda n:f'{n}일 전 예측',index=0,key='history_horizon')
            selected=history[history['horizon']==horizon].sort_values(['target_date','issued_at']).drop_duplicates('target_date',keep='first').sort_values('target_date',ascending=False).copy()
            if selected.empty:
                st.info('선택한 예측일차의 기록이 아직 없습니다.')
            else:
                scored=selected.dropna(subset=['actual_tmax','absolute_error'])
                col1,col2,col3=st.columns(3)
                col1.metric('비교 완료',f'{len(scored)}일')
                col2.metric('평균 절대오차 (MAE)',f"{scored['absolute_error'].mean():.2f}°C" if len(scored) else '계산 대기')
                col3.metric('관측 대기',f'{len(selected)-len(scored)}일')
                table=selected[['target_date','predicted_tmax','actual_tmax','absolute_error']].copy()
                table['상태']=table['actual_tmax'].apply(lambda v:'관측 대기' if pd.isna(v) else '비교 완료')
                table['target_date']=table['target_date'].dt.strftime('%Y-%m-%d')
                table.columns=['날짜','AI 예측 (°C)','실제 최고기온 (°C)','오차 (°C)','상태']
                st.dataframe(table,hide_index=True,use_container_width=True,
                    column_config={
                        'AI 예측 (°C)':st.column_config.NumberColumn(format='%.1f'),
                        '실제 최고기온 (°C)':st.column_config.NumberColumn(format='%.1f'),
                        '오차 (°C)':st.column_config.NumberColumn(format='%.2f'),
                    })
                if len(scored):
                    st.markdown('**날짜별 예측 오차 추이**')
                    trend=scored.sort_values('target_date').set_index('target_date')['absolute_error']
                    st.line_chart(trend,y_label='절대오차 (°C)',x_label='실제 기온 날짜')
                    st.markdown('**AI 예측 기온과 실제 기온 비교**')
                    comparison=scored.sort_values('target_date').set_index('target_date')[['predicted_tmax','actual_tmax']]
                    comparison.columns=['AI 예측 최고기온','실제 최고기온']
                    st.line_chart(comparison,y_label='최고기온 (°C)',x_label='날짜')
                st.download_button('날짜별 예측·관측 비교 CSV 다운로드',table.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f'forecast_vs_actual_{horizon}day.csv',mime='text/csv')
            st.caption(f'전체 예측 기록 {len(history)}건 · 관측자료가 갱신되면 매일 자동 비교됩니다. 실제 관측이 없는 날짜는 오차 계산에서 제외됩니다.')
    except Exception as exc:
        st.warning(f'예측 기록을 읽지 못했습니다: {type(exc).__name__}')
else:
    st.info('아직 예측 기록이 없습니다. GitHub Actions → AI daily improvement → Run workflow를 실행하면 기록이 시작됩니다.')

st.header('3. 계절 맞춤 행동요령')
focus,_,_=season_focus(forecast)
if focus=='저온':
    st.info('추위·결빙에 대비해 방한용품을 준비하고, 빙판길 보행에 주의하세요. 공식 한파특보를 확인하세요.')
elif focus=='고온':
    st.info('더운 시간대 야외활동을 줄이고, 수분 섭취와 시원한 장소에서의 휴식을 챙기세요. 공식 폭염특보를 확인하세요.')
else:
    st.info('일교차에 맞는 옷차림을 준비하고, 강수·강풍 예보와 기상청 특보를 확인하세요.')
st.caption('계절 맞춤 안내는 기상조건에 따른 참고 정보이며 공식 기상특보가 아닙니다.')

st.header('4. 쉼터 지도 · 내 위치')
st.caption('쉼터 정보는 행정안전부/지자체 공식 자료만 사용합니다. 지도에 보이는 위치와 실제 운영 여부는 다를 수 있으니 방문 전에 확인하세요.')
st.markdown('[전국무더위쉼터표준데이터 (공공데이터포털)](https://www.data.go.kr/data/15013199/standard.do)')
st.write('**내 위치 표시**: 버튼을 누르면 브라우저가 위치 권한을 요청합니다. 위치는 지도와 거리 계산에만 사용합니다.')
if 'geo_enabled' not in st.session_state:
    st.session_state.geo_enabled = False
if 'geo_coords' not in st.session_state:
    st.session_state.geo_coords = None
left, right = st.columns([2, 1])
with left:
    if st.button('📍 내 위치 가져오기', type='primary', use_container_width=True):
        st.session_state.geo_enabled = True
with right:
    if st.button('위치 표시 해제', use_container_width=True):
        st.session_state.geo_enabled = False
        st.session_state.geo_coords = None
if st.session_state.geo_enabled:
    st.caption('위치 요청 창이 나타나면 허용을 누르세요. GPS가 켜져 있어야 합니다.')
    try:
        location = streamlit_geolocation()
        if isinstance(location, dict) and location.get('latitude') is not None and location.get('longitude') is not None:
            lat, lon = float(location['latitude']), float(location['longitude'])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                st.session_state.geo_coords = (lat, lon)
    except Exception as exc:
        st.warning('위치 구성요소를 불러오지 못했습니다. 페이지를 새로고침하고 다시 시도하세요.')
user_lat = user_lon = None
if st.session_state.geo_enabled and st.session_state.geo_coords is not None:
    user_lat, user_lon = st.session_state.geo_coords
    st.success('현재 위치를 확인했습니다. 지도에서 파란색 점으로 표시합니다.')
elif st.session_state.geo_enabled:
    st.warning('위치 응답을 기다리는 중입니다. 태블릿의 위치 서비스와 Chrome 사이트 위치 권한을 확인한 뒤 새로고침해 보세요.')
else:
    st.info('위치 확인은 선택 사항입니다. 현재는 칠성동 중심 지도를 표시합니다.')
shelter_path=Path(__file__).parent/'data'/'shelters.csv'
sh=None
if shelter_path.exists():
    try:sh=parse_shelters(shelter_path.read_bytes())
    except Exception:st.warning('저장소의 쉼터 CSV 형식을 확인하세요.')
if sh is None:
    with st.expander('운영자용 공식 쉼터 CSV 시험 업로드'):
        uploaded=st.file_uploader('쉼터명/주소/위도/경도 열이 있는 공식 CSV',type='csv',key='shelter_csv')
        if uploaded:
            try:sh=parse_shelters(uploaded.getvalue())
            except ValueError as exc:st.warning(str(exc))
    st.caption('방문자 모두에게 쉼터를 자동 표시하려면 공식 CSV를 GitHub의 data/shelters.csv에 저장하세요. 이 페이지에서 올린 파일은 영구 저장되지 않습니다.')
# Map always renders, even without shelter data.
center_lat,center_lon=(user_lat,user_lon) if user_lat is not None else (35.879,128.602)
if sh is not None and len(sh):
    st.success(f'공식 쉼터 좌표 {len(sh):,}개 연결됨')
    if user_lat is not None:
        sh=sh.copy();sh['직선거리(km)']=distances_km(sh,user_lat,user_lon).round(2)
        sh=sh.sort_values('직선거리(km)')
        st.subheader('현재 위치에서 가까운 쉼터')
        st.dataframe(sh[['이름','주소','운영정보','직선거리(km)']].head(30),hide_index=True,use_container_width=True)
        st.caption('표시된 거리는 직선거리이며 실제 도보 이동거리와 다릅니다.')
    else:
        st.dataframe(sh[['이름','주소','운영정보']].head(40),hide_index=True,use_container_width=True)
else:
    st.warning('공식 쉼터 CSV가 없어 쉼터 점을 표시할 수 없습니다. 확인되지 않은 쉼터를 임의로 표시하지 않습니다.')
layers=[]
if sh is not None and len(sh):
    layers.append(pdk.Layer('ScatterplotLayer',sh,get_position='[lon, lat]',get_fill_color='[235, 96, 63, 210]',get_radius=7,radius_min_pixels=5,radius_max_pixels=9,pickable=True))
if user_lat is not None:
    layers.append(pdk.Layer('ScatterplotLayer',pd.DataFrame([{'lat':user_lat,'lon':user_lon,'이름':'내 위치','주소':'사용자 기기 위치','운영정보':''}]),get_position='[lon, lat]',get_fill_color='[30, 110, 240, 230]',get_radius=8,radius_min_pixels=6,radius_max_pixels=10,pickable=True))
view=pdk.ViewState(latitude=center_lat,longitude=center_lon,zoom=12.5 if user_lat is None else 13.5,pitch=0)
st.pydeck_chart(pdk.Deck(map_style=None,initial_view_state=view,layers=layers,tooltip={'text':'{이름}\n{주소}\n{운영정보}'}),use_container_width=True)
st.caption('주황색: 공식 자료의 무더위쉼터 / 파란색: 권한을 허용한 사용자의 현재 위치. 위치 정확도는 기기와 브라우저에 따라 달라집니다.')

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
