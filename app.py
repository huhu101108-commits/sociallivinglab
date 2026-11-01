"""칠성동 AI 폭염 안전지킴이: official forecast + validated research ML."""
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st
import plotly.express as px
from model import parse_csv, normalize, evaluate, predict_next
from kma_api import fetch_asos
from forecast import fetch_forecast

st.set_page_config(page_title='칠성동 AI 폭염 안전지킴이',page_icon='☀️',layout='wide')
st.title('☀️ 칠성동 AI 폭염 안전지킴이')
st.caption('대구 북구 칠성동 · 공식 기상예보 자동 조회 + 2006~2025년 대구 ASOS 관측자료 기반 AI 연구')
st.warning('대구 ASOS 관측소 자료와 기상청 예보 격자는 칠성동 개별 주소의 실측값이 아닙니다. 공식 특보·의료 판단을 대체하지 않습니다.')

def secret(name):
    try: return str(st.secrets.get(name,'')).strip()
    except Exception: return ''

@st.cache_data(ttl=3600,show_spinner='기상청 공식 예보를 조회하고 있습니다…')
def cached_forecast(key): return fetch_forecast(key)

@st.cache_data(show_spinner='기상 관측자료를 정리하고 있습니다…')
def cached_csv(content): return normalize([parse_csv(content)])

@st.cache_data(show_spinner='연도별 시간순 검증으로 머신러닝 모델을 학습하고 있습니다…')
def cached_model(df): return evaluate(df)

@st.cache_data(ttl=86400,show_spinner='기상청 ASOS 20년 관측자료를 수집하고 있습니다…')
def cached_asos(key): return fetch_asos(key,start='2006-01-01',end='2025-12-31')

st.header('1. 기상청 공식 단기예보')
key=secret('KMA_FORECAST_KEY')
forecast=None
if key:
    try:
        forecast,issued,grid_xy=cached_forecast(key)
        st.caption(f'발표: {issued:%Y-%m-%d %H:%M} KST · 예보 격자 {grid_xy[0]}, {grid_xy[1]} · 최대 1시간 캐시')
        now=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None)
        future=forecast[forecast.index>=now.floor('h')]
        if future.empty: future=forecast
        a,b,c=st.columns(3)
        temp=future['TMP'].dropna() if 'TMP' in future else pd.Series(dtype=float)
        hum=future['REH'].dropna() if 'REH' in future else pd.Series(dtype=float)
        highs=future['TMX'].dropna() if 'TMX' in future else pd.Series(dtype=float)
        a.metric('다음 시간대 기온',f'{temp.iloc[0]:.1f}°C' if len(temp) else '자료 없음')
        b.metric('예보 기간 최고기온',f'{highs.max():.1f}°C' if len(highs) else '자료 없음')
        c.metric('다음 시간대 습도',f'{hum.iloc[0]:.0f}%' if len(hum) else '자료 없음')
        if len(temp):
            st.line_chart(temp.rename('기온(°C)'))
        if (len(temp) and temp.ge(33).any()) or (len(highs) and highs.ge(33).any()):
            st.warning('조회된 예보에 33°C 이상의 기온이 포함됩니다. 공식 폭염특보와 행동요령을 확인하세요.')
        else:
            st.info('조회된 예보에서 33°C 이상 기온이 확인되지 않았습니다. 공식 폭염특보는 별도로 확인하세요.')
    except Exception:
        st.error('공식 예보를 불러오지 못했습니다. API 승인·인증 설정 또는 네트워크를 확인하세요. 인증키는 표시하지 않습니다.')
else:
    st.info('운영자가 Streamlit Secrets에 KMA_FORECAST_KEY를 등록하면 공식 예보가 자동 표시됩니다.')
st.markdown('[기상청 공식 예보·특보 확인](https://www.weather.go.kr/)')

st.header('2. 20년 관측자료 기반 AI 최고기온 예측')
st.caption('방문자가 인증키를 입력할 필요 없이 운영자가 등록한 Streamlit Secrets를 서버에서 읽습니다.')
st.caption('공식 예보와 별개인 실험적 머신러닝 모델입니다. 과거 데이터로 다음 날 최고기온을 추정하며, 예측 정확도를 시간순으로 평가합니다.')
local=Path(__file__).parent/'data'/'asos_143.csv'
data=None
source=None
if local.exists():
    try:
        data=cached_csv(local.read_bytes()); source='저장소에 포함된 기상청 ASOS 데이터'
    except Exception: st.warning('저장된 ASOS 데이터 파일을 읽을 수 없습니다.')
if data is None:
    asos_key=secret('ASOS_SERVICE_KEY')
    if asos_key:
        try:
            data=cached_asos(asos_key);source='공공데이터포털 ASOS 일자료 API (2006~2025)'
        except Exception:
            st.warning('ASOS 자동 수집에 실패했습니다. 인증키·활용 승인·API 응답 및 서버 연결을 확인해 주세요. 인증키는 표시하지 않습니다.')
if data is None:
    st.info('자동 AI 학습을 활성화하려면 저장소의 data/asos_143.csv에 실제 대구 ASOS 일자료를 올리거나 Streamlit Secrets에 ASOS_SERVICE_KEY를 추가하세요. 기상청 API허브 예보 키와 공공데이터포털 ASOS 키는 서로 다릅니다.')
    with st.expander('연구용 기상청 CSV를 직접 업로드하기'):
        upload=st.file_uploader('대구(143) ASOS 일자료 CSV',type=['csv'])
        if upload:
            try: data=cached_csv(upload.getvalue());source='사용자가 업로드한 관측자료'
            except Exception as exc: st.error(f'CSV 열 형식을 확인해 주세요: {type(exc).__name__}')

if data is not None:
    st.success(f'관측자료 연결: {source} · {data.index.min().date()} ~ {data.index.max().date()} · {data.tmax.notna().sum():,}일 최고기온')
    try:
        result=cached_model(data)
        st.subheader('시간순 교차 검증')
        st.caption('최근 3개 연도를 대상으로 해당 연도 이전 데이터만 학습합니다. MAE는 낮을수록 정확하며, 단순 전날 기온 기준과 비교합니다.')
        st.dataframe(result['metrics'],use_container_width=True,hide_index=True)
        ev=result['evaluation']; selected=result['selected']
        view=ev[ev.model==selected].tail(120)
        st.plotly_chart(px.line(view,x='date',y=['actual','pred'],title=f'{selected}: 실제값과 예측값 비교',labels={'date':'날짜','value':'최고기온(°C)'}),use_container_width=True)
        yearly=(data.tmax.ge(33)&data.tmax.notna()).groupby(data.index.year).sum().rename('33°C 이상 일수').reset_index(names='연도')
        st.plotly_chart(px.bar(yearly,x='연도',y='33°C 이상 일수',title='대구 관측소 연도별 33°C 이상 일수'),use_container_width=True)
        predicted=predict_next(result)
        if predicted:
            target,temp=predicted
            st.metric(f'AI 추정 대상: {target.date()} 최고기온',f'{temp:.1f}°C')
            today=pd.Timestamp.now(tz='Asia/Seoul').date()
            if target.date()<today:
                st.warning('학습자료 종료 다음 날에 대한 과거 시점 재현 결과입니다. 현재 기온 예보가 아닙니다.')
            elif target.date()>(pd.Timestamp(today)+pd.Timedelta(days=2)).date():
                st.warning('학습자료의 마지막 날과 현재 날짜가 맞지 않습니다. 실제 현재 예보로 해석하면 안 됩니다.')
            else:
                st.info('AI 통계 추정이며 기상청 공식 예보와 동일한 예측 대상·방법이 아닙니다.')
            if forecast is not None:
                forecast_day=forecast.loc[forecast.index.date==target.date()]
                values=forecast_day['TMX'].dropna() if 'TMX' in forecast_day else pd.Series(dtype=float)
                if len(values):
                    st.metric('같은 날짜의 기상청 공식 최고기온 예보',f'{values.max():.1f}°C',help='동일 날짜일 때만 비교하며 지점/격자와 예측 방식은 다릅니다.')
        st.download_button('검증 지표 CSV 내려받기',result['metrics'].to_csv(index=False).encode('utf-8-sig'),'ai_validation.csv','text/csv')
    except Exception as exc:
        st.warning(f'모델 학습에 필요한 데이터가 부족하거나 형식이 맞지 않습니다: {type(exc).__name__}')

st.header('3. 폭염 위험 안내')
st.caption('예보 온도를 이용한 참고용 안내이며 공식 폭염특보·온열질환 위험등급이 아닙니다.')
if forecast is not None and 'TMP' in forecast:
    future=forecast[forecast.index>=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).floor('h')]
    mx=future['TMP'].max() if len(future) else float('nan')
    if pd.notna(mx):
        if mx>=33: st.error('참고 안내: 예보에 33°C 이상 시간대가 있습니다. 야외활동을 줄이고 시원한 곳에서 휴식하세요.')
        elif mx>=30: st.warning('참고 안내: 예보에 30°C 이상 시간대가 있습니다. 수분 섭취와 휴식에 신경 쓰세요.')
        else: st.info('조회된 시간대 기온은 30°C 미만입니다. 날씨 변화와 공식 특보를 계속 확인하세요.')
else: st.info('예보 데이터가 연결되면 온도 기반 참고 안내가 표시됩니다.')
st.markdown('**건강 수칙:** 물 자주 마시기, 그늘이나 시원한 장소에서 쉬기, 더운 시간대 야외활동 줄이기. 의식 변화 등 위급한 상황에서는 119에 신고하세요.')

st.header('4. 칠성동 무더위쉼터 안내')
st.caption('쉼터 위치와 운영시간은 지자체 공식 자료를 확인한 뒤에만 표시합니다. 임의의 장소는 만들지 않습니다.')
shelter_path=Path(__file__).parent/'data'/'shelters.csv'
sh=None
if shelter_path.exists():
    try: sh=pd.read_csv(shelter_path,encoding='utf-8-sig')
    except Exception: st.warning('쉼터 파일을 읽지 못했습니다.')
if sh is None:
    sh_file=st.file_uploader('검증된 쉼터 CSV 업로드 (이름, 주소, 위도, 경도, 운영시간)',type='csv',key='shelter')
    if sh_file:
        try: sh=parse_csv(sh_file.getvalue())
        except Exception: st.warning('쉼터 CSV 형식을 확인해 주세요.')
if sh is not None:
    st.dataframe(sh,use_container_width=True,hide_index=True)
    lat=next((x for x in ('위도','lat','latitude') if x in sh),None)
    lon=next((x for x in ('경도','lon','longitude') if x in sh),None)
    if lat and lon:
        geo=pd.DataFrame({'lat':pd.to_numeric(sh[lat],errors='coerce'),'lon':pd.to_numeric(sh[lon],errors='coerce')}).dropna()
        geo=geo[geo.lat.between(35,36)&geo.lon.between(128,129)]
        if len(geo):st.map(geo)
else:
    st.info('쉼터 위치 데이터가 아직 연결되지 않았습니다. 실제 운영 여부는 지자체에서 확인하세요.')
st.header('5. 지역사회 정책 제안')
st.markdown('고령자·거동불편자의 쉼터 접근성을 조사하고, 폭염 시 방문 돌봄과 이동 지원을 검토할 수 있습니다. 세부 동네 위험지도는 별도의 공간·인구·현장 관측자료를 확보한 뒤 제작해야 합니다.')
st.caption('자료 출처: 기상청 공식 단기예보, 기상청 ASOS 일자료(연결 시). AI 성능 수치는 실제 자료로 검증된 경우에만 표시됩니다.')
