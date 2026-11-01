"""칠성동 AI 폭염 안전지킴이: official forecast + validated research ML."""
from pathlib import Path
from datetime import date, timedelta
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation
from future7 import predict_seven_days
from shelters import parse_shelters, distances_km
import pandas as pd
import streamlit as st
import plotly.express as px
from model import parse_csv, normalize, evaluate, predict_next
from kma_api import fetch_asos
from forecast import fetch_forecast

st.set_page_config(page_title='칠성동 AI 폭염 안전지킴이',page_icon='☀️',layout='wide')
st.title('☀️ 칠성동 AI 폭염 안전지킴이')
st.caption('대구 북구 칠성동 · 공식 기상예보 자동 조회 + 2006년 이후 대구 ASOS 관측자료 기반 AI 연구')
st.warning('대구 ASOS 관측소 자료와 기상청 예보 격자는 칠성동 개별 주소의 실측값이 아닙니다. 공식 특보·의료 판단을 대체하지 않습니다.')

def secret(name):
    try: return str(st.secrets.get(name,'')).strip()
    except Exception: return ''

@st.cache_data(ttl=3600,show_spinner='기상청 공식 예보를 조회하고 있습니다…')
def cached_forecast(key): return fetch_forecast(key)

@st.cache_data(show_spinner='기상 관측자료를 정리하고 있습니다…')
def cached_csv(content): return normalize([parse_csv(content)])

@st.cache_resource(show_spinner='최초 실행 시 경량 AI 모델을 학습하고 있습니다…')
def cached_model(df): return evaluate(df)

@st.cache_data(ttl=86400,show_spinner='기상청 ASOS 20년 관측자료를 수집하고 있습니다…')
def cached_asos(key): return fetch_asos(key,start='2006-01-01',end='2025-12-31')

@st.cache_data(ttl=21600,show_spinner='올해 최신 ASOS 관측자료를 조회하고 있습니다…')
def cached_recent_asos(key,year,end):
    return fetch_asos(key,start=f'{year}-01-01',end=end)

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

st.header('2. AI 일주일 최고기온 예측 (오늘 포함 7일)')
st.caption('실험적 AI 예측: 최신 대구 ASOS 관측자료 + 과거 20년 데이터로 학습합니다. 7일 예측은 최신 실측자료를 기준으로 각 날짜를 독립적으로 추정하는 직접 다중시점 모델입니다. 예측값을 다음 날 입력으로 재사용하지 않습니다. 공식 기상청 예보는 아닙니다.')
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
    st.success(f'ASOS 관측자료: {source} · {data.index.min().date()} ~ {data.index.max().date()}')
    try:
        result=cached_model(data)
        st.subheader('1일 예측 모델의 시간순 검증 성능')
        st.caption('아래 검증 지표는 1일 예측 모델의 성능이며, 7일 직접 예측 모델의 정확도를 보장하지 않습니다. 추가 특성은 성능 개선을 목표로 도입했으며 실제 개선 여부는 별도 시간순 검증이 필요합니다.')
        st.dataframe(result['metrics'],hide_index=True,use_container_width=True)
        ev=result['evaluation'];selected=result['selected']
        view=ev[ev['model']==selected].tail(120)
        st.plotly_chart(px.line(view,x='date',y=['actual','pred'],title=f'{selected} 실제값 vs 예측값'),use_container_width=True)
        yearly=pd.DataFrame({'연도':data.index.year,'폭염일':(data.tmax>=33).fillna(False).astype(int).to_numpy()})
        yearly=yearly.groupby('연도',as_index=False)['폭염일'].sum().rename(columns={'폭염일':'33°C 이상 일수'})
        st.plotly_chart(px.bar(yearly,x='연도',y='33°C 이상 일수',title='대구 ASOS 연도별 33°C 이상 일수'),use_container_width=True)
        st.subheader('오늘부터 7일간 AI 최고기온 추정')
        try:
            week=predict_seven_days(result)
            st.caption(f'최종 실측 날짜: {week.iloc[0]["관측 기반 마지막 날짜"]} · 예측 모델: {selected} · 단위: °C')
            st.plotly_chart(px.line(week,x='날짜',y='AI 예상 최고기온(°C)',markers=True,title='AI 7일 최고기온 예측'),use_container_width=True)
            st.dataframe(week[['날짜','AI 예상 최고기온(°C)']].style.format({'AI 예상 최고기온(°C)':'{:.1f}'}),hide_index=True,use_container_width=True)
            st.info('AI 예측값을 다음 날 입력으로 재사용하지 않는 직접 예측 방식입니다. 다만 먼 날짜의 예측은 불확실성이 클 수 있으며 공식 폭염특보를 대신하지 않습니다.')
            st.download_button('AI 7일 예측 CSV 내려받기',week.to_csv(index=False).encode('utf-8-sig'),'ai_7day.csv','text/csv')
        except ValueError as exc:
            st.error(str(exc))
        st.download_button('AI 검증 지표 CSV 내려받기',result['metrics'].to_csv(index=False).encode('utf-8-sig'),'ai_validation.csv','text/csv')
    except Exception as exc:
        st.error(f'AI 학습/검증 오류: {type(exc).__name__}. 관측자료 누락 및 model.py 버전을 확인하세요.')

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

st.header('4. 무더위쉼터 지도 · 내 위치')
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

st.header('5. 폭염 시 행동요령')
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
st.caption('자료 출처: 기상청 공식 단기예보, 기상청 ASOS 일자료(연결 시). AI 성능 수치는 실제 자료로 검증된 경우에만 표시됩니다.')
