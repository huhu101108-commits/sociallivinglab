import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import parse_csv,normalize,evaluate,predict_next
from kma_api import fetch_asos
from forecast import fetch_forecast

st.set_page_config(page_title='칠성동 AI 폭염 안전지킴이',page_icon='☀️',layout='wide')
st.title('☀️ 칠성동 AI 폭염 안전지킴이')
st.markdown('**칠성동의 폭염을 데이터로 이해하고, 안전하게 대비하는 연구용 웹사이트**')
st.caption('대구 북구 칠성동 전용 연구용 웹앱 | 대구 ASOS 143 관측자료를 대리 지표로 사용')
st.warning('대구 관측소의 온도는 칠성동 실측값이 아닙니다. 통계 예측은 기상청 공식 예보·폭염특보나 의료 판단을 대체하지 않습니다.')

@st.cache_data(show_spinner='기상청 CSV 자료 분석 중…')
def from_upload(files): return normalize([parse_csv(raw) for name,raw in files])
@st.cache_data(show_spinner='모델 학습과 시간순 검증 중…')
def run_model(data): return evaluate(data)

# 공식 단기예보는 머신러닝 결과와 구분해서 표시합니다.
@st.cache_data(ttl=3600,show_spinner='기상청 공식 단기예보 불러오는 중…')
def official_forecast(key):
    return fetch_forecast(key)

st.subheader('🌤️ 칠성동 인근 기상청 공식 단기예보')
st.caption('기상청 격자 예보이며 칠성동 특정 주소의 실측 온도가 아닙니다. AI 자체 예측과 구분됩니다.')
try:
    secret_key=st.secrets.get('KMA_FORECAST_KEY','')
except Exception:
    secret_key=''
if secret_key:
    try:
        forecast,issued,grid_xy=official_forecast(secret_key)
        st.caption(f'기상청 발표: {issued:%Y-%m-%d %H:%M} KST · 예보 격자 {grid_xy[0]}, {grid_xy[1]} · 1시간 캐시')
        cols=st.columns(3)
        today=pd.Timestamp.now(tz='Asia/Seoul').date()
        future=forecast[forecast.index.date>=today]
        if not future.empty:
            if 'TMP' in future:
                next_t=future['TMP'].dropna()
                if not next_t.empty:cols[0].metric('가장 가까운 시간대 예보 기온',f'{next_t.iloc[0]:.1f}°C')
            if 'TMX' in future:
                mx=future['TMX'].dropna()
                if not mx.empty:cols[1].metric('예보 기간 최고기온',f'{mx.max():.1f}°C')
            if 'REH' in future:
                rh=future['REH'].dropna()
                if not rh.empty:cols[2].metric('가장 가까운 시간대 예보 습도',f'{rh.iloc[0]:.0f}%')
            if 'TMP' in future:
                graph=future[['TMP']].dropna().reset_index()
                st.line_chart(graph,x='date',y='TMP',x_label='예보 시각',y_label='기온(°C)')
            high=bool(('TMX' in future and future['TMX'].ge(33).any()) or ('TMP' in future and future['TMP'].ge(33).any()))
            if high:st.warning('예보에 33°C 이상인 시간대가 포함됩니다. 기상청 폭염특보도 확인하세요.')
            else:st.info('현재 조회된 예보에서 33°C 이상인 값은 확인되지 않았습니다. 공식 폭염특보는 별도로 확인하세요.')
    except Exception as e:
        st.warning(f'기상청 예보 연결 실패: {e}')
else:
    st.info('관리자가 Streamlit 앱 Secrets에 KMA_FORECAST_KEY(API허브 인증키)를 설정하면, 방문자에게 기상청 단기예보가 자동 표시됩니다. 방문자는 인증키를 입력할 필요가 없습니다.')
st.markdown('[기상청 날씨누리에서 공식 예보·특보 확인](https://www.weather.go.kr/)')
st.divider()

with st.sidebar:
    st.header('기상 자료 연결')
    mode=st.radio('데이터 방식',['기상청 CSV 업로드','공공데이터포털 API 자동 수집'])
    data=None
    if mode=='기상청 CSV 업로드':
        uploaded=st.file_uploader('대구 ASOS(143) 일자료 CSV',type='csv',accept_multiple_files=True)
        st.markdown('[기상청 ASOS 일자료](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)')
        if uploaded:
            try: data=from_upload(tuple((f.name,f.getvalue()) for f in uploaded))
            except Exception as e: st.error(str(e))
    else:
        key=st.text_input('공공데이터포털 서비스키',type='password',help='키는 저장하지 않습니다. 발급받은 키를 직접 입력하세요.')
        if st.button('2006~2025년 20년치 가져오기',type='primary'):
            try:
                progress=st.progress(0)
                data=fetch_asos(key,progress=progress.progress)
                st.session_state['asos_data']=data
                st.success(f'{len(data):,}일 자료를 수집했습니다.')
            except Exception as e: st.error(f'수집 실패: {e}')
        if 'asos_data' in st.session_state:data=st.session_state['asos_data']
    st.divider()
    st.caption('데이터 기준: 2006~2025년 학습 권장. 2026년 예보를 표시하려면 최신 관측자료와 공식 예보를 별도로 연동해야 합니다.')

if data is None:
    st.subheader('시작하기')
    st.markdown('① 왼쪽 메뉴에서 **기상청 CSV 업로드** 또는 **API 자동 수집**을 선택하세요. ② 분석이 끝나면 AI 검증 결과와 폭염 대응 정보를 볼 수 있습니다.')
    st.info('왼쪽에서 기상청 CSV를 올리거나 API 인증키로 20년치 데이터를 수집하면 AI가 학습을 시작합니다. 자료가 없을 때는 임의의 예측을 표시하지 않습니다.')
    st.markdown('**앱 기능**: ① 과거 폭염 변화 ② 시간순 검증된 AI 예측 ③ 칠성동 쉼터 안내 ④ 온열질환 예방 ⑤ 정책 분석')
    st.stop()

try:
    data=data.loc['2006-01-01':].copy()
    st.subheader('① 관측자료 품질')
    a,b,c=st.columns(3)
    a.metric('관측 시작',str(data.index.min().date()))
    b.metric('관측 종료',str(data.index.max().date()))
    c.metric('최고기온 관측률',f'{data.tmax.notna().mean():.1%}')
    result=run_model(data)
    st.subheader('② 실제 과거 자료로 검증한 AI')
    st.write('마지막 3개 연도를 각각 **이전 연도 자료만으로 학습**해 검증합니다. 임의로 데이터를 섞지 않아 미래 정보 누출을 방지합니다.')
    st.dataframe(result['metrics'],hide_index=True,use_container_width=True)
    st.caption('MAE/RMSE는 낮을수록 좋고, 폭염 재현율은 실제 33°C 이상인 날을 얼마나 포착했는지 나타냅니다. 단순 전날 기온 기준과 비교하세요.')
    ev=result['evaluation']; name=result['selected']
    chart=ev[ev.model==name].tail(120).copy()
    st.plotly_chart(px.line(chart,x='date',y=['actual','pred'],title=f'{name}: 마지막 평가일 120일 실제값 vs 예측값',labels={'value':'최고기온(°C)','date':'날짜'}),use_container_width=True)
    if result['metrics'].iloc[0]['모델']=='전날 기온 기준': st.warning('AI가 단순 기준보다 나은 성능을 보이지 않았습니다. 모델을 실사용 예보로 간주하지 마세요.')
    st.subheader('③ 지난 20년 폭염 패턴')
    yearly=(data.tmax.ge(33)&data.tmax.notna()).groupby(data.index.year).sum().reset_index()
    yearly.columns=['연도','33°C 이상 일수']
    st.plotly_chart(px.bar(yearly,x='연도',y='33°C 이상 일수',title='대구 관측소 연도별 폭염일수'),use_container_width=True)
    st.subheader('④ 다음 날 최고기온 추정')
    next_pred=predict_next(result)
    if next_pred:
        date,temp=next_pred
        st.metric(f'{date.date()} 통계적 최고기온 추정',f'{temp:.1f}°C')
        if date.date()<pd.Timestamp.now(tz='Asia/Seoul').date():
            st.warning('마지막 업로드 자료 다음 날에 대한 **과거 시점 시연**입니다. 현재 또는 미래 날씨 예보가 아닙니다.')
        else: st.info('기상청 공식 단기예보와 함께 확인하세요. 장기 기후 패턴을 학습한 모형만으로 폭염 경보를 발령하지 않습니다.')
        if temp>=33:st.error('33°C 이상으로 추정: 더위 노출을 줄이고 공식 특보를 확인하세요.')
    else:st.info('최근 28일 중 결측이 있어 다음 날 추정을 표시하지 않습니다.')
    st.subheader('⑤ 칠성동 무더위쉼터 안내')
    st.write('지자체에서 검증된 쉼터 자료를 올리면 위치와 운영정보를 확인할 수 있습니다.')
    shelter=st.file_uploader('쉼터 CSV (이름·주소·위도·경도·운영시간)',type='csv',key='shelter')
    if shelter:
        sh=parse_csv(shelter.getvalue());st.dataframe(sh,use_container_width=True)
        lat=next((k for k in ['위도','lat','latitude'] if k in sh.columns),None)
        lon=next((k for k in ['경도','lon','longitude'] if k in sh.columns),None)
        if lat and lon:
            loc=sh.copy();loc['lat']=pd.to_numeric(loc[lat],errors='coerce');loc['lon']=pd.to_numeric(loc[lon],errors='coerce')
            loc=loc.dropna(subset=['lat','lon']);loc=loc[loc.lat.between(35,36)&loc.lon.between(128,129)]
            if len(loc):st.map(loc[['lat','lon']])
    else:st.info('검증되지 않은 쉼터 이름이나 위치를 임의로 표시하지 않습니다.')
    st.subheader('⑥ 주민 건강 보호 및 행정 활용')
    st.markdown('**폭염 대응**: 공식 특보 확인, 시원한 장소에서 휴식, 물 자주 마시기, 한낮 야외활동 줄이기. 심한 이상 증상이나 의식 변화가 있으면 즉시 119에 연락하세요.')
    st.markdown('**정책 제안**: 쉼터 운영시간과 고령층 접근성을 확인하고, 방문 돌봄·이동지원 필요 지역을 조사하세요. 현재 모델은 칠성동 내부 블록별 기온을 구분하지 않으므로 세부 위험지도는 현장 관측·추가 공간자료 확보 후에만 제작합니다.')
    st.download_button('AI 평가 결과 CSV 저장',result['metrics'].to_csv(index=False).encode('utf-8-sig'),'model_evaluation.csv','text/csv')
except Exception as e:
    st.error(f'분석 오류: {e}')

# 폭염 안전 행동 안내
st.divider()
st.header("🌡️ 폭염 안전 행동 안내")

st.info(
    "이 안내는 폭염 예방을 위한 일반 정보이며, "
    "기상청 공식 폭염특보를 대체하지 않습니다."
)

st.markdown("""
### 폭염 예방 수칙
- 물을 자주 마시고 충분히 휴식하기
- 한낮의 장시간 야외활동 줄이기
- 그늘이나 냉방시설 이용하기
- 고령자와 건강 취약계층의 상태 살피기

### 응급상황 대응
의식 변화 등 위급한 상황에서는 즉시 119에 신고하세요.
""")

# =====================================
# AI 폭염 예측 및 학습
# =====================================

st.divider()
st.header("🤖 AI 폭염 예측 시스템")

st.write(
    "2006~2025년 대구 기상 관측자료를 "
    "학습하여 다음 날 최고기온을 예측합니다."
)

try:
    asos_key = st.secrets.get(
        "ASOS_SERVICE_KEY", ""
    )
except Exception:
    asos_key = ""

if not asos_key:
    st.warning(
        "AI 학습을 시작하려면 Streamlit Secrets에 "
        "ASOS_SERVICE_KEY를 등록해야 합니다."
    )
else:
    if st.button("🚀 AI 학습 시작"):
        try:
            from kma_api import fetch_asos

            with st.spinner(
                "20년 기상자료를 수집하고 AI를 학습하는 중..."
            ):
                data = fetch_asos(
                    asos_key,
                    start="2006-01-01",
                    end="2025-12-31"
                )
                result = evaluate(data)

            st.success("AI 학습 완료!")

            st.write("AI 모델 검증 결과")
            st.write(result)

        except Exception:
            st.error(
                "기상자료 수집 또는 AI 학습에 실패했습니다. "
                "API 승인 상태와 데이터 형식을 확인해 주세요."
            )
