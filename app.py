
"""칠성동 AI 폭염 안전지킴이"""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from model import parse_csv, normalize, evaluate, predict_next
from kma_api import fetch_asos
from forecast import fetch_forecast


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="칠성동 AI 폭염 안전지킴이",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 칠성동 AI 폭염 안전지킴이")

st.caption(
    "대구 북구 칠성동 | 기상청 공식 예보와 "
    "2006~2025년 ASOS 관측자료를 활용한 AI 연구"
)

st.warning(
    "대구 ASOS 관측소 자료는 칠성동 특정 주소의 실측값이 아닙니다. "
    "AI 예측은 연구용이며 기상청 공식 특보나 의료 판단을 대체하지 않습니다."
)


# =========================================================
# 2. 인증키 자동 로드
# =========================================================

def get_secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


FORECAST_KEY = get_secret("KMA_FORECAST_KEY")
ASOS_KEY = get_secret("ASOS_SERVICE_KEY")


# =========================================================
# 3. 데이터 캐시
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner="기상청 공식 예보를 조회하고 있습니다..."
)
def load_forecast(key):
    return fetch_forecast(key)


@st.cache_data(
    ttl=86400,
    show_spinner="2006~2025년 ASOS 관측자료를 수집하고 있습니다..."
)
def load_asos(key):
    return fetch_asos(
        key,
        start="2006-01-01",
        end="2025-12-31"
    )


@st.cache_data(
    show_spinner="관측자료를 정리하고 있습니다..."
)
def load_csv(content):
    return normalize([parse_csv(content)])


@st.cache_data(
    show_spinner="AI 모델을 학습하고 검증하고 있습니다..."
)
def train_model(data):
    return evaluate(data)


# =========================================================
# 4. 기상청 공식 단기예보
# =========================================================

st.header("1. 칠성동 인근 기상청 공식 단기예보")

st.caption(
    "기상청 예보 격자 자료이며 "
    "AI 자체 예측 결과와는 구분됩니다."
)

forecast = None

if FORECAST_KEY:

    try:
        forecast, issued, grid_xy = load_forecast(
            FORECAST_KEY
        )

        st.caption(
            f"기상청 발표: {issued:%Y-%m-%d %H:%M} KST "
            f"| 예보 격자: {grid_xy[0]}, {grid_xy[1]}"
        )

        now = (
            pd.Timestamp.now(tz="Asia/Seoul")
            .tz_localize(None)
        )

        future = forecast[
            forecast.index >= now.floor("h")
        ]

        if future.empty:
            future = forecast

        temperature = (
            future["TMP"].dropna()
            if "TMP" in future
            else pd.Series(dtype=float)
        )

        humidity = (
            future["REH"].dropna()
            if "REH" in future
            else pd.Series(dtype=float)
        )

        highest = (
            future["TMX"].dropna()
            if "TMX" in future
            else pd.Series(dtype=float)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "가장 가까운 시간대 예보 기온",
                (
                    f"{temperature.iloc[0]:.1f}°C"
                    if len(temperature)
                    else "자료 없음"
                )
            )

        with col2:
            st.metric(
                "예보 기간 최고기온",
                (
                    f"{highest.max():.1f}°C"
                    if len(highest)
                    else "자료 없음"
                )
            )

        with col3:
            st.metric(
                "가장 가까운 시간대 예보 습도",
                (
                    f"{humidity.iloc[0]:.0f}%"
                    if len(humidity)
                    else "자료 없음"
                )
            )

        if len(temperature):

            chart_data = temperature.rename(
                "예보 기온(°C)"
            )

            st.line_chart(chart_data)

        hot_forecast = (
            (
                len(temperature) > 0
                and temperature.ge(33).any()
            )
            or
            (
                len(highest) > 0
                and highest.ge(33).any()
            )
        )

        if hot_forecast:
            st.warning(
                "예보에 33°C 이상 기온이 포함되어 있습니다. "
                "기상청 공식 폭염특보를 확인하세요."
            )
        else:
            st.info(
                "조회된 예보에서 33°C 이상 기온이 "
                "확인되지 않았습니다. "
                "공식 폭염특보는 별도로 확인하세요."
            )

    except Exception:
        st.error(
            "기상청 공식 예보를 불러오지 못했습니다. "
            "인증키, API 승인 상태 또는 서버 연결을 확인하세요."
        )

else:
    st.info(
        "Streamlit Secrets에 "
        "KMA_FORECAST_KEY가 등록되어 있지 않습니다."
    )

st.markdown(
    "[기상청 공식 날씨 및 특보 확인]"
    "(https://www.weather.go.kr/)"
)


# =========================================================
# 5. ASOS 관측자료 자동 수집
# =========================================================

st.divider()

st.header("2. 20년 관측자료 기반 AI 최고기온 예측")

st.caption(
    "대구 ASOS 143 관측소의 과거 관측자료를 이용합니다. "
    "인증키는 서버에서 자동으로 읽으므로 "
    "방문자가 직접 입력할 필요가 없습니다."
)

data = None
data_source = None

data_path = (
    Path(__file__).parent
    / "data"
    / "asos_143.csv"
)

# 저장소에 공식 관측자료가 있으면 우선 사용
if data_path.exists():

    try:
        data = load_csv(
            data_path.read_bytes()
        )

        data_source = (
            "저장소에 포함된 ASOS 관측자료"
        )

    except Exception:
        st.warning(
            "저장된 ASOS CSV 파일을 읽지 못했습니다."
        )


# 저장된 자료가 없으면 API 자동 수집
if data is None and ASOS_KEY:

    try:
        data = load_asos(ASOS_KEY)

        data_source = (
            "공공데이터포털 ASOS 일자료 API"
        )

    except Exception:
        st.warning(
            "ASOS 관측자료 자동 수집에 실패했습니다. "
            "공공데이터포털 인증키와 API 응답을 확인하세요."
        )


# 운영자용 대체 업로드
if data is None:

    st.info(
        "ASOS 자료가 아직 연결되지 않았습니다. "
        "운영자는 CSV 파일을 이용해 자료를 연결할 수도 있습니다."
    )

    with st.expander("연구용 ASOS CSV 업로드"):

        uploaded = st.file_uploader(
            "대구 ASOS 관측자료 CSV",
            type=["csv"],
            key="asos_upload"
        )

        if uploaded is not None:

            try:
                data = load_csv(
                    uploaded.getvalue()
                )

                data_source = (
                    "업로드된 ASOS 관측자료"
                )

            except Exception:
                st.error(
                    "CSV 파일의 형식을 확인하세요."
                )


# =========================================================
# 6. AI 모델 학습 및 평가
# =========================================================

if data is not None:

    st.success(
        f"관측자료 연결 성공: {data_source}"
    )

    st.write(
        f"관측 기간: "
        f"{data.index.min().date()} ~ "
        f"{data.index.max().date()}"
    )

    st.write(
        f"최고기온 유효 데이터: "
        f"{data['tmax'].notna().sum():,}일"
    )

    try:

        result = train_model(data)

        st.subheader("AI 모델 성능 비교")

        st.caption(
            "과거 데이터를 시간순으로 나누어 "
            "AI 예측값과 실제 관측값을 비교합니다."
        )

        metrics = result["metrics"]

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # 실제 기온과 AI 예측 기온 비교
        # -------------------------------------------------

        evaluation = result["evaluation"]
        selected = result["selected"]

        comparison = evaluation[
            evaluation["model"] == selected
        ].tail(120)

        if not comparison.empty:

            fig = px.line(
                comparison,
                x="date",
                y=["actual", "pred"],
                title=(
                    f"{selected}: 실제값과 예측값 비교"
                ),
                labels={
                    "date": "날짜",
                    "value": "최고기온(°C)",
                    "variable": "구분"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # -------------------------------------------------
        # 연도별 폭염일수
        # TypeError 수정 부분
        # -------------------------------------------------

        st.subheader("연도별 폭염일수 분석")

        yearly = pd.DataFrame({
            "연도": data.index.year,
            "폭염일": (
                data["tmax"] >= 33
            ).astype(int).to_numpy()
        })

        yearly = (
            yearly
            .groupby(
                "연도",
                as_index=False
            )["폭염일"]
            .sum()
            .rename(
                columns={
                    "폭염일": "33°C 이상 일수"
                }
            )
        )

        fig_yearly = px.bar(
            yearly,
            x="연도",
            y="33°C 이상 일수",
            title=(
                "대구 ASOS 관측소 "
                "연도별 33°C 이상 일수"
            ),
            labels={
                "연도": "연도",
                "33°C 이상 일수": "일수"
            }
        )

        st.plotly_chart(
            fig_yearly,
            use_container_width=True
        )

        # -------------------------------------------------
        # AI 다음 날 최고기온 추정
        # -------------------------------------------------

        st.subheader("AI 최고기온 추정 결과")

        predicted = predict_next(result)

        if predicted is not None:

            target, predicted_temp = predicted

            target_date = pd.Timestamp(
                target
            ).date()

            st.metric(
                f"AI 추정 대상: {target_date}",
                f"{predicted_temp:.1f}°C"
            )

            today = pd.Timestamp.now(
                tz="Asia/Seoul"
            ).date()

            if target_date < today:

                st.warning(
                    "이 결과는 과거 관측자료 종료일 "
                    "다음 날에 대한 추정입니다. "
                    "현재 날짜의 실시간 AI 예보가 아닙니다."
                )

            elif target_date > (
                pd.Timestamp(today)
                + pd.Timedelta(days=2)
            ).date():

                st.warning(
                    "학습자료의 마지막 날짜와 현재 날짜가 "
                    "일치하지 않습니다."
                )

            else:

                st.info(
                    "AI 통계 예측 결과입니다. "
                    "기상청 공식 예보와 구분해 해석하세요."
                )

            # 같은 날짜의 공식 예보가 있을 때만 비교
            if forecast is not None:

                same_day = forecast.loc[
                    forecast.index.date == target_date
                ]

                if (
                    not same_day.empty
                    and "TMX" in same_day.columns
                ):

                    official = (
                        same_day["TMX"]
                        .dropna()
                    )

                    if len(official):

                        st.metric(
                            "같은 날짜 기상청 공식 최고기온",
                            f"{official.max():.1f}°C"
                        )

        # -------------------------------------------------
        # 성능 결과 다운로드
        # -------------------------------------------------

        st.download_button(
            label="AI 검증 결과 CSV 다운로드",
            data=metrics.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="ai_validation.csv",
            mime="text/csv"
        )

    except Exception as exc:

        st.error(
            "AI 모델 처리 중 오류가 발생했습니다."
        )

        st.caption(
            f"오류 유형: {type(exc).__name__}"
        )

        # 인증키나 전체 요청 URL은 출력하지 않음
        st.info(
            "모델 데이터 형식 또는 "
            "model.py 함수 반환값을 확인하세요."
        )


# =========================================================
# 7. 폭염 위험 안내
# =========================================================

st.divider()

st.header("3. 폭염 위험 안내")

st.caption(
    "기상청 예보 기온을 이용한 참고용 안내입니다. "
    "공식 폭염특보와는 다릅니다."
)

if forecast is not None and "TMP" in forecast.columns:

    now = (
        pd.Timestamp.now(tz="Asia/Seoul")
        .tz_localize(None)
        .floor("h")
    )

    future = forecast[
        forecast.index >= now
    ]

    if not future.empty:

        max_temp = pd.to_numeric(
            future["TMP"],
            errors="coerce"
        ).max()

        if pd.notna(max_temp):

            if max_temp >= 33:

                st.error(
                    "예보에 33°C 이상 시간대가 있습니다. "
                    "야외활동을 줄이고 시원한 장소에서 쉬세요."
                )

            elif max_temp >= 30:

                st.warning(
                    "예보에 30°C 이상 시간대가 있습니다. "
                    "수분 섭취와 휴식에 신경 쓰세요."
                )

            else:

                st.info(
                    "조회된 예보의 최고 시간대 기온은 "
                    "30°C 미만입니다. "
                    "공식 특보는 별도로 확인하세요."
                )

        else:

            st.info(
                "온도 예보 데이터가 부족합니다."
            )

    else:

        st.info(
            "현재 이후의 기온 예보가 없습니다."
        )

else:

    st.info(
        "기상청 예보가 연결되면 "
        "폭염 위험 참고 안내가 표시됩니다."
    )


st.markdown(
    """
    **폭염 대비 행동요령**

    - 물을 자주 마시기
    - 그늘이나 시원한 장소에서 충분히 쉬기
    - 더운 시간대의 야외활동 줄이기
    - 고령자와 거동이 불편한 이웃의 안전 확인하기
    - 의식 변화 등 위급한 상황에서는 119에 신고하기
    """
)


# =========================================================
# 8. 칠성동 무더위쉼터 안내
# =========================================================

st.divider()

st.header("4. 칠성동 무더위쉼터 안내")

st.caption(
    "실제 지자체 자료로 확인된 쉼터만 표시합니다. "
    "운영시간과 이용 가능 여부는 방문 전에 확인하세요."
)

shelter_path = (
    Path(__file__).parent
    / "data"
    / "shelters.csv"
)

shelters = None


# 저장소의 쉼터 CSV 자동 로드
if shelter_path.exists():

    try:
        shelters = pd.read_csv(
            shelter_path,
            encoding="utf-8-sig"
        )

    except Exception:

        st.warning(
            "쉼터 데이터 파일을 읽지 못했습니다."
        )


# 파일이 없을 때만 수동 업로드
if shelters is None:

    with st.expander("운영자용 쉼터 CSV 업로드"):

        shelter_file = st.file_uploader(
            "검증된 쉼터 CSV",
            type=["csv"],
            key="shelter_upload"
        )

        if shelter_file is not None:

            try:
                shelters = pd.read_csv(
                    shelter_file,
                    encoding="utf-8-sig"
                )

            except Exception:
                st.error(
                    "쉼터 CSV 형식을 확인하세요."
                )


if shelters is not None and not shelters.empty:

    st.success(
        f"무더위쉼터 {len(shelters)}개 자료 연결"
    )

    st.dataframe(
        shelters,
        use_container_width=True,
        hide_index=True
    )

    lat_col = next(
        (
            col
            for col in [
                "위도",
                "lat",
                "latitude"
            ]
            if col in shelters.columns
        ),
        None
    )

    lon_col = next(
        (
            col
            for col in [
                "경도",
                "lon",
                "longitude"
            ]
            if col in shelters.columns
        ),
        None
    )

    if lat_col and lon_col:

        locations = pd.DataFrame({
            "lat": pd.to_numeric(
                shelters[lat_col],
                errors="coerce"
            ),
            "lon": pd.to_numeric(
                shelters[lon_col],
                errors="coerce"
            )
        }).dropna()

        # 대구 인근의 정상 좌표만 표시
        locations = locations[
            locations["lat"].between(35, 36)
            &
            locations["lon"].between(128, 129)
        ]

        if not locations.empty:

            st.map(locations)

        else:

            st.info(
                "표시 가능한 쉼터 좌표가 없습니다."
            )

    else:

        st.info(
            "지도 표시에는 위도와 경도 열이 필요합니다."
        )

else:

    st.info(
        "무더위쉼터 위치 데이터가 아직 연결되지 않았습니다. "
        "운영자가 공식 쉼터 CSV를 등록하면 "
        "방문자에게 자동 표시됩니다."
    )


# =========================================================
# 9. 지역사회 정책 제안
# =========================================================

st.divider()

st.header("5. 지역사회 정책 제안")

st.markdown(
    """
    ### ① 폭염 취약계층 보호

    고령자와 거동이 불편한 주민을 대상으로
    폭염 시 안부 확인과 방문 돌봄을 강화한다.

    ### ② 무더위쉼터 접근성 개선

    칠성동의 쉼터 위치와 운영시간을 조사하고,
    이동이 어려운 주민을 위한 지원 방안을 검토한다.

    ### ③ AI 기반 폭염 대응

    장기간 기온 관측자료와 AI 분석을 활용해
    폭염 발생 특성을 연구하고,
    향후 인구·녹지·건물 자료를 결합하여
    지역별 취약도를 분석한다.

    ### ④ 데이터 기반 행정

    폭염 위험이 높은 시기에
    취약계층 지원과 쉼터 운영을 강화할 수 있도록
    객관적인 자료를 활용한다.
    """
)


# =========================================================
# 10. 출처
# =========================================================

st.divider()

st.caption(
    "자료 출처: 기상청 공식 단기예보, "
    "기상청 ASOS 일자료(연결 시)"
)

st.caption(
    "AI 분석 결과는 교육·연구 목적으로 제공됩니다. "
    "실제 기상특보와 응급 대응은 공식 기관 안내를 따르세요."
)
