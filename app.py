
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation

from model import parse_csv, normalize, evaluate
from kma_api import fetch_asos
from forecast import fetch_forecast
from future7 import predict_seven_days
from shelters import parse_shelters, distances_km


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="칠성동 AI 폭염 안전지킴이",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 칠성동 AI 폭염 안전지킴이")

st.caption(
    "기상청 공식 예보 · 대구 ASOS 관측자료 · "
    "AI 7일 최고기온 예측 · 무더위쉼터 지도"
)

st.warning(
    "대구 ASOS 관측자료는 칠성동 개별 주소의 실측값이 아닙니다. "
    "AI 예측은 연구용이며 공식 기상특보를 대체하지 않습니다."
)


def secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


FORECAST_KEY = secret("KMA_FORECAST_KEY")
ASOS_KEY = secret("ASOS_SERVICE_KEY")


# =========================================================
# 캐시
# =========================================================

@st.cache_data(ttl=3600)
def load_forecast(key):
    return fetch_forecast(key)


@st.cache_data(ttl=86400)
def load_historical(key):
    return fetch_asos(
        key,
        start="2006-01-01",
        end="2025-12-31"
    )


@st.cache_data(ttl=21600)
def load_recent(key, year, end):
    return fetch_asos(
        key,
        start=f"{year}-01-01",
        end=end
    )


@st.cache_data
def load_csv(content):
    return normalize([parse_csv(content)])


@st.cache_data
def train_model(data):
    return evaluate(data)


# =========================================================
# 1. 공식 기상예보
# =========================================================

st.header("1. 기상청 공식 단기예보")

forecast = None

if FORECAST_KEY:
    try:
        forecast, issued, grid_xy = load_forecast(
            FORECAST_KEY
        )

        st.caption(
            f"발표: {issued:%Y-%m-%d %H:%M} KST | "
            f"예보 격자: {grid_xy[0]}, {grid_xy[1]}"
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

        temp = (
            future["TMP"].dropna()
            if "TMP" in future
            else pd.Series(dtype=float)
        )

        hum = (
            future["REH"].dropna()
            if "REH" in future
            else pd.Series(dtype=float)
        )

        highs = (
            future["TMX"].dropna()
            if "TMX" in future
            else pd.Series(dtype=float)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "다음 시간대 예보 기온",
            f"{temp.iloc[0]:.1f}°C"
            if len(temp) else "자료 없음"
        )

        c2.metric(
            "예보 기간 최고기온",
            f"{highs.max():.1f}°C"
            if len(highs) else "자료 없음"
        )

        c3.metric(
            "다음 시간대 습도",
            f"{hum.iloc[0]:.0f}%"
            if len(hum) else "자료 없음"
        )

        if len(temp):
            st.line_chart(
                temp.rename("기온(°C)")
            )

    except Exception:
        st.error(
            "기상청 예보 조회에 실패했습니다. "
            "API 설정 또는 서버 연결을 확인하세요."
        )
else:
    st.warning(
        "KMA_FORECAST_KEY가 등록되지 않았습니다."
    )

st.markdown(
    "[기상청 공식 예보 확인]"
    "(https://www.weather.go.kr/)"
)


# =========================================================
# 2. AI 일주일 예측
# =========================================================

st.divider()
st.header("2. AI 일주일 최고기온 예측")

st.caption(
    "오늘을 포함한 7일간의 최고기온을 AI로 추정합니다. "
    "최신 관측자료가 필요하며 공식 예보와 구분됩니다."
)

data = None
source = None

historical_path = (
    Path(__file__).parent
    / "data"
    / "asos_143.csv"
)

if historical_path.exists():
    try:
        data = load_csv(
            historical_path.read_bytes()
        )
        source = "저장소 ASOS CSV"
    except Exception:
        st.warning(
            "저장된 ASOS CSV를 읽지 못했습니다."
        )

if data is None and ASOS_KEY:
    try:
        data = load_historical(ASOS_KEY)
        source = "공공데이터포털 ASOS API"
    except Exception:
        st.warning(
            "과거 ASOS 관측자료 수집에 실패했습니다."
        )


# 최신 연도 관측자료 갱신
if data is not None and ASOS_KEY:
    try:
        today = pd.Timestamp.now(
            tz="Asia/Seoul"
        ).date()

        yesterday = today - timedelta(days=1)
        last_year = data.index.max().year

        recent = []

        for year in range(
            last_year + 1,
            yesterday.year + 1
        ):
            finish = min(
                yesterday,
                date(year, 12, 31)
            )

            new_data = load_recent(
                ASOS_KEY,
                year,
                finish.isoformat()
            )

            recent.append(new_data)

        if yesterday.year <= last_year:
            recent.append(
                load_recent(
                    ASOS_KEY,
                    yesterday.year,
                    yesterday.isoformat()
                )
            )

        if recent:
            data = pd.concat(
                [data] + recent
            ).sort_index()

            data = data[
                ~data.index.duplicated(
                    keep="last"
                )
            ]

            data = data.asfreq("D")
            source += " + 최신 관측자료"

    except Exception:
        st.warning(
            "최신 관측자료 갱신에 실패했습니다. "
            "AI 예측 날짜가 제한될 수 있습니다."
        )


if data is None:
    st.error(
        "AI 학습용 ASOS 자료가 없습니다."
    )

else:
    st.success(
        f"자료 연결: {source}"
    )

    st.caption(
        f"관측자료 기간: "
        f"{data.index.min().date()} ~ "
        f"{data.index.max().date()}"
    )

    try:
        result = train_model(data)

        st.subheader("AI 모델 검증")

        st.dataframe(
            result["metrics"],
            hide_index=True,
            use_container_width=True
        )

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
                    f"{selected} 실제 기온과 "
                    "예측 기온 비교"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # 연도별 33°C 이상 일수
        yearly = pd.DataFrame({
            "연도": data.index.year,
            "폭염일": (
                data["tmax"] >= 33
            ).fillna(False).astype(int).to_numpy()
        })

        yearly = (
            yearly
            .groupby("연도", as_index=False)["폭염일"]
            .sum()
            .rename(
                columns={
                    "폭염일": "33°C 이상 일수"
                }
            )
        )

        st.plotly_chart(
            px.bar(
                yearly,
                x="연도",
                y="33°C 이상 일수",
                title="연도별 33°C 이상 일수"
            ),
            use_container_width=True
        )

        st.subheader("오늘부터 7일 AI 예측")

        try:
            week = predict_seven_days(result)

            st.caption(
                "마지막 실측일: "
                f"{week.iloc[0]['관측 기반 마지막 날짜']}"
            )

            st.plotly_chart(
                px.line(
                    week,
                    x="날짜",
                    y="AI 예상 최고기온(°C)",
                    markers=True,
                    title="AI 7일 최고기온 예측"
                ),
                use_container_width=True
            )

            st.dataframe(
                week[
                    [
                        "날짜",
                        "AI 예상 최고기온(°C)"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "미래 날짜의 AI 예측을 다음 날짜의 "
                "입력값으로 다시 사용하므로 "
                "예측 기간이 길어질수록 오차가 "
                "누적될 수 있습니다."
            )

            st.download_button(
                "AI 7일 예측 CSV 다운로드",
                week.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="ai_7day.csv",
                mime="text/csv"
            )

        except ValueError as exc:
            st.warning(str(exc))

    except Exception as exc:
        st.error(
            "AI 모델 실행 오류: "
            f"{type(exc).__name__}"
        )


# =========================================================
# 3. 폭염 위험 안내
# =========================================================

st.divider()
st.header("3. 폭염 위험 안내")

if forecast is not None and "TMP" in forecast:
    now = (
        pd.Timestamp.now(tz="Asia/Seoul")
        .tz_localize(None)
        .floor("h")
    )

    upcoming = forecast[
        forecast.index >= now
    ]

    max_temp = (
        upcoming["TMP"].max()
        if len(upcoming)
        else float("nan")
    )

    if pd.notna(max_temp):
        if max_temp >= 33:
            st.error(
                "예보에 33°C 이상 시간대가 있습니다. "
                "시원한 장소에서 충분히 쉬세요."
            )
        elif max_temp >= 30:
            st.warning(
                "예보에 30°C 이상 시간대가 있습니다. "
                "수분 섭취와 휴식에 신경 쓰세요."
            )
        else:
            st.info(
                "조회된 예보의 최고 시간대 기온은 "
                "30°C 미만입니다."
            )
    else:
        st.info("현재 이후의 기온 예보가 없습니다.")
else:
    st.info(
        "기상청 예보가 연결되면 위험 안내가 표시됩니다."
    )

st.markdown(
    """
    **폭염 대비 행동요령**

    - 물을 자주 마시기
    - 시원한 장소에서 충분히 쉬기
    - 더운 시간대 야외활동 줄이기
    - 고령자와 거동이 불편한 이웃의 안전 확인
    - 위급한 상황에서는 119 신고
    """
)


# =========================================================
# 4. 무더위쉼터 지도
# =========================================================

st.divider()
st.header("4. 무더위쉼터 지도 · 내 위치")

st.caption(
    "주황색: 공식 쉼터 | 파란색: 사용자 위치"
)

st.markdown(
    "[전국무더위쉼터표준데이터 출처]"
    "(https://www.data.go.kr/data/15013199/standard.do)"
)

st.write(
    "아래 위치 버튼을 누르고 브라우저의 "
    "위치 권한을 허용하면 내 위치를 표시합니다."
)

location = streamlit_geolocation()

user_lat = None
user_lon = None

if (
    isinstance(location, dict)
    and location.get("latitude") is not None
    and location.get("longitude") is not None
):
    try:
        user_lat = float(location["latitude"])
        user_lon = float(location["longitude"])

        if not (
            -90 <= user_lat <= 90
            and -180 <= user_lon <= 180
        ):
            user_lat = None
            user_lon = None

    except (TypeError, ValueError):
        user_lat = None
        user_lon = None


# ---------------------------------------------------------
# 공식 쉼터 CSV 자동 연결
# ---------------------------------------------------------

shelter_path = (
    Path(__file__).parent
    / "data"
    / "shelters.csv"
)

shelters = None

if shelter_path.exists():
    try:
        shelters = parse_shelters(
            shelter_path.read_bytes()
        )
    except Exception:
        st.warning(
            "쉼터 CSV 형식 또는 좌표를 확인하세요."
        )


# 운영자 시험 업로드
if shelters is None:
    with st.expander("운영자용 공식 쉼터 CSV 업로드"):
        uploaded = st.file_uploader(
            "공식 쉼터 CSV",
            type=["csv"],
            key="shelter_csv"
        )

        if uploaded is not None:
            try:
                shelters = parse_shelters(
                    uploaded.getvalue()
                )
            except ValueError as exc:
                st.warning(str(exc))


# ---------------------------------------------------------
# 지도 중심 및 쉼터 정보
# ---------------------------------------------------------

if user_lat is not None:
    center_lat = user_lat
    center_lon = user_lon

    st.success(
        "사용자 위치가 확인되었습니다."
    )
else:
    center_lat = 35.879
    center_lon = 128.602

    st.info(
        "위치 권한을 허용하지 않아 "
        "칠성동 인근을 중심으로 표시합니다."
    )


if shelters is not None and not shelters.empty:

    st.success(
        f"공식 쉼터 {len(shelters):,}개 연결됨"
    )

    if user_lat is not None:

        shelters = shelters.copy()

        shelters["직선거리(km)"] = (
            distances_km(
                shelters,
                user_lat,
                user_lon
            ).round(2)
        )

        shelters = shelters.sort_values(
            "직선거리(km)"
        )

        st.subheader("가까운 무더위쉼터")

        st.dataframe(
            shelters[
                [
                    "이름",
                    "주소",
                    "운영정보",
                    "직선거리(km)"
                ]
            ].head(30),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "거리는 직선거리이며 실제 도보 이동거리와 다릅니다."
        )

    else:
        st.dataframe(
            shelters[
                ["이름", "주소", "운영정보"]
            ].head(40),
            hide_index=True,
            use_container_width=True
        )

else:
    st.warning(
        "공식 쉼터 CSV가 아직 연결되지 않았습니다. "
        "쉼터 위치를 임의로 표시하지 않습니다."
    )


# ---------------------------------------------------------
# PyDeck 지도 마커
# ---------------------------------------------------------

layers = []

# 주황색 쉼터: 5~9픽셀
if shelters is not None and not shelters.empty:

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=shelters,
            get_position="[lon, lat]",
            get_fill_color="[235, 96, 63, 220]",
            get_radius=7,
            radius_min_pixels=5,
            radius_max_pixels=9,
            pickable=True
        )
    )


# 파란색 사용자 위치: 6~10픽셀
if user_lat is not None:

    user_point = pd.DataFrame([
        {
            "lat": user_lat,
            "lon": user_lon,
            "이름": "내 위치",
            "주소": "사용자 기기 위치",
            "운영정보": ""
        }
    ])

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=user_point,
            get_position="[lon, lat]",
            get_fill_color="[30, 110, 240, 240]",
            get_radius=8,
            radius_min_pixels=6,
            radius_max_pixels=10,
            pickable=True
        )
    )


view = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=13.5,
    pitch=0
)

deck = pdk.Deck(
    map_style=None,
    initial_view_state=view,
    layers=layers,
    tooltip={
        "text": "{이름}\n{주소}\n{운영정보}"
    }
)

st.pydeck_chart(
    deck,
    use_container_width=True
)

st.caption(
    "쉼터 운영 여부와 이용 가능 시간은 "
    "방문 전 지자체에 확인하세요. "
    "위치 정확도는 기기와 브라우저에 따라 달라집니다."
)


# =========================================================
# 5. 지역사회 정책 제안
# =========================================================

st.divider()
st.header("5. 지역사회 정책 제안")

st.markdown(
    """
    ### 폭염 취약계층 보호

    고령자와 거동이 불편한 주민을 대상으로
    폭염 시 안부 확인과 방문 돌봄을 강화한다.

    ### 무더위쉼터 접근성 개선

    쉼터 위치와 접근성을 분석하여
    이동 지원이 필요한 지역을 파악한다.

    ### AI 기반 폭염 대응

    기상 관측자료와 AI 예측을 활용해
    폭염 발생 특성을 분석하고,
    향후 인구·녹지·건물 자료와 결합하여
    지역별 취약도를 연구한다.
    """
)

st.divider()

st.caption(
    "자료 출처: 기상청 공식 단기예보, "
    "기상청 ASOS 일자료, "
    "공공데이터포털 전국무더위쉼터표준데이터"
)
