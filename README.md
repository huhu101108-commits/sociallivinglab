# 칠성동 AI 폭염 안전지킴이 - 자동 인증키 버전

기존 `sociallivinglab` 저장소의 `app.py`, `model.py`, `forecast.py`, `kma_api.py`, `requirements.txt`를 교체하세요.
GitHub 저장소 루트에 각 파일을 직접 올리세요. ZIP 자체를 올리면 실행되지 않습니다.

Streamlit Cloud > App > Settings > Secrets:
```toml
KMA_FORECAST_KEY = "기상청 API허브 인증키"
ASOS_SERVICE_KEY = "공공데이터포털 ASOS 일자료 일반 인증키"
```
키는 GitHub에 올리지 마세요. 공개 오류 화면에 인증키나 요청 URL을 출력하지 않습니다.

앱에 접속하면 공식 예보를 자동 조회하고, 실제 ASOS 자료가 수집되면 모델을 자동 학습합니다.
20년치 API 요청은 최초 실행 시 시간이 걸릴 수 있고, 이후 24시간 캐시합니다.
대구 ASOS 143 관측소는 칠성동 실측이 아닙니다. 2006~2025 자료로 추정한 2026-01-01 최고기온은 현재 날씨 예보가 아닙니다.
쉼터 지도는 `data/shelters.csv`에 공식 자료가 있을 때만 표시합니다.
