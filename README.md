# 칠성동 AI 폭염 안전지킴이 — 웹 배포판

대구 북구 칠성동 대상 연구용 Streamlit 웹사이트입니다. **대구 ASOS 143 관측값은 칠성동 실측값이 아닙니다.** 관측 자료가 없으면 예측 수치를 만들어 보여주지 않습니다.

## 누구나 접속할 수 있는 웹주소 만들기 (Streamlit Community Cloud)
1. GitHub에서 새 공개 저장소를 만듭니다. 이 폴더의 `app.py`, `model.py`, `kma_api.py`, `requirements.txt`, `.streamlit/config.toml`을 업로드합니다.
2. https://share.streamlit.io 에서 GitHub로 로그인하고 **Create app**을 누릅니다.
3. 저장소와 브랜치를 선택하고 **Main file path**를 `app.py`로 설정한 후 **Deploy**를 누릅니다.
4. 생성된 `https://...streamlit.app` 주소를 공유하면 휴대폰·태블릿·PC에서 사용할 수 있습니다.
5. 기상청 CSV를 업로드하거나 공공데이터포털 서비스키로 2006~2025 자료를 불러옵니다. 키를 공개 GitHub 저장소에 올리지 마세요.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터와 한계
- 대구 ASOS 일자료 (지점번호 143), 2006~2025년 학습 권장.
- 학습 검증은 시간 순서대로 수행하며, 모델의 33°C 이상 예측 능력을 비교합니다.
- 데이터 마지막 날짜 다음 날의 통계적 추정값만 산출하며, **현재 실시간 예보나 의료 판단을 대체하지 않습니다.**
- 무더위쉼터 정보는 검증된 CSV를 올려야 표시됩니다.
- 칠성동 블록별 지도 예측은 현장 센서나 공간자료 확보 전에는 제공하지 않습니다.

## 3.0 공식 예보 자동 표시 설정

1. [공공데이터포털](https://www.data.go.kr/)에서 '기상청_단기예보 조회서비스' 활용 신청 후 일반 인증키(Decoding)를 확인합니다.
2. Streamlit Community Cloud → 해당 앱 → Settings → Secrets에 다음을 입력합니다. 인증키를 GitHub에 올리지 마세요.

```toml
KMA_FORECAST_KEY = "발급받은_일반_인증키"
```

3. GitHub 저장소 최상위에 `app.py`, `model.py`, `kma_api.py`, `forecast.py`, `requirements.txt`를 업로드한 뒤 재배포합니다.
4. 공개 사이트의 공식 단기예보 섹션에 데이터가 나타나는지 확인합니다. API 승인 상태/발표 시각에 따라 일시적으로 실패할 수 있습니다.

이 기능은 **기상청 공식 예보를 자동으로 보여주는 기능**이지, AI 모델의 실시간 자동 재학습 기능은 아닙니다. 기존 AI 모델의 학습/검증에는 별도로 ASOS CSV 업로드 또는 ASOS API 인증키가 필요합니다. 방문자가 데이터 없이도 공식 예보를 볼 수 있지만, AI 자체 예측을 즉시 표시하려면 관리자가 검증된 관측자료를 사전 제공하는 추가 작업이 필요합니다.
