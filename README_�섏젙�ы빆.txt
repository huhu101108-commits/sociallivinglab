칠성동 AI 폭염 안전지킴이 - CPU 제한 완화판

1. model.py: 랜덤포레스트 반복 학습을 제거하고 HistGradientBoosting 70회 반복, 단일 연도 시간순 검증 + 전체 학습으로 변경.
2. app.py: AI 모델을 st.cache_resource에 보관하여 동일 관측자료 재실행 때 학습하지 않음. 최신 관측자료가 변경되면 재학습함.
3. 위치 확인은 선택 체크박스를 눌렀을 때만 실행해 브라우저 연결 오류 가능성을 줄임.
4. data/shelters.csv: 이전에 제공한 북구 쉼터 130곳 데이터를 포함.
5. Streamlit Secrets의 KMA_FORECAST_KEY, ASOS_SERVICE_KEY는 기존 설정을 그대로 유지. ZIP에 키를 넣지 않음.

GitHub 저장소의 같은 경로에 ZIP 안 파일을 업로드하세요. ZIP 자체를 업로드하면 앱에 반영되지 않습니다.
주의: 최초 학습과 신규 관측자료 갱신 때 CPU가 사용됩니다. CPU 제한이 완전히 사라진다는 보장은 없습니다.
