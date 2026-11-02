# v12 — 사용자 화면 개선 및 시간별 공식 예보 비교

- 기존 v11의 모든 Python 모듈과 GitHub Actions 설정 유지.
- 페이지와 탭, 섹션 제목, 버튼·설명에서 그림문자 제거. 단정한 3개 탭 유지.
- 모바일/태블릿 가독성을 고려한 여백·지표 카드·탭 CSS 적용.
- 첫 화면에 간단한 이용 방법 추가.
- 시간당 강수 AI 실행 시 동일 시각의 기상청 격자 예보와 AI 관측소 예측을 나란히 표시하고 CSV 다운로드.
- 공식 강수량이 `1mm 미만` 같은 범위 표현이면 정량 강수량으로 변환하지 않음.
- 실제 관측이 없는 비교는 정확도 비교가 아니며, 관측소와 격자 공간 차이 주의.

## 설정
기존 Streamlit Secrets `KMA_FORECAST_KEY`, `KMA_ASOS_KEY`(또는 `ASOS_SERVICE_KEY`), `SAFETYDATA_SERVICE_KEY`, `SAFETYDATA_COLD_SERVICE_KEY` 유지.

## 배포
ZIP을 풀어 기존 GitHub 저장소 파일을 덮어쓰고 커밋. Streamlit 앱 재시작.
