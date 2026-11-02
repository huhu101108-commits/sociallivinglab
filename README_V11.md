# v11: 전국 관측소 기반 시간당 강수 AI (실험)

v10 파일을 그대로 보존하고 `hourly_rain_ai.py`와 날씨 탭의 접이식 섹션을 추가했습니다. 기존 3개 탭, 전국 7일 기온·강수 AI, 대구 기존 AI, 기상청 예보, 전국 무더위/한파쉼터 지도, 내 위치 노란색 마커, 자동 학습 코드 모두 유지됩니다.

- 기상청 ASOS **시간자료 조회서비스**: `https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList`
- 요청: `ServiceKey`, `pageNo`, `numOfRows`, `dataType=JSON`, `dataCd=ASOS`, `dateCd=HR`, `startDt`, `startHh`, `endDt`, `endHh`, `stnIds`
- 응답 필드: `tm`(관측시각), `rn`(시간강수량), `ta`(기온), `hm`(습도), `ws`(풍속), `pa`(현지기압).
- 최근 90일 시간자료, 1/3/6시간 후 **각 시각의 1시간 강수량** 및 강수 발생 확률 직접 모델. 80/20 시간순 분리, 경계 6시간 제외, Brier 점수·MAE 검증.
- 자료 결측/지연, 부족한 학습 사례는 예측을 중단. 10mm/h·20mm/h 이상 월별 과거 발생 횟수는 관측 통계이며 예측값이 아닙니다.
- 전국 모든 지점이 아닌 기존 22개 ASOS 관측소에 연결된 지역에서만 실행. 선택 지역 전체·도로 단위의 예측이 아닙니다.
- 기존 Streamlit Secrets의 `KMA_ASOS_KEY` 또는 `ASOS_SERVICE_KEY` 사용. 해당 **ASOS 시간자료 API 활용신청 권한**이 별도로 필요할 수 있습니다.
- Streamlit 앱의 날씨 예측 탭에서 `v11 전국 관측소 기반 시간당 강수 AI` 확장 영역을 열고 버튼을 누릅니다.
- 실측 인증키를 이용한 실제 API 호출/배포 테스트는 미실시. 공식 예보나 호우특보 대체 불가. 기상청 예보보다 정확하다는 주장은 검증하지 않았습니다.
