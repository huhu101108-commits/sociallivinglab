# v7 전국 한파쉼터 공식 API 연동

- 기존 v6 전체 프로젝트에서 최고·최저기온 예측, 오차 분석, 무더위쉼터 API를 유지했습니다.
- 한파쉼터 공식 데이터: 행정안전부 재난안전데이터공유플랫폼 `DSSP-IF-10804` (https://www.safetydata.go.kr/disaster-data/view?dataSn=966)
- 공식 응답 필드: `REARE_NM`(쉼터명), `LAT`(위도), `LOT`(경도), `RONA_DADDR`(도로명상세주소), `DADDR`(상세주소), `REARE_FCLT_NO`(시설번호). 
- 별도 인증키: 한파쉼터 이용신청 승인 후 Streamlit Cloud > App settings > Secrets에 다음을 추가하세요.

```toml
SAFETYDATA_COLD_SERVICE_KEY = "한파쉼터_전용_발급키"
```

- 기존 `SAFETYDATA_SERVICE_KEY`(무더위쉼터)는 삭제하지 마세요.
- GitHub 루트에 ZIP 내부 전체 파일을 덮어쓰고 Commit 후 Streamlit 앱을 Reboot 하세요.
- `쉼터 API 처리 코드 v7` 표시를 확인하세요.
- 데이터는 최대 24시간 캐시합니다. 지도에서 파란색 마커가 한파쉼터입니다. 종류별 표시를 선택할 수 있습니다.
- `data/cold_shelters.csv`를 자동으로 사용하지 않으며 공식 API를 우선합니다.
- 인증키가 없으므로 실제 인증 API의 실시간 호출 성공 여부는 검증하지 못했습니다. 로컬 모의 응답과 문법 검사는 통과해야 합니다.
- API 응답의 시설 운영시간은 실시간 개방 보장이 아닙니다. 방문 전 확인하세요.
