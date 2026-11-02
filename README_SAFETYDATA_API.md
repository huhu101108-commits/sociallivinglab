# 행정안전부 전국 무더위쉼터 API 연동

- API: https://www.safetydata.go.kr/V2/api/DSSP-IF-10942
- Streamlit Cloud > App settings > Secrets 에 `SAFETYDATA_SERVICE_KEY = "발급받은_새_키"` 설정
- 이전 data/shelters.csv 파일은 제거했고 무더위쉼터는 API에서만 불러옵니다.
- API 페이지별 요청 후 좌표가 있는 시설만 표시합니다. 하루 동안 Streamlit 캐시를 사용합니다.
- 인증키로 실제 API 응답을 확인하지 못했으므로 필드명에 따라 `safetydata_api.py`의 FIELDS 수정이 필요할 수 있습니다.
- API가 실패하면 오류를 표시하며 과거 CSV를 대체 데이터로 사용하지 않습니다.
- 최고/최저기온 AI와 예측 정확도 기록 기능은 이전 버전 그대로 유지합니다.
- 다른 종류 시설은 기존 별도 CSV 파일이 있으면 표시합니다.
- 보안: 기존 화면에 노출된 키는 재발급 권장.
