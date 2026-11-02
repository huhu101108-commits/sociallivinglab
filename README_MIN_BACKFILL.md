# 최저기온 AI 학습자료 복구

기존 CSV에 최저기온이 155일만 저장된 문제를 해결합니다. `auto_train.py`가 2006년 이후 각 연도의 최저기온 자료 보유율을 검사하고, 90% 미만인 연도를 공공데이터포털 ASOS 일자료 API에서 다시 수집합니다. 기존 최고기온 관측값은 보존하고 누락된 최저기온을 보충합니다.

## 적용
1. GitHub에서 `auto_train.py`를 교체합니다. 기존 `.github/workflows/train.yml`은 그대로 사용합니다.
2. GitHub Actions → AI daily improvement → Run workflow를 실행합니다.
3. `Backfilling YYYY minimum temperature` 및 `Available tmin observations` 로그를 확인합니다.
4. 성공하면 Actions가 `data/asos_143.csv`, `artifacts/min_model.joblib`, `artifacts/min_manifest.json`을 커밋합니다.
5. Streamlit 웹사이트를 새로고침합니다.

## 주의
- 기상청 API에서 `minTa` 값이 실제로 제공되어야 합니다. 응답에 값이 없다면 오류로 원인을 알려줍니다.
- 연도별 재수집이 발생하는 최초 실행은 평소보다 오래 걸릴 수 있습니다.
- 과거 최저기온 자료가 충분해도 실제 예측 정확도는 별도 검증해야 합니다.
- 이 ZIP에는 실제 ASOS API 인증키나 학습된 모델이 포함되어 있지 않습니다.
