# v4: 최고·최저기온 예측 정확도 + 전국 시설 지도

- daily_monitor.py: 신규 발행 예측에 최고기온·최저기온 모두 기록하고 실제 관측값과 비교
- app.py: 최고/최저 선택, 1~7일차 선택, MAE·그래프·CSV
- 기존 data/prediction_history.csv는 자동으로 새로운 열을 추가하며 유지
- 전국 시설: data/shelters.csv(무더위쉼터), cold_shelters.csv(한파쉼터), evacuation.csv(대피시설), hospitals.csv(응급의료기관), safety_facilities.csv(기타 안전시설)
- 각 시설 CSV는 시설명·위도·경도 필요. 전국 시설 데이터는 이 ZIP에 포함되어 있지 않으며, 공식 데이터 추가 전에는 기존 등록 시설만 표시됨
- 적용: app.py, daily_monitor.py 교체. 기존 facilities.py, shelters.py 유지
- GitHub Actions train.yml에서 python daily_monitor.py 실행되는지 확인
- 실제 예측 오차는 새로 기록된 미래 예측이 실측과 비교된 후 계산됨. 과거 누락 예측은 생성하지 않음
