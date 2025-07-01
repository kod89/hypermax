# Batch Defect Prediction App

SK케미칼 의약품 공정관리 인턴 포트폴리오용 웹 앱입니다. 이 애플리케이션은 생산 공정 데이터를 분석하여 불량 여부를 예측하고 주요 영향을 미치는 변수들을 시각화합니다.

## 📂 기능
- CSV 업로드를 통한 배치 단위 불량 예측
- 예측 결과 및 불량 가능성(%) 시각화
- 주요 변수 중요도 그래프 제공
- CSV로 결과 다운로드 가능

## ▶️ 실행 방법 (로컬)
```bash
pip install -r requirements.txt
streamlit run kodaehong_batch_defect_predictor.py
```

## 🌐 웹 배포 (Streamlit Cloud)
1. GitHub에 이 파일들과 함께 업로드
2. [Streamlit Cloud](https://streamlit.io/cloud) 접속 → GitHub 연동 → 저장소 선택 후 배포
3. 생성된 URL을 포트폴리오로 사용 가능

## 📑 예시 CSV 형식
| Temperature_C | Pressure_bar | MixingSpeed_rpm | pH | Yield_percent | Contaminant_ppm |
|---------------|--------------|------------------|----|----------------|------------------|
| 25.3          | 1.02         | 118              | 7.0 | 94.5           | 0.5              |
