# Multivariate Time Series Anomaly Detection

다변량 시계열 CSV를 업로드하면 **예측모델 × Scorer 격자**로 이상치를 탐지하는 Streamlit 대시보드입니다.
`darts`의 `ForecastingAnomalyModel` 기반이며, 0/1 정답 라벨을 기준으로 탐지 결과를 정량 평가합니다.

## Features
- **이상탐지(지도식 평가)** — 0/1 이상 라벨 기준 AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy 평가
- **백테스팅·잔차분석** — 예측 vs 실제·잔차 분포로 이상 점수의 원천(예측오차)을 점검 (잔차가 큰 구간이 이상 후보)

## Models
- **ML**: LinearRegression, RandomForest, XGBoost, LightGBM, CatBoost
- **DL**: RNN, LSTM, GRU, BlockRNN, Transformer, NBEATS, NHiTS

## Scorers
Norm · KMeans · Wasserstein

## Tech Stack
Python · Streamlit · darts · scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch

## Run Locally
```bash
docker build -t ts-anomaly .
docker run -p 8501:8501 ts-anomaly
# http://localhost:8501
```

## Usage
1. 사이드바에서 시계열 CSV 업로드 (시간 컬럼 + 숫자 변수들 + 0/1 이상 라벨 컬럼)
2. 정답 라벨 컬럼(0/1) 지정
3. 설정 탭에서 예측모델(ML 5종 / DL 7종)과 Scorer(Norm/KMeans/Wasserstein) 선택
4. 탐지 결과 · 성능 비교 탭에서 결과 확인
