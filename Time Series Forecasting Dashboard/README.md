# Time Series Forecasting Dashboard

임의의 단변량 시계열 CSV를 업로드하면 다양한 예측 알고리즘의 결과를 비교·평가할 수 있는 인터랙티브 Streamlit 대시보드입니다.

## Features
- **다중 모델 비교**: 한 번의 실행으로 여러 예측 모델을 동시에 학습·비교
- **정량 평가**: 성능 지표와 편향 진단 지표를 함께 제공
- **유연한 예측 모드**: 단일 학습 / Expanding Window / Rolling Window 백테스팅

## Models
Naive, SMA, ExpSmoothing, Holt, Holt-Winters, STL, AutoARIMA, Theta, Prophet

## Metrics
- **성능**: MSE, RMSE, MAE, MAPE, MASE, MdRAE
- **편향 진단**: RSFE, Tracking Signal (TS)

## Tech Stack
Python · Streamlit · sktime · statsmodels · pmdarima · Prophet · Plotly

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

또는 Docker로 실행:
```bash
docker build -t ts-forecasting .
docker run -p 7860:7860 ts-forecasting
# http://localhost:7860
```

## Usage
1. 좌측 사이드바에서 CSV 업로드 (날짜 컬럼 + 숫자 컬럼 1개 이상)
2. 예측 horizon, 모델, 하이퍼파라미터 설정
3. 학습 실행 후 모델별 예측·평가 결과 확인
