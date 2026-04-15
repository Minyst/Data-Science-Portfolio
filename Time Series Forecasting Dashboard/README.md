---
title: 시계열 예측 대시보드
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 시계열 예측 스트림릿 웹 애플리케이션 과제

## 프로젝트 개요
임의의 단변량 시계열 CSV 파일을 업로드하면, 다양한 시계열 알고리즘의 예측 결과를 비교·평가할 수 있는 동적 대시보드입니다.

**지원 모델**: Naive, SMA, ExpSmoothing, Holt, HoltWinters, STL, ARIMA, AutoARIMA, Theta, Prophet
**평가 지표**: MSE, RMSE, MAE, MAPE, MASE, MdRAE (정확도) + RSFE, TS (편향 진단)
**예측 모드**: 단일 학습 / Expanding Window / Rolling Window

## 파일 구성
- `app.py` — Streamlit 웹 어플리케이션 메인 코드
- `requirements.txt` — 의존 패키지 목록
- `.streamlit/config.toml` — Streamlit UI 설정

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 사용 방법
1. 좌측 사이드바에서 CSV 업로드 (날짜 컬럼 + 숫자 컬럼 1개 이상 포함)
2. 예측 horizon, 모델, 하이퍼파라미터 설정
3. "학습 실행" 버튼 클릭 → 모델별 예측·평가 결과 확인
