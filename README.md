# 🎓 Data Scientist Portfolio

## 🏆 Flagship — Real-Time On-Device Semantic Segmentation System for Recycling Waste Sorting

> 📄 **KCI 등재 논문** — 한국산업경영시스템학회지(JKSIE) Vol.49, No.1, pp.10–20 (2026.03)
> *"실시간 온디바이스 재활용품 분리배출 세그먼테이션 시스템"*, 김민준·하정훈
> DOI: [10.11627/jksie.2026.49.1.010](https://doi.org/10.11627/jksie.2026.49.1.010)

**[▶ 프로젝트 폴더 바로가기](./Real-Time%20On-Device%20Semantic%20Segmentation%20System%20for%20Recycling%20Waste%20Sorting)**

스마트폰으로 재활용품을 촬영하면 **서버 없이 기기 내부에서** 캔·유리·종이·플라스틱·비닐을 픽셀 단위로 구분해
카메라 화면 위에 반투명 마스크로 오버레이해주는 온디바이스 AI 시스템입니다.

### 무엇을 직접 했는가

- **데이터를 발로 뛰며 직접 수집** — 아파트 분리수거장, 편의점, 대형마트 등 실제 배출 현장을 돌아다니며 재활용품 이미지 200장을 직접 촬영·수집
- **픽셀 단위 라벨링 직접 수행** — 재활용품용 공개 마스크 데이터셋이 없어, Roboflow에서 Polygon 방식으로 5개 클래스를 한 장 한 장 직접 라벨링
- **데이터 증강으로 150장 → 750장 확장** — Albumentations 확률적 파이프라인(HorizontalFlip·Rotate ±10°·ColorJitter)으로 증강, 증강 전후 통제 실험으로 **Test Dice 0.8673 → 0.9402 (+8.41%)** 개선을 정량 입증
- **경량 모델 파인튜닝** — DeepLabV3 + MobileViT-x-small(**2.9M 파라미터**)을 직접 구축한 데이터셋으로 재학습. Dice(0.6) + 가중 Cross-Entropy(0.4) 복합 손실, 클래스 픽셀 수 역비례 가중치, EMA·AMP·Warmup+Cosine 스케줄까지 학습 파이프라인 전체 설계
- **아키텍처 비교 실험** — 동일 데이터·동일 파라미터 규모에서 YOLO11n-seg와 정면 비교(Dice 0.9402 vs 0.7793) 후 최종 모델 선정. 이를 위해 semantic mask → YOLO polygon 포맷 변환기도 OpenCV contour 기반으로 직접 구현
- **온디바이스 배포까지 완주** — PyTorch → **ONNX 변환** → Flutter 앱에 모델 내장, 네이티브 플랫폼 채널로 ONNX Runtime(C++) 호출. Galaxy Tab S7 **CPU만으로** 촬영→오버레이 전 과정 추론

### 핵심 결과

| 항목 | 수치 |
|---|---|
| Test Dice Score | **0.9402** (mIoU 0.8884, Pixel Acc 0.9878) |
| 증강 전후 개선 | 0.8673 → 0.9402 (**+8.41%**) |
| vs YOLO11n-seg | Dice 0.9402 vs 0.7793 |
| 모델 크기 | 2.9M 파라미터 (MobileViT-x-small 백본) |

### 왜 온디바이스인가 (이 프로젝트의 차별점)

- **네트워크가 필요 없다** — 모델이 기기 안에서 실행되므로 Wi-Fi가 안 터지는 지하주차장에서도, 야외 어디서든 동작
- **개인정보 보호** — 촬영 이미지가 서버로 전송되지 않아 유출 위험 원천 차단
- **플랫폼 비종속** — 특정 벤더 가속기(Core ML, NNAPI 등)에 묶이지 않는 ONNX Runtime 기반이라 중저가 안드로이드 기기까지 폭넓게 지원
- **작은 모델로 실용 성능** — 150장이라는 극소 데이터와 2.9M 경량 모델의 제약 속에서 데이터 증강·복합 손실·클래스 가중치 설계로 Dice 0.94를 달성

---

## 📈 Time Series

### [Multivariate Time Series Anomaly Detection](./Multivariate%20Time%20Series%20Anomaly%20Detection)

다변량 시계열 CSV를 업로드하면 **예측모델 12종 × Scorer 3종 격자**로 이상탐지를 수행·비교하는 Streamlit 대시보드.

- **접근법**: `darts` ForecastingAnomalyModel — 예측 잔차(forecast error)를 이상 점수로 변환하는 방식. ML 5종(LinearRegression·RandomForest·XGBoost·LightGBM·CatBoost) + DL 7종(RNN·LSTM·GRU·BlockRNN·Transformer·NBEATS·NHiTS)을 Norm·KMeans·Wasserstein 3가지 Scorer와 조합해 총 36개 구성을 한 번에 평가
- **자동화**: 시간 컬럼·라벨 컬럼·샘플링 주기·계절성을 자동 감지, 주기에 맞는 lag와 cyclic datetime encoder 자동 설정
- **엄밀한 평가**: 데이터 누수 없는 rolling 1-step 백테스트, AUC-ROC/AUC-PR/F1/Precision/Recall 정량 비교, 잔차 분석 시각화
- **설계**: 탐지 로직(`anomaly_core.py`)과 UI(`app.py`)를 완전 분리한 재사용 가능한 구조, Docker로 Hugging Face Spaces 배포
- **Stack**: Python · darts · PyTorch · Streamlit · Plotly · Docker

### [Time Series Forecasting Dashboard](./Time%20Series%20Forecasting%20Dashboard)

단변량 시계열 CSV를 업로드하면 **9개 예측 모델을 3가지 방식으로 학습·비교**하는 인터랙티브 대시보드.

- **모델 9종**: Naive · SMA · Exponential Smoothing · Holt · Holt-Winters · STL · AutoARIMA · Theta · Prophet (sktime + Prophet)
- **3가지 예측 방식**: 단일 학습 예측뿐 아니라 **Expanding Window / Rolling Window 백테스트**까지 구현해 실제 운영 상황에 가까운 평가 제공
- **예측 이론에 충실한 평가**: 성능 6종(MSE·RMSE·MAE·MAPE·MASE·MdRAE)에 더해 **RSFE·Tracking Signal로 예측 편향 진단**, Ljung-Box·Jarque-Bera 잔차 검정, AIC/BIC 정보량 기준까지 — 교과서(FPP3) 수준의 진단 체계를 대시보드로 구현
- **UX 엔지니어링**: 계절분해·ACF/PACF 등 Plotly 차트 15종, 모델별 하이퍼파라미터 조절 패널, 학습 진행률·ETA 실시간 표시, 순위표가 포함된 자동 분석 리포트 생성
- **Stack**: Python · Streamlit · sktime · Prophet · statsmodels · Plotly · Docker (Hugging Face Spaces)

---

## 💻 Web Application

### [Coding Test Prep Platform](./Coding%20Test%20Prep%20Platform)

브라우저 안에서 Python을 직접 실행하고 자동 채점까지 해주는 코딩테스트 학습 웹앱. **직접 기획·개발·배포·사용까지** 하고 있는 서비스.

- **서버 없는 코드 실행**: **Pyodide(WebAssembly Python)** 를 브라우저에서 구동해 별도 코드 실행 서버·샌드박스 없이 안전하게 Python 실행 — 서버 비용 0원으로 채점 시스템 구현
- **자동 채점 엔진 직접 구현**: 함수형·클래스형·스크립트형 3가지 문제 유형을 자동 인식하고, TreeNode/ListNode 같은 자료구조를 Python↔JSON으로 직렬화·비교, 테스트 간 네임스페이스 오염 정리까지 처리하는 테스트 러너 설계
- **콘텐츠 파이프라인**: Notion에 정리해둔 문제를 Supabase(Postgres + Storage)로 동기화하는 스크립트 파이프라인 구축 — "Notion을 원본(source of truth)으로" 유지하는 구조
- **학습 기능**: 카테고리(자료구조/알고리즘 12종)·난이도 필터, Monaco 에디터, 데일리 퀴즈(매일 2문제 자동 선정), 달력형 학습 기록, 내 코드 vs 정답 코드 diff 비교 오답노트
- **Stack**: Next.js (App Router) · TypeScript · React · Tailwind CSS · Supabase · Pyodide · Monaco Editor · Vercel

---

## 🧠 Computer Vision — Object Detection & Segmentation

### [YOLOv10 / Detectron2 / SAM2 비교 실험](./Object%20Detection%20%26%20Segmentation)

주요 객체 탐지·세그멘테이션 프레임워크 3종을 **Pretrained vs Custom 파인튜닝** 구도로 직접 학습·비교한 실험 모음.

- **YOLOv10** — Roboflow 커스텀 데이터셋으로 파인튜닝, 비디오 프레임 추출 → 추론 → 재합성 파이프라인 구축
- **Detectron2 (Faster R-CNN R50-FPN)** — COCO 포맷 커스텀 데이터셋 등록(`register_coco_instances`) 후 전이학습
- **SAM2 + YOLO 조합** — YOLO 탐지 박스의 중심점을 SAM2의 포인트 프롬프트로 넘겨 클래스별 색상 마스크를 생성하는 **모델 합성(composition) 파이프라인** 구현
- **웹 데모 2종** — 파인튜닝한 YOLO 모델을 Flask로 감싼 이미지/비디오 탐지 웹앱. 비디오는 FFmpeg으로 H.264 재인코딩해 브라우저에서 바로 재생되도록 처리
- **Stack**: PyTorch · Ultralytics · Detectron2 · SAM2 · OpenCV · supervision · Flask · FFmpeg

---

## 🔢 Tabular / Classical ML

### [Credit Card Fraud Detection](./Credit%20Card%20Fraud%20Detection)

사기 거래 비율 **0.17%(284,807건 중 492건)** 극단 불균형 데이터에서, 불균형 처리 전략이 검출 성능에 미치는 영향을 체계적으로 비교한 실험.

- **"Accuracy의 함정" 정면 검증**: 모든 모델이 Accuracy 99.8%+가 나오는 데이터에서 ROC-AUC·Recall·Precision 중심으로 평가 체계를 설계
- **대규모 조합 실험**: 오버샘플링 3종(SMOTE·BorderlineSMOTE·ADASYN) × 차원축소 3종(PCA·t-SNE·UMAP) × 모델 6계열(RandomForest·XGBoost·LightGBM·CatBoost·TensorFlow·PyTorch)을 조합해 성능 순위표 도출
- **결과**: LightGBM + SMOTE 조합이 **ROC-AUC 0.9751**로 최고 성능 — 오버샘플링 적용 전(0.9664) 대비 개선을 정량 확인. 오버샘플링은 학습 데이터에만 적용해 평가 누수 방지
- **Stack**: scikit-learn · imbalanced-learn · XGBoost · LightGBM · CatBoost · TensorFlow · PyTorch · UMAP

---

## 📈 Skills

- **Languages**: Python, TypeScript, Dart
- **ML/DL**: PyTorch, TensorFlow, scikit-learn, Transformers(Hugging Face), Ultralytics, Detectron2, darts, sktime, OpenCV
- **Model Deployment**: ONNX / ONNX Runtime, Flutter(On-Device AI), Docker, Hugging Face Spaces, Vercel
- **Data**: Pandas, NumPy, Albumentations, Roboflow(직접 라벨링), Plotly, Matplotlib
- **App/Web**: Next.js, React, Supabase, Streamlit, Flask
- **Tools**: Git, Jupyter, Google Colab

---

## 🛠️ Tools & Technologies

<p>
  <img src="https://img.shields.io/badge/Pytorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="Pytorch" width="120" height="30"/>
  <img src="https://img.shields.io/badge/Tensorflow-FF6F00.svg?style=for-the-badge&logo=Tensorflow&logoColor=white" alt="Tensorflow" width="120" height="30"/>
  <img src="https://img.shields.io/badge/ONNX-005CED.svg?style=for-the-badge&logo=ONNX&logoColor=white" alt="ONNX" width="120" height="30"/>
  <img src="https://img.shields.io/badge/Flutter-02569B.svg?style=for-the-badge&logo=Flutter&logoColor=white" alt="Flutter" width="120" height="30"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV" width="120" height="30"/>
</p>
