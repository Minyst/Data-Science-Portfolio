# 🎓 Data Scientist Portfolio

머신러닝·딥러닝 모델링부터 배포 가능한 엔드투엔드 애플리케이션까지 다룬 프로젝트 모음입니다.
중요도·완성도 순으로 정리했습니다.

---

## 🏆 Flagship

### [Real-Time On-Device Semantic Segmentation System for Recycling Waste Sorting](./Real-Time%20On-Device%20Semantic%20Segmentation%20System%20for%20Recycling%20Waste%20Sorting)
**On-device** 실시간 시맨틱 세그멘테이션으로 재활용 폐기물을 분류하는 모바일 시스템.
- DeepLabV3 + MobileViT, YOLO11n-seg 학습 → **ONNX 변환** → 모바일(Flutter/Android) 온디바이스 추론
- 서버 없이 기기에서 직접 추론하는 엣지 AI 파이프라인 (학습 → 경량화 → 배포)

---

## 📦 Applications & Dashboards

### [Coding Test Prep Platform](./Coding%20Test%20Prep%20Platform)
브라우저에서 Python을 직접 실행·자동 채점하는 코딩테스트 학습 웹앱.
- Next.js 15 + TypeScript + Supabase, **Pyodide** 기반 브라우저 내 Python 실행, 테스트 케이스 자동 채점
- Notion → Supabase 콘텐츠 동기화 파이프라인, Vercel 배포

### [Time Series Forecasting Dashboard](./Time%20Series%20Forecasting%20Dashboard)
단변량 시계열 CSV 업로드 → 9개 예측 모델 비교·평가하는 인터랙티브 대시보드.
- Streamlit · sktime · Prophet, 성능(RMSE/MAE/MAPE 등) + 편향(RSFE/TS) 진단, Docker

### [Multivariate Time Series Anomaly Detection](./Multivariate%20Time%20Series%20Anomaly%20Detection)
다변량 시계열 이상탐지 대시보드 (예측모델 × Scorer 격자 평가).
- `darts` ForecastingAnomalyModel, ML 5종 + DL 7종, AUC-ROC/AUC-PR/F1 정량 평가, Docker

---

## 🧠 Computer Vision

### [YOLOv10 — Pretrained vs Custom](./Computer%20Vision/YOLO)
Pretrained YOLOv10과 Custom 학습 모델의 탐지 성능 비교. 비디오 프레임 추출 → 학습 → 재합성 파이프라인.

### [Detectron2 — Pretrained vs Custom](./Computer%20Vision/Detectron)
Faster R-CNN 기반 Detectron2의 pretrained vs custom 비교.

### [SAM2 + YOLO Segmentation](./Computer%20Vision/SAM)
YOLO로 객체 탐지 후 SAM2로 세그멘테이션 마스크 생성·오버레이.

### [CLIP — Zero-Shot Image Classification](./Computer%20Vision/CLIP)
CLIP(Contrastive Language-Image Pretraining)으로 학습하지 않은 클래스도 텍스트 설명만으로 분류하는 zero-shot 분류.

---

## 🔢 Tabular / Classical ML

### [Credit Card Fraud Detection](./Credit%20Card%20Fraud%20Detection)
차원 축소(PCA/t-SNE/UMAP) vs 증강(SMOTE/BorderlineSMOTE/ADASYN)이 모델 검출 성능에 미치는 영향 비교.
- ML(RandomForest/XGBoost/CatBoost/LightGBM) + DL(TensorFlow/PyTorch)로 조합별 성능 순위표 도출

---

## 🎙️ Generative AI

### AI Cover (RVC)
RVC 모델로 한 가수의 목소리로 다른 곡을 부르게 하는 AI 커버. (음원 분리 → 슬라이싱 → 학습 → 생성)
> 외부(Colab) 작업 — 코드 폴더 없음.

---

## 🧱 Deep Learning Foundations

### [CIFAR-10 CNN](./Deep%20Learning%20Foundations/CIFAR)
TensorFlow·PyTorch로 구현한 CNN 전 과정 (Data Augmentation, Conv/BatchNorm/Pooling/Dropout 등).

### [MNIST](./Deep%20Learning%20Foundations/MNIST)
SimpleCNN 기반 손글씨 숫자 분류 — 전처리·학습·추론 기본 파이프라인.

---

## 📈 Skills

- **Languages**: Python, TypeScript
- **ML/DL**: scikit-learn, TensorFlow, PyTorch, OpenCV, sktime, darts
- **Data**: Pandas, NumPy, Matplotlib, Plotly
- **App/Deploy**: Streamlit, Next.js, Supabase, Docker, ONNX, Vercel, Hugging Face Spaces
- **Tools**: Jupyter, Google Colab, Git

---

## 🛠️ Tools & Technologies

<p>
  <img src="https://img.shields.io/badge/Tensorflow-FF6F00.svg?style=for-the-badge&logo=Tensorflow&logoColor=white" alt="Tensorflow" width="120" height="30"/>
  <img src="https://img.shields.io/badge/Pytorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="Pytorch" width="120" height="30"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV" width="120" height="30"/>
</p>

---
