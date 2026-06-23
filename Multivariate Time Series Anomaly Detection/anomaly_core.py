"""다변량 시계열 이상 탐지 — 순수 로직 (Streamlit 의존성 없음).

darts ForecastingAnomalyModel 흐름을 다변량으로 일반화한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import NormScorer, KMeansScorer, WassersteinScorer
from darts.ad import ForecastingAnomalyModel, QuantileDetector
from darts.ad.utils import eval_metric_from_scores, eval_metric_from_binary_prediction

_TIME_NAME_HINTS = {"date", "time", "timestamp", "datetime", "ds", "dt", "날짜", "시간"}
_LABEL_NAME_HINTS = {
    "label", "labels", "anomaly", "anomalies", "is_anomaly", "y_true",
    "ground_truth", "gt", "target", "이상", "라벨",
}


def detect_time_column(df: pd.DataFrame) -> str | None:
    """시간 컬럼 추정: 이름 힌트 우선, 없으면 datetime 파싱 성공률이 가장 높은 컬럼."""
    # 1) 이름 힌트
    for col in df.columns:
        if str(col).strip().lower() in _TIME_NAME_HINTS:
            return col
    # 2) 파싱 성공률
    best_col, best_ratio = None, 0.0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    return best_col if best_ratio >= 0.8 else None


def detect_label_column(df: pd.DataFrame, exclude: set) -> str | None:
    """이진(0/1) 라벨 컬럼 추정: 이름 힌트 우선, 없으면 값이 {0,1}인 컬럼."""
    candidates = [c for c in df.columns if c not in exclude]
    # 1) 이름 힌트 + 이진 검증
    for col in candidates:
        if str(col).strip().lower() in _LABEL_NAME_HINTS and _is_binary(df[col]):
            return col
    # 2) 값이 이진인 컬럼
    for col in candidates:
        if _is_binary(df[col]):
            return col
    return None


def _is_binary(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1})


def _freq_bucket(freq: str | None) -> str:
    """darts/pandas freq 문자열을 굵은 분류로 환원."""
    if not freq:
        return "unknown"
    f = str(freq).upper().split("-")[0]  # 'QE-DEC' -> 'QE'
    if f in {"H", "BH"} or f.endswith("H"):
        return "hourly"
    if f in {"T", "MIN"} or "MIN" in f or f.endswith("T"):
        return "subhourly"
    if f in {"S"}:
        return "subhourly"
    if f in {"D", "B", "C"}:
        return "daily"
    if f.startswith("W"):
        return "weekly"
    if f in {"M", "MS", "ME", "BM", "BMS"}:
        return "monthly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("Y") or f.startswith("A"):
        return "yearly"
    return "unknown"


def freq_to_encoder_attrs(freq: str | None) -> list:
    """cyclic future 인코더에 쓸 datetime 속성 목록 (hour, dayofweek 등)."""
    return {
        "subhourly": ["hour", "minute", "dayofweek"],
        "hourly": ["hour", "dayofweek"],
        "daily": ["dayofweek", "month"],
        "weekly": ["week", "month"],
        "monthly": ["month"],
        "quarterly": ["quarter"],
        "yearly": [],
        "unknown": [],
    }[_freq_bucket(freq)]


def freq_to_default_lags(freq: str | None) -> int:
    """예측 모델의 기본 lags / input_chunk_length (한 계절 주기 근사)."""
    return {
        "subhourly": 60,
        "hourly": 24,
        "daily": 7,
        "weekly": 8,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 3,
        "unknown": 12,
    }[_freq_bucket(freq)]


def prepare_dataframe(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """시간 컬럼을 DatetimeIndex로 설정하고 정렬한 DataFrame 반환."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).set_index(time_col).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def numeric_feature_columns(df: pd.DataFrame, exclude: set) -> list:
    """exclude를 제외한 숫자형 컬럼 목록."""
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if c not in exclude]


def build_timeseries(df: pd.DataFrame, value_cols: list, label_col: str | None):
    """다변량 TimeSeries와 (있으면) 0/1 이상 라벨 TimeSeries를 만든다.

    반환: (series, anomalies | None)
    """
    feat = df[value_cols].apply(pd.to_numeric, errors="coerce").interpolate().ffill().bfill()
    series = TimeSeries.from_dataframe(feat)
    anomalies = None
    if label_col is not None and label_col in df.columns:
        lab = pd.to_numeric(df[label_col], errors="coerce")
        unique_vals = set(lab.dropna().unique())
        if unique_vals.issubset({0, 1}):
            lab = lab.fillna(0)
            anomalies = TimeSeries.from_times_and_values(
                df.index, lab.values.reshape(-1, 1).astype(float)
            )
    return series, anomalies


def split_train_test(series: TimeSeries, ratio: float):
    """비율 기반 train/test 분할 (시간순 인덱스 분할)."""
    n = len(series)
    cut = max(2, int(round(n * ratio)))
    cut = min(cut, n - 1)
    return series[:cut], series[cut:]


ML_MODELS = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DL_MODELS = ["RNN", "LSTM", "GRU", "BlockRNN", "Transformer", "NBEATS", "NHiTS"]
ALL_MODELS = ML_MODELS + DL_MODELS


def _encoders(encoder_attrs: list):
    return {"cyclic": {"future": list(encoder_attrs)}} if encoder_attrs else None


def build_forecasting_model(name: str, *, lags: int, encoder_attrs: list, n_epochs: int = 30):
    """ML/DL 예측모델을 생성. ML은 lags, DL은 input_chunk_length 사용."""
    enc = _encoders(encoder_attrs)
    # ── ML 회귀 — lags_future_covariates=[0] 사용 ──
    if name == "LinearRegression":
        from darts.models import LinearRegressionModel
        return LinearRegressionModel(lags=lags, lags_future_covariates=[0],
                                     output_chunk_length=1, add_encoders=enc)
    if name == "RandomForest":
        from darts.models import RandomForestModel
        return RandomForestModel(lags=lags, lags_future_covariates=[0],
                                 output_chunk_length=1, n_estimators=100, add_encoders=enc)
    if name == "XGBoost":
        from darts.models import XGBModel
        return XGBModel(lags=lags, lags_future_covariates=[0],
                        output_chunk_length=1, add_encoders=enc)
    if name == "LightGBM":
        from darts.models import LightGBMModel
        return LightGBMModel(lags=lags, lags_future_covariates=[0],
                             output_chunk_length=1, add_encoders=enc, verbose=-1)
    if name == "CatBoost":
        from darts.models import CatBoostModel
        return CatBoostModel(lags=lags, lags_future_covariates=[0],
                             output_chunk_length=1, add_encoders=enc)
    # ── DL — input_chunk_length 사용, CPU 강제 + 진행바 off ──
    pl_kwargs = {"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False}
    common = dict(input_chunk_length=lags, output_chunk_length=1, n_epochs=n_epochs,
                  random_state=42, add_encoders=enc, pl_trainer_kwargs=pl_kwargs)
    if name in ("RNN", "LSTM", "GRU"):
        # 순환형 RNNModel(model="RNN"/"LSTM"/"GRU"). output_chunk_length 대신
        # training_length 사용(input 의 2배). future 인코더를 실제로 활용.
        from darts.models import RNNModel
        return RNNModel(model=name, input_chunk_length=lags, training_length=2 * lags,
                        n_epochs=n_epochs, random_state=42, add_encoders=enc,
                        pl_trainer_kwargs=pl_kwargs)
    if name == "BlockRNN":
        from darts.models import BlockRNNModel
        return BlockRNNModel(model="RNN", **common)
    if name == "Transformer":
        from darts.models import TransformerModel
        return TransformerModel(**common)
    if name == "NBEATS":
        from darts.models import NBEATSModel
        return NBEATSModel(**common)
    if name == "NHiTS":
        from darts.models import NHiTSModel
        return NHiTSModel(**common)
    raise ValueError(f"Unknown model: {name}")


def build_scorers(specs: list):
    """specs: [("Norm",{}), ("KMeans",{"k":2}), ("Wasserstein",{"window":10})]"""
    out = []
    for kind, kw in specs:
        if kind == "Norm":
            out.append(NormScorer())
        elif kind == "KMeans":
            out.append(KMeansScorer(k=int(kw.get("k", 2))))
        elif kind == "Wasserstein":
            out.append(WassersteinScorer(window=int(kw.get("window", 10))))
        else:
            raise ValueError(f"Unknown scorer: {kind}")
    return out


SCORER_LABELS = ["Norm", "KMeans", "Wasserstein"]


def run_detection(train, test, model_name, scorer_specs, *, lags, encoder_attrs,
                  n_epochs=30):
    """예측모델 사전학습 → ForecastingAnomalyModel로 Scorer 학습 → test 점수화.

    반환: Scorer 수만큼의 이상 점수 TimeSeries 리스트.
    """
    model = build_forecasting_model(
        model_name, lags=lags, encoder_attrs=encoder_attrs, n_epochs=n_epochs
    )
    model.fit(train)
    scorers = build_scorers(scorer_specs)
    anomaly_model = ForecastingAnomalyModel(model=model, scorer=scorers)
    anomaly_model.fit(train, allow_model_training=False)
    scores = anomaly_model.score(test)
    # darts returns a TimeSeries (single scorer) or tuple/list (multiple scorers)
    scores = list(scores) if isinstance(scores, (list, tuple)) else [scores]
    return scores


def compute_score_metrics(anomalies, score, quantile: float = 0.95) -> dict:
    """라벨 대비 6개 지표 (darts.ad 범위 내).

    AUC_ROC / AUC_PR: 점수 기반. F1/Precision/Recall/Accuracy: QuantileDetector 이진화 후.
    """
    # 점수와 라벨의 공통 시간 구간으로 정렬
    anom = anomalies.slice_intersect(score)
    sc = score.slice_intersect(anomalies)

    # 테스트 구간에 양성(1)이 없거나 전부 양성이면 AUC는 정의되지 않음 → NaN 처리
    def _safe_auc(metric):
        try:
            return float(eval_metric_from_scores(anom, sc, metric=metric))
        except Exception:
            return float("nan")

    auc_roc = _safe_auc("AUC_ROC")
    auc_pr = _safe_auc("AUC_PR")

    detector = QuantileDetector(high_quantile=quantile)
    pred = detector.fit_detect(sc)
    pred = pred.slice_intersect(anom)
    anom2 = anom.slice_intersect(pred)

    def _m(metric):
        try:
            return float(eval_metric_from_binary_prediction(anom2, pred, metric=metric))
        except Exception:
            return float("nan")

    return {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "F1": _m("f1"),
        "Precision": _m("precision"),
        "Recall": _m("recall"),
        "Accuracy": _m("accuracy"),
    }


def detect_binary(score, quantile: float = 0.95):
    """점수 → 0/1 탐지 TimeSeries (시각화/오버레이용)."""
    return QuantileDetector(high_quantile=quantile).fit_detect(score)


def backtest_residuals(train, test, model_name, *, lags, encoder_attrs, n_epochs=30):
    """예측모델을 train 으로 학습한 뒤 test 구간을 rolling 1-step 백테스팅 예측한다.

    백테스팅(historical_forecasts) + 잔차분석(model.residuals) 개념을
    이상탐지에 적용: 예측오차(잔차)가 곧 이상 점수의 원천이다.

    데이터 누수 방지: 모델은 train 으로만 fit, test 구간은 retrain=False 로 예측만 수행
    (각 시점 예측에 과거 값만 사용). train→test 가 시간순으로 이어져 있어야 한다.

    반환: (pred, resid, actual) — 모두 test 와 정렬된 다변량 TimeSeries
          pred=예측, resid=actual-pred(잔차), actual=정렬된 실제값
    """
    model = build_forecasting_model(
        model_name, lags=lags, encoder_attrs=encoder_attrs, n_epochs=n_epochs
    )
    model.fit(train)
    # train+test 를 이어 붙여 test 구간만 rolling 예측 (test 앞의 train 이 lags 히스토리 제공)
    full = train.append(test)
    pred = model.historical_forecasts(
        full, start=test.start_time(), forecast_horizon=1, stride=1,
        retrain=False, last_points_only=True, verbose=False,
    )
    pred = pred.slice_intersect(test)
    actual = test.slice_intersect(pred)
    resid = actual - pred
    return pred, resid, actual
