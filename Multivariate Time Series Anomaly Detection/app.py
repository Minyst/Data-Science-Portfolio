# -*- coding: utf-8 -*-
import os
import sys
import time
import hashlib
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
import anomaly_core as ac
from darts.dataprocessing.transformers import Scaler

st.set_page_config(page_title="다변량 이상 탐지 대시보드", layout="wide", page_icon="🚨")

# ── 세션 상태 ──
if "detected" not in st.session_state:
    st.session_state.detected = False
    st.session_state.detect_results = {}

# ── CSS (project.py 계승) ──
st.markdown("""
<style>
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #ef553b11, #764ba211);
  border: 1px solid #e0e0e0; border-radius: 12px; padding: 12px 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label { font-size: 0.85rem !important; color: #555 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
button[data-baseweb="tab"] { font-size: 0.95rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚨 다변량 시계열 이상 탐지 대시보드")

# 톤다운(muted) 팔레트 — 너무 밝지도 어둡지도 않은 중간 채도
MODEL_COLORS = {
    "LinearRegression": "#5B7DB1", "RandomForest": "#6FA877", "XGBoost": "#C77B7B",
    "LightGBM": "#9079B0", "CatBoost": "#E0A458",
    "RNN": "#6E93A8", "LSTM": "#73B0C4", "GRU": "#88B0A0",
    "BlockRNN": "#CB8BB0", "Transformer": "#93A65B", "NBEATS": "#B089B0", "NHiTS": "#C7B05B",
}
# 자동 색상 순환용(변수별 라인 등) — 동일 계열의 톤다운 색
MUTED_PALETTE = [
    "#5B7DB1", "#E0A458", "#6FA877", "#C77B7B", "#9079B0",
    "#A1785A", "#CB8BB0", "#8A8A8A", "#C7B05B", "#73B0C4",
]
# 톤다운 강조색
MUTED_RED = "#C2504D"      # 정답 라벨 음영
MUTED_ORANGE = "#C97E2C"   # 탐지 구간 음영
MUTED_SCORE = "#B0504D"    # 이상 점수 라인
MUTED_HEAT = [[0.0, "#eef1f4"], [0.5, "#a9bcd0"], [1.0, "#5e7d9e"]]  # 히트맵

# ── 사이드바: 업로드 ──
st.sidebar.header("📂 데이터 업로드")
uploaded = st.sidebar.file_uploader("다변량 시계열 CSV", type="csv")

if uploaded is None:
    st.info("**왼쪽에서 다변량 CSV를 업로드하면 이상 탐지를 시작합니다.**")
    st.markdown("---")
    st.markdown("""
### 주요 기능
- **다변량 자동 처리**: 숫자 컬럼 전부를 하나의 다변량 시계열로 묶어 분석
- **자동 EDA/진단**: 상관 히트맵 · ACF 자기상관 · 계절성 검출(check_seasonality)
- **예측모델 × Scorer 격자 비교**: ML(LinearRegression·RandomForest·XGBoost·LightGBM·CatBoost) / DL(RNN·LSTM·GRU·BlockRNN·Transformer·NBEATS·NHiTS) × Norm·KMeans·Wasserstein
- **지도식 이상탐지**: 0/1 정답 라벨을 기준으로 탐지 결과를 정량 평가
- **백테스팅·잔차분석**: 선택 모델의 예측 vs 실제·잔차 분포로 이상 점수의 원천(예측오차) 점검
- **평가지표**: AUC-ROC·AUC-PR·F1·정밀도·재현율·정확도
- **탐지 적절성 점검**: 탐지 비율·이상 점수 요약으로 과탐·미탐 판단
- **파일 변경 자동 감지**: 내용 해시로 값 변경까지 감지해 자동 재탐지
""")
    st.stop()

# ── 데이터 로드 ──
try:
    uploaded.seek(0)
    raw = pd.read_csv(uploaded)
    # 파일 "내용" 해시 — 이름/행수/기간이 같아도 값이 바뀌면 재탐지하도록
    uploaded.seek(0)
    file_hash = hashlib.md5(uploaded.read()).hexdigest()

    time_col_guess = ac.detect_time_column(raw)
    if time_col_guess is None:
        st.error("⚠️ 시간 컬럼을 찾지 못했습니다. (날짜/시간 형식 컬럼 필요)")
        st.stop()

    st.sidebar.subheader("🧭 컬럼 매핑")
    time_col = st.sidebar.selectbox(
        "시간 컬럼", list(raw.columns),
        index=list(raw.columns).index(time_col_guess), key="time_col",
    )
    # ── 정답 라벨 컬럼 선택 ──
    # 이상탐지(분류)는 0/1 정답 라벨이 반드시 있어야 한다(=평가 기준). 이름/값 힌트로
    # 기본값을 제안하되, 사용자가 직접 라벨 컬럼을 지정한다.
    label_guess = ac.detect_label_column(raw, exclude={time_col})
    label_options = [c for c in raw.columns if c != time_col]
    if label_guess in label_options:
        label_default = label_options.index(label_guess)
    else:
        label_default = 0
    label_col = st.sidebar.selectbox("정답 라벨 컬럼 (0/1)", label_options,
                                     index=label_default, key="label_col")

    df = ac.prepare_dataframe(raw, time_col)
    feature_cols = ac.numeric_feature_columns(df, exclude={label_col} if label_col else set())
    if len(feature_cols) == 0:
        st.error("⚠️ 분석할 숫자형 변수 컬럼이 없습니다.")
        st.stop()

    series, anomalies = ac.build_timeseries(df, feature_cols, label_col)
    if anomalies is None:
        st.error("⚠️ 선택한 정답 라벨 컬럼이 0/1 이진값이 아닙니다. "
                 "0/1 이상 라벨 컬럼을 지정하세요.")
        st.stop()
    inferred_freq = pd.infer_freq(df.index)

    # ── fingerprint: 파일 변경 시 캐시 무효화 (내용 해시 포함 → 값만 바뀌어도 감지) ──
    fp = (uploaded.name, file_hash, tuple(feature_cols), label_col)
    if st.session_state.get("_csv_fingerprint") != fp:
        st.session_state.detected = False
        st.session_state.detect_results = {}
        st.session_state["_csv_fingerprint"] = fp
        # 데이터셋에 종속된 위젯 키 초기화 — 옛 선택값이 새 옵션에 없으면 오류나므로
        for _k in ("diag_var", "result_pick"):
            st.session_state.pop(_k, None)

    st.sidebar.caption(
        f"변수 {len(feature_cols)}개 · {len(df)}행 · 주기 {inferred_freq or '불명'}"
    )

    # 탭 자리 (이후 Task에서 채움)
    tab_overview, tab_settings, tab_results, tab_metrics = st.tabs(
        ["📊 데이터 개요", "⚙️ 설정 & 탐지", "🎯 이상 탐지 결과", "💡 성능 비교"]
    )
    with tab_overview:
        st.subheader("다변량 시계열 원본")
        st.caption(f"변수 {len(feature_cols)}개 · 총 {len(df)}개 · 주기 {inferred_freq or '불명'}")

        # 변수별 subplot
        fig = make_subplots(rows=len(feature_cols), cols=1, shared_xaxes=True,
                            subplot_titles=feature_cols, vertical_spacing=0.04)
        for i, col in enumerate(feature_cols, 1):
            fig.add_trace(go.Scatter(x=df.index, y=df[col].values, mode="lines",
                                     line=dict(width=1.1), showlegend=False), row=i, col=1)
            # 정답 라벨 이상 구간 음영 (분류 모드 = 라벨 있을 때만)
            if label_col is not None:
                lab = pd.to_numeric(df[label_col], errors="coerce").fillna(0).values
                in_anom = False
                for k in range(len(lab)):
                    if lab[k] == 1 and not in_anom:
                        start_idx = df.index[k]; in_anom = True
                    elif lab[k] == 0 and in_anom:
                        fig.add_vrect(x0=start_idx, x1=df.index[k], fillcolor=MUTED_RED,
                                      opacity=0.15, line_width=0, row=i, col=1)
                        in_anom = False
                if in_anom:
                    fig.add_vrect(x0=start_idx, x1=df.index[-1], fillcolor=MUTED_RED,
                                  opacity=0.15, line_width=0, row=i, col=1)
        fig.update_layout(height=max(260, 150 * len(feature_cols)), template="plotly_white",
                          colorway=MUTED_PALETTE,
                          margin=dict(l=50, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
        if label_col is not None:
            st.caption("🔴 빨강 음영 = 정답 라벨로 표시된 이상 구간")

        st.markdown("---")
        st.markdown("**기초 통계량**")
        st.dataframe(df[feature_cols].describe().round(3), use_container_width=True)

        # ── 변수 간 상관관계 (다변량 분석) ──
        st.markdown("---")
        st.markdown("**🔗 변수 간 상관관계**")
        if len(feature_cols) >= 2:
            corr = df[feature_cols].corr()
            figc = go.Figure(go.Heatmap(
                z=corr.values, x=feature_cols, y=feature_cols,
                zmin=-1, zmax=1, colorscale=MUTED_HEAT,
                text=corr.round(2).values, texttemplate="%{text:.2f}", textfont=dict(size=16),
                colorbar=dict(title="corr"),
            ))
            figc.update_layout(height=min(720, 160 + 60 * len(feature_cols)),
                               template="plotly_white", margin=dict(l=90, r=20, t=20, b=50))
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.caption("변수가 1개뿐이라 상관 히트맵은 생략합니다.")

        # ── 변수별 자기상관(ACF) · 계절성 검출 ──
        st.markdown("---")
        st.markdown("**🔬 변수별 자기상관(ACF) · 계절성 검출**")
        st.caption("ACF로 자기상관 구조를 보고 check_seasonality로 계절 주기를 찾습니다. "
                   "(예측모델의 lags·계절주기 설정 근거)")
        diag_var = st.selectbox("진단할 변수 선택", feature_cols, key="diag_var")
        yv = pd.to_numeric(df[diag_var], errors="coerce").interpolate().ffill().bfill()

        # 1) ACF — 자기상관함수 (ACF로 계절성 주기 찾기)
        with st.expander("🔍 ACF (자기상관함수)", expanded=True):
            if len(yv) >= 10:
                try:
                    from statsmodels.tsa.stattools import acf as _acf_fn
                    nlags = min(40, len(yv) // 2 - 1)
                    av = _acf_fn(yv.values, nlags=nlags)
                    ci = 1.96 / np.sqrt(len(yv))
                    figa = go.Figure()
                    for i, v in enumerate(av):
                        figa.add_trace(go.Scatter(x=[i, i], y=[0, v], mode="lines",
                                                  line=dict(color="#5B7DB1", width=1.4),
                                                  showlegend=False, hoverinfo="skip"))
                    figa.add_trace(go.Scatter(x=list(range(len(av))), y=av, mode="markers",
                                              marker=dict(color="#5B7DB1", size=5), showlegend=False))
                    figa.add_shape(type="rect", x0=0, x1=nlags, y0=-ci, y1=ci,
                                   fillcolor="rgba(150,150,150,0.18)", line_width=0, layer="below")
                    figa.update_yaxes(range=[-1.05, 1.05])
                    figa.update_xaxes(title_text="Lag")
                    figa.update_layout(height=320, template="plotly_white",
                                       margin=dict(l=50, r=20, t=30, b=40))
                    st.plotly_chart(figa, use_container_width=True)
                    st.caption("회색 음영은 95% 신뢰구간 — 막대가 음영을 벗어나면 유의한 자기상관(계절성/주기 단서).")
                except Exception as e:
                    st.caption(f"ACF 생략: {e}")
            else:
                st.info("ACF에는 최소 10개 데이터가 필요합니다.")

        # 2) 계절성 검출 — check_seasonality
        try:
            from darts.utils.statistics import check_seasonality
            comp = series[diag_var]
            # 일(24h) 주기까지 보도록 상한을 24로 설정. check_seasonality 는 후보 m 이
            # 국소 최대인지 확인하므로 max_lag 가 m 보다 커야 경계값(24)도 잡힌다.
            m_max = min(24, len(yv) // 2 - 1)
            found = []
            for m in range(2, m_max + 1):
                try:
                    is_seasonal, period = check_seasonality(
                        comp, m=m, max_lag=m_max + 1, alpha=0.05)
                except Exception:
                    is_seasonal = False
                if is_seasonal:
                    found.append(int(period))
            found = sorted(set(found))
            if found:
                st.success("📌 검출된 계절 주기: " + ", ".join(str(p) for p in found)
                           + " — 예측모델 lags·계절주기 설정에 참고하세요.")
            else:
                st.info("뚜렷한 계절 주기가 검출되지 않았습니다 (데이터가 짧거나 비계절적).")
        except Exception as e:
            st.caption(f"계절성 검출 생략: {e}")

    with tab_settings:
        st.subheader("⚙️ 탐지 설정")
        split_ratio = st.slider("학습 비율 (Train ratio)", 0.5, 0.95, 0.8, 0.05, key="split_ratio")
        default_lags = ac.freq_to_default_lags(inferred_freq)
        max_lags = max(2, len(series) // 3)
        lags = st.slider("lags / input_chunk_length", 2, max_lags,
                         min(default_lags, max_lags), key="lags")

        # ── 예측모델 선택 (기본: 아무것도 선택 안 함, 사용자가 직접 선택) ──
        n_ml, n_dl = len(ac.ML_MODELS), len(ac.DL_MODELS)

        # 세션 상태로 선택값 관리 (전체 선택 버튼이 위젯 생성 전에 값을 세팅할 수 있도록)
        if "ml_models" not in st.session_state:
            st.session_state["ml_models"] = []
        if "dl_models" not in st.session_state:
            st.session_state["dl_models"] = []

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("ML 전체", use_container_width=True, help="ML 회귀 모델 전체 선택"):
            st.session_state["ml_models"] = list(ac.ML_MODELS); st.rerun()
        if b2.button("DL 전체", use_container_width=True, help="DL 신경망 모델 전체 선택"):
            st.session_state["dl_models"] = list(ac.DL_MODELS); st.rerun()
        if b3.button(f"{n_ml + n_dl}개 전부", use_container_width=True, help="ML + DL 전체 선택"):
            st.session_state["ml_models"] = list(ac.ML_MODELS)
            st.session_state["dl_models"] = list(ac.DL_MODELS); st.rerun()
        if b4.button("전체 해제", use_container_width=True):
            st.session_state["ml_models"] = []; st.session_state["dl_models"] = []; st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            ml_sel = st.multiselect(f"ML 예측모델 (총 {n_ml}종)", ac.ML_MODELS, key="ml_models")
        with c2:
            dl_sel = st.multiselect(f"DL 예측모델 (총 {n_dl}종)", ac.DL_MODELS, key="dl_models")
        selected_models = ml_sel + dl_sel
        st.caption(
            f"✅ 선택된 **{len(selected_models)}개** 모델이 모두 동일한 `ForecastingAnomalyModel` "
            f"파이프라인으로 실행됩니다. (선택 가능: ML {n_ml} + DL {n_dl} = {n_ml + n_dl}개)"
        )

        scorer_sel = st.multiselect("Scorer", ac.SCORER_LABELS,
                                    default=list(ac.SCORER_LABELS), key="scorers")
        cc1, cc2, cc3 = st.columns(3)
        kmeans_k = cc1.number_input("KMeans k", 2, 20, 2, key="kmeans_k")
        wass_win = cc2.number_input("Wasserstein window", 2, 100, 10, key="wass_win")
        quantile = cc3.slider("탐지 임계 분위수 q", 0.80, 0.999, 0.95, 0.005, key="quantile")
        n_epochs = st.number_input("DL 학습 epoch", 5, 300, 30, key="n_epochs")

        if not selected_models:
            st.warning("최소 1개 예측모델을 선택하세요.")
        if not scorer_sel:
            st.warning("최소 1개 Scorer를 선택하세요.")

        st.markdown("---")
        run_clicked = st.button("🚀 이상 탐지 실행", type="primary",
                                use_container_width=True, key="run_btn",
                                disabled=(not selected_models or not scorer_sel))
        if st.session_state.detected:
            st.success("✅ 탐지 완료 — 결과/성능 탭에서 확인하세요.")

        # ── scorer_specs 구성 ──
        scorer_specs = []
        for s in scorer_sel:
            if s == "KMeans":
                scorer_specs.append(("KMeans", {"k": int(kmeans_k)}))
            elif s == "Wasserstein":
                scorer_specs.append(("Wasserstein", {"window": int(wass_win)}))
            else:
                scorer_specs.append(("Norm", {}))
        encoder_attrs = ac.freq_to_encoder_attrs(inferred_freq)

    # ── 탐지 실행 ──
    if run_clicked:
        # ── 시간순 분할을 먼저 한 뒤 스케일러를 train 으로만 fit ──
        # (전체로 fit_transform 하면 test 분포가 전처리에 새어드는 데이터 누수)
        train_raw, test_raw = ac.split_train_test(series, split_ratio)
        sc = Scaler()
        train = sc.fit_transform(train_raw)
        test = sc.transform(test_raw)
        anom_test = anomalies.slice_intersect(test) if anomalies is not None else None
        if anom_test is not None and float(anom_test.values().sum()) == 0:
            tab_settings.info(
                "ℹ️ 테스트 구간에 라벨 이상치(1)가 없어 AUC 등 지표가 NaN으로 나올 수 있습니다. "
                "학습 비율을 낮춰 이상 구간이 테스트에 포함되도록 조정해 보세요."
            )

        results_scores = {}     # (model, scorer_label) -> score TimeSeries
        results_detect = {}     # (model, scorer_label) -> 0/1 TimeSeries
        metrics_rows = []       # (model, scorer)별 AUC/F1 등
        total = len(selected_models)
        prog = tab_settings.progress(0.0, text=f"⏳ 준비 중... (0/{total})")
        start_t = time.time()

        for mi, mname in enumerate(selected_models):
            prog.progress(mi / total, text=f"⏳ {mname} 학습/점수화... ({mi}/{total})")
            try:
                scores = ac.run_detection(train, test, mname, scorer_specs,
                                          lags=int(lags), encoder_attrs=encoder_attrs,
                                          n_epochs=int(n_epochs))
                for slabel_spec, score in zip(scorer_specs, scores):
                    slabel = slabel_spec[0]
                    results_scores[(mname, slabel)] = score
                    results_detect[(mname, slabel)] = ac.detect_binary(score, quantile)
                    if anom_test is not None:
                        m = ac.compute_score_metrics(anom_test, score, quantile)
                        metrics_rows.append({"Model": mname, "Scorer": slabel, **m})
            except Exception as e:
                tab_settings.warning(f"⚠️ {mname} 실행 오류: {e}")
            prog.progress((mi + 1) / total, text=f"⏳ {mname} 완료 ({mi + 1}/{total})")
        prog.progress(1.0, text=f"✅ 완료 ({total}/{total}) · {int(time.time()-start_t)}초")

        st.session_state.detected = True
        st.session_state.detect_results = {
            "scores": results_scores, "detect": results_detect,
            "metrics": metrics_rows,
            "test": test, "train": train,
            "anom_test": anom_test, "feature_cols": feature_cols,
            "lags": int(lags), "encoder_attrs": encoder_attrs, "n_epochs": int(n_epochs),
        }
        # 백테스팅/잔차 캐시 무효화 (새 탐지 결과이므로)
        st.session_state["_bt_cache"] = {}

    # 결과 로드
    R = st.session_state.detect_results if st.session_state.detected else {}

    with tab_results:
        if not st.session_state.detected or not R.get("scores"):
            st.warning("아직 탐지 결과가 없습니다. '설정 & 탐지' 탭에서 실행하세요.")
        else:
            combos = list(R["scores"].keys())
            labels = [f"{m} · {s}" for (m, s) in combos]
            pick = st.selectbox("조합 선택 (모델 · Scorer)", labels, key="result_pick")
            mname, slabel = combos[labels.index(pick)]
            score = R["scores"][(mname, slabel)]
            detect = R["detect"][(mname, slabel)]
            test_ts = R["test"]

            # 아래 패널에 표시할 변수 1개 선택 (단변량 시계열 + 탐지 음영 그림을 다변량으로 일반화)
            view_cols = R.get("feature_cols") or list(test_ts.to_dataframe().columns)
            view_var = st.selectbox("아래 패널에 표시할 변수", view_cols, key="result_view_var")

            # 상단: 이상 점수 + 임계 탐지
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=["이상 점수 (Anomaly Score)",
                                                f"{view_var} 시계열 + 탐지 구간"])
            sx = score.time_index
            fig.add_trace(go.Scatter(x=sx, y=score.values().ravel(), mode="lines",
                                     line=dict(color=MUTED_SCORE, width=1.6), name="score"),
                          row=1, col=1)
            # 탐지 구간 음영 (위=점수 / 아래=변수 시계열을 더 진하게)
            dvals = detect.values().ravel()
            dx = detect.time_index
            row_op = ((1, 0.38), (2, 0.55))
            in_a = False
            for k in range(len(dvals)):
                if dvals[k] == 1 and not in_a:
                    s0 = dx[k]; in_a = True
                elif dvals[k] == 0 and in_a:
                    for rr, op in row_op:
                        fig.add_vrect(x0=s0, x1=dx[k], fillcolor=MUTED_ORANGE, opacity=op,
                                      line_width=0, row=rr, col=1)
                    in_a = False
            if in_a:
                for rr, op in row_op:
                    fig.add_vrect(x0=s0, x1=dx[-1], fillcolor=MUTED_ORANGE, opacity=op,
                                  line_width=0, row=rr, col=1)
            # 하단: score 와 동일 구간의 "선택한 변수 1개" 시계열 (정규화 스케일된 series 사용)
            # 변수마다 스케일이 달라 여러 개를 한 축에 겹치면 큰 변수가 작은 변수를 눌러 평평해 보임 → 1개씩 표시
            test_df = test_ts.slice_intersect(score).to_dataframe()
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df[view_var].values, mode="lines",
                                     line=dict(color=MUTED_PALETTE[0], width=1.2),
                                     name=str(view_var)), row=2, col=1)
            # 정답 라벨 이상 구간도 표시(빨강) — 분류 모드(라벨 있을 때)만
            if R.get("anom_test") is not None:
                av = R["anom_test"].values().ravel(); ax = R["anom_test"].time_index
                in_l = False
                for k in range(len(av)):
                    if av[k] == 1 and not in_l:
                        l0 = ax[k]; in_l = True
                    elif av[k] == 0 and in_l:
                        fig.add_vrect(x0=l0, x1=ax[k], fillcolor=MUTED_RED, opacity=0.14,
                                      line_width=0, row=2, col=1)
                        in_l = False
                if in_l:
                    fig.add_vrect(x0=l0, x1=ax[-1], fillcolor=MUTED_RED, opacity=0.14,
                                  line_width=0, row=2, col=1)
            fig.update_layout(height=620, template="plotly_white",
                              colorway=MUTED_PALETTE,
                              margin=dict(l=50, r=20, t=50, b=30),
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟧 주황 = 모델이 탐지한 이상 구간"
                       + ("  |  🔴 빨강 = 정답 라벨 이상 구간" if R.get("anom_test") is not None else ""))

            # ── 탐지 요약: 탐지 적절성(과탐/미탐)을 판단하기 위한 지표 ──
            dv = detect.values().ravel().astype(float)
            sc_vals = score.values().ravel().astype(float)
            n_anom = int(np.nansum(dv))
            n_tot = int(len(dv))
            ratio = (n_anom / n_tot * 100) if n_tot else 0.0
            s1, s2, s3 = st.columns(3)
            s1.metric("탐지된 이상 시점", f"{n_anom} / {n_tot}")
            s2.metric("이상 비율", f"{ratio:.2f}%")
            s3.metric("이상 점수 (평균 / 최대)", f"{np.nanmean(sc_vals):.3f} / {np.nanmax(sc_vals):.3f}")
            st.caption("탐지 임계 분위수 q를 조절해 이상 비율이 데이터 특성에 맞는지(과탐·미탐) 판단하세요. "
                       "'성능 비교' 탭에서 정답 라벨 기준 지표(AUC·F1 등)로 정량 평가할 수 있습니다.")

            # ── 백테스팅 · 잔차분석: 예측오차(잔차)가 곧 이상 점수의 원천 ──
            st.markdown("---")
            st.markdown("**🔁 백테스팅 · 잔차분석** — 선택한 예측모델의 test 구간 1-step 예측")
            st.caption("백테스팅(historical_forecasts)·잔차분석을 이상탐지에 적용합니다. "
                       "모델은 train 으로만 학습하고 test 는 과거값만으로 예측 → 예측오차(잔차)가 큰 구간이 이상 후보입니다.")
            bt_cache = st.session_state.setdefault("_bt_cache", {})
            if st.button(f"▶ '{mname}' 백테스팅·잔차 계산", key="bt_btn"):
                with st.spinner(f"{mname} rolling 예측 중..."):
                    try:
                        bt_cache[mname] = ac.backtest_residuals(
                            R["train"], R["test"], mname,
                            lags=R["lags"], encoder_attrs=R["encoder_attrs"],
                            n_epochs=R["n_epochs"])
                    except Exception as e:
                        bt_cache[mname] = e

            bt = bt_cache.get(mname)
            if isinstance(bt, Exception):
                st.warning(f"백테스팅 생략: {bt}")
            elif bt is not None:
                pred, resid, actual = bt
                a_df = actual.to_dataframe(); p_df = pred.to_dataframe(); r_df = resid.to_dataframe()
                bx = a_df.index
                figb = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                     subplot_titles=[f"{view_var}: 실제 vs 예측 (백테스팅)",
                                                     f"{view_var}: 잔차 (실제 − 예측)"])
                figb.add_trace(go.Scatter(x=bx, y=a_df[view_var].values, mode="lines",
                                          line=dict(color=MUTED_PALETTE[0], width=1.4), name="실제"),
                               row=1, col=1)
                figb.add_trace(go.Scatter(x=bx, y=p_df[view_var].values, mode="lines",
                                          line=dict(color=MUTED_ORANGE, width=1.4, dash="dot"),
                                          name="예측"), row=1, col=1)
                figb.add_trace(go.Scatter(x=bx, y=r_df[view_var].values, mode="lines",
                                          line=dict(color=MUTED_SCORE, width=1.3), name="잔차",
                                          showlegend=False), row=2, col=1)
                figb.add_hline(y=0, line=dict(color="#999", width=1), row=2, col=1)
                figb.update_layout(height=520, template="plotly_white",
                                   margin=dict(l=50, r=20, t=50, b=30),
                                   legend=dict(orientation="h", y=1.12))
                st.plotly_chart(figb, use_container_width=True)

                # 잔차 분포 (정규분포에 가까울수록 모델 적합 良) + 요약
                rv = r_df[view_var].values.astype(float)
                rv = rv[~np.isnan(rv)]
                figh = go.Figure(go.Histogram(x=rv, nbinsx=40, marker_color=MUTED_PALETTE[4]))
                figh.update_layout(height=260, template="plotly_white",
                                   title=f"{view_var} 잔차 분포", bargap=0.05,
                                   margin=dict(l=50, r=20, t=40, b=30))
                st.plotly_chart(figh, use_container_width=True)
                b1, b2, b3 = st.columns(3)
                b1.metric("잔차 평균", f"{np.mean(rv):+.4f}")
                b2.metric("잔차 표준편차", f"{np.std(rv):.4f}")
                b3.metric("최대 |잔차|", f"{np.max(np.abs(rv)):.4f}")
                st.caption("잔차가 0 주변에 고르게(정규분포에 가깝게) 분포할수록 예측모델이 정상 패턴을 잘 학습한 것이고, "
                           "특정 구간에서 잔차가 크게 튀면 그 구간이 이상 후보입니다 — 이 잔차가 Scorer를 거쳐 이상 점수가 됩니다.")
            else:
                st.caption("위 버튼을 누르면 선택한 모델의 백테스팅 예측과 잔차분석을 계산합니다.")

    with tab_metrics:
        if not st.session_state.detected:
            st.warning("아직 탐지 결과가 없습니다.")
        elif not R.get("metrics"):
            st.warning("평가 지표가 없습니다.")
        else:
            # ══ 이상탐지 평가 — 정답 0/1 라벨 기준 AUC·F1 등 ══
            st.caption("정답 0/1 라벨 기준 지도식 평가입니다. "
                       "AUC·F1·정밀도·재현율·정확도는 모두 1에 가까울수록 좋습니다.")
            dfm = pd.DataFrame(R["metrics"])
            dfm["조합"] = dfm["Model"] + " · " + dfm["Scorer"]
            metric_cols = ["AUC_ROC", "AUC_PR", "F1", "Precision", "Recall", "Accuracy"]

            # 테스트 구간에 양성(1) 라벨이 없으면 AUC/F1 등이 모든 행에서 NaN →
            # idxmax()가 NaN을 반환해 .loc[NaN]이 KeyError가 나므로, NaN 여부를 먼저 가드한다.
            if not dfm[["AUC_ROC", "F1"]].notna().any().any():
                st.info(
                    "테스트 구간에 양성(이상=1) 라벨이 없어 AUC·F1 등 지도식 지표를 계산할 수 없습니다. "
                    "'설정 & 탐지' 탭에서 학습 비율을 낮춰 이상 구간이 테스트에 포함되도록 조정해 보세요. "
                    "(아래 표/히트맵은 참고용으로 그대로 표시됩니다.)",
                    icon="🔴",
                )
            k1, k2, k3 = st.columns(3)
            if dfm["AUC_ROC"].notna().any():
                best = dfm.loc[dfm["AUC_ROC"].idxmax()]
                k1.metric("🥇 최고 AUC-ROC 조합", f"{best['조합']}", f"{best['AUC_ROC']:.3f}")
            else:
                k1.metric("🥇 최고 AUC-ROC 조합", "—", "계산 불가", delta_color="off")
            if dfm["F1"].notna().any():
                k2.metric("최고 F1", f"{dfm['F1'].max():.3f}",
                          dfm.loc[dfm['F1'].idxmax(), '조합'], delta_color="off")
            else:
                k2.metric("최고 F1", "—", "계산 불가", delta_color="off")
            k3.metric("평가 조합 수", f"{len(dfm)}개")

            st.markdown("---")
            st.subheader("📈 지표별 조합 비교")
            fig = make_subplots(rows=2, cols=3, subplot_titles=metric_cols,
                                horizontal_spacing=0.08, vertical_spacing=0.30)
            pos = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            for mi, mc in enumerate(metric_cols):
                r, c = pos[mi]
                fig.add_trace(go.Bar(x=dfm["조합"], y=dfm[mc], showlegend=False,
                                     marker_color=[MODEL_COLORS.get(m, "#5B7DB1") for m in dfm["Model"]]),
                              row=r, col=c)
            fig.update_xaxes(tickangle=-40, tickfont=dict(size=8))
            fig.update_layout(height=720, template="plotly_white",
                              margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🗺️ 조합 × 지표 히트맵")
            heat = dfm.set_index("조합")[metric_cols]
            figh = go.Figure(go.Heatmap(
                z=heat.values, x=metric_cols, y=heat.index.tolist(),
                text=heat.values.round(3), texttemplate="%{text}",
                textfont=dict(size=13),
                colorscale=MUTED_HEAT, colorbar=dict(title="값"),
            ))
            figh.update_layout(height=min(620, max(260, 30 * len(heat))),
                               template="plotly_white", font=dict(size=13),
                               margin=dict(l=150, r=20, t=20, b=55))
            st.plotly_chart(figh, use_container_width=True)

            st.subheader("📋 지표 상세 (AUC-ROC 내림차순)")
            show = dfm[["조합"] + metric_cols].sort_values("AUC_ROC", ascending=False).reset_index(drop=True)
            st.dataframe(show.style.format({c: "{:.3f}" for c in metric_cols}),
                         use_container_width=True, hide_index=True)

except Exception:
    st.error("데이터 처리 중 오류가 발생했습니다.")
    st.code(traceback.format_exc())
