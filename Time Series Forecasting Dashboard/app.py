import math
import threading
import time
import traceback

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.theta import ThetaForecaster
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
from prophet import Prophet

from sktime.performance_metrics.forecasting import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_absolute_scaled_error,
)
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="시계열 예측 대시보드", layout="wide", page_icon="📈")

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.train_settings = {}
    st.session_state.train_results = {}

# ──────────────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* KPI 카드 */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea11, #764ba211);
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label {
    font-size: 0.85rem !important;
    color: #555 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
/* 탭 스타일 */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
/* 사이드바 헤더 */
section[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
}
/* expander 내부 패딩 */
div[data-testid="stExpander"] details {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 시계열 예측 멀티모델 대시보드")

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
MODEL_COLORS = {
    "Naive": "#636EFA",
    "SMA": "#19D3F3",
    "ExpSmoothing": "#EF553B",
    "Holt": "#FF6692",
    "HoltWinters": "#B6E880",
    "STL": "#FF97FF",
    "AutoARIMA": "#AB63FA",
    "Theta": "#FFA15A",
    "Prophet": "#00CC96",
}

INTERVAL_CAPABLE = {"AutoARIMA", "Theta", "Prophet"}
ALL_MODEL_NAMES = list(MODEL_COLORS.keys())

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
st.sidebar.header("📂 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("단변량 시계열 CSV 파일을 올려주세요", type="csv")

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        # ──────────────────────────────────────
        # 컬럼 자동 감지 & 전처리
        # ──────────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.error("⚠️ 데이터에 숫자형(Target) 컬럼이 없습니다.")
            st.stop()

        target_col = numeric_cols[-1]
        time_options = [c for c in df.columns if c != target_col]
        time_col = time_options[0] if time_options else df.columns[0]

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.dropna(subset=[time_col], inplace=True)
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)

        # 주기(freq) 자동 추론
        inferred_freq = pd.infer_freq(df.index)
        # Period용 freq 변환 (ME→M, QE→Q, YE→Y 등)
        _freq_map = {"ME": "M", "MS": "M", "QE": "Q", "QS": "Q", "YE": "Y", "YS": "Y",
                     "BYE": "BY", "BQE": "BQ", "BME": "BM", "BMS": "BM",
                     "QE-DEC": "Q-DEC", "QE-MAR": "Q-MAR", "QS-DEC": "Q-DEC",
                     "YE-DEC": "Y-DEC", "YE-JAN": "Y-JAN", "YS-DEC": "Y-DEC"}
        period_freq = _freq_map.get(inferred_freq, inferred_freq) if inferred_freq else None

        # 데이터 주기 라벨
        _freq_label = {
            "M": "월", "MS": "월", "ME": "월",
            "W": "주", "D": "일", "B": "영업일",
            "Q": "분기", "QS": "분기", "QE": "분기",
            "Y": "연", "YS": "연", "YE": "연", "H": "시간",
        }
        freq_txt = _freq_label.get(inferred_freq, inferred_freq if inferred_freq else "불명")

        # ──────────────────────────────────────
        # CSV fingerprint 변경 감지 — 데이터셋 의존 위젯의 옛 상태가
        # 새 옵션 목록에 없으면 Streamlit이 오류를 내므로 사전 무효화한다.
        # ──────────────────────────────────────
        _dynamic_keys = ["date_range", "horizon_preset", "horizon_custom"]
        _csv_fp = (uploaded_file.name, len(df), str(df.index[0]), str(df.index[-1]))
        if st.session_state.get("_csv_fingerprint") != _csv_fp:
            for _k in _dynamic_keys:
                st.session_state.pop(_k, None)
            # 새 CSV의 옛 학습 결과 캐시가 hit되어 stale 예측이 노출되지 않도록 함께 초기화
            st.session_state.trained = False
            st.session_state.train_settings = {}
            st.session_state.train_results = {}
            st.session_state["_csv_fingerprint"] = _csv_fp

        # ──────────────────────────────────────
        # 날짜 범위 필터 (데이터에 존재하는 날짜 기반)
        # ──────────────────────────────────────
        st.sidebar.subheader("📅 데이터 범위")
        all_dates = df.index.sort_values()
        date_labels = [d.strftime("%Y-%m-%d") for d in all_dates]

        # 양쪽 끝 슬라이더로 범위 선택
        start_label, end_label = st.sidebar.select_slider(
            "사용할 구간",
            options=date_labels,
            value=(date_labels[0], date_labels[-1]),
            key="date_range",
        )
        start_date = pd.Timestamp(start_label)
        end_date = pd.Timestamp(end_label)
        df = df.loc[start_date:end_date]

        y = df[target_col].dropna()
        data_size = len(y)

        if data_size < 3:
            st.error("선택 구간의 데이터가 너무 적습니다 (최소 3개 필요).")
            st.stop()

        st.sidebar.caption(f"선택: {start_label} ~ {end_label} ({data_size}개)")

        # ──────────────────────────────────────
        # 분석 설정
        # ──────────────────────────────────────
        st.sidebar.header("⚙️ 분석 설정")

        # 시평 프리셋 + 직접 입력
        max_horizon = data_size - 2
        _freq_counter = {"월": "개월", "주": "주", "일": "일", "영업일": "영업일",
                         "분기": "분기", "연": "년", "시간": "시간"}
        freq_unit = _freq_counter.get(freq_txt, freq_txt)

        preset_list = []
        for v in [3, 6, 12, 24, 36]:
            if v <= max_horizon:
                preset_list.append(f"{v}스텝 ({v}{freq_unit})")
        preset_list.append("직접 입력")

        preset_choice = st.sidebar.selectbox("예측 시평", preset_list, key="horizon_preset")

        if preset_choice == "직접 입력":
            horizon = st.sidebar.slider(
                "시평 값 설정", min_value=1, max_value=max_horizon,
                value=min(12, max_horizon), step=1,
                key="horizon_custom",
            )
        else:
            horizon = int(preset_choice.split("스텝")[0])

        # 학습/테스트 분할 시각화 바
        train_remaining = data_size - horizon
        train_pct = train_remaining / data_size * 100
        test_pct = 100 - train_pct
        st.sidebar.markdown(
            f'<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;font-size:11px;text-align:center;line-height:18px;">'
            f'<div style="width:{train_pct}%;background:#636EFA;color:white;">학습 {train_remaining}</div>'
            f'<div style="width:{test_pct}%;background:#EF553B;color:white;">테스트 {horizon}</div>'
            f'</div>', unsafe_allow_html=True,
        )
        st.sidebar.caption(f"전체 {data_size}개 = 학습 {train_remaining}개 + 테스트 {horizon}개 (주기: {freq_txt})")

        sp = st.sidebar.number_input("계절 주기 (Seasonal Period)", min_value=1, value=12, step=1, key="sp")
        # default 대신 session_state로 초기값 관리 — Session State API와 default 동시 사용 시
        # Streamlit이 경고를 출력하므로 default 인자는 제거.
        if "selected_models" not in st.session_state:
            st.session_state["selected_models"] = list(ALL_MODEL_NAMES)
        selected_models = st.sidebar.multiselect("사용할 모델 선택", ALL_MODEL_NAMES, key="selected_models")
        forecast_mode = st.sidebar.selectbox(
            "예측 방식",
            ["기본 (단일 학습)", "Expanding Window (확장 윈도우)", "Rolling Window (고정 윈도우)"],
            index=0,
            help=(
                "기본: 학습 데이터로 한 번 학습 후 전체 구간 예측.\n"
                "Expanding: 1-step씩 실제값을 추가하며 반복 예측 (학습 데이터 증가).\n"
                "Rolling: 고정 크기 윈도우가 1-step씩 이동하며 반복 예측."
            ),
            key="forecast_mode",
        )
        use_expanding = forecast_mode.startswith("Expanding")
        use_rolling = forecast_mode.startswith("Rolling")
        show_ci = st.sidebar.checkbox("신뢰구간 표시 (지원 모델만)", value=False, key="show_ci")
        if len(selected_models) == 0:
            st.warning("최소 1개 이상의 모델을 선택해주세요.")
            st.stop()

        st.sidebar.markdown("---")
        _reset_keys = [
            "date_range", "horizon_preset", "horizon_custom",
            "sp", "selected_models", "forecast_mode", "show_ci",
            "naive_strat", "sma_win", "es_alpha", "holt_alpha", "holt_beta",
            "hw_alpha", "hw_beta", "hw_seasonal",
            "prophet_season_mode", "prophet_cp_scale", "prophet_season_scale",
        ]
        if st.sidebar.button("🔄 설정 초기화", use_container_width=True):
            for k in _reset_keys:
                if k in st.session_state:
                    del st.session_state[k]
            # default가 있는 위젯은 명시적으로 default를 다시 set —
            # Streamlit 위젯 캐시 quirk로 `del` 만으로는 복귀가 보장되지 않는 경우가 있음.
            st.session_state["selected_models"] = list(ALL_MODEL_NAMES)
            st.session_state.trained = False
            st.session_state.train_settings = {}
            st.session_state.train_results = {}
            st.rerun()

        st.sidebar.markdown(
            "<p style='text-align:center; margin-top:8px; color:rgba(49, 51, 63, 0.6); font-size:14px;'>모델 설정탭으로 이동</p>",
            unsafe_allow_html=True,
        )

        # ──────────────────────────────────────
        # Train / Test 분할
        # ──────────────────────────────────────
        train_size = int(len(y) - horizon)
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        fh = list(range(1, horizon + 1))

        # 모델 학습용 PeriodIndex 시리즈 (Theta, STL 등에 필요)
        if period_freq:
            y_train_m = y_train.copy()
            y_train_m.index = y_train_m.index.to_period(period_freq)
            y_test_m = y_test.copy()
            y_test_m.index = y_test_m.index.to_period(period_freq)
            y_m = y.copy()
            y_m.index = y_m.index.to_period(period_freq)
        else:
            # freq 추론 실패 시 RangeIndex로 변환 (Theta 등 freq 필수 모델 대응)
            y_train_m = y_train.reset_index(drop=True)
            y_test_m = y_test.reset_index(drop=True)
            y_m = y.reset_index(drop=True)

        # ──────────────────────────────────────
        # 탭 구성 (5탭)
        # ──────────────────────────────────────
        tab_settings, tab_overview, tab_forecast, tab_metrics = st.tabs(
            ["⚙️ 모델 설정", "📊 데이터 개요", "🎯 모델별 예측", "💡 성능 비교"]
        )

        # ============================================================
        # 탭 1: 데이터 개요
        # ============================================================
        with tab_overview:
            st.subheader("시계열 원본 데이터")
            st.caption(f"데이터 주기: {freq_txt} | 총 {len(y)}개 | 훈련 {len(y_train)}개 · 테스트 {len(y_test)}개")

            fig_hist = go.Figure()
            # 훈련 데이터 끝 + 테스트 첫 포인트를 이어서 끊김 방지
            train_x = list(y_train.index) + [y_test.index[0]]
            train_y = list(y_train.values) + [y_test.values[0]]
            fig_hist.add_trace(go.Scatter(
                x=train_x, y=train_y,
                mode="lines+markers", name="훈련 데이터",
                marker=dict(size=3), line=dict(color="#636EFA", width=1.5),
            ))
            fig_hist.add_trace(go.Scatter(
                x=y_test.index, y=y_test.values,
                mode="lines+markers", name="테스트 데이터",
                marker=dict(size=3), line=dict(color="#EF553B", width=1.5),
            ))
            split_x = y_test.index[0].isoformat()
            fig_hist.add_shape(
                type="line", x0=split_x, x1=split_x, y0=0, y1=1,
                yref="paper", line=dict(color="gray", width=1.5, dash="dash"),
            )
            fig_hist.add_annotation(
                x=split_x, y=1, yref="paper",
                text="Train / Test 분할", showarrow=False,
                xanchor="left", yanchor="bottom", font=dict(size=11, color="gray"),
            )
            fig_hist.update_layout(
                height=380, template="plotly_white",
                margin=dict(l=50, r=20, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown("**데이터 정보**")
            st.write(f"- 전체 데이터 수: **{len(y)}** 개")
            st.write(f"- 기간: {y.index.min().strftime('%Y-%m-%d')} ~ {y.index.max().strftime('%Y-%m-%d')}")
            st.write(f"- 훈련 세트: **{len(y_train)}** 개 / 테스트 세트: **{len(y_test)}** 개")

            st.markdown("---")
            st.markdown(f"**기초 통계량** (`{target_col}` 컬럼)")
            desc_df = y.describe().round(2).reset_index()
            desc_df.columns = ["", target_col]
            st.markdown(
                desc_df.style.set_properties(**{
                    "text-align": "center",
                    "font-size": "14px",
                }).set_table_styles([
                    {"selector": "th", "props": [("text-align", "center"), ("font-size", "14px")]},
                ]).hide(axis="index").to_html(),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            # 계절 분해
            with st.expander("📈 계절 분해 (Seasonal Decomposition)", expanded=True):
                if len(y) >= sp * 2:
                    try:
                        decomp = seasonal_decompose(y, model="additive", period=sp)
                        fig_decomp = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                            shared_xaxes=True, vertical_spacing=0.06,
                        )
                        for i, (comp, clr) in enumerate([
                            (decomp.observed, "#636EFA"), (decomp.trend, "#EF553B"),
                            (decomp.seasonal, "#00CC96"), (decomp.resid, "#AB63FA"),
                        ], 1):
                            fig_decomp.add_trace(go.Scatter(
                                x=comp.index, y=comp.values, mode="lines",
                                line=dict(color=clr, width=1.2), showlegend=False,
                            ), row=i, col=1)
                        fig_decomp.update_layout(height=480, template="plotly_white", margin=dict(l=50, r=20, t=30, b=20))
                        st.plotly_chart(fig_decomp, use_container_width=True)
                    except Exception as e:
                        st.warning(f"계절 분해 실패: {e}")
                else:
                    st.info(f"계절 분해를 위해 최소 {sp * 2}개 이상의 데이터가 필요합니다.")

            st.markdown("---")
            # ACF / PACF 자기상관 분석
            with st.expander("🔍 ACF / PACF (자기상관 분석)", expanded=True):
                st.caption("자기상관함수(ACF) / 부분자기상관함수(PACF) — 시계열의 정상성 및 계절성 패턴 진단. 회색 음영은 95% 신뢰구간이며, 막대가 음영을 벗어나면 유의한 자기상관.")
                y_acf = y.dropna()
                if len(y_acf) >= 10:
                    try:
                        from statsmodels.tsa.stattools import acf as _acf_fn, pacf as _pacf_fn
                        nlags = min(40, len(y_acf) // 2 - 1)
                        acf_vals = _acf_fn(y_acf, nlags=nlags)
                        pacf_vals = _pacf_fn(y_acf, nlags=nlags)
                        ci_band = 1.96 / np.sqrt(len(y_acf))

                        fig_acf = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"],
                            horizontal_spacing=0.1,
                        )
                        # ACF: stems + markers
                        for i, v in enumerate(acf_vals):
                            fig_acf.add_trace(go.Scatter(
                                x=[i, i], y=[0, v], mode="lines",
                                line=dict(color="#636EFA", width=1.5),
                                showlegend=False, hoverinfo="skip",
                            ), row=1, col=1)
                        fig_acf.add_trace(go.Scatter(
                            x=list(range(len(acf_vals))), y=acf_vals,
                            mode="markers", marker=dict(color="#636EFA", size=6),
                            showlegend=False,
                            hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
                        ), row=1, col=1)
                        # PACF: stems + markers
                        for i, v in enumerate(pacf_vals):
                            fig_acf.add_trace(go.Scatter(
                                x=[i, i], y=[0, v], mode="lines",
                                line=dict(color="#EF553B", width=1.5),
                                showlegend=False, hoverinfo="skip",
                            ), row=1, col=2)
                        fig_acf.add_trace(go.Scatter(
                            x=list(range(len(pacf_vals))), y=pacf_vals,
                            mode="markers", marker=dict(color="#EF553B", size=6),
                            showlegend=False,
                            hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
                        ), row=1, col=2)
                        # 95% 신뢰구간 음영 + 0 기준선
                        for col_idx in [1, 2]:
                            fig_acf.add_shape(
                                type="rect", xref=f"x{col_idx}", yref=f"y{col_idx}",
                                x0=0, x1=nlags, y0=-ci_band, y1=ci_band,
                                fillcolor="rgba(150,150,150,0.18)", line_width=0, layer="below",
                            )
                            fig_acf.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=col_idx)

                        fig_acf.update_yaxes(range=[-1.05, 1.05])
                        fig_acf.update_xaxes(title_text="Lag")
                        fig_acf.update_layout(
                            height=340, template="plotly_white",
                            margin=dict(l=50, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig_acf, use_container_width=True)
                    except Exception as e:
                        st.warning(f"ACF/PACF 계산 실패: {e}")
                else:
                    st.info("ACF/PACF 분석을 위해 최소 10개 이상의 데이터가 필요합니다.")

        # ============================================================
        # 탭 2: 모델 하이퍼파라미터 설정
        # ============================================================
        with tab_settings:
            st.subheader("⚙️ 모델별 하이퍼파라미터 설정")
            st.caption("선택된 모델의 핵심 파라미터를 조절할 수 있습니다. 변경 후 하단 **🚀 학습 시작** 버튼을 누르면 적용됩니다.")

            # 파라미터 수집용 딕셔너리
            hp = {}
            active_models = [m for m in selected_models]

            # 2열 그리드로 배치
            setting_cols = st.columns(2)
            for idx, name in enumerate(active_models):
                with setting_cols[idx % 2]:
                    with st.expander(f"🔧 {name}", expanded=True):
                        if name == "Naive":
                            hp["Naive_strategy"] = st.selectbox(
                                "전략", ["last", "mean"], index=0, key="naive_strat",
                                help="last: 마지막 값 반복, mean: 평균값 반복"
                            )
                        elif name == "SMA":
                            hp["SMA_window"] = st.slider(
                                "이동평균 윈도우 (window_length)", 2, min(50, train_size), 5, key="sma_win",
                                help="최근 n개 관측치의 평균으로 예측"
                            )
                        elif name == "ExpSmoothing":
                            hp["ES_alpha"] = st.slider(
                                "평활계수 α (smoothing_level)", 0.01, 1.0, 0.3, 0.01, key="es_alpha",
                                help="1에 가까울수록 최근 데이터에 민감"
                            )
                        elif name == "Holt":
                            hp["Holt_alpha"] = st.slider("평활계수 α", 0.01, 1.0, 0.3, 0.01, key="holt_alpha")
                            hp["Holt_beta"] = st.slider("추세 평활계수 β", 0.01, 1.0, 0.05, 0.01, key="holt_beta")
                        elif name == "HoltWinters":
                            hp["HW_alpha"] = st.slider("평활계수 α", 0.01, 1.0, 0.3, 0.01, key="hw_alpha")
                            hp["HW_beta"] = st.slider("추세 평활계수 β", 0.01, 1.0, 0.05, 0.01, key="hw_beta")
                            hp["HW_seasonal"] = st.selectbox(
                                "계절성 타입", ["multiplicative", "additive"], index=0, key="hw_seasonal",
                                help="승법적: 진폭이 추세에 비례 / 가법적: 진폭 일정"
                            )
                        elif name == "STL":
                            st.info(f"계절 주기(sp={sp})가 사이드바에서 자동 연동됩니다.")
                        elif name == "AutoARIMA":
                            st.info(f"자동 파라미터 탐색 (sp={sp})")
                        elif name == "Theta":
                            st.info(f"Theta 모델 (sp={sp})")
                        elif name == "Prophet":
                            hp["Prophet_seasonality_mode"] = st.selectbox(
                                "계절성 모드", ["multiplicative", "additive"], index=0,
                                key="prophet_season_mode",
                                help="승법적: 진폭이 추세에 비례 / 가법적: 진폭 일정"
                            )
                            hp["Prophet_changepoint_scale"] = st.slider(
                                "변화점 민감도 (changepoint_prior_scale)",
                                0.001, 0.5, 0.05, 0.001, key="prophet_cp_scale",
                                help="높을수록 추세 변화에 민감 (과적합 주의), 낮을수록 부드러운 추세"
                            )
                            hp["Prophet_seasonality_scale"] = st.slider(
                                "계절성 강도 (seasonality_prior_scale)",
                                0.01, 10.0, 10.0, 0.1, key="prophet_season_scale",
                                help="높을수록 계절 패턴을 강하게 반영"
                            )

            # ── 학습 시작 버튼 (모델 설정 탭 하단, 가운데 배치) ──
            st.markdown("---")
            _btn_l, _btn_c, _btn_r = st.columns([1, 2, 1])
            with _btn_c:
                run_clicked = st.button(
                    "🚀 학습 시작", type="primary", use_container_width=True,
                    key="run_train_btn",
                )
                if st.session_state.trained:
                    st.success("✅ 학습 완료 — 결과 탭에서 확인하세요. 파라미터 변경 시 다시 클릭.")

        # ──────────────────────────────────────
        # 설정 변경 감지 (사이드바 + 하이퍼파라미터 모두 포함)
        # ──────────────────────────────────────
        _current_settings = {
            "horizon": horizon, "sp": sp, "models": sorted(selected_models),
            "mode": forecast_mode, "ci": show_ci,
            "date_range": (str(start_date), str(end_date)),
            "hp": dict(sorted(hp.items())),
        }

        # ──────────────────────────────────────
        # 모델 팩토리 (하이퍼파라미터 반영)
        # ──────────────────────────────────────
        def _build_holtwinters(data):
            seasonal_type = hp.get("HW_seasonal", "multiplicative")
            alpha = hp.get("HW_alpha", 0.3)
            beta = hp.get("HW_beta", 0.05)
            for seasonal in [seasonal_type, "additive" if seasonal_type == "multiplicative" else "multiplicative", None]:
                try:
                    m = ExponentialSmoothing(
                        trend="add", seasonal=seasonal, sp=sp if seasonal else 1,
                        smoothing_level=alpha, smoothing_trend=beta,
                    )
                    m.fit(data)
                    return m
                except Exception:
                    continue
            return None

        def _build_theta(data):
            """Theta 빌더.
            Hyndman FPP3 §8.10 / Assimakopoulos & Nikolopoulos(2000):
            표준 Theta는 multiplicative 계절분해를 가정 → y>0 필요.
            데이터에 0/음수가 있으면 R `forecast::thetaf`와 동일하게 비계절 Theta
            (deseasonalize=False)로 fallback. 이는 SES + drift 조합으로 환원되어
            계절성을 잃지만 정의상 유효한 Theta 변형이다."""
            y_arr = np.asarray(data.values, dtype=float)
            if np.any(y_arr <= 0):
                m = ThetaForecaster(sp=sp, deseasonalize=False)
            else:
                m = ThetaForecaster(sp=sp)
            m.fit(data)
            return m

        model_factories = {
            "Naive": lambda: NaiveForecaster(strategy=hp.get("Naive_strategy", "last")),
            "SMA": lambda: NaiveForecaster(strategy="mean", window_length=hp.get("SMA_window", 5)),
            "ExpSmoothing": lambda: ExponentialSmoothing(
                trend=None, seasonal=None, smoothing_level=hp.get("ES_alpha", 0.3),
            ),
            "Holt": lambda: ExponentialSmoothing(
                trend="add", seasonal=None,
                smoothing_level=hp.get("Holt_alpha", 0.3),
                smoothing_trend=hp.get("Holt_beta", 0.05),
            ),
            # STL: Cleveland(1990) 표준 jump=ceil(window/10) — R `stl()` 기본값과 동일.
            # statsmodels 기본 jump=1보다 20배 이상 빠르며 예측값은 사실상 동일.
            "STL": lambda: STLForecaster(
                sp=sp,
                seasonal_jump=1,
                trend_jump=max(1, math.ceil(
                    (math.ceil((1.5 * sp) / (1 - 1.5 / 7)) | 1) / 10
                )),
                low_pass_jump=max(1, math.ceil((sp + (0 if sp % 2 else 1)) / 10)),
            ),
            "AutoARIMA": lambda: AutoARIMA(
                sp=sp,
                seasonal=(sp > 1),
                stepwise=True,
                information_criterion="aic",
                n_jobs=1,
                random=False,
                random_state=42,
                error_action="ignore",
                suppress_warnings=True,
            ),
            # Theta는 데이터 부호에 따라 deseasonalize 분기가 필요하므로
            # _build_theta(data) 헬퍼로 호출 시점에 빌드 (HoltWinters 패턴과 동일)
        }

        # ──────────────────────────────────────
        # Prophet 헬퍼 함수
        # ──────────────────────────────────────
        def _prophet_fit_predict(y_data, h, return_ci=False, return_in_sample=False):
            """Prophet 학습 & 예측. y_data는 DatetimeIndex를 가진 Series."""
            # Prophet은 DatetimeIndex 필요 — PeriodIndex → Timestamp 변환
            idx = y_data.index
            if hasattr(idx, 'to_timestamp'):
                idx = idx.to_timestamp()
            pdf = pd.DataFrame({"ds": idx, "y": y_data.values})

            # 하이퍼파라미터 반영
            season_mode = hp.get("Prophet_seasonality_mode", "multiplicative")
            cp_scale = hp.get("Prophet_changepoint_scale", 0.05)
            season_scale = hp.get("Prophet_seasonality_scale", 10.0)

            m = Prophet(
                yearly_seasonality="auto",
                weekly_seasonality="auto",
                daily_seasonality=False,
                seasonality_mode=season_mode,
                changepoint_prior_scale=cp_scale,
                seasonality_prior_scale=season_scale,
            )
            m.fit(pdf)
            future = m.make_future_dataframe(periods=h, freq=inferred_freq or "MS")
            fc = m.predict(future)
            fc_tail = fc.tail(h)
            y_pred = pd.Series(fc_tail["yhat"].values, index=fc_tail["ds"].values)

            extras = []
            if return_ci:
                ci_df = pd.DataFrame({
                    "lower": fc_tail["yhat_lower"].values,
                    "upper": fc_tail["yhat_upper"].values,
                }, index=fc_tail["ds"].values)
                extras.append(ci_df)
            if return_in_sample:
                # 학습 구간 적합값 → in-sample 잔차
                fc_in = fc.head(len(pdf))
                in_resid = (np.asarray(y_data.values, dtype=float) - fc_in["yhat"].values).astype(float)
                in_resid = in_resid[~np.isnan(in_resid)]
                extras.append(in_resid)

            if extras:
                return (y_pred, *extras)
            return y_pred

        # ──────────────────────────────────────
        # In-sample 잔차 추출 헬퍼
        # Hyndman, Forecasting: Principles and Practice §3.3, §5.4 정의
        # ──────────────────────────────────────
        def _compute_in_sample_residuals(forecaster, name, y, _hp=hp):
            """학습된 forecaster로부터 학습 구간 in-sample 잔차(numpy array)를 추출.
            실패 시 None 반환."""
            try:
                y_arr = np.asarray(y.values, dtype=float)

                # AutoARIMA: statsmodels SARIMAXResults.resid 직접 사용 (one-step innovations)
                if name == "AutoARIMA":
                    pm = getattr(forecaster, "_forecaster", forecaster)
                    arima_res = getattr(pm, "arima_res_", None)
                    if arima_res is None and hasattr(pm, "model_"):
                        arima_res = getattr(pm.model_, "arima_res_", None)
                    if arima_res is not None:
                        resid = np.asarray(arima_res.resid, dtype=float)
                        return resid[~np.isnan(resid)]

                # Naive: 전략별 textbook 정의 (Hyndman §3.3)
                if name == "Naive":
                    strategy = _hp.get("Naive_strategy", "last")
                    if strategy == "last":
                        # ŷ_t = y_{t-1}  →  e_t = y_t - y_{t-1}
                        resid = y_arr[1:] - y_arr[:-1]
                        return resid[~np.isnan(resid)]
                    if strategy == "mean":
                        # ŷ_t = mean(y_{1..t-1})  →  e_t = y_t - 누적평균
                        cumsum = np.cumsum(y_arr)
                        means = cumsum[:-1] / np.arange(1, len(y_arr))
                        resid = y_arr[1:] - means
                        return resid[~np.isnan(resid)]
                    if strategy == "drift":
                        # ŷ_t = y_{t-1} + (y_{t-1} - y_1)/(t-2)
                        # 단순화: t=2부터 drift 계산
                        n = len(y_arr)
                        resid_list = []
                        for t in range(2, n):
                            slope = (y_arr[t-1] - y_arr[0]) / (t - 1)
                            pred = y_arr[t-1] + slope
                            resid_list.append(y_arr[t] - pred)
                        return np.array(resid_list, dtype=float)

                # SMA: ŷ_t = mean(y_{t-k..t-1})  →  e_t = y_t - rolling mean
                if name == "SMA":
                    window = int(_hp.get("SMA_window", 5))
                    if len(y_arr) <= window:
                        return None
                    resid_list = []
                    for t in range(window, len(y_arr)):
                        pred = float(np.mean(y_arr[t-window:t]))
                        resid_list.append(y_arr[t] - pred)
                    return np.array(resid_list, dtype=float)

                # STL: 학습 시 분해된 remainder(R_t)를 그대로 반환.
                # Hyndman FPP3 §6.6: y_t = T_t + S_t + R_t, 적합값=T+S, 잔차=R.
                # forecaster.predict(fh=전체훈련인덱스) 일반 fallback은 sktime 내부에서
                # O(n²) 재예측이 발생하여 매우 느리므로 우회한다.
                if name == "STL":
                    resid_attr = getattr(forecaster, "resid_", None)
                    if resid_attr is not None:
                        resid = np.asarray(resid_attr, dtype=float)
                        return resid[~np.isnan(resid)]

                # ETS / Holt / HoltWinters / Theta 등:
                # statsmodels 백엔드의 fittedvalues 우선 시도, 없으면 sktime predict
                if hasattr(forecaster, "_fitted_forecaster"):
                    inner = forecaster._fitted_forecaster
                    fv = getattr(inner, "fittedvalues", None)
                    if fv is not None:
                        fv_arr = np.asarray(fv, dtype=float)
                        if len(fv_arr) == len(y_arr):
                            resid = y_arr - fv_arr
                            return resid[~np.isnan(resid)]
                    res = getattr(inner, "resid", None)
                    if res is not None:
                        res_arr = np.asarray(res, dtype=float)
                        if len(res_arr) == len(y_arr):
                            return res_arr[~np.isnan(res_arr)]

                # 일반 sktime fallback: 학습 구간 fh로 in-sample 예측
                from sktime.forecasting.base import ForecastingHorizon
                fh_in = ForecastingHorizon(y.index, is_relative=False)
                y_fit = forecaster.predict(fh=fh_in)
                fit_arr = np.asarray(y_fit.values, dtype=float)
                if len(fit_arr) != len(y_arr):
                    return None
                resid = y_arr - fit_arr
                return resid[~np.isnan(resid)]
            except Exception:
                return None

        # ──────────────────────────────────────
        # 정보 기준(AIC/BIC/HQIC) + sigma2 추출
        # FPP3 §8.6 (ETS) / §9.7 (ARIMA): likelihood 기반 모델만 정의됨.
        # Naive/SMA/STL/Prophet은 likelihood 미정의 → 빈 dict 반환.
        # ──────────────────────────────────────
        def _extract_ic(forecaster, name, residuals_arr):
            ic = {}
            try:
                if name == "AutoARIMA":
                    pm = getattr(forecaster, "_forecaster", forecaster)
                    ar = getattr(pm, "arima_res_", None)
                    if ar is None and hasattr(pm, "model_"):
                        ar = getattr(pm.model_, "arima_res_", None)
                    if ar is not None:
                        for k in ("aic", "bic", "hqic"):
                            v = getattr(ar, k, None)
                            try:
                                vf = float(v)
                                if np.isfinite(vf):
                                    ic[k] = vf
                            except (TypeError, ValueError):
                                pass
                        try:
                            if "sigma2" in ar.params.index:
                                ic["sigma2"] = float(ar.params["sigma2"])
                        except Exception:
                            pass
                    return ic

                # ETS-family + Theta: statsmodels 백엔드
                inner = getattr(forecaster, "_fitted_forecaster", None)
                if inner is not None:
                    for k in ("aic", "bic", "hqic"):
                        v = getattr(inner, k, None)
                        try:
                            vf = float(v)
                            if np.isfinite(vf):
                                ic[k] = vf
                        except (TypeError, ValueError):
                            pass
                    # sigma2: statsmodels ETSResults는 sse/nobs 또는 .params['sigma2']
                    sse = getattr(inner, "sse", None)
                    nobs = getattr(inner, "nobs", None)
                    try:
                        if sse is not None and nobs is not None and float(nobs) > 1:
                            ic["sigma2"] = float(sse) / float(nobs)
                    except (TypeError, ValueError):
                        pass

                # 잔차 분산 fallback (모든 모델)
                if "sigma2" not in ic and residuals_arr is not None and len(residuals_arr) > 1:
                    ic["sigma2"] = float(np.var(residuals_arr, ddof=1))
            except Exception:
                pass
            return ic

        # ──────────────────────────────────────
        # 모델별 최소 데이터 조건 체크
        # ──────────────────────────────────────
        def _check_min_data(name, n, _hp=hp):
            """학습 데이터 크기 n이 모델 최소 요구를 충족하는지 확인"""
            if name == "SMA":
                return n >= _hp.get("SMA_window", 5)
            if name == "STL":
                return n >= sp * 2
            if name == "Theta":
                return n >= sp * 2
            if name == "AutoARIMA":
                return n >= sp * 2 + 1
            if name == "HoltWinters":
                return n >= sp * 2
            if name == "Prophet":
                return n >= max(10, sp * 2)
            return n >= 2

        # ──────────────────────────────────────
        # 모델 학습 & 평가
        # ──────────────────────────────────────
        # NaN 제거된 학습 데이터 준비
        y_train_m_clean = y_train_m.dropna()
        y_m_clean = y_m.dropna()

        if run_clicked:
            # 새로 학습 실행
            predictions_test = {}
            predictions_future = {}
            intervals_future = {}
            metrics_results = []
            fitted_arima = None  # AutoARIMA 적합 모델 보관 (잔차 진단용)
            in_sample_residuals = {}  # 모델별 in-sample 잔차 (잔차 진단용)
            in_sample_ic = {}  # 모델별 정보 기준 (AIC/BIC/HQIC/sigma2)

            if use_expanding:
                mode_msg = "Expanding Window Forecast"
            elif use_rolling:
                mode_msg = "Rolling Window Forecast"
            else:
                mode_msg = "모델 학습"

            total_models = len(selected_models)
            # 진행 UI는 모델 설정 탭 안에서만 렌더 (다른 탭에 새지 않게)
            overall_progress = tab_settings.progress(0.0, text=f"⏳ {mode_msg} 준비 중... (0/{total_models})")
            train_start_time = time.time()

            def _fmt_secs(s: float) -> str:
                s = int(max(0, s))
                if s >= 60:
                    return f"{s // 60}분 {s % 60}초"
                return f"{s}초"

            # 진행률 ticker용 공유 상태 (메인 스레드 = 쓰기, 백그라운드 = 읽기)
            ticker_state = {
                "done": 0,
                "name": selected_models[0] if selected_models else "",
                "model_started_at": train_start_time,
                "stop": False,
            }

            def _render_progress():
                now = time.time()
                elapsed_total = now - train_start_time
                done = ticker_state["done"]
                name_cur = ticker_state["name"]
                model_started = ticker_state["model_started_at"]

                if done >= total_models:
                    overall_progress.progress(
                        1.0,
                        text=f"✅ 학습 완료 ({total_models}/{total_models}) · 총 소요 {_fmt_secs(elapsed_total)}",
                    )
                    return

                if done == 0:
                    pct = 0.0
                    text = (
                        f"⏳ [0%] {name_cur} 학습 중... (0/{total_models}) · "
                        f"경과 {_fmt_secs(elapsed_total)}"
                    )
                else:
                    avg_per_model = (model_started - train_start_time) / done
                    elapsed_current = max(0.0, now - model_started)
                    within = min(elapsed_current / avg_per_model, 0.99) if avg_per_model > 0 else 0.0
                    pct = (done + within) / total_models
                    text = (
                        f"⏳ [{int(pct * 100)}%] {name_cur} 학습 중... "
                        f"({done}/{total_models}) · 경과 {_fmt_secs(elapsed_total)}"
                    )
                overall_progress.progress(min(pct, 0.999), text=text)

            def _ticker():
                while not ticker_state["stop"]:
                    try:
                        _render_progress()
                    except Exception:
                        pass
                    time.sleep(1.0)

            ticker_thread = threading.Thread(target=_ticker, daemon=True)
            add_script_run_ctx(ticker_thread)
            ticker_thread.start()

            for model_idx, name in enumerate(selected_models):
                ticker_state["name"] = name
                ticker_state["model_started_at"] = time.time()
                _render_progress()
                try:
                    if not _check_min_data(name, len(y_train_m_clean)):
                        tab_settings.warning(f"⚠️ {name} — 학습 데이터({len(y_train_m_clean)}개)가 최소 요구 조건에 부족하여 건너뜁니다.")
                        continue

                    if use_expanding or use_rolling:
                        # ── Expanding / Rolling Window: 1-step 반복 예측 ──
                        iter_preds = []
                        mode_label = "Expanding" if use_expanding else "Rolling"
                        progress_bar = tab_settings.progress(0, text=f"{name} {mode_label}...")
                        for step in range(horizon):
                            if use_expanding:
                                y_window = y_m_clean.iloc[:train_size + step]
                            else:
                                y_window = y_m_clean.iloc[step:train_size + step]
                            if not _check_min_data(name, len(y_window)):
                                iter_preds.append(np.nan)
                                progress_bar.progress((step + 1) / horizon)
                                continue
                            if name == "Prophet":
                                # Prophet은 DatetimeIndex 필요 — 원본 y에서 동일 범위 슬라이스
                                if use_expanding:
                                    y_window_dt = y.iloc[:train_size + step]
                                else:
                                    y_window_dt = y.iloc[step:train_size + step]
                                pred_1 = _prophet_fit_predict(y_window_dt, 1)
                                iter_preds.append(pred_1.values[0])
                            elif name == "HoltWinters":
                                forecaster = _build_holtwinters(y_window)
                                if forecaster is None:
                                    iter_preds.append(np.nan)
                                    progress_bar.progress((step + 1) / horizon)
                                    continue
                                pred_1 = forecaster.predict(fh=[1])
                                iter_preds.append(pred_1.values[0])
                            elif name == "Theta":
                                forecaster = _build_theta(y_window)
                                pred_1 = forecaster.predict(fh=[1])
                                iter_preds.append(pred_1.values[0])
                            else:
                                forecaster = model_factories[name]()
                                forecaster.fit(y_window)
                                pred_1 = forecaster.predict(fh=[1])
                                iter_preds.append(pred_1.values[0])
                            progress_bar.progress((step + 1) / horizon)
                        progress_bar.empty()

                        if all(np.isnan(v) for v in iter_preds):
                            tab_settings.warning(f"⚠️ {name} {mode_label} Forecast 실패 — 모든 스텝에서 데이터 부족")
                            continue

                        y_pred_test = pd.Series(iter_preds, index=y_test.index, name=target_col).interpolate()
                    else:
                        # ── 기본: 한 번에 전체 예측 ──
                        if name == "Prophet":
                            y_pred_test = _prophet_fit_predict(y_train, horizon)
                            y_pred_test.index = y_test.index
                        elif name == "HoltWinters":
                            forecaster = _build_holtwinters(y_train_m_clean)
                            if forecaster is None:
                                tab_settings.warning(f"⚠️ {name} 모델 학습 실패")
                                continue
                            y_pred_test = forecaster.predict(fh=fh)
                            y_pred_test.index = y_test.index
                        elif name == "Theta":
                            forecaster = _build_theta(y_train_m_clean)
                            y_pred_test = forecaster.predict(fh=fh)
                            y_pred_test.index = y_test.index
                        else:
                            forecaster = model_factories[name]()
                            forecaster.fit(y_train_m_clean)
                            y_pred_test = forecaster.predict(fh=fh)
                            y_pred_test.index = y_test.index

                    predictions_test[name] = y_pred_test

                    rmse = root_mean_squared_error(y_test, y_pred_test)
                    mae = mean_absolute_error(y_test, y_pred_test)
                    mse = mean_squared_error(y_test, y_pred_test)
                    mape = mean_absolute_percentage_error(y_test, y_pred_test, symmetric=False)
                    mase = mean_absolute_scaled_error(y_test, y_pred_test, y_train=y_train)

                    # 편향 진단: RSFE(누적 오차) 및 TS(Tracking Signal)
                    errors = y_test.values - y_pred_test.values
                    rsfe = float(np.sum(errors))
                    ts = float(rsfe / mae) if mae > 0 else 0.0

                    # MdRAE: naive(마지막 관측치 유지) 벤치마크 대비 상대오차의 중앙값
                    naive_pred = np.full_like(y_test.values, y_train.iloc[-1], dtype=float)
                    naive_errors = y_test.values - naive_pred
                    denom = np.where(np.abs(naive_errors) < 1e-10, 1e-10, np.abs(naive_errors))
                    mdrae = float(np.median(np.abs(errors) / denom))

                    metrics_results.append({
                        "Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae,
                        "MAPE(%)": mape * 100, "MASE": mase,
                        "MdRAE": mdrae, "RSFE": rsfe, "TS": ts,
                    })

                    # 전체 데이터로 재학습 → 미래 예측 + in-sample 잔차
                    if name == "Prophet":
                        if show_ci:
                            y_fut, ci_prophet, in_resid_p = _prophet_fit_predict(
                                y, horizon, return_ci=True, return_in_sample=True,
                            )
                            intervals_future[name] = ci_prophet
                        else:
                            y_fut, in_resid_p = _prophet_fit_predict(
                                y, horizon, return_in_sample=True,
                            )
                        predictions_future[name] = y_fut
                        _prophet_resid = (
                            np.asarray(in_resid_p, dtype=float)
                            if in_resid_p is not None and len(in_resid_p) > 0
                            else None
                        )
                        if _prophet_resid is not None:
                            in_sample_residuals[name] = _prophet_resid
                        # Prophet은 likelihood 기반 IC 미정의 → sigma2(잔차 분산)만 fallback
                        _ic = _extract_ic(None, name, _prophet_resid)
                        if _ic:
                            in_sample_ic[name] = _ic
                    elif name == "HoltWinters":
                        forecaster2 = _build_holtwinters(y_m_clean)
                        if forecaster2 is None:
                            continue
                        y_fut = forecaster2.predict(fh=fh)
                        if hasattr(y_fut.index, 'to_timestamp'):
                            y_fut.index = y_fut.index.to_timestamp()
                        predictions_future[name] = y_fut

                        in_resid = _compute_in_sample_residuals(forecaster2, name, y_m_clean)
                        if in_resid is not None and len(in_resid) > 0:
                            in_sample_residuals[name] = in_resid
                        _ic = _extract_ic(forecaster2, name, in_resid)
                        if _ic:
                            in_sample_ic[name] = _ic
                    elif name == "Theta":
                        forecaster2 = _build_theta(y_m_clean)
                        y_fut = forecaster2.predict(fh=fh)
                        if hasattr(y_fut.index, 'to_timestamp'):
                            y_fut.index = y_fut.index.to_timestamp()
                        predictions_future[name] = y_fut

                        in_resid = _compute_in_sample_residuals(forecaster2, name, y_m_clean)
                        if in_resid is not None and len(in_resid) > 0:
                            in_sample_residuals[name] = in_resid
                        _ic = _extract_ic(forecaster2, name, in_resid)
                        if _ic:
                            in_sample_ic[name] = _ic

                        if show_ci:
                            try:
                                ci = forecaster2.predict_interval(fh=fh, coverage=0.9)
                                if hasattr(ci.index, 'to_timestamp'):
                                    ci.index = ci.index.to_timestamp()
                                intervals_future[name] = ci
                            except Exception:
                                pass
                    else:
                        forecaster2 = model_factories[name]()
                        forecaster2.fit(y_m_clean)
                        y_fut = forecaster2.predict(fh=fh)
                        if hasattr(y_fut.index, 'to_timestamp'):
                            y_fut.index = y_fut.index.to_timestamp()
                        predictions_future[name] = y_fut

                        if name == "AutoARIMA":
                            fitted_arima = forecaster2

                        in_resid = _compute_in_sample_residuals(forecaster2, name, y_m_clean)
                        if in_resid is not None and len(in_resid) > 0:
                            in_sample_residuals[name] = in_resid
                        _ic = _extract_ic(forecaster2, name, in_resid)
                        if _ic:
                            in_sample_ic[name] = _ic

                        if show_ci and name in INTERVAL_CAPABLE:
                            try:
                                ci = forecaster2.predict_interval(fh=fh, coverage=0.9)
                                if hasattr(ci.index, 'to_timestamp'):
                                    ci.index = ci.index.to_timestamp()
                                intervals_future[name] = ci
                            except Exception:
                                pass

                except Exception as e:
                    tab_settings.warning(f"⚠️ {name} 모델 실행 오류: {e}")
                finally:
                    ticker_state["done"] = model_idx + 1

            # ticker 정지 후 최종 100% 렌더
            ticker_state["stop"] = True
            ticker_thread.join(timeout=2.0)
            _render_progress()

            # 학습 결과를 세션에 저장
            st.session_state.trained = True
            st.session_state.train_settings = _current_settings
            st.session_state.train_results = {
                "predictions_test": predictions_test,
                "predictions_future": predictions_future,
                "intervals_future": intervals_future,
                "metrics_results": metrics_results,
                "fitted_arima": fitted_arima,
                "in_sample_residuals": in_sample_residuals,
                "in_sample_ic": in_sample_ic,
            }
        elif st.session_state.trained:
            # 기존 결과 로드
            predictions_test = st.session_state.train_results["predictions_test"]
            predictions_future = st.session_state.train_results["predictions_future"]
            intervals_future = st.session_state.train_results["intervals_future"]
            metrics_results = st.session_state.train_results["metrics_results"]
            fitted_arima = st.session_state.train_results.get("fitted_arima")
            in_sample_residuals = st.session_state.train_results.get("in_sample_residuals", {})
            in_sample_ic = st.session_state.train_results.get("in_sample_ic", {})
        else:
            # 학습 전 — 빈 상태로 초기화 (결과 탭들이 NameError 없이 렌더되도록)
            predictions_test = {}
            predictions_future = {}
            intervals_future = {}
            metrics_results = []
            fitted_arima = None
            in_sample_residuals = {}
            in_sample_ic = {}

        # ============================================================
        # 탭 3: 모델별 예측
        # ============================================================
        with tab_forecast:
            if not predictions_future:
                st.warning("학습된 모델이 없습니다.")
            else:
                _mode_short = "Expanding" if use_expanding else ("Rolling" if use_rolling else "단일 학습")
                st.subheader("통합 예측 비교")
                st.caption(f"예측 방식: {_mode_short} | 주기: {freq_txt} | Horizon: {horizon}스텝({horizon}{freq_txt})")
                fig_combined = go.Figure()
                context_len_combined = min(len(y), horizon * 3)
                y_ctx_combined = y.iloc[-context_len_combined:]
                fig_combined.add_trace(go.Scatter(
                    x=y_ctx_combined.index, y=y_ctx_combined.values, mode="lines", name="실제 데이터",
                    line=dict(color="#B2BABB", width=2), showlegend=False,
                ))
                _label_info = []  # (끝 y값, 모델명, 색상) 수집
                for name, y_future in predictions_future.items():
                    color = MODEL_COLORS[name]
                    hex_c = color.lstrip("#")
                    r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                    # 연결선
                    fig_combined.add_trace(go.Scatter(
                        x=[y.index[-1], y_future.index[0]],
                        y=[y.values[-1], y_future.values[0]],
                        mode="lines", line=dict(color=color, width=2, dash="dot"),
                        showlegend=False,
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=y_future.index, y=y_future.values,
                        mode="lines+markers", name=name,
                        marker=dict(size=4), line=dict(color=color, width=2.5),
                        showlegend=False,
                    ))
                    _label_info.append((y_future.values[-1], name, color, y_future.index[-1]))
                    # 신뢰구간 (경계선 없이 영역만)
                    if name in intervals_future:
                        ci = intervals_future[name]
                        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
                        fig_combined.add_trace(go.Scatter(
                            x=list(y_future.index) + list(y_future.index[::-1]),
                            y=list(upper.values) + list(lower.values[::-1]),
                            fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False,
                        ))

                # 라벨 겹침 방지: y값 기준 정렬 후 최소 간격 확보
                _label_info.sort(key=lambda t: t[0])
                all_y_vals = [v for v, _, _, _ in _label_info]
                y_range = max(all_y_vals) - min(all_y_vals) if len(all_y_vals) > 1 else 1
                min_gap = y_range * 0.04  # 전체 범위의 4%를 최소 간격으로

                adjusted_y = [_label_info[0][0]]
                for i in range(1, len(_label_info)):
                    prev_y = adjusted_y[-1]
                    curr_y = _label_info[i][0]
                    if curr_y - prev_y < min_gap:
                        adjusted_y.append(prev_y + min_gap)
                    else:
                        adjusted_y.append(curr_y)

                for i, (orig_y, lname, lcolor, lx) in enumerate(_label_info):
                    fig_combined.add_annotation(
                        x=lx, y=adjusted_y[i],
                        text=f"<b>{lname}</b>", showarrow=False,
                        xanchor="left", yanchor="middle",
                        xshift=6, font=dict(size=10, color=lcolor),
                    )
                fig_combined.update_layout(
                    height=500, template="plotly_white",
                    margin=dict(l=50, r=120, t=30, b=30),
                    showlegend=False,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_combined, use_container_width=True)

                # 개별 모델 차트
                st.subheader("개별 모델 예측")
                n_models = len(predictions_future)
                n_cols = min(n_models, 4 if n_models > 6 else 3)
                model_items = list(predictions_future.items())

                for row_start in range(0, n_models, n_cols):
                    row_items = model_items[row_start:row_start + n_cols]
                    cols = st.columns(n_cols)
                    for col, (name, y_future) in zip(cols, row_items):
                        with col:
                            color = MODEL_COLORS[name]
                            hex_c = color.lstrip("#")
                            rv, gv, bv = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                            fig_ind = go.Figure()
                            context_len = min(len(y), horizon * 2)
                            y_ctx = y.iloc[-context_len:]
                            fig_ind.add_trace(go.Scatter(
                                x=y_ctx.index, y=y_ctx.values,
                                mode="lines", name="실제",
                                line=dict(color="#B2BABB", width=1.5),
                            ))
                            fig_ind.add_trace(go.Scatter(
                                x=[y.index[-1], y_future.index[0]],
                                y=[y.values[-1], y_future.values[0]],
                                mode="lines", showlegend=False,
                                line=dict(color=color, width=2, dash="dot"),
                            ))
                            fig_ind.add_trace(go.Scatter(
                                x=y_future.index, y=y_future.values,
                                mode="lines+markers", name=name,
                                marker=dict(size=4), line=dict(color=color, width=2.5),
                            ))
                            if name in intervals_future:
                                ci = intervals_future[name]
                                fig_ind.add_trace(go.Scatter(
                                    x=list(y_future.index) + list(y_future.index[::-1]),
                                    y=list(ci.iloc[:, 1].values) + list(ci.iloc[:, 0].values[::-1]),
                                    fill="toself", showlegend=False,
                                    fillcolor=f"rgba({rv},{gv},{bv},0.15)",
                                    line=dict(color="rgba(0,0,0,0)"),
                                ))
                            fig_ind.update_layout(
                                height=280, template="plotly_white",
                                title=dict(text=name, font=dict(size=14)),
                                margin=dict(l=40, r=10, t=40, b=20),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_ind, use_container_width=True)

        # ============================================================
        # 탭 4: 성능 비교
        # ============================================================
        with tab_metrics:
            if not metrics_results:
                st.warning("평가 결과가 없습니다.")
            else:
                _mode_short2 = "Expanding" if use_expanding else ("Rolling" if use_rolling else "단일 학습")
                st.caption(f"예측 방식: {_mode_short2} | 주기: {freq_txt} | Horizon: {horizon}스텝({horizon}{freq_txt}) | 평가 모델: {len(metrics_results)}개")

                df_metrics = pd.DataFrame(metrics_results).reset_index(drop=True)
                # 성능 평가 지표 6개 (RSFE, TS는 편향 진단 전용으로 분리)
                metric_names = ["MSE", "RMSE", "MAE", "MAPE(%)", "MASE", "MdRAE"]

                # 순위 계산용 데이터 (낮을수록 우수)
                rank_input = df_metrics.set_index("Model")[metric_names].copy()
                rank_df = rank_input.rank()
                rank_df["평균 순위"] = rank_df.mean(axis=1)
                rank_df = rank_df.sort_values("평균 순위")
                best_overall = rank_df.index[0]
                best_mape_row = df_metrics.loc[df_metrics["MAPE(%)"].idxmin()]
                best_rmse_row = df_metrics.loc[df_metrics["RMSE"].idxmin()]

                # ── KPI 카드 ──
                st.subheader("🏆 핵심 성능 요약")
                kpi_cols = st.columns(4)
                with kpi_cols[0]:
                    st.metric("🥇 최우수 모델", best_overall, help="전체 지표 평균 순위 기준")
                with kpi_cols[1]:
                    st.metric("MAPE 최저", f"{best_mape_row['MAPE(%)']:.2f}%", delta=best_mape_row["Model"], delta_color="off")
                with kpi_cols[2]:
                    st.metric("RMSE 최저", f"{best_rmse_row['RMSE']:.2f}", delta=best_rmse_row["Model"], delta_color="off")
                with kpi_cols[3]:
                    st.metric("평가 모델 수", f"{len(metrics_results)}개")

                st.markdown("")

                # ── 1) 지표별 Bar Chart ──
                st.subheader("📈 지표별 모델 비교")
                fig_bars = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=metric_names,
                    horizontal_spacing=0.08, vertical_spacing=0.18,
                )
                positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
                for m_idx, metric in enumerate(metric_names):
                    r, c = positions[m_idx]
                    for _, row in df_metrics.iterrows():
                        fig_bars.add_trace(
                            go.Bar(x=[row["Model"]], y=[row[metric]], name=row["Model"],
                                   marker_color=MODEL_COLORS[row["Model"]], showlegend=(m_idx == 0)),
                            row=r, col=c,
                        )
                fig_bars.update_layout(
                    height=600, template="plotly_white",
                    margin=dict(l=40, r=20, t=100, b=20), barmode="group",
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                )
                fig_bars.update_xaxes(tickangle=-45, tickfont=dict(size=9))
                fig_bars.update_traces(hovertemplate="%{x}: %{y:.4f}<extra></extra>")
                st.plotly_chart(fig_bars, use_container_width=True)

                # ── 2) 히트맵 (모델 × 지표 정규화) ──
                st.subheader("🗺️ 모델별 성능 히트맵")
                st.caption("각 지표를 0~1로 정규화 (낮을수록 우수 → 진한 초록).")
                heat_data = df_metrics.set_index("Model")[metric_names].copy()
                heat_norm = heat_data.copy()
                for col in metric_names:
                    min_v, max_v = heat_data[col].min(), heat_data[col].max()
                    heat_norm[col] = 0.0 if max_v == min_v else (heat_data[col] - min_v) / (max_v - min_v)
                fig_heat = go.Figure(go.Heatmap(
                    z=heat_norm.values,
                    x=metric_names,
                    y=heat_norm.index.tolist(),
                    text=heat_data.values.round(3),
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    colorscale=[[0, "#2d6a4f"], [0.5, "#ffffbf"], [1, "#d73027"]],
                    showscale=True, colorbar=dict(title="정규화"),
                    hovertemplate="모델: %{y}<br>지표: %{x}<br>값: %{text}<extra></extra>",
                ))
                fig_heat.update_layout(
                    height=max(300, len(heat_norm) * 45),
                    template="plotly_white",
                    margin=dict(l=100, r=20, t=30, b=60),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # ── 3) 범프 차트 ──
                st.subheader("📊 모델 순위 변동 차트")
                st.caption("각 지표마다 모델이 몇 등인지 선으로 이어 보여줍니다. 위쪽일수록 우수 (1등). 선이 출렁이면 그 모델은 지표마다 강약이 갈린다는 뜻.")

                bump_metrics = metric_names  # 성능 평가 6개 지표
                # 지표별 순위 계산 (값이 낮을수록 우수)
                bump_src = df_metrics.set_index("Model")[bump_metrics].copy()
                bump_ranks = bump_src.rank(method="min", ascending=True)
                bump_models = bump_ranks.index.tolist()
                n_bump_models = len(bump_models)

                fig_bump = go.Figure()
                last_metric = bump_metrics[-1]

                for m in bump_models:
                    ranks = bump_ranks.loc[m].values
                    color = MODEL_COLORS[m]
                    fig_bump.add_trace(go.Scatter(
                        x=bump_metrics,
                        y=ranks,
                        mode="lines+markers+text",
                        name=m,
                        showlegend=False,
                        line=dict(color=color, width=3, shape="spline", smoothing=0.6),
                        marker=dict(
                            size=17,
                            color=color,
                            line=dict(width=1.5, color="white"),
                        ),
                        text=[str(int(r)) for r in ranks],
                        textposition="middle center",
                        textfont=dict(color="white", size=9, family="Arial Black"),
                        hovertemplate=f"<b>{m}</b><br>지표: %{{x}}<br>순위: %{{y:.0f}}등<extra></extra>",
                    ))

                # 마지막 지표 위치에 모델명 인라인 라벨 (선 색상과 동일)
                # 동일 순위 충돌 시 수직 분산
                last_rank_groups = {}
                for m in bump_models:
                    r_val = int(bump_ranks.loc[m, last_metric])
                    last_rank_groups.setdefault(r_val, []).append(m)
                for rank_val, group in last_rank_groups.items():
                    n_grp = len(group)
                    for idx, m in enumerate(group):
                        y_offset = 0 if n_grp == 1 else (idx - (n_grp - 1) / 2) * 0.38
                        fig_bump.add_annotation(
                            x=last_metric,
                            y=rank_val + y_offset,
                            text=m,
                            showarrow=False,
                            xanchor="left",
                            xshift=16,
                            font=dict(color=MODEL_COLORS[m], size=12, family="Arial Black"),
                        )

                fig_bump.update_layout(
                    height=560, template="plotly_white",
                    margin=dict(l=80, r=170, t=40, b=40),
                    xaxis=dict(
                        title="", tickfont=dict(size=12),
                        showgrid=False, zeroline=False,
                    ),
                    yaxis=dict(
                        title="순위 (1등이 최상위)",
                        autorange="reversed",
                        tickmode="linear", tick0=1, dtick=1,
                        range=[n_bump_models + 0.5, 0.5],
                        gridcolor="#eee", zeroline=False,
                        tickfont=dict(size=11),
                    ),
                    showlegend=False,
                    hovermode="closest",
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_bump, use_container_width=True)

                # ── 편향 진단 (RSFE / TS) ──
                st.subheader("⚖️ 편향 진단 (RSFE / TS)")
                st.caption("RSFE = Σ(실제 − 예측). 양수면 과소예측, 음수면 과대예측. TS = RSFE / MAE, |TS| > 4 이면 편향 의심.")
                bias_df = df_metrics[["Model", "RSFE", "TS"]].copy()
                bias_df["편향 판정"] = bias_df["TS"].apply(
                    lambda t: "✗ 편향 의심" if abs(t) > 4 else ("⚠ 주의" if abs(t) > 2 else "✓ 양호")
                )
                def _bias_color(v):
                    if isinstance(v, str):
                        if v.startswith("✗"):
                            return "color: #d62728; font-weight: 600"
                        if v.startswith("⚠"):
                            return "color: #ff9f1c; font-weight: 600"
                        if v.startswith("✓"):
                            return "color: #2ca02c; font-weight: 600"
                    return ""

                bias_style = (
                    bias_df.style
                    .format({"RSFE": "{:+.3f}", "TS": "{:+.3f}"})
                    .map(_bias_color, subset=["편향 판정"])
                    .hide(axis="index")
                )
                st.dataframe(bias_style, hide_index=True, use_container_width=True)

                # ── 성능 테이블 (성능순 정렬 + 금은동) ──
                st.subheader("📋 성능 지표 상세 테이블")
                df_table = df_metrics.drop(columns=["RSFE", "TS"]).copy()
                df_table["평균 순위"] = rank_df.reindex(df_table["Model"])["평균 순위"].values
                df_table = df_table.sort_values("평균 순위").reset_index(drop=True)
                df_table.insert(0, "순위", range(1, len(df_table) + 1))

                medal_colors = {
                    1: "background-color: #FFD700",  # 금
                    2: "background-color: #C0C0C0",  # 은
                    3: "background-color: #CD7F32; color: white",  # 동
                }
                num_cols = [c for c in df_table.columns if c not in ("Model", "순위")]

                def _highlight_medal(row):
                    rank = row["순위"]
                    style = medal_colors.get(rank, "")
                    return [style] * len(row)

                metrics_style = (
                    df_table.style
                    .format({c: "{:.3f}" for c in num_cols})
                    .apply(_highlight_medal, axis=1)
                    .hide(axis="index")
                )
                st.dataframe(metrics_style, hide_index=True, use_container_width=True)

                # ── 잔차 진단 ──
                st.markdown("---")
                st.subheader("🔍 잔차 진단")
                st.caption("선택한 모델의 잔차를 분석합니다. 잔차가 백색잡음에 가까울수록 모델이 시계열 구조를 잘 파악한 것.")

                # 진단 결과 해석 블록에서 사용 — predictions_test 분기와 무관하게 항상 정의되어 있어야 함
                residuals = None
                resid_model = None
                arima_res = None
                lb_lag = 1

                if predictions_test:
                    # in-sample 잔차가 추출된 모델만 선택 가능
                    _resid_keys = [k for k in predictions_test.keys() if k in in_sample_residuals]
                    if not _resid_keys:
                        st.warning("in-sample 잔차를 추출할 수 있는 모델이 없습니다.")
                        residuals = None
                        resid_model = None
                        arima_res = None
                    else:
                        _default_idx = _resid_keys.index("AutoARIMA") if "AutoARIMA" in _resid_keys else 0
                        resid_model = st.selectbox(
                            "진단할 모델 선택",
                            _resid_keys,
                            index=_default_idx,
                            key="resid_model_select",
                        )

                        residuals = np.asarray(in_sample_residuals[resid_model], dtype=float)
                        residuals = residuals[~np.isnan(residuals)]
                        lb_lag = 1

                        # AutoARIMA는 SARIMAX summary 출력용 arima_res 도 추출
                        arima_res = None
                        if resid_model == "AutoARIMA" and fitted_arima is not None:
                            pm = getattr(fitted_arima, "_forecaster", fitted_arima)
                            arima_res = getattr(pm, "arima_res_", None)
                            if arima_res is None and hasattr(pm, "model_"):
                                arima_res = getattr(pm.model_, "arima_res_", None)

                    # 1) 잔차 ACF / PACF
                    if residuals is not None and len(residuals) >= 5:
                        try:
                            from statsmodels.tsa.stattools import acf as _acf_r, pacf as _pacf_r
                            nlags_r = min(20, max(2, len(residuals) // 2 - 1))
                            acf_r_vals = _acf_r(residuals, nlags=nlags_r)
                            pacf_r_vals = _pacf_r(residuals, nlags=nlags_r)
                            ci_r = 1.96 / np.sqrt(len(residuals))

                            fig_resid_acf = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=["잔차 ACF", "잔차 PACF"],
                                horizontal_spacing=0.1,
                            )
                            for i, v in enumerate(acf_r_vals):
                                fig_resid_acf.add_trace(go.Scatter(
                                    x=[i, i], y=[0, v], mode="lines",
                                    line=dict(color="#636EFA", width=1.5),
                                    showlegend=False, hoverinfo="skip",
                                ), row=1, col=1)
                            fig_resid_acf.add_trace(go.Scatter(
                                x=list(range(len(acf_r_vals))), y=acf_r_vals,
                                mode="markers", marker=dict(color="#636EFA", size=6),
                                showlegend=False,
                                hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
                            ), row=1, col=1)
                            for i, v in enumerate(pacf_r_vals):
                                fig_resid_acf.add_trace(go.Scatter(
                                    x=[i, i], y=[0, v], mode="lines",
                                    line=dict(color="#EF553B", width=1.5),
                                    showlegend=False, hoverinfo="skip",
                                ), row=1, col=2)
                            fig_resid_acf.add_trace(go.Scatter(
                                x=list(range(len(pacf_r_vals))), y=pacf_r_vals,
                                mode="markers", marker=dict(color="#EF553B", size=6),
                                showlegend=False,
                                hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
                            ), row=1, col=2)
                            for col_idx in [1, 2]:
                                fig_resid_acf.add_shape(
                                    type="rect", xref=f"x{col_idx}", yref=f"y{col_idx}",
                                    x0=0, x1=nlags_r, y0=-ci_r, y1=ci_r,
                                    fillcolor="rgba(150,150,150,0.18)", line_width=0, layer="below",
                                )
                                fig_resid_acf.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=col_idx)
                            fig_resid_acf.update_yaxes(range=[-1.05, 1.05])
                            fig_resid_acf.update_xaxes(title_text="Lag")
                            fig_resid_acf.update_layout(
                                height=300, template="plotly_white",
                                margin=dict(l=50, r=20, t=40, b=30),
                            )
                            st.plotly_chart(fig_resid_acf, use_container_width=True)
                        except Exception as e:
                            st.warning(f"잔차 ACF/PACF 계산 실패: {e}")

                    # 3) AutoARIMA: statsmodels SARIMAX summary 그대로 출력 (textbook 표준 reference)
                    #    그 외 모델의 잔차 검정은 아래 "📖 진단 결과 해석" 블록에서 일괄 처리.
                    if arima_res is not None:
                        try:
                            st.text(str(arima_res.summary()))
                        except Exception as e:
                            st.warning(f"ARIMA summary 출력 실패: {e}")
                else:
                    st.info("학습된 모델이 없어 잔차 진단을 표시할 수 없습니다.")

                # ── 진단 결과 해석 (본 실행 계산값 기반) ──
                if residuals is not None and len(residuals) >= 5:
                    st.markdown("##### 📖 진단 결과 해석")

                    _n_r = len(residuals)

                    try:
                        from statsmodels.stats.diagnostic import acorr_ljungbox as _interp_lb, het_arch as _interp_arch
                        from scipy.stats import jarque_bera as _interp_jb, skew as _interp_sk, kurtosis as _interp_ku

                        # p-value 표시: 매우 작은 값(< 1e-4)은 "< 0.0001"로 표기 (0.0000으로 보이는 것 방지)
                        def _fp(p):
                            if not np.isfinite(p):
                                return "—"
                            if p < 1e-4:
                                return "< 0.0001"
                            return f"{p:.4f}"

                        # 2자리 표기 (Ljung-Box·Hetero·JB 등 — 반올림 후 0.00이면 "< 0.01")
                        def _fp2(p):
                            if not np.isfinite(p):
                                return "—"
                            if round(p, 2) < 0.01:
                                return "< 0.01"
                            return f"{round(p, 2):.2f}"

                        # 2자리 반올림 (판정 비교용 — 표시값과 동일한 수치로 비교)
                        def _r2(x):
                            return x if not np.isfinite(x) else round(x, 2)

                        # Ljung-Box: AutoARIMA는 SARIMAX standardized residuals 기반 (summary와 동일)
                        # 그 외 모델은 raw 잔차에 acorr_ljungbox (burn-in 영향 없음)
                        if arima_res is not None:
                            _lb_arr = arima_res.test_serial_correlation(
                                method="ljungbox", lags=int(lb_lag),
                            )
                            # shape = (k_endog, 2, n_lags). [0,0,-1] = lag=lb_lag Q stat, [0,1,-1] = p-value
                            _lb_q = float(_lb_arr[0, 0, -1])
                            _lb_pv = float(_lb_arr[0, 1, -1])
                        else:
                            _lb_df = _interp_lb(residuals, lags=[int(lb_lag)], return_df=True)
                            _lb_q = float(_lb_df["lb_stat"].iloc[0])
                            _lb_pv = float(_lb_df["lb_pvalue"].iloc[0])

                        # Jarque-Bera·skew·kurt: AutoARIMA는 standardized residuals 기반 (summary와 동일)
                        if arima_res is not None:
                            _norm_arr = arima_res.test_normality(method="jarquebera")
                            # shape = (k_endog, 4): [jb, jbpv, skew, kurt(raw)]
                            _jb_q = float(_norm_arr[0, 0])
                            _jb_pv = float(_norm_arr[0, 1])
                            _sk_v = float(_norm_arr[0, 2])
                            _ku_v = float(_norm_arr[0, 3])
                        else:
                            _jb_q, _jb_pv = _interp_jb(residuals)
                            _jb_q = float(_jb_q); _jb_pv = float(_jb_pv)
                            _sk_v = float(_interp_sk(residuals))
                            _ku_v = float(_interp_ku(residuals, fisher=False))

                        # Heteroskedasticity: AutoARIMA는 breakvar (summary의 H 통계량과 동일),
                        # 그 외 모델은 ARCH-LM (raw 잔차)
                        _het_ok = False
                        _het_label = None
                        _het_stat = None
                        _het_pv = None
                        if arima_res is not None:
                            try:
                                _het_arr = arima_res.test_heteroskedasticity(method="breakvar")
                                # shape = (k_endog, 2): [stat, pvalue]. p-value는 two-sided.
                                _het_stat = float(_het_arr[0, 0])
                                _het_pv = float(_het_arr[0, 1])
                                _het_label = "Heteroskedasticity"
                                _het_ok = True
                            except Exception:
                                pass
                        else:
                            try:
                                _arch_lag_i = min(12, max(1, _n_r // 5))
                                _arch_q, _arch_pv, _, _ = _interp_arch(residuals, nlags=_arch_lag_i)
                                _het_stat = float(_arch_q)
                                _het_pv = float(_arch_pv)
                                _het_label = "Heteroskedasticity"
                                _het_ok = True
                            except Exception:
                                pass

                        interp_rows = []

                        # 정보 기준 (likelihood 기반 모델만): ARIMA + ETS-family + Theta
                        # FPP3 §8.6 (ETS), §9.7 (ARIMA). Naive/SMA/STL/Prophet은 미정의 → 행 생략.
                        _ic = in_sample_ic.get(resid_model, {}) if in_sample_ic else {}
                        if "aic" in _ic:
                            interp_rows.append({
                                "검정·통계": "AIC",
                                "값": f"{_ic['aic']:.2f}",
                                "해석·판정": "정보 기준 — 절댓값만으로는 판단 불가. 동일 모델군 후보 간 비교 시 낮을수록 좋음.",
                            })
                        if "bic" in _ic:
                            interp_rows.append({
                                "검정·통계": "BIC",
                                "값": f"{_ic['bic']:.2f}",
                                "해석·판정": "AIC보다 복잡도 페널티 강함. 표본이 클수록 단순 모델 선호. 모델 비교용.",
                            })
                        if "hqic" in _ic:
                            interp_rows.append({
                                "검정·통계": "HQIC",
                                "값": f"{_ic['hqic']:.2f}",
                                "해석·판정": "AIC와 BIC의 중간 페널티. 모델 비교용.",
                            })
                        if "sigma2" in _ic:
                            _sig = _ic["sigma2"]
                            interp_rows.append({
                                "검정·통계": "sigma2",
                                "값": f"{_sig:.4f}",
                                "해석·판정": f"모델이 설명하지 못한 변동의 분산 (잔차 표준편차 ≈ {_sig**0.5:.4f}). 동일 모델군 후보 간 비교 시 작을수록 적합 양호.",
                            })

                        # AutoARIMA 한정 — 계수 (intercept, ar*, ma*) + p-value
                        if arima_res is not None:
                            for _coef_name in arima_res.params.index:
                                if _coef_name == "sigma2":
                                    continue  # 위 통합 sigma2 행으로 이미 표시됨
                                _coef_val = float(arima_res.params[_coef_name])
                                _coef_p = float(arima_res.pvalues.get(_coef_name, np.nan))
                                if np.isnan(_coef_p):
                                    _coef_verdict = "p-value 산출 불가."
                                elif _coef_p < 0.05:
                                    _coef_verdict = f"✓ p = {_fp(_coef_p)} < 0.05 → 5% 수준에서 통계적으로 유의."
                                else:
                                    _coef_verdict = f"⚠ p = {_fp(_coef_p)} ≥ 0.05 → 5% 수준에서 비유의 (제거 후보)."
                                interp_rows.append({
                                    "검정·통계": _coef_name,
                                    "값": f"{_coef_val:+.4f}",
                                    "해석·판정": _coef_verdict,
                                })

                        # 모든 모델 공통 — 잔차 검정
                        # Ljung-Box (선택 lag) — AutoARIMA는 SARIMAX summary와 동일 검정
                        _lb_pv_r = _r2(_lb_pv)
                        if _lb_pv_r > 0.05:
                            _lb_verdict = f"✓ p = {_fp2(_lb_pv)} > 0.05 → lag-{int(lb_lag)} 자기상관 없음 (백색잡음 가설 채택)."
                        else:
                            _lb_verdict = f"✗ p = {_fp2(_lb_pv)} ≤ 0.05 → lag-{int(lb_lag)} 자기상관 잔존 (백색잡음 기각)."
                        interp_rows.append({
                            "검정·통계": "Ljung-Box",
                            "값": f"{_lb_q:.2f}, p = {_fp2(_lb_pv)}",
                            "해석·판정": _lb_verdict,
                        })

                        if _het_ok:
                            _het_pv_r = _r2(_het_pv)
                            if _het_pv_r > 0.05:
                                _het_verdict = f"✓ p = {_fp2(_het_pv)} > 0.05 → 등분산 (분산이 시간에 따라 안정적)."
                            else:
                                _het_verdict = f"✗ p = {_fp2(_het_pv)} ≤ 0.05 → 이분산 (변동성 군집 가능, 신뢰구간 정확도 저하 우려)."
                            interp_rows.append({
                                "검정·통계": _het_label,
                                "값": f"{_het_stat:.2f}, p = {_fp2(_het_pv)}",
                                "해석·판정": _het_verdict,
                            })
                        else:
                            interp_rows.append({
                                "검정·통계": "Heteroskedasticity",
                                "값": "—",
                                "해석·판정": "계산 불가 (표본 부족 또는 수렴 실패).",
                            })

                        _jb_pv_r = _r2(_jb_pv)
                        if _jb_pv_r > 0.05:
                            _jb_verdict = f"✓ p = {_fp2(_jb_pv)} > 0.05 → 정규성 만족 (잔차가 정규분포에 근접)."
                        else:
                            _jb_verdict = f"✗ p = {_fp2(_jb_pv)} ≤ 0.05 → 정규성 위배 (신뢰구간·예측구간 정확도 저하 우려)."
                        interp_rows.append({
                            "검정·통계": "Jarque-Bera",
                            "값": f"{_jb_q:.2f}, p = {_fp2(_jb_pv)}",
                            "해석·판정": _jb_verdict,
                        })

                        _sk_v_r = _r2(_sk_v)
                        _abs_sk_r = abs(_sk_v_r)
                        if _abs_sk_r < 0.5:
                            _sk_verdict = f"✓ |왜도| = {_abs_sk_r:.2f} < 0.5 → 분포 대칭."
                        elif _abs_sk_r < 1:
                            _sk_verdict = f"⚠ |왜도| = {_abs_sk_r:.2f} (0.5 ≤ |왜도| < 1) → 약한 비대칭."
                        else:
                            _sk_verdict = f"✗ |왜도| = {_abs_sk_r:.2f} ≥ 1 → 강한 비대칭."
                        interp_rows.append({
                            "검정·통계": "Skew",
                            "값": f"{_sk_v_r:+.2f}  (정규분포 = 0)",
                            "해석·판정": _sk_verdict,
                        })

                        _ku_v_r = _r2(_ku_v)
                        _ku_dev = abs(_ku_v_r - 3) if np.isfinite(_ku_v_r) else float("nan")
                        if np.isfinite(_ku_dev) and _ku_dev < 1:
                            _ku_verdict = f"✓ |첨도 − 3| = {_ku_dev:.2f} < 1 → 정규에 근사."
                        else:
                            _ku_verdict = f"⚠ |첨도 − 3| = {_ku_dev:.2f} ≥ 1 → 정규에서 이탈 (꼬리 두꺼움/얇음)."
                        interp_rows.append({
                            "검정·통계": "Kurtosis",
                            "값": f"{_ku_v_r:.2f}  (정규분포 = 3)",
                            "해석·판정": _ku_verdict,
                        })

                        interp_df = pd.DataFrame(interp_rows)

                        def _interp_color(v):
                            if isinstance(v, str):
                                if v.startswith("✗"):
                                    return "color: #d62728; font-weight: 600"
                                if v.startswith("⚠"):
                                    return "color: #ff9f1c; font-weight: 600"
                                if v.startswith("✓"):
                                    return "color: #2ca02c; font-weight: 600"
                            return ""

                        interp_style = (
                            interp_df.style
                            .map(_interp_color, subset=["해석·판정"])
                            .hide(axis="index")
                        )
                        st.dataframe(interp_style, hide_index=True, use_container_width=True)
                    except Exception as _interp_err:
                        st.warning(f"진단 결과 해석 계산 실패: {_interp_err}")

                # ── 자동 분석 리포트 ──
                st.markdown("---")
                st.subheader("📝 자동 분석 리포트")

                # 지표별 최적 모델 추출
                best_by = {}
                for m in metric_names:
                    idx = df_metrics[m].idxmin()
                    best_by[m] = (df_metrics.loc[idx, "Model"], df_metrics.loc[idx, m])

                # 상위 3개 모델
                top3 = df_table.head(3)

                step_txt = f"1-step(={freq_txt})"

                # 예측 방식 텍스트
                if use_expanding:
                    method_txt = (
                        f"Expanding Window — 초기 학습 크기: {train_size}개, "
                        f"{step_txt}씩 확장하며 {horizon}회 반복 예측"
                    )
                elif use_rolling:
                    method_txt = (
                        f"Rolling Window — 고정 윈도우 크기: {train_size}개, "
                        f"{step_txt}씩 이동하며 {horizon}회 반복 예측"
                    )
                else:
                    method_txt = f"기본 (단일 학습) — 학습 데이터 {train_size}개로 한 번 학습"

                start_txt = y.index.min().strftime("%Y-%m-%d")
                end_txt = y.index.max().strftime("%Y-%m-%d")

                # 지표별 최적 모델 테이블 행
                metric_rows = ""
                for m in metric_names:
                    name, val = best_by[m]
                    fmt = f"{val:.3f}%" if "%" in m else f"{val:.3f}"
                    metric_rows += f"| {m} | **{name}** | {fmt} |\n"

                # 상위 모델 분석
                top_analysis = ""
                for _, r in top3.iterrows():
                    rank = int(r["순위"])
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}[rank]
                    strengths = []
                    for m in metric_names:
                        if r["Model"] == best_by[m][0]:
                            strengths.append(m)
                    strength_txt = ", ".join(strengths) if strengths else "균형 잡힌 성능"
                    top_analysis += (
                        f"**{medal} {rank}위: {r['Model']}** (평균 순위 {r['평균 순위']:.2f})<br>\n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;강점 지표: {strength_txt} | "
                        f"MAPE: {r['MAPE(%)']:.2f}% | RMSE: {r['RMSE']:.3f}<br><br>\n"
                    )

                # 전통 vs ML 비교 분석 텍스트 생성
                _ml_models = {"Prophet"}
                _traditional_models = set(MODEL_COLORS.keys()) - _ml_models
                ml_in_results = [r for r in metrics_results if r["Model"] in _ml_models]
                trad_in_results = [r for r in metrics_results if r["Model"] in _traditional_models]

                ml_vs_trad_txt = ""
                if ml_in_results and trad_in_results:
                    # ML 모델 평균 순위
                    ml_avg_ranks = []
                    trad_avg_ranks = []
                    for r in metrics_results:
                        avg_r = rank_df.loc[r["Model"], "평균 순위"] if r["Model"] in rank_df.index else None
                        if avg_r is not None:
                            if r["Model"] in _ml_models:
                                ml_avg_ranks.append((r["Model"], avg_r, r["MAPE(%)"]))
                            else:
                                trad_avg_ranks.append((r["Model"], avg_r, r["MAPE(%)"]))

                    best_trad = min(trad_avg_ranks, key=lambda x: x[1])
                    best_ml = min(ml_avg_ranks, key=lambda x: x[1]) if ml_avg_ranks else None

                    if best_ml:
                        if best_ml[1] > best_trad[1]:
                            ml_verdict = (
                                f"ML 모델(**{best_ml[0]}**, 평균 순위 {best_ml[1]:.2f})은 "
                                f"전통 최우수 모델(**{best_trad[0]}**, 평균 순위 {best_trad[1]:.2f})에 비해 "
                                f"종합 성능이 낮았습니다.<br>"
                                f"이는 단변량 시계열의 특성상, 피처 엔지니어링 없이 시간축만으로 학습하는 ML 모델보다 "
                                f"시계열 구조(자기상관, 계절성, 추세)를 직접 모델링하는 전통 통계 모델이 유리하기 때문입니다."
                            )
                        else:
                            ml_verdict = (
                                f"ML 모델(**{best_ml[0]}**, 평균 순위 {best_ml[1]:.2f})이 "
                                f"전통 최우수 모델(**{best_trad[0]}**, 평균 순위 {best_trad[1]:.2f})과 "
                                f"대등하거나 우수한 성능을 보였습니다.<br>"
                                f"Prophet의 추세·계절성 자동 분해 기능이 해당 데이터의 패턴을 효과적으로 포착한 것으로 판단됩니다."
                            )
                        ml_vs_trad_txt = f"""
<h4 style="color:#0c5460;">전통 통계 모델 vs ML 모델 비교</h4>

본 실험에서는 전통 통계 모델 {len(trad_in_results)}개와 ML 기반 모델 {len(ml_in_results)}개를 동일 조건에서 비교하였습니다.<br><br>
{ml_verdict}<br><br>
<b>시사점:</b> 단변량 시계열 예측에서는 데이터의 시간적 구조를 명시적으로 활용하는 전통 통계 모델이
일반적으로 강점을 보이며, ML 모델은 다변량·비선형 패턴이 존재하는 경우에 더 큰 이점을 가집니다.
모델 선택 시 데이터 특성과 예측 목적에 따른 적합성을 종합적으로 고려해야 합니다.
"""

                report_html = f"""<div style="background:#f8f9fa; border-left:5px solid #17a2b8; padding:20px 24px; border-radius:8px; line-height:1.9; color:#212529;">

<h4 style="margin-top:0; color:#0c5460;">실험 조건</h4>

- 데이터 기간: **{start_txt} ~ {end_txt}** (총 {len(y)}개 관측치)<br>
- 훈련 / 테스트: **{len(y_train)}개 / {len(y_test)}개**<br>
- 예측 시평: **{horizon} 스텝** | 계절 주기: **{sp}**<br>
- 예측 방식: **{method_txt}**<br>
- 평가 모델: **{len(metrics_results)}개**{f" (전통 통계 {len(trad_in_results)}개 + ML {len(ml_in_results)}개)" if ml_in_results else ""} | 평가 지표: **6개** (+ 편향 진단 2개)

<h4 style="color:#0c5460;">지표별 최적 모델</h4>

| 지표 | 최적 모델 | 값 |
|:----:|:--------:|:---:|
{metric_rows}

<h4 style="color:#0c5460;">상위 모델 분석</h4>

{top_analysis}

{ml_vs_trad_txt}

<h4 style="color:#0c5460;">최종 결론</h4>

본 실험에서 **{top3.iloc[0]['Model']}** 모델이 6개 성능 지표의 평균 순위 **{top3.iloc[0]['평균 순위']:.2f}위**로
가장 높은 종합 성능을 기록하였습니다.<br><br>
특히 MAPE {df_metrics.loc[df_metrics['Model']==top3.iloc[0]['Model'], 'MAPE(%)'].values[0]:.2f}%,
RMSE {df_metrics.loc[df_metrics['Model']==top3.iloc[0]['Model'], 'RMSE'].values[0]:.3f}를 달성하여
예측 정확도와 오차 크기 모두에서 우수한 결과를 보였습니다.<br><br>
따라서 현재 데이터와 실험 조건({method_txt})에서는
**{top3.iloc[0]['Model']}**의 예측 결과를 최우선으로 활용할 것을 권장합니다.

</div>"""

                st.markdown(report_html, unsafe_allow_html=True)

    except Exception:
        st.error("데이터 처리 중 오류가 발생했습니다.")
        st.code(traceback.format_exc())

else:
    # ──────────────────────────────────────────────
    # 랜딩 페이지
    # ──────────────────────────────────────────────
    st.info("**왼쪽 패널에서 CSV 파일을 업로드하여 시계열 예측을 시작하세요.**")

    st.markdown("---")
    st.markdown("### 주요 기능")
    st.markdown("""
- **9가지 예측 모델**: Naive, SMA, ExpSmoothing, Holt, Holt-Winters, STL, AutoARIMA, Theta, Prophet
- **6가지 평가 지표**: MSE, RMSE, MAE, MAPE, MASE, MdRAE + 편향 진단(RSFE, TS)
- **하이퍼파라미터 튜닝**: 모델별 핵심 파라미터 실시간 조절
- **심층 분석**: 계절 분해, 잔차 분석, 레이더 차트
""")

    st.markdown("---")
    st.markdown("### 모델 분류")
    st.markdown("""
**평활법 (Smoothing)**
- **SMA**: 단순이동평균 — 최근 n개 관측치의 평균
- **ExpSmoothing**: 지수평활 — 가중 평균 (α 파라미터)
- **Holt**: 선형추세지수평활 — 추세 반영 (α, β)
- **Holt-Winters**: 계절반영 — 추세 + 계절성 (α, β, γ)

**분해법 (Decomposition)**
- **STL**: Loess 기반 계절-추세 분해

**통계 모델**
- **AutoARIMA**: 자동 ARIMA 파라미터 탐색
- **Theta**: 지수평활 + 곡률 결합

**ML 기반 모델**
- **Prophet**: Facebook 개발, 추세 + 계절성 자동 분해
""")

    st.markdown("---")
    st.markdown("### 평가 지표 가이드")
    st.markdown("""
| 지표 | 특성 | 용도 |
|------|------|------|
| **MSE** | 제곱 오차 평균 | 수학적 분석 기본 |
| **RMSE** | 큰 오차에 패널티 | 이상치 민감 평가 |
| **MAE** | 스케일 의존, 직관적 | 일반적인 오차 크기 |
| **MAPE** | 백분율, 스케일 무관 | 서로 다른 척도 비교 |
| **MASE** | 벤치마크 대비 | 스케일 무관, 0 근방 안정 |
| **MdRAE** | naive 대비 상대오차의 중앙값 | 이상치에 강건한 비교 |
| **RSFE** | 오차 누적합(부호 유지) | 편향(bias) 진단 — 0에서 멀수록 편향 |
| **TS** | RSFE / MAE | &#124;TS&#124; > 4 이면 편향 의심 |
""")

