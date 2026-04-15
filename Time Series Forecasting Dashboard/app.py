import traceback

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA, ARIMA
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
    "ARIMA": "#2CA02C",
    "SARIMA": "#D62728",
    "AutoARIMA": "#AB63FA",
    "Theta": "#FFA15A",
    "Prophet": "#00CC96",
}

INTERVAL_CAPABLE = {"ARIMA", "SARIMA", "AutoARIMA", "Theta", "Prophet"}
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

        preset_choice = st.sidebar.selectbox("예측 시평", preset_list)

        if preset_choice == "직접 입력":
            horizon = st.sidebar.slider(
                "시평 값 설정", min_value=1, max_value=max_horizon,
                value=min(12, max_horizon), step=1,
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

        sp = st.sidebar.number_input("계절 주기 (Seasonal Period)", min_value=1, value=12, step=1)
        selected_models = st.sidebar.multiselect("사용할 모델 선택", ALL_MODEL_NAMES, default=ALL_MODEL_NAMES)
        forecast_mode = st.sidebar.selectbox(
            "예측 방식",
            ["기본 (단일 학습)", "Expanding Window (확장 윈도우)", "Rolling Window (고정 윈도우)"],
            index=0,
            help=(
                "기본: 학습 데이터로 한 번 학습 후 전체 구간 예측.\n"
                "Expanding: 1-step씩 실제값을 추가하며 반복 예측 (학습 데이터 증가).\n"
                "Rolling: 고정 크기 윈도우가 1-step씩 이동하며 반복 예측."
            ),
        )
        use_expanding = forecast_mode.startswith("Expanding")
        use_rolling = forecast_mode.startswith("Rolling")
        show_ci = st.sidebar.checkbox("신뢰구간 표시 (지원 모델만)", value=False)
        if len(selected_models) == 0:
            st.warning("최소 1개 이상의 모델을 선택해주세요.")
            st.stop()

        st.sidebar.markdown("---")
        _reset_keys = [
            "naive_strat", "sma_win", "es_alpha", "holt_alpha", "holt_beta",
            "hw_alpha", "hw_beta", "hw_seasonal",
            "arima_p", "arima_d", "arima_q",
            "prophet_season_mode", "prophet_cp_scale", "prophet_season_scale",
        ]
        if st.sidebar.button("🔄 설정 초기화", use_container_width=True):
            for k in _reset_keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.trained = False
            st.session_state.train_settings = {}
            st.session_state.train_results = {}
            st.rerun()
        run_clicked = st.sidebar.button("🚀 학습 시작", use_container_width=True, type="primary")

        # 학습 시작 또는 기존 결과 표시 판단 (설정 변경 감지는 hp 수집 후 아래에서 처리)
        if not run_clicked and not st.session_state.trained:
            st.info("**왼쪽 패널에서 설정을 완료한 후 '🚀 학습 시작' 버튼을 눌러주세요.**")
            st.stop()

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
        tab_overview, tab_settings, tab_forecast, tab_metrics = st.tabs(
            ["📊 데이터 개요", "⚙️ 모델 설정", "🎯 모델별 예측", "💡 성능 비교"]
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

        # ============================================================
        # 탭 2: 모델 하이퍼파라미터 설정
        # ============================================================
        with tab_settings:
            st.subheader("⚙️ 모델별 하이퍼파라미터 설정")
            st.caption("선택된 모델의 핵심 파라미터를 조절할 수 있습니다. 변경 시 자동으로 재학습됩니다.")

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
                        elif name == "ARIMA":
                            hp["ARIMA_p"] = st.slider("AR 차수 (p)", 0, 10, 1, key="arima_p", help="자기회귀 차수")
                            hp["ARIMA_d"] = st.slider("차분 차수 (d)", 0, 3, 1, key="arima_d", help="비정상성 제거를 위한 차분 횟수")
                            hp["ARIMA_q"] = st.slider("MA 차수 (q)", 0, 10, 0, key="arima_q", help="이동평균 차수")
                        elif name == "SARIMA":
                            hp["SARIMA_p"] = st.slider("AR 차수 (p)", 0, 5, 1, key="sarima_p", help="비계절 자기회귀 차수")
                            hp["SARIMA_d"] = st.slider("차분 차수 (d)", 0, 2, 1, key="sarima_d", help="비계절 차분 횟수")
                            hp["SARIMA_q"] = st.slider("MA 차수 (q)", 0, 5, 1, key="sarima_q", help="비계절 이동평균 차수")
                            hp["SARIMA_P"] = st.slider("계절 AR 차수 (P)", 0, 2, 1, key="sarima_P", help="계절 자기회귀 차수")
                            hp["SARIMA_D"] = st.slider("계절 차분 차수 (D)", 0, 1, 1, key="sarima_D", help="계절 차분 횟수")
                            hp["SARIMA_Q"] = st.slider("계절 MA 차수 (Q)", 0, 2, 1, key="sarima_Q", help="계절 이동평균 차수")
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
            "STL": lambda: STLForecaster(sp=sp),
            "ARIMA": lambda: ARIMA(
                order=(hp.get("ARIMA_p", 1), hp.get("ARIMA_d", 1), hp.get("ARIMA_q", 0)),
                suppress_warnings=True,
            ),
            "SARIMA": lambda: ARIMA(
                order=(hp.get("SARIMA_p", 1), hp.get("SARIMA_d", 1), hp.get("SARIMA_q", 1)),
                seasonal_order=(hp.get("SARIMA_P", 1), hp.get("SARIMA_D", 1), hp.get("SARIMA_Q", 1), sp),
                suppress_warnings=True,
            ),
            "AutoARIMA": lambda: AutoARIMA(suppress_warnings=True, sp=sp),
            "Theta": lambda: ThetaForecaster(sp=sp),
        }

        # ──────────────────────────────────────
        # Prophet 헬퍼 함수
        # ──────────────────────────────────────
        def _prophet_fit_predict(y_data, h, return_ci=False):
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
            if return_ci:
                ci_df = pd.DataFrame({
                    "lower": fc_tail["yhat_lower"].values,
                    "upper": fc_tail["yhat_upper"].values,
                }, index=fc_tail["ds"].values)
                return y_pred, ci_df
            return y_pred

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
            if name == "ARIMA":
                return n >= _hp.get("ARIMA_p", 1) + _hp.get("ARIMA_d", 1) + 1
            if name == "SARIMA":
                return n >= sp * 2 + _hp.get("SARIMA_p", 1) + _hp.get("SARIMA_d", 1) + 1
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

            if use_expanding:
                spinner_msg = "Expanding Window Forecast 진행 중..."
            elif use_rolling:
                spinner_msg = "Rolling Window Forecast 진행 중..."
            else:
                spinner_msg = "모델 학습 중..."
            with st.spinner(spinner_msg):
                for name in selected_models:
                    try:
                        if not _check_min_data(name, len(y_train_m_clean)):
                            st.warning(f"⚠️ {name} — 학습 데이터({len(y_train_m_clean)}개)가 최소 요구 조건에 부족하여 건너뜁니다.")
                            continue

                        if use_expanding or use_rolling:
                            # ── Expanding / Rolling Window: 1-step 반복 예측 ──
                            iter_preds = []
                            mode_label = "Expanding" if use_expanding else "Rolling"
                            progress_bar = st.progress(0, text=f"{name} {mode_label}...")
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
                                else:
                                    forecaster = model_factories[name]()
                                    forecaster.fit(y_window)
                                    pred_1 = forecaster.predict(fh=[1])
                                    iter_preds.append(pred_1.values[0])
                                progress_bar.progress((step + 1) / horizon)
                            progress_bar.empty()

                            if all(np.isnan(v) for v in iter_preds):
                                st.warning(f"⚠️ {name} {mode_label} Forecast 실패 — 모든 스텝에서 데이터 부족")
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
                                    st.warning(f"⚠️ {name} 모델 학습 실패")
                                    continue
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

                        # 전체 데이터로 재학습 → 미래 예측
                        if name == "Prophet":
                            if show_ci:
                                y_fut, ci_prophet = _prophet_fit_predict(y, horizon, return_ci=True)
                                intervals_future[name] = ci_prophet
                            else:
                                y_fut = _prophet_fit_predict(y, horizon)
                            predictions_future[name] = y_fut
                        elif name == "HoltWinters":
                            forecaster2 = _build_holtwinters(y_m_clean)
                            if forecaster2 is None:
                                continue
                            y_fut = forecaster2.predict(fh=fh)
                            if hasattr(y_fut.index, 'to_timestamp'):
                                y_fut.index = y_fut.index.to_timestamp()
                            predictions_future[name] = y_fut
                        else:
                            forecaster2 = model_factories[name]()
                            forecaster2.fit(y_m_clean)
                            y_fut = forecaster2.predict(fh=fh)
                            if hasattr(y_fut.index, 'to_timestamp'):
                                y_fut.index = y_fut.index.to_timestamp()
                            predictions_future[name] = y_fut

                            if show_ci and name in INTERVAL_CAPABLE:
                                try:
                                    ci = forecaster2.predict_interval(fh=fh, coverage=0.9)
                                    if hasattr(ci.index, 'to_timestamp'):
                                        ci.index = ci.index.to_timestamp()
                                    intervals_future[name] = ci
                                except Exception:
                                    pass

                    except Exception as e:
                        st.warning(f"⚠️ {name} 모델 실행 오류: {e}")

            # 학습 결과를 세션에 저장
            st.session_state.trained = True
            st.session_state.train_settings = _current_settings
            st.session_state.train_results = {
                "predictions_test": predictions_test,
                "predictions_future": predictions_future,
                "intervals_future": intervals_future,
                "metrics_results": metrics_results,
            }
        else:
            # 기존 결과 로드
            predictions_test = st.session_state.train_results["predictions_test"]
            predictions_future = st.session_state.train_results["predictions_future"]
            intervals_future = st.session_state.train_results["intervals_future"]
            metrics_results = st.session_state.train_results["metrics_results"]

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
                # 전체 평가 지표 8개 (RSFE, TS는 절대값 기준으로 순위/정규화)
                metric_names = ["MSE", "RMSE", "MAE", "MAPE(%)", "MASE", "MdRAE", "RSFE", "TS"]
                bias_names = ["RSFE", "TS"]

                # 순위 계산용 데이터 — RSFE, TS는 |값| 사용 (0에 가까울수록 우수)
                rank_input = df_metrics.set_index("Model")[metric_names].copy()
                for c in bias_names:
                    rank_input[c] = rank_input[c].abs()
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
                    rows=2, cols=4,
                    subplot_titles=metric_names,
                    horizontal_spacing=0.06, vertical_spacing=0.18,
                )
                positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
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
                st.caption("각 지표를 0~1로 정규화 (낮을수록 우수 → 진한 초록). RSFE·TS는 |값| 기준.")
                heat_data = df_metrics.set_index("Model")[metric_names].copy()
                heat_norm_src = heat_data.copy()
                for c in bias_names:
                    heat_norm_src[c] = heat_norm_src[c].abs()
                heat_norm = heat_norm_src.copy()
                for col in metric_names:
                    min_v, max_v = heat_norm_src[col].min(), heat_norm_src[col].max()
                    heat_norm[col] = 0.0 if max_v == min_v else (heat_norm_src[col] - min_v) / (max_v - min_v)
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

                bump_metrics = metric_names  # 전체 8개 지표
                # 지표별 순위 계산 (RSFE, TS는 |값| 기준, 나머지는 값이 낮을수록 우수)
                bump_src = df_metrics.set_index("Model")[bump_metrics].copy()
                for c in bias_names:
                    bump_src[c] = bump_src[c].abs()
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
                bias_df = df_metrics[["Model"] + bias_names].copy()
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
                df_table = df_metrics.copy()
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

                # ── 자동 분석 리포트 ──
                st.markdown("---")
                st.subheader("📝 자동 분석 리포트")

                # 지표별 최적 모델 추출
                best_by = {}
                for m in metric_names:
                    if m in bias_names:
                        idx = df_metrics[m].abs().idxmin()
                    else:
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
- 평가 모델: **{len(metrics_results)}개**{f" (전통 통계 {len(trad_in_results)}개 + ML {len(ml_in_results)}개)" if ml_in_results else ""} | 평가 지표: **8개**

<h4 style="color:#0c5460;">지표별 최적 모델</h4>

| 지표 | 최적 모델 | 값 |
|:----:|:--------:|:---:|
{metric_rows}

<h4 style="color:#0c5460;">상위 모델 분석</h4>

{top_analysis}

{ml_vs_trad_txt}

<h4 style="color:#0c5460;">최종 결론</h4>

본 실험에서 **{top3.iloc[0]['Model']}** 모델이 전체 8개 평가 지표의 평균 순위 **{top3.iloc[0]['평균 순위']:.2f}위**로
가장 높은 종합 성능을 기록하였습니다.<br><br>
특히 MAPE {df_metrics.loc[df_metrics['Model']==top3.iloc[0]['Model'], 'MAPE(%)'].values[0]:.2f}%,
RMSE {df_metrics.loc[df_metrics['Model']==top3.iloc[0]['Model'], 'RMSE'].values[0]:.3f}를 달성하여
예측 정확도와 오차 크기 모두에서 우수한 결과를 보였습니다.<br><br>
따라서 현재 데이터와 실험 조건({method_txt}, horizon={horizon})에서는
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
- **11가지 예측 모델**: Naive, SMA, ExpSmoothing, Holt, Holt-Winters, STL, ARIMA, SARIMA, AutoARIMA, Theta, Prophet
- **8가지 평가 지표**: MSE, RMSE, MAE, MAPE, MASE, MdRAE (정확도) + RSFE, TS (편향 진단)
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
- **ARIMA**: 사용자 지정 (p,d,q) 파라미터
- **SARIMA**: 계절 ARIMA, (p,d,q)(P,D,Q,sp) 파라미터
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

