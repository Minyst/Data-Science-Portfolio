"""Streamlit 앱: 원예장비 총괄생산계획 최적화 대시보드."""
import streamlit as st

from data import APPParams, PRESETS
from optimizer import solve
import charts

st.set_page_config(layout="wide", page_title="총괄생산계획 최적화", page_icon="🏭")
st.title("🏭 원예장비 제조업체 총괄생산계획")
st.caption("Pyomo + HiGHS 기반 최적화 · 수요·비용·자원 파라미터를 조정하면 즉시 재계획")

# ────────────────────────── 사이드바 ──────────────────────────
with st.sidebar:
    st.header("⚙️ 파라미터")
    mode = st.radio("최적화 모드", ["LP", "IP"], horizontal=True,
                    help="LP=연속 완화, IP=정수 계획")

    preset_name = st.selectbox("수요 프리셋", list(PRESETS.keys()))
    preset_demand = PRESETS[preset_name]
    horizon = st.number_input("계획기간(월)", 1, 24, len(preset_demand))

    with st.expander("📈 월별 수요 (ea)", expanded=True):
        demand = []
        for t in range(horizon):
            default = preset_demand[t] if t < len(preset_demand) else 2000
            demand.append(
                st.number_input(f"{t+1}월", value=float(default), min_value=0.0,
                                step=100.0, key=f"d_{t}")
            )

    with st.expander("💰 비용 (천원)"):
        price = st.number_input("판매단가", value=40.0)
        material = st.number_input("재료비/개", value=10.0)
        holding = st.number_input("재고유지비/개·월", value=2.0)
        shortage = st.number_input("부재고비/개·월", value=5.0)
        wage_reg = st.number_input("정규임금/Hr", value=4.0)
        wage_ot = st.number_input("초과임금/Hr", value=6.0)
        hire_cost = st.number_input("고용비/인", value=300.0)
        fire_cost = st.number_input("해고비/인", value=500.0)
        subcontract_unit_cost = st.number_input("하청 추가비용/개", value=30.0)

    with st.expander("👷 생산능력"):
        work_days = st.number_input("작업일/월", value=20, min_value=1)
        work_hours = st.number_input("작업시간/일", value=8, min_value=1)
        ot_limit_per_worker = st.number_input("초과시간 상한/인·월", value=10, min_value=0)
        std_hours_per_unit = st.number_input("표준작업시간/개 (Hr)", value=4.0, min_value=0.1)

    with st.expander("🏁 초기/최종 조건"):
        W0 = st.number_input("초기 인원 W₀", value=80, min_value=0)
        I0 = st.number_input("초기 재고 I₀", value=1000, min_value=0)
        S0 = st.number_input("초기 부재고 S₀", value=0, min_value=0)
        IT_min = st.number_input("최종 재고 하한 Iᴛ", value=500, min_value=0)
        ST_end = st.number_input("최종 부재고 Sᴛ", value=0, min_value=0)

    run = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

# ────────────────────────── 실행 ──────────────────────────
if run:
    params = APPParams(
        demand=demand,
        price=price, material=material, holding=holding, shortage=shortage,
        wage_reg=wage_reg, wage_ot=wage_ot, hire_cost=hire_cost, fire_cost=fire_cost,
        subcontract_unit_cost=subcontract_unit_cost,
        work_days=int(work_days), work_hours=int(work_hours),
        ot_limit_per_worker=int(ot_limit_per_worker),
        std_hours_per_unit=float(std_hours_per_unit),
        W0=int(W0), I0=int(I0), S0=int(S0),
        IT_min=int(IT_min), ST_end=int(ST_end),
    )
    try:
        with st.spinner("최적화 중..."):
            result = solve(params, mode)
        st.session_state["result"] = result
        st.session_state["params"] = params
        st.session_state["mode"] = mode
        if "optimal" not in result.status.lower():
            st.warning(f"솔버 상태: {result.status}")
    except Exception as e:
        st.error(f"최적화 실패: {e}")

# ────────────────────────── 결과 대시보드 ──────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]
    p = st.session_state["params"]
    mode_used = st.session_state["mode"]

    st.success(f"✅ 최적화 완료 ({mode_used}, status={r.status})")

    tabs = st.tabs([
        "📊 KPI", "📦 수요·생산", "🏬 재고·부재고",
        "👷 인력", "💸 비용", "🔬 민감도", "📋 원본 표",
    ])

    with tabs[0]:
        k = charts.kpi_values(r, p)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총비용", k["총비용"])
        c2.metric("서비스 수준", k["서비스 수준"])
        c3.metric("평균 재고", k["평균 재고"])
        c4.metric("평균 가동률", k["평균 가동률"])
        st.divider()
        # 3등분 균형 레이아웃: 게이지 | 게이지 | 도넛 (모두 동일 높이 320)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(charts.cost_breakdown(r),
                            use_container_width=True, key="kpi_donut")
        with g2:
            st.plotly_chart(charts.service_level_gauge(r),
                            use_container_width=True, key="kpi_service")
        with g3:
            st.plotly_chart(charts.utilization_gauge(r, p),
                            use_container_width=True, key="kpi_util")

    with tabs[1]:
        st.plotly_chart(charts.demand_vs_supply(r), use_container_width=True, key="demand_supply")

    with tabs[2]:
        st.plotly_chart(charts.inventory_backlog(r), use_container_width=True, key="inv_backlog")

    with tabs[3]:
        st.plotly_chart(charts.workforce(r), use_container_width=True, key="workforce")

    with tabs[4]:
        st.plotly_chart(charts.cost_by_month(r, p), use_container_width=True, key="cost_area")
        st.plotly_chart(charts.cost_breakdown(r), use_container_width=True, key="cost_donut")

    with tabs[5]:
        if r.shadow_prices:
            st.caption(
                "**잠재가격(Shadow Price)**: 해당 제약이 한 단위 완화될 때 총비용이 줄어드는 양 (천원). "
                "값이 클수록 그 제약이 이익에 크게 걸리고 있음을 의미합니다. "
                "(LP 모드에서만 계산됨)"
            )
            st.plotly_chart(charts.shadow_price_chart(r), use_container_width=True, key="shadow_chart")
            st.dataframe(charts.shadow_price_table(r), use_container_width=True, hide_index=True)
        else:
            st.info("민감도 분석은 **LP 모드**에서만 제공됩니다. 사이드바에서 LP를 선택 후 재실행하세요.")

    with tabs[6]:
        st.plotly_chart(charts.decision_heatmap(r), use_container_width=True, key="heatmap")
        df = charts.result_dataframe(r)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ CSV 다운로드",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="app_result.csv",
            mime="text/csv",
        )
else:
    st.info("👈 좌측에서 파라미터를 설정하고 **최적화 실행** 버튼을 누르세요.")
