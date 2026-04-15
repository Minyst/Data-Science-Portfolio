"""Plotly 차트 빌더 (다양한 시각화 타입 조합)."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import APPResult
from data import APPParams

BAR_WIDTH = 0.45  # 얇은 바 폭


def _months(n: int) -> list:
    return [f"{t}월" for t in range(1, n + 1)]


# ────────── 1. 수요 vs 공급 ──────────
def demand_vs_supply(r: APPResult) -> go.Figure:
    """라인(수요) + 얇은 바(생산) + 얇은 바(하청) + 누적 공급 에어리어."""
    x = _months(len(r.demand))
    supply = [p + c for p, c in zip(r.P, r.C)]
    fig = go.Figure()
    fig.add_scatter(name="누적공급(P+C)", x=x, y=supply, mode="lines",
                    fill="tozeroy", line=dict(color="#B0C4DE", width=0),
                    fillcolor="rgba(176,196,222,0.35)", hoverinfo="skip")
    fig.add_bar(name="생산 P", x=x, y=r.P, marker_color="#4C78A8", width=BAR_WIDTH)
    fig.add_bar(name="하청 C", x=x, y=r.C, marker_color="#F58518", width=BAR_WIDTH)
    fig.add_scatter(name="수요 D", x=x, y=r.demand, mode="lines+markers",
                    line=dict(color="#E45756", width=3, dash="dot"),
                    marker=dict(size=10, symbol="diamond"))
    fig.update_layout(barmode="stack", title="수요 vs 공급(생산+하청)",
                      yaxis_title="ea", legend_orientation="h", height=420)
    return fig


# ────────── 2. 재고·부재고 (영역 + 라인) ──────────
def inventory_backlog(r: APPResult) -> go.Figure:
    x = _months(len(r.demand))
    neg_S = [-s for s in r.S]
    fig = go.Figure()
    fig.add_scatter(name="재고 I", x=x, y=r.I, mode="lines+markers",
                    fill="tozeroy", line=dict(color="#54A24B", width=2),
                    fillcolor="rgba(84,162,75,0.35)",
                    marker=dict(size=8))
    fig.add_scatter(name="부재고 S", x=x, y=neg_S, mode="lines+markers",
                    fill="tozeroy", line=dict(color="#E45756", width=2),
                    fillcolor="rgba(228,87,86,0.35)",
                    marker=dict(size=8, symbol="x"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="재고 vs 부재고 (영역 차트)",
                      yaxis_title="ea (부재고는 음수)",
                      legend_orientation="h", height=420)
    return fig


# ────────── 3. 인력 (이중 축, 스텝 + 산점도) ──────────
def workforce(r: APPResult) -> go.Figure:
    """상단: 인원 W 추이 영역+라인 / 하단: 고용(+)·해고(−) 대칭 바."""
    x = _months(len(r.demand))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.08,
        subplot_titles=("인원 W 추이", "월별 고용(+) · 해고(−)"),
    )
    fig.add_scatter(
        name="인원 W", x=x, y=r.W, mode="lines+markers",
        line=dict(color="#2E86AB", width=3),
        marker=dict(size=10, color="#2E86AB"),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.2)",
        row=1, col=1,
    )
    fig.add_bar(name="고용 H", x=x, y=r.H,
                marker_color="#4C78A8", width=BAR_WIDTH, row=2, col=1)
    fig.add_bar(name="해고 L", x=x, y=[-l for l in r.L],
                marker_color="#E45756", width=BAR_WIDTH, row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="인원 (명)", row=1, col=1)
    fig.update_yaxes(title_text="변동 (명)", row=2, col=1)
    fig.update_layout(title="인력 변동", legend_orientation="h",
                      height=500, barmode="relative")
    return fig


# ────────── 4. 비용 도넛 ──────────
def cost_breakdown(r: APPResult) -> go.Figure:
    items = list(r.cost_breakdown.keys())
    values = list(r.cost_breakdown.values())
    total = sum(values)
    fig = go.Figure(go.Pie(
        labels=items, values=values, hole=0.6,
        textinfo="none", hoverinfo="label+value+percent",
        pull=[0.02] * len(items), sort=False,
        domain=dict(x=[0.0, 1.0], y=[0.25, 1.0]),
    ))
    fig.add_annotation(
        text=f"<b>{total:,.0f}</b><br><sub>천원</sub>",
        x=0.5, y=0.625, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16), align="center",
    )
    fig.update_layout(
        title=dict(text="<b>비용 구성</b>", x=0.5, xanchor="center",
                   y=0.95, yanchor="top", font=dict(size=16)),
        height=320,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05,
                    xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig


# ────────── 5. 월별 비용 누적 영역 차트 ──────────
def cost_by_month(r: APPResult, p: APPParams) -> go.Figure:
    x = _months(len(r.demand))
    reg = p.wage_reg * p.work_hours * p.work_days
    series = {
        "정규임금": [reg * w for w in r.W],
        "초과근무": [p.wage_ot * o for o in r.O],
        "고용비": [p.hire_cost * h for h in r.H],
        "해고비": [p.fire_cost * l for l in r.L],
        "재고유지": [p.holding * i for i in r.I],
        "부재고": [p.shortage * s for s in r.S],
        "재료비": [p.material * pp for pp in r.P],
        "하청비": [p.subcontract_unit_cost * c for c in r.C],
    }
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_scatter(name=name, x=x, y=ys, mode="lines",
                        stackgroup="one", hoverinfo="x+y+name")
    fig.update_layout(title="월별 비용 구성 (누적 영역 차트)",
                      yaxis_title="천원", legend_orientation="h", height=420)
    return fig


# ────────── 6. KPI 게이지 4종 ──────────
def _single_gauge(value: float, title: str, bar_color: str,
                  axis_max: float, steps: list) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=22)),
        gauge=dict(axis=dict(range=[0, axis_max]),
                   bar=dict(color=bar_color),
                   steps=steps),
        domain=dict(x=[0.1, 0.9], y=[0.05, 0.9]),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   y=0.92, yanchor="top", font=dict(size=16)),
        height=320,
        margin=dict(t=35, b=20, l=20, r=20),
    )
    return fig


def service_level_gauge(r: APPResult) -> go.Figure:
    total_demand = sum(r.demand) or 1.0
    val = (1.0 - sum(r.S) / total_demand) * 100
    return _single_gauge(
        val, "서비스 수준", "#54A24B", 100,
        steps=[dict(range=[0, 80], color="#F8D7DA"),
               dict(range=[80, 95], color="#FFF3CD"),
               dict(range=[95, 100], color="#D4EDDA")],
    )


def utilization_gauge(r: APPResult, p: APPParams) -> go.Figure:
    reg_cap = (1.0 / p.std_hours_per_unit) * p.work_hours * p.work_days
    rate = 1.0 / p.std_hours_per_unit
    utils = []
    for w, o, pp in zip(r.W, r.O, r.P):
        cap = reg_cap * w + rate * o
        if cap > 0:
            utils.append(pp / cap * 100)
    val = sum(utils) / len(utils) if utils else 0.0
    return _single_gauge(
        val, "평균 가동률", "#4C78A8", 120,
        steps=[dict(range=[0, 70], color="#FFF3CD"),
               dict(range=[70, 100], color="#D4EDDA"),
               dict(range=[100, 120], color="#F8D7DA")],
    )


def kpi_gauge_pair(r: APPResult, p: APPParams) -> go.Figure:
    """서비스 수준 + 평균 가동률, 2개 게이지."""
    total_demand = sum(r.demand) or 1.0
    service_level = (1.0 - sum(r.S) / total_demand) * 100
    reg_cap = (1.0 / p.std_hours_per_unit) * p.work_hours * p.work_days
    rate = 1.0 / p.std_hours_per_unit
    utils = []
    for w, o, pp in zip(r.W, r.O, r.P):
        cap = reg_cap * w + rate * o
        if cap > 0:
            utils.append(pp / cap * 100)
    avg_util = sum(utils) / len(utils) if utils else 0.0

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("<b>서비스 수준</b>", "<b>평균 가동률</b>"),
        horizontal_spacing=0.15,
    )
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=service_level,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=26)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#54A24B"),
                   steps=[dict(range=[0, 80], color="#F8D7DA"),
                          dict(range=[80, 95], color="#FFF3CD"),
                          dict(range=[95, 100], color="#D4EDDA")])),
        row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_util,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=26)),
        gauge=dict(axis=dict(range=[0, 120]), bar=dict(color="#4C78A8"),
                   steps=[dict(range=[0, 70], color="#FFF3CD"),
                          dict(range=[70, 100], color="#D4EDDA"),
                          dict(range=[100, 120], color="#F8D7DA")])),
        row=1, col=2)
    fig.update_annotations(font_size=18, yshift=-5)
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def kpi_gauges(r: APPResult, p: APPParams) -> go.Figure:
    total_demand = sum(r.demand) or 1.0
    service_level = (1.0 - sum(r.S) / total_demand) * 100
    avg_inv = sum(r.I) / len(r.I) if r.I else 0.0
    max_inv = max(r.I) if r.I else 1.0
    reg_cap = (1.0 / p.std_hours_per_unit) * p.work_hours * p.work_days
    rate = 1.0 / p.std_hours_per_unit
    utils = []
    for w, o, pp in zip(r.W, r.O, r.P):
        cap = reg_cap * w + rate * o
        if cap > 0:
            utils.append(pp / cap * 100)
    avg_util = sum(utils) / len(utils) if utils else 0.0

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=("총비용", "서비스 수준", "평균 재고", "평균 가동률"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Indicator(
        mode="number", value=r.total_cost,
        number=dict(suffix=" 천원", valueformat=",.0f", font=dict(size=28))),
        row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=service_level,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=22)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#54A24B"),
                   steps=[dict(range=[0, 80], color="#F8D7DA"),
                          dict(range=[80, 95], color="#FFF3CD"),
                          dict(range=[95, 100], color="#D4EDDA")])),
        row=1, col=2)
    fig.add_trace(go.Indicator(
        mode="number", value=avg_inv,
        number=dict(suffix=" ea", valueformat=",.0f", font=dict(size=28))),
        row=1, col=3)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=avg_util,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=22)),
        gauge=dict(axis=dict(range=[0, 120]), bar=dict(color="#4C78A8"),
                   steps=[dict(range=[0, 70], color="#FFF3CD"),
                          dict(range=[70, 100], color="#D4EDDA"),
                          dict(range=[100, 120], color="#F8D7DA")])),
        row=1, col=4)
    fig.update_annotations(font_size=16, yshift=10)
    fig.update_layout(height=360, margin=dict(t=80, b=30, l=30, r=30))
    return fig


# ────────── 7. 민감도 ──────────
def shadow_price_chart(r: APPResult) -> go.Figure:
    x = _months(len(r.demand))
    fig = go.Figure()
    colors = {"수요(재고균형)": "#E45756", "생산능력": "#4C78A8",
              "노동력": "#54A24B", "초과근무 상한": "#F58518"}
    for name, series in r.shadow_prices.items():
        fig.add_scatter(name=name, x=x, y=series, mode="lines+markers",
                        line=dict(width=2, color=colors.get(name)),
                        marker=dict(size=9))
    fig.update_layout(title="월별 잠재가격 (Shadow Price)",
                      yaxis_title="천원 / 제약단위 1 증가 시",
                      legend_orientation="h", height=420)
    return fig


def shadow_price_table(r: APPResult) -> pd.DataFrame:
    if not r.shadow_prices:
        return pd.DataFrame()
    data = {"월": _months(len(r.demand))}
    data.update(r.shadow_prices)
    return pd.DataFrame(data)


# ────────── 8. 결정변수 히트맵 ──────────
def decision_heatmap(r: APPResult) -> go.Figure:
    """월×변수 정규화 히트맵 (변수별로 max로 정규화)."""
    x = _months(len(r.demand))
    rows = {"W": r.W, "H": r.H, "L": r.L, "P": r.P,
            "I": r.I, "S": r.S, "C": r.C, "O": r.O}
    z_raw, z_norm = [], []
    for name, vals in rows.items():
        z_raw.append(vals)
        m = max(vals) or 1.0
        z_norm.append([v / m for v in vals])
    fig = go.Figure(go.Heatmap(
        z=z_norm, x=x, y=list(rows.keys()),
        text=[[f"{v:.0f}" for v in row] for row in z_raw],
        texttemplate="%{text}", colorscale="Blues",
        colorbar=dict(title="정규화"),
    ))
    fig.update_layout(title="결정변수 월별 히트맵 (각 변수의 최댓값 대비)",
                      height=380)
    return fig


def result_dataframe(r: APPResult) -> pd.DataFrame:
    return pd.DataFrame({
        "월": _months(len(r.demand)),
        "수요 D": r.demand,
        "인원 W": r.W, "고용 H": r.H, "해고 L": r.L,
        "생산 P": r.P, "재고 I": r.I, "부재고 S": r.S,
        "하청 C": r.C, "초과시간 O": r.O,
    })


def kpi_values(r: APPResult, p: APPParams) -> dict:
    total_demand = sum(r.demand) or 1.0
    service_level = 1.0 - (sum(r.S) / total_demand)
    avg_inventory = sum(r.I) / len(r.I) if r.I else 0.0
    reg_cap = (1.0 / p.std_hours_per_unit) * p.work_hours * p.work_days
    rate = 1.0 / p.std_hours_per_unit
    utils = []
    for w, o, pp in zip(r.W, r.O, r.P):
        cap = reg_cap * w + rate * o
        if cap > 0:
            utils.append(pp / cap)
    avg_util = sum(utils) / len(utils) if utils else 0.0
    return {
        "총비용": f"{r.total_cost:,.0f} 천원",
        "서비스 수준": f"{service_level*100:.1f} %",
        "평균 재고": f"{avg_inventory:,.0f} ea",
        "평균 가동률": f"{avg_util*100:.1f} %",
    }
