"""원예장비 총괄생산계획 Pyomo 모델 (LP/IP).

PDF 강의록(스마트제조_06_총괄생산계획.pdf)의 수식을 그대로 구현한다.

결정변수 (t = 1..T):
  W_t  종업원 수          H_t  신규 고용         L_t  해고
  P_t  생산량             I_t  월말 재고         S_t  월말 부재고
  C_t  하청 계약량        O_t  총 초과근무시간

비용 최소화:
  Z = Σ (wage_reg * work_hours * work_days) * W_t
    + Σ wage_ot * O_t
    + Σ hire_cost * H_t + Σ fire_cost * L_t
    + Σ holding * I_t + Σ shortage * S_t
    + Σ material * P_t + Σ subcontract_unit_cost * C_t
"""
from dataclasses import dataclass, field
from typing import List, Dict, Literal

from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, NonNegativeReals, NonNegativeIntegers,
    SolverFactory, Suffix, minimize, value,
)

from data import APPParams


@dataclass
class APPResult:
    status: str
    total_cost: float
    W: List[float] = field(default_factory=list)
    H: List[float] = field(default_factory=list)
    L: List[float] = field(default_factory=list)
    P: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    C: List[float] = field(default_factory=list)
    O: List[float] = field(default_factory=list)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    demand: List[float] = field(default_factory=list)
    # 민감도 분석 결과 (LP 전용; IP에서는 빈 dict)
    shadow_prices: Dict[str, List[float]] = field(default_factory=dict)


def build_model(p: APPParams, mode: Literal["LP", "IP"] = "LP") -> ConcreteModel:
    T = len(p.demand)
    TIME = list(range(0, T + 1))   # 0..T (0은 초기월)
    PERIODS = list(range(1, T + 1))

    domain = NonNegativeIntegers if mode == "IP" else NonNegativeReals

    m = ConcreteModel()
    if mode == "LP":
        m.dual = Suffix(direction=Suffix.IMPORT)
    m.W = Var(TIME, domain=domain)
    m.H = Var(TIME, domain=domain)
    m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain)
    m.I = Var(TIME, domain=domain)
    m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain)
    m.O = Var(TIME, domain=domain)

    # 파생 상수
    reg_labor_cost = p.wage_reg * p.work_hours * p.work_days   # 기본 640
    rate = 1.0 / p.std_hours_per_unit                          # 0.25 ea/Hr
    reg_capacity_per_worker = rate * p.work_hours * p.work_days  # 40 ea/인·월

    # 목적함수
    m.Cost = Objective(
        expr=sum(
            reg_labor_cost * m.W[t]
            + p.wage_ot * m.O[t]
            + p.hire_cost * m.H[t]
            + p.fire_cost * m.L[t]
            + p.holding * m.I[t]
            + p.shortage * m.S[t]
            + p.material * m.P[t]
            + p.subcontract_unit_cost * m.C[t]
            for t in PERIODS
        ),
        sense=minimize,
    )

    # 제약조건
    m.labor = Constraint(
        PERIODS, rule=lambda m, t: m.W[t] == m.W[t - 1] + m.H[t] - m.L[t]
    )
    m.capacity = Constraint(
        PERIODS,
        rule=lambda m, t: m.P[t] <= reg_capacity_per_worker * m.W[t] + rate * m.O[t],
    )
    m.inventory = Constraint(
        PERIODS,
        rule=lambda m, t: m.I[t] == m.I[t - 1] + m.P[t] + m.C[t]
        - p.demand[t - 1] - m.S[t - 1] + m.S[t],
    )
    m.overtime = Constraint(
        PERIODS, rule=lambda m, t: m.O[t] <= p.ot_limit_per_worker * m.W[t]
    )

    # 초기/최종 고정
    m.W0 = Constraint(rule=lambda m: m.W[0] == p.W0)
    m.I0 = Constraint(rule=lambda m: m.I[0] == p.I0)
    m.S0 = Constraint(rule=lambda m: m.S[0] == p.S0)
    m.H0 = Constraint(rule=lambda m: m.H[0] == 0)
    m.L0 = Constraint(rule=lambda m: m.L[0] == 0)
    m.P0 = Constraint(rule=lambda m: m.P[0] == 0)
    m.C0 = Constraint(rule=lambda m: m.C[0] == 0)
    m.O0 = Constraint(rule=lambda m: m.O[0] == 0)
    m.last_inventory = Constraint(rule=lambda m: m.I[T] >= p.IT_min)
    m.last_shortage = Constraint(rule=lambda m: m.S[T] == p.ST_end)

    return m


def _get_solver():
    """HiGHS 솔버를 우선 시도. 실패 시 명확한 오류."""
    for name in ("appsi_highs", "highs"):
        try:
            solver = SolverFactory(name)
            if solver.available(exception_flag=False):
                return solver
        except Exception:
            continue
    raise RuntimeError(
        "HiGHS 솔버를 찾을 수 없습니다. `pip install highspy` 후 다시 시도하세요."
    )


def solve(p: APPParams, mode: Literal["LP", "IP"] = "LP") -> APPResult:
    m = build_model(p, mode)
    solver = _get_solver()
    res = solver.solve(m)
    status = str(res.solver.termination_condition) if hasattr(res, "solver") else "unknown"

    T = len(p.demand)
    PERIODS = list(range(1, T + 1))

    def extract(var):
        return [float(value(var[t]) or 0.0) for t in PERIODS]

    W = extract(m.W); H = extract(m.H); L = extract(m.L)
    P = extract(m.P); I = extract(m.I); S = extract(m.S)
    C = extract(m.C); O = extract(m.O)

    reg_labor_cost = p.wage_reg * p.work_hours * p.work_days
    breakdown = {
        "정규임금": sum(reg_labor_cost * w for w in W),
        "초과근무": sum(p.wage_ot * o for o in O),
        "고용비": sum(p.hire_cost * h for h in H),
        "해고비": sum(p.fire_cost * l for l in L),
        "재고유지": sum(p.holding * i for i in I),
        "부재고": sum(p.shortage * s for s in S),
        "재료비": sum(p.material * pp for pp in P),
        "하청비": sum(p.subcontract_unit_cost * c for c in C),
    }

    shadow = {}
    if mode == "LP" and hasattr(m, "dual"):
        def duals(con):
            out = []
            for t in PERIODS:
                try:
                    out.append(float(m.dual[con[t]]))
                except Exception:
                    out.append(0.0)
            return out
        try:
            shadow = {
                "수요(재고균형)": duals(m.inventory),
                "생산능력": duals(m.capacity),
                "노동력": duals(m.labor),
                "초과근무 상한": duals(m.overtime),
            }
        except Exception:
            shadow = {}

    return APPResult(
        status=status,
        total_cost=float(value(m.Cost)),
        W=W, H=H, L=L, P=P, I=I, S=S, C=C, O=O,
        cost_breakdown=breakdown,
        demand=list(p.demand),
        shadow_prices=shadow,
    )


if __name__ == "__main__":
    # PDF 재현 검증
    cases = [
        ("LP 6개월", [1600, 3000, 3200, 3800, 2200, 2200], "LP", 422275.0),
        ("IP 6개월", [1600, 3000, 3200, 3800, 2200, 2200], "IP", 422660.0),
        ("IP 8개월", [1600, 5000, 3200, 5800, 2200, 2200, 6500, 2300], "IP", 763220.0),
    ]
    for name, d, mode, expected in cases:
        r = solve(APPParams(demand=d), mode)
        ok = abs(r.total_cost - expected) < 1.0
        print(f"[{name}] cost={r.total_cost:.1f} expected={expected} {'OK' if ok else 'MISMATCH'}")
