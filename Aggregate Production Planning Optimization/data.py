"""기본 파라미터와 수요 프리셋."""
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class APPParams:
    demand: List[float] = field(default_factory=lambda: [1600, 3000, 3200, 3800, 2200, 2200])

    # 단가 (천원 단위)
    price: float = 40.0
    material: float = 10.0
    holding: float = 2.0
    shortage: float = 5.0
    wage_reg: float = 4.0
    wage_ot: float = 6.0
    hire_cost: float = 300.0
    fire_cost: float = 500.0
    subcontract_unit_cost: float = 30.0  # 하청 추가비용(하청단가 - 재료비)

    # 생산능력
    work_days: int = 20
    work_hours: int = 8
    ot_limit_per_worker: int = 10
    std_hours_per_unit: float = 4.0  # ⇒ rate = 1/4 ea/Hr

    # 초기/최종 조건
    W0: int = 80
    I0: int = 1000
    S0: int = 0
    IT_min: int = 500
    ST_end: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "기본 (6개월)": [1600, 3000, 3200, 3800, 2200, 2200],
    "변동 수요 (8개월)": [1600, 5000, 3200, 5800, 2200, 2200, 6500, 2300],
    "균등 수요 (6개월)": [2500] * 6,
    "성수기 집중 (6개월)": [1000, 1500, 5000, 5500, 1500, 1000],
}
