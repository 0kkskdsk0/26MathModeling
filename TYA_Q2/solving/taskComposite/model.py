# -*- coding: utf-8 -*-
"""问题二两阶段随机线性规划的稀疏装配与求解。

决策层：第 d 天 0:00 求解，第一阶段变量为对所有情景共享的计划购电量 G，
第二阶段变量为逐情景的充放电、储电量、紧急购电量与弃置电量：

    min  sum_t P_t G_t + sum_w pi_w sum_t 5 P_t E_wt
    s.t. G_t + E_wt + R_wt + D_wt = L_wt + C_wt + W_wt          (能量平衡)
         S_w,t+1 = S_wt + eta_c C_wt - D_wt / eta_d              (SOC 递推)
         S_w,1 = S_start
         C_wt <= max(0, R_wt - L_wt) + G_t                      (充电来源约束)
         Smin <= S_wt <= Smax, 0 <= C_wt, D_wt <= e_bar
         G, E, W >= 0

执行层：固定决策层的 G，把当日实测负载与光伏代入同一组物理约束，
以 min sum_t 5 P_t E_t 求解实际执行路径 C, D, S, E, W，即填报 result2 的量。
执行层的 G 是常数，因此"负载优先"可以写成精确的上下界：

    C_t <= max(0, R_t + G_t - L_t)        (缺口时段不得充电)
    E_t <= max(0, L_t - R_t - G_t)        (富余时段不得紧急购电)

两条合起来推出 C_t * E_t = 0，即不存在"一边紧急购电一边充电"的时段。

**关于充电来源约束的动机。** 仅靠 G + E + R + D = L + C + W 一条平衡式，
线性规划会允许"用紧急购电支持储能充电"：充电量 C 是情景特定变量，计划购电量
G 是全情景共享变量，于是"在低价时段用该情景的紧急电充电、再到高价时段放电替代
紧急购电"成为套利路径。Q2Project.md 的假设 3 曾认为 5P_t > P_t 会自动排除它，
该判断不成立：跨时段比较的是"低价时段 5P"与"高价时段 5P"，而不是与同时刻的 P 比。
上述约束把充电能力与共享的计划购电绑定，执行层再以精确上下界落实物理互斥。

变量索引一律通过 `VariableLayout` 计算，避免分块顺序与索引映射错位。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

SLOTS = 144
SOLVER_OPTIONS = {"primal_feasibility_tolerance": 1e-10,
                  "dual_feasibility_tolerance": 1e-10}
RESIDUAL_TOL = 1e-6


@dataclass(frozen=True)
class ModelParams:
    """问题二的物理与价格参数，默认值取自 notation.md 与附录 1。

    `degeneracy_penalty` 是去退化项的单价（元/kWh），量级取 1e-6，比最低电价低六个
    数量级。线性规划在"紧急购电总量相同"的前提下对充放电路径存在大量等价最优解，
    不加该项时求解器可能返回"同时充放电"的退化解。该系数只用于在等价最优解中挑出
    充放电动作最少的路径，不改变主成本与计划购电量。
    """

    price: np.ndarray
    delta: float = 1.0 / 6.0
    s_min: float = 1200.0
    s_max: float = 10800.0
    max_power_kw: float = 5000.0
    eta_c: float = 0.9
    eta_d: float = 0.9
    emergency_multiplier: float = 5.0
    degeneracy_penalty: float = 1e-6

    @property
    def max_energy(self) -> float:
        return self.max_power_kw * self.delta

    def __post_init__(self):
        price = np.asarray(self.price, dtype=float)
        if price.shape != (SLOTS,) or not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError("电价必须是 144 维严格正有限向量")
        object.__setattr__(self, "price", price)


@dataclass(frozen=True)
class LinearProgram:
    """装配好的线性规划，含等式块、不等式块、边界与变量布局。"""

    c: np.ndarray
    A_eq: csr_matrix
    b_eq: np.ndarray
    A_ub: csr_matrix
    b_ub: np.ndarray
    bounds: list
    layout: object
    kind: str

    @property
    def size(self) -> int:
        return len(self.c)


@dataclass(frozen=True)
class VariableLayout:
    """决策层变量分块与索引映射。

    分块顺序为 [G(144), GC(144), C(K*144), D(K*144), S(K*145), E(K*144), W(K*144)]。
    GC_t 是"计划购电中划给储能充电的份额"，也是第一阶段变量，因此对所有情景共享。
    """

    scenarios: int

    @property
    def n_G(self) -> int:
        return SLOTS

    @property
    def offset_G(self) -> int:
        return 0

    @property
    def offset_GC(self) -> int:
        return self.n_G

    @property
    def offset_C(self) -> int:
        return self.offset_GC + SLOTS

    @property
    def offset_D(self) -> int:
        return self.offset_C + self.scenarios * SLOTS

    @property
    def offset_S(self) -> int:
        return self.offset_D + self.scenarios * SLOTS

    @property
    def offset_E(self) -> int:
        return self.offset_S + self.scenarios * (SLOTS + 1)

    @property
    def offset_W(self) -> int:
        return self.offset_E + self.scenarios * SLOTS

    @property
    def size(self) -> int:
        return self.offset_W + self.scenarios * SLOTS

    def G(self, t: int) -> int:
        return self.offset_G + t

    def GC(self, t: int) -> int:
        return self.offset_GC + t

    def C(self, k: int, t: int) -> int:
        return self.offset_C + k * SLOTS + t

    def D(self, k: int, t: int) -> int:
        return self.offset_D + k * SLOTS + t

    def S(self, k: int, t: int) -> int:
        return self.offset_S + k * (SLOTS + 1) + t

    def E(self, k: int, t: int) -> int:
        return self.offset_E + k * SLOTS + t

    def W(self, k: int, t: int) -> int:
        return self.offset_W + k * SLOTS + t


@dataclass(frozen=True)
class RecourseLayout:
    """执行层变量分块：G 已定，其余同决策层单一情景。"""

    @property
    def size(self) -> int:
        return 4 * SLOTS + SLOTS + 1

    def C(self, t: int) -> int:
        return t

    def D(self, t: int) -> int:
        return SLOTS + t

    def S(self, t: int) -> int:
        return 2 * SLOTS + t

    def E(self, t: int) -> int:
        return 3 * SLOTS + 1 + t

    def W(self, t: int) -> int:
        return 4 * SLOTS + 1 + t


def validate_scenarios(load: np.ndarray, pv: np.ndarray, weights: np.ndarray) -> tuple:
    """核验情景矩阵与权重形状、有限性与归一化。"""
    load, pv, weights = (np.asarray(a, dtype=float) for a in (load, pv, weights))
    if load.ndim != 2 or load.shape != pv.shape or load.shape[1] != SLOTS:
        raise ValueError("情景负载与光伏必须是同形状的 (K,144) 矩阵")
    if weights.shape != (len(load),):
        raise ValueError("情景权重长度必须等于情景数")
    if not (np.isfinite(load).all() and np.isfinite(pv).all() and np.isfinite(weights).all()):
        raise ValueError("情景与权重必须为有限值")
    if (load < 0).any() or (pv < 0).any() or (weights < 0).any():
        raise ValueError("情景功率与权重必须非负")
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"情景权重必须归一化，当前和为 {weights.sum()!r}")
    return load, pv, weights


def build_decision_lp(params: ModelParams, load: np.ndarray, pv: np.ndarray,
                      weights: np.ndarray, s_start: float) -> LinearProgram:
    """装配第 d 天的两阶段随机线性规划。

    充电来源用显式的一阶段份额 GC_t 表达：

        C_wt <= max(0, R_wt - L_wt) + GC_t          (充电只能来自光伏富余与计划电份额)
        GC_t <= G_t - Delta_t                       (计划电先覆盖期望净负荷缺口)
        0 <= GC_t <= G_t

    其中 Delta_t = sum_w pi_w max(0, L_wt - R_wt) 是该时段的期望净负荷缺口。
    因为 GC_t 对所有情景共享，规划不能再"用某个情景自己的紧急购电去充电"，
    紧急购电与充电的时段级互斥由此在决策层也得到保证。
    """
    load, pv, weights = validate_scenarios(load, pv, weights)
    if not (params.s_min <= s_start <= params.s_max):
        raise ValueError("日初储电量必须落在 [Smin, Smax] 内")
    scenarios = len(load)
    layout = VariableLayout(scenarios)
    net = load - pv
    deficit = np.maximum(0.0, load - pv)
    expected_deficit = weights @ deficit
    rows, cols, data = [], [], []
    ub_rows, ub_cols, ub_data, ub_bound = [], [], [], []

    def add(row, col, value):
        rows.append(row)
        cols.append(col)
        data.append(value)

    def add_ub(row, col, value):
        ub_rows.append(row)
        ub_cols.append(col)
        ub_data.append(value)

    # 目标：计划购电费 + 紧急购电费的经验期望 + 去退化项
    c = np.zeros(layout.size)
    c[:layout.n_G] = params.price
    emergency_price = params.emergency_multiplier * params.price
    for k in range(scenarios):
        for t in range(SLOTS):
            c[layout.E(k, t)] = weights[k] * emergency_price[t]
            c[layout.C(k, t)] = weights[k] * params.degeneracy_penalty
            c[layout.D(k, t)] = weights[k] * params.degeneracy_penalty

    # 能量平衡：G + E + D - C - W = L - R
    for k in range(scenarios):
        for t in range(SLOTS):
            row = k * SLOTS + t
            add(row, layout.G(t), 1.0)
            add(row, layout.E(k, t), 1.0)
            add(row, layout.D(k, t), 1.0)
            add(row, layout.C(k, t), -1.0)
            add(row, layout.W(k, t), -1.0)
    # SOC 递推：S_{t+1} - S_t - eta C + D / eta = 0
    soc_offset = scenarios * SLOTS
    for k in range(scenarios):
        for t in range(SLOTS):
            row = soc_offset + k * SLOTS + t
            add(row, layout.S(k, t + 1), 1.0)
            add(row, layout.S(k, t), -1.0)
            add(row, layout.C(k, t), -params.eta_c)
            add(row, layout.D(k, t), 1.0 / params.eta_d)
    # 共同出发点
    start_offset = 2 * scenarios * SLOTS
    for k in range(scenarios):
        add(start_offset + k, layout.S(k, 0), 1.0)
    # 充电来源约束：C_wt - GC_t <= max(0, R_wt - L_wt)
    for k in range(scenarios):
        for t in range(SLOTS):
            row = k * SLOTS + t
            add_ub(row, layout.C(k, t), 1.0)
            add_ub(row, layout.GC(t), -1.0)
            ub_bound.append(max(0.0, pv[k, t] - load[k, t]))
    # 计划电份额上限：GC_t - G_t <= -Delta_t，同时给出 G_t >= Delta_t
    for t in range(SLOTS):
        row = scenarios * SLOTS + t
        add_ub(row, layout.GC(t), 1.0)
        add_ub(row, layout.G(t), -1.0)
        ub_bound.append(-float(expected_deficit[t]))

    A_eq = coo_matrix((data, (rows, cols)),
                      shape=(2 * scenarios * SLOTS + scenarios, layout.size)).tocsr()
    b_eq = np.concatenate([net.ravel(),
                           np.zeros(scenarios * SLOTS),
                           np.full(scenarios, float(s_start))])
    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)),
                      shape=(scenarios * SLOTS + SLOTS, layout.size)).tocsr()
    bounds = ([(0.0, None)] * (layout.n_G + SLOTS)
              + [(0.0, params.max_energy)] * (2 * scenarios * SLOTS)
              + [(params.s_min, params.s_max)] * (scenarios * (SLOTS + 1))
              + [(0.0, None)] * (2 * scenarios * SLOTS))
    return LinearProgram(c, A_eq, b_eq, A_ub, np.asarray(ub_bound), bounds, layout,
                         "decision")


def build_recourse_lp(params: ModelParams, plan: np.ndarray, load: np.ndarray,
                      pv: np.ndarray, s_start: float) -> LinearProgram:
    """固定计划购电量后的实际执行补救问题。"""
    plan = np.asarray(plan, dtype=float)
    load, pv = np.asarray(load, dtype=float), np.asarray(pv, dtype=float)
    if plan.shape != (SLOTS,) or load.shape != (SLOTS,) or pv.shape != (SLOTS,):
        raise ValueError("计划购电、负载与光伏都必须为 144 维")
    if (plan < 0).any() or (load < 0).any() or (pv < 0).any():
        raise ValueError("计划购电量、负载与光伏必须非负")
    if not (params.s_min <= s_start <= params.s_max):
        raise ValueError("日初储电量必须落在 [Smin, Smax] 内")
    layout = RecourseLayout()
    net = load - pv
    rows, cols, data = [], [], []
    ub_rows, ub_cols, ub_data, ub_bound = [], [], [], []

    def add(row, col, value):
        rows.append(row)
        cols.append(col)
        data.append(value)

    def add_ub(row, col, value):
        ub_rows.append(row)
        ub_cols.append(col)
        ub_data.append(value)

    c = np.zeros(layout.size)
    for t in range(SLOTS):
        c[layout.E(t)] = params.emergency_multiplier * params.price[t]
        c[layout.C(t)] = params.degeneracy_penalty
        c[layout.D(t)] = params.degeneracy_penalty
    # 能量平衡：E + D - C - W = L - R - G
    for t in range(SLOTS):
        add(t, layout.E(t), 1.0)
        add(t, layout.D(t), 1.0)
        add(t, layout.C(t), -1.0)
        add(t, layout.W(t), -1.0)
    for t in range(SLOTS):
        row = SLOTS + t
        add(row, layout.S(t + 1), 1.0)
        add(row, layout.S(t), -1.0)
        add(row, layout.C(t), -params.eta_c)
        add(row, layout.D(t), 1.0 / params.eta_d)
    add(2 * SLOTS, layout.S(0), 1.0)
    # 负载优先：A_t = R_t + G_t - L_t 为常数
    surplus = np.maximum(0.0, pv + plan - load)
    deficit = np.maximum(0.0, load - pv - plan)
    for t in range(SLOTS):
        add_ub(t, layout.C(t), 1.0)
        ub_bound.append(float(surplus[t]))
    for t in range(SLOTS):
        add_ub(SLOTS + t, layout.E(t), 1.0)
        ub_bound.append(float(deficit[t]))

    A_eq = coo_matrix((data, (rows, cols)), shape=(2 * SLOTS + 1, layout.size)).tocsr()
    b_eq = np.concatenate([net - plan, np.zeros(SLOTS), [float(s_start)]])
    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)),
                      shape=(2 * SLOTS, layout.size)).tocsr()
    bounds = ([(0.0, params.max_energy)] * (2 * SLOTS)
              + [(params.s_min, params.s_max)] * (SLOTS + 1)
              + [(0.0, None)] * (2 * SLOTS))
    return LinearProgram(c, A_eq, b_eq, A_ub, np.asarray(ub_bound), bounds, layout,
                         "recourse")


def solve(program: LinearProgram, tag: str):
    """调用 SciPy 的 HiGHS 接口求解，非最优直接报错而不静默返回。"""
    result = linprog(program.c, A_ub=program.A_ub, b_ub=program.b_ub,
                     A_eq=program.A_eq, b_eq=program.b_eq, bounds=program.bounds,
                     method="highs", options=SOLVER_OPTIONS)
    if not result.success or result.status != 0:
        raise RuntimeError(f"{tag} 未达到最优状态：{result.message}")
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"{tag} 的解包含非有限值")
    return result


def unpack_decision(result, layout: VariableLayout, scenarios: int) -> dict:
    """按布局拆解决策层解向量，返回逐情景数组与第一阶段计划。"""
    x = result.x
    G = x[layout.offset_G:layout.offset_G + layout.n_G]
    shape = (scenarios, SLOTS)
    return {
        "G": G,
        "GC": x[layout.offset_GC:layout.offset_GC + SLOTS],
        "C": x[layout.offset_C:layout.offset_C + scenarios * SLOTS].reshape(shape),
        "D": x[layout.offset_D:layout.offset_D + scenarios * SLOTS].reshape(shape),
        "S": x[layout.offset_S:layout.offset_S + scenarios * (SLOTS + 1)].reshape(
            (scenarios, SLOTS + 1)),
        "E": x[layout.offset_E:layout.offset_E + scenarios * SLOTS].reshape(shape),
        "W": x[layout.offset_W:layout.offset_W + scenarios * SLOTS].reshape(shape),
        "objective": float(result.fun),
    }


def unpack_recourse(result, layout: RecourseLayout) -> dict:
    """按布局拆解执行层解向量。"""
    x = result.x
    return {"C": x[:SLOTS], "D": x[SLOTS:2 * SLOTS], "S": x[2 * SLOTS:3 * SLOTS + 1],
            "E": x[3 * SLOTS + 1:4 * SLOTS + 1], "W": x[4 * SLOTS + 1:5 * SLOTS + 1],
            "objective": float(result.fun)}
