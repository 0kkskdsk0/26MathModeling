# -*- coding: utf-8 -*-
"""含终端价值项的问题二稀疏线性规划装配（SciPy + HiGHS）。

模型相对 `taskComposite/model.py` 的唯一差别是在目标函数中恢复终端价值项

    min  sum_t P_t G_t + sum_w pi_w sum_t 5 P_t E_wt + sum_w pi_w theta_w
         + eps sum_w pi_w sum_t (C_wt + D_wt)

并新增分段线性下界约束，自下方支撑次日价值函数 V_{d+1}(s)：

    theta_w >= a_q S_{w,145} + b_q,     q = 0,...,Q-1,  w in Omega_d

其中 (a_q, b_q) 由 `value_function.py` 在同一情景池 Omega_d 与同一核权重 pi 下
因果地估计得到，且 a_q <= 0（V 关于开局储电量单调不增）、a_q 随 q 单调不减（V 凸）。
因为 theta_w 以正权重 pi_w 进入最小化目标，线性规划会自动取
theta_w = max_q (a_q S_{w,145} + b_q)，即切线的上包络；由值函数的次梯度性质，
每条切线都在 V_{d+1} 下方，所以包络自下方支撑 V_{d+1}，不会高估留电的价值。

**退化开关。** `tangents=None`（或长度为 0）时不添加 theta 变量与约束，装配结果与
`taskComposite/model.py` 的 `build_decision_lp` 逐元素相同。`tangents` 全为零矩阵时
会添加 theta_k >= 0 与目标系数 pi_w，优化后 theta 取 0，计划购电量与近视模型一致。

**接口兼容性。** `ModelParams`、`LinearProgram`、`VariableLayout`、`RecourseLayout`、
`build_recourse_lp`、`solve`、`unpack_recourse` 直接从 `taskComposite/model.py` 复用；
本模块只新增 `FarVariableLayout`、`build_decision_lp_far`、`unpack_decision_far`
与终端价值约束的残差检查。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack

if __package__:
    from .value_function import S_MAX, S_MIN
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taskCompositeFar.value_function import S_MAX, S_MIN

from taskComposite.model import (SLOTS, LinearProgram, ModelParams, VariableLayout,  # noqa: E402
                                 build_decision_lp, build_recourse_lp, solve,  # noqa: E402,F401
                                 unpack_decision, unpack_recourse, validate_scenarios)  # noqa: E402,F401

TOL = 1e-6
_TOL_PAIR = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
FAR_SOLVER_LADDER = (
    ("highs+presolve", _TOL_PAIR, "highs"),
    ("highs-no-presolve", {**_TOL_PAIR, "presolve": False}, "highs"),
    ("highs-ds-no-presolve", {**_TOL_PAIR, "presolve": False}, "highs-ds"),
    ("highs-ipm", {"ipm_optimality_tolerance": 1e-10}, "highs-ipm"),
)


def solve_far(program: LinearProgram, tag: str, trace: list | None = None):
    """远视决策层与远视执行层的求解入口：按固定顺序回退的多级求解。

    终端价值约束把自由变量 $\\theta$（数万元量级）、斜率 $a_q$（约 $-0.5$）与大右端项
    $-b_q$（约 $-5\\times10^4$）放进同一个模型后，数值条件明显变差：2025-03-25 在
    HiGHS 默认 presolve 下返回 status 15（model_status Unknown、primal_status
    Infeasible），而 2025-02-11 在关闭 presolve 后反而返回同样的状态——**两种设置各有
    失败的个案**，模型本身在两处都可行。因此这里按事先固定的顺序逐级回退：

    1. `highs` + presolve（与 `taskComposite.model.solve` 逐元素相同，绝大多数日走这一级）；
    2. `highs` + `presolve=False`；
    3. `highs-ds` + `presolve=False`；
    4. `highs-ipm`。

    实际走了哪一级逐日记录在 `dispatch_year_far.csv` 的 `solver_path` 列。
    无论走哪一级，解都要通过 C1—C14 的残差与终端价值核验，因此求解路径的差异
    不会改变模型的定义与结论。
    """
    from scipy.optimize import linprog
    last = None
    for name, options, method in FAR_SOLVER_LADDER:
        last = linprog(program.c, A_ub=program.A_ub, b_ub=program.b_ub,
                       A_eq=program.A_eq, b_eq=program.b_eq, bounds=program.bounds,
                       method=method, options=options)
        if last.success and last.status == 0 and np.isfinite(last.x).all():
            if trace is not None:
                trace.append(name)
            return last
    raise RuntimeError(f"{tag} 在四级求解路径上均未达到最优状态：{last.message}")


@dataclass(frozen=True)
class FarVariableLayout:
    """在近视布局之后追加 theta 块：分块顺序 [G, GC, C, D, S, E, W, theta]。"""

    base: VariableLayout
    n_theta: int = 0

    @property
    def scenarios(self) -> int:
        return self.base.scenarios

    @property
    def n_G(self) -> int:
        return self.base.n_G

    @property
    def offset_G(self) -> int:
        return self.base.offset_G

    @property
    def offset_GC(self) -> int:
        return self.base.offset_GC

    @property
    def offset_C(self) -> int:
        return self.base.offset_C

    @property
    def offset_D(self) -> int:
        return self.base.offset_D

    @property
    def offset_S(self) -> int:
        return self.base.offset_S

    @property
    def offset_E(self) -> int:
        return self.base.offset_E

    @property
    def offset_W(self) -> int:
        return self.base.offset_W

    @property
    def offset_theta(self) -> int:
        return self.base.size

    @property
    def size(self) -> int:
        return self.base.size + self.n_theta

    def G(self, t: int) -> int:
        return self.base.G(t)

    def GC(self, t: int) -> int:
        return self.base.GC(t)

    def C(self, k: int, t: int) -> int:
        return self.base.C(k, t)

    def D(self, k: int, t: int) -> int:
        return self.base.D(k, t)

    def S(self, k: int, t: int) -> int:
        return self.base.S(k, t)

    def E(self, k: int, t: int) -> int:
        return self.base.E(k, t)

    def W(self, k: int, t: int) -> int:
        return self.base.W(k, t)

    def theta(self, k: int) -> int:
        return self.offset_theta + k


def normalise_tangents(tangents) -> np.ndarray:
    """校验并规范化切线矩阵，返回 (Q,2) 的 float 数组，Q 可以为 0。"""
    if tangents is None:
        return np.zeros((0, 2))
    array = np.asarray(tangents, dtype=float).reshape((-1, 2))
    if len(array) == 0:
        return array
    if not np.isfinite(array).all():
        raise ValueError("切线系数必须为有限值")
    if (array[:, 0] > TOL).any():
        raise ValueError("终端价值是费用函数，切线斜率必须非正（a_q <= 0）")
    # 切线的标号 q 本身不带语义，按斜率升序重排，使 q 与次梯度单调性一致
    return array[np.argsort(array[:, 0], kind="stable")]


def build_decision_lp_far(params: ModelParams, load: np.ndarray, pv: np.ndarray,
                          weights: np.ndarray, s_start: float,
                          tangents=None) -> LinearProgram:
    """装配第 d 天的远视两阶段随机线性规划。

    `tangents=None` 或不含任何行时退化为 `taskComposite/model.py` 的近视装配
    （不添加 theta 变量与约束，矩阵逐元素相同）。
    """
    base = build_decision_lp(params, load, pv, weights, s_start)
    coefficients = normalise_tangents(tangents)
    scenarios = base.layout.scenarios
    if len(coefficients) == 0:
        return base
    if not (S_MIN - TOL <= s_start <= S_MAX + TOL):
        raise ValueError("日初储电量必须落在 [Smin, Smax] 内")
    quality = len(coefficients)
    layout = FarVariableLayout(base.layout, scenarios)

    c = np.concatenate([base.c, np.asarray(weights, dtype=float)])
    # 追加行：[0 | theta 块] 与 [S_{k,145} 块 | theta 块] 上下拼接，
    # 使新增约束 -theta_k + a_q S_{k,145} <= -b_q 同时引用旧列与新列。
    top = hstack([base.A_ub, coo_matrix((base.A_ub.shape[0], scenarios))],
                 format="csr")
    storage_rows, storage_cols, storage_data = [], [], []
    theta_rows, theta_cols, theta_data = [], [], []
    for k in range(scenarios):
        for q in range(quality):
            row = k * quality + q
            storage_rows.append(row)
            storage_cols.append(base.layout.S(k, SLOTS))
            storage_data.append(float(coefficients[q, 0]))
            theta_rows.append(row)
            theta_cols.append(k)
            theta_data.append(-1.0)
    storage_block = coo_matrix((storage_data, (storage_rows, storage_cols)),
                               shape=(scenarios * quality, base.size))
    theta_block = coo_matrix((theta_data, (theta_rows, theta_cols)),
                             shape=(scenarios * quality, scenarios))
    bottom = hstack([storage_block, theta_block], format="csr")
    A_ub = vstack([top, bottom], format="csr")
    b_ub = np.concatenate([base.b_ub,
                           np.tile(-coefficients[:, 1], scenarios)])
    A_eq = hstack([base.A_eq,
                   coo_matrix((base.A_eq.shape[0], scenarios))], format="csr")
    bounds = list(base.bounds) + [(None, None)] * scenarios
    return LinearProgram(c, A_eq, base.b_eq, A_ub, b_ub, bounds, layout, "decision_far")


def unpack_decision_far(result, layout: FarVariableLayout, scenarios: int) -> dict:
    """按布局拆解远视决策层解向量，近视分块与 `unpack_decision` 完全一致。"""
    base_layout = layout.base if isinstance(layout, FarVariableLayout) else layout
    view = _SolutionView(np.asarray(result.x, dtype=float)[:base_layout.size],
                         float(result.fun))
    solution = unpack_decision(view, base_layout, scenarios)
    if isinstance(layout, FarVariableLayout) and layout.n_theta:
        solution["theta"] = np.asarray(result.x[layout.offset_theta:layout.size],
                                       dtype=float)
    else:
        solution["theta"] = np.zeros(0)
    return solution


class _SolutionView:
    """把解向量的前 base.size 个分量与目标值暴露给 `unpack_decision`。

    这样远视与近视的分块拆解共用同一段代码，退化核验的"逐元素相同"才有意义。
    """

    __slots__ = ("x", "fun")

    def __init__(self, x: np.ndarray, fun: float):
        self.x = x
        self.fun = fun


@dataclass(frozen=True)
class FarRecourseLayout:
    """在近视执行层布局之后追加一个 theta 变量：分块顺序 [C, D, S, E, W, theta]。"""

    base: object
    n_theta: int = 0

    @property
    def size(self) -> int:
        return self.base.size + self.n_theta

    @property
    def offset_theta(self) -> int:
        return self.base.size

    def C(self, t: int) -> int:
        return self.base.C(t)

    def D(self, t: int) -> int:
        return self.base.D(t)

    def S(self, t: int) -> int:
        return self.base.S(t)

    def E(self, t: int) -> int:
        return self.base.E(t)

    def W(self, t: int) -> int:
        return self.base.W(t)

    def theta(self) -> int:
        return self.offset_theta


def build_recourse_lp_far(params: ModelParams, plan: np.ndarray, load: np.ndarray,
                          pv: np.ndarray, s_start: float,
                          tangents=None) -> LinearProgram:
    """在**执行层**也计入终端价值项的附加口径（不用于主对照，见 comparison 报告第五节）。

    主对照按要求让两个模型共用逐元素相同的执行层；本函数只用于诊断
    "执行层的近视性是否掩盖了终端价值的作用"。它在补救问题的目标里追加
    $\\theta$ 与同一组切线给出的下界，使实际执行路径也愿意为次日留电。
    `tangents=None` 时退化为 `taskComposite` 的执行层。
    """
    from taskComposite.model import build_recourse_lp
    base = build_recourse_lp(params, plan, load, pv, s_start)
    coefficients = normalise_tangents(tangents)
    if len(coefficients) == 0:
        return base
    quality = len(coefficients)
    layout = FarRecourseLayout(base.layout, 1)
    c = np.concatenate([base.c, [1.0]])
    top = hstack([base.A_ub, coo_matrix((base.A_ub.shape[0], 1))], format="csr")
    storage_col = base.layout.S(SLOTS)
    storage_block = coo_matrix((coefficients[:, 0],
                                (np.arange(quality), np.full(quality, storage_col))),
                               shape=(quality, base.size))
    theta_block = coo_matrix((-np.ones(quality), (np.arange(quality), np.zeros(quality, int))),
                             shape=(quality, 1))
    bottom = hstack([storage_block, theta_block], format="csr")
    A_ub = vstack([top, bottom], format="csr")
    b_ub = np.concatenate([base.b_ub, -coefficients[:, 1]])
    A_eq = hstack([base.A_eq, coo_matrix((base.A_eq.shape[0], 1))], format="csr")
    bounds = list(base.bounds) + [(None, None)]
    return LinearProgram(c, A_eq, base.b_eq, A_ub, b_ub, bounds, layout, "recourse_far")


def unpack_recourse_far(result, layout) -> dict:
    """按布局拆解远视执行层的解向量。"""
    base_layout = layout.base if isinstance(layout, FarRecourseLayout) else layout
    view = _SolutionView(np.asarray(result.x, dtype=float)[:base_layout.size],
                         float(result.fun))
    solution = unpack_recourse(view, base_layout)
    solution["theta"] = (float(result.x[layout.offset_theta])
                         if isinstance(layout, FarRecourseLayout) and layout.n_theta
                         else 0.0)
    return solution


def terminal_value_residual(solution: dict, tangents: np.ndarray,
                            scenarios: int) -> dict:
    """终端价值约束的残差：max_{w,q} [a_q S_w,145 + b_q - theta_w]，应 <= 0。"""
    coefficients = normalise_tangents(tangents)
    if len(coefficients) == 0:
        return {"terminal_support_violation": 0.0, "terminal_slack_min": 0.0,
                "terminal_objective_gap": 0.0, "tangent_count": 0}
    storage_end = np.asarray(solution["S"], dtype=float)[:, -1]
    theta = np.asarray(solution["theta"], dtype=float)
    if theta.size != scenarios:
        raise ValueError("theta 维度必须等于情景数")
    lower = (coefficients[:, 0:1] * storage_end[None, :]
             + coefficients[:, 1:2])                     # (Q, K)
    residual = lower - theta[None, :]
    return {
        "terminal_support_violation": float(np.max(np.maximum(0.0, residual))),
        "terminal_slack_min": float(np.min(theta[None, :] - lower)),
        "terminal_objective_gap": float((theta - lower.max(axis=0)).max()),
        "tangent_count": int(len(coefficients)),
    }
