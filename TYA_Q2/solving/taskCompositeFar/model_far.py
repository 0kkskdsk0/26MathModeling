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
from scipy.sparse import coo_matrix, hstack

if __package__:
    from .value_function import S_MAX, S_MIN
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taskCompositeFar.value_function import S_MAX, S_MIN

from taskComposite.model import (SLOTS, LinearProgram, ModelParams, VariableLayout,  # noqa: E402
                                 build_decision_lp, build_recourse_lp, solve,  # noqa: E402,F401
                                 unpack_decision, unpack_recourse, validate_scenarios)  # noqa: E402,F401

TOL = 1e-6


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
    rows, cols, data = [], [], []
    for k in range(scenarios):
        for q in range(quality):
            row = k * quality + q
            rows.append(row)
            cols.append(k)
            data.append(-1.0)
            rows.append(row)
            cols.append(base.layout.S(k, SLOTS))
            data.append(float(coefficients[q, 0]))
    theta_block = coo_matrix((data, (rows, cols)), shape=(scenarios * quality, scenarios))
    A_ub = hstack([base.A_ub, theta_block], format="csr")
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
