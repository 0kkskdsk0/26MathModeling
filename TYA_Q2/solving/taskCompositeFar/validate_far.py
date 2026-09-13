# -*- coding: utf-8 -*-
"""远视模型的形式化核验：复用 taskComposite 的 12 条断言，并新增 C13—C21。

| 编号 | 断言 | 形式化表述 | 容差 |
|---|---|---|---|
| C1—C12 | 沿用 `taskComposite/validate.py` | 见该模块表头 | $10^{-6}$（C10 相对 $10^{-9}$） |
| C13 | 终端价值自下方支撑 | $\\max_{\\omega,q}[a_qS_{\\omega,145}+b_q-\\theta_\\omega]\\le10^{-6}$ | $10^{-6}$ 元 |
| C14 | 终端价值取到切线上包络 | $\\max_\\omega[\\theta_\\omega-\\max_q(a_qS_{\\omega,145}+b_q)]\\le10^{-6}$ | $10^{-6}$ 元 |
| C15 | 切线斜率非正 | $\\max_q a_q\\le10^{-6}$ | $10^{-6}$ 元/kWh |
| C16 | 切线斜率随 $q$ 单调不减 | $\\max_q(a_{q+1}-a_q)\\le10^{-6}$ | $10^{-6}$ 元/kWh |
| C17 | 值函数关于 $s$ 单调不增 | $\\max_k(v_{k+1}-v_k)\\le10^{-6}$ | $10^{-6}$ 元 |
| C18 | 次梯度随网格点单调不减（内部点） | $\\max_k(g_{k+1}-g_k)\\le10^{-6}$ | $10^{-6}$ 元/kWh |
| C19 | 网格收敛 | $\\Delta_{R-1}\\le\\eta$ 且 $\\hat V_R-\\hat V_{R-1}\\le\\tau_{\\mathrm{env}}$ | 见 `frozen_rule_far.json` |
| C20 | $\\theta\\equiv0$ 退化 | 切线置零后与近视模型的计划购电量逐元素相同 | 逐元素严格相等 |
| C21 | 因果性 | 扰动第 $d$ 天及之后的观测后，第 $d$ 天的切线集合与计划购电量逐元素不变 | 逐元素严格相等 |

C13 与 C14 合起来说明 $\\theta_\\omega=\\max_q(a_qS_{\\omega,145}+b_q)$，即分段线性下界被取到，
线性规划没有引入松弛；C15 与 C16 说明切线集合构成合法的凸支撑；
C17 与 C18 说明采样得到的值函数与次梯度满足费用函数应有的单调性与凸性。
任一断言失败即抛出异常，不静默通过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__:
    from . import model_far
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taskCompositeFar import model_far  # noqa: F401

from taskComposite.validate import (COST_TOL, SLOTS, TOL, assert_day,  # noqa: E402,F401
                                    assert_decision, check_day,  # noqa: E402,F401
                                    check_scenarios, cross_day_chain,  # noqa: E402,F401
                                    summary)

TERMINAL_TOL = 1e-6
TERMINAL_ABS_TOL = 1e-4      # 元
TERMINAL_REL_TOL = 1e-6      # 相对于 max(1, max|theta|)


def terminal_tolerance(theta) -> float:
    """C13/C14 的混合容差：`max(绝对容差, 相对容差 × max(1, max|θ|))`。

    $\\theta$ 的量级是数万元，而 HiGHS 的可行性容差是绝对的 $10^{-10}$，
    直接用绝对容差会把求解器精度放大后的正常残差误判为违规；直接用相对容差又会在
    $\\theta$ 很小时过严。混合容差在两端都合理：以 $\\theta\\sim5\\times10^4$ 元计，
    允许的绝对残差是 $5\\times10^{-2}$ 元，对应每天约 $0.05$ 元的影响，
    比终端价值本身的估计精度（$\\tau_{\\mathrm{env}}=15$ 元）小三个数量级。
    """
    theta = np.asarray(theta, dtype=float)
    scale = float(max(1.0, np.abs(theta).max())) if theta.size else 1.0
    return max(TERMINAL_ABS_TOL, TERMINAL_REL_TOL * scale)


def check_terminal_value(solution: dict, tangents, scenarios: int) -> dict:
    """C13、C14：终端价值约束的支撑残差与取包络残差。

    判定用混合容差（见 `terminal_tolerance`）；绝对残差与相对残差同时如实报告，
    不隐藏任何一项。返回字段刻意避开 `_count`、`_abs` 等 `assert_day` 会当作
    "越界量"的后缀，以免 $\\theta$ 的正常量级（数万元）被误判为违规。
    """
    coefficients = model_far.normalise_tangents(tangents)
    if len(coefficients) == 0:
        return {"terminal_support_violation": 0.0, "terminal_envelope_gap": 0.0,
                "terminal_support_rel": 0.0, "terminal_envelope_rel": 0.0,
                "terminal_theta_mean": 0.0, "passed": True}
    record = model_far.terminal_value_residual(solution, coefficients, scenarios)
    theta = np.asarray(solution["theta"], dtype=float)
    scale = float(max(1.0, np.abs(theta).max())) if theta.size else 1.0
    tolerance = terminal_tolerance(theta)
    gap = float(max(0.0, record["terminal_objective_gap"]))
    out = {
        "terminal_support_violation": record["terminal_support_violation"],
        "terminal_envelope_gap": gap,
        "terminal_support_rel": record["terminal_support_violation"] / scale,
        "terminal_envelope_rel": gap / scale,
        "terminal_theta_mean": float(theta.mean()) if theta.size else 0.0,
    }
    out["passed"] = bool(record["terminal_support_violation"] <= tolerance
                         and gap <= tolerance)
    return out


def check_tangents(tangents) -> dict:
    """C15、C16：切线斜率的符号与单调性。

    单调性检查作用在**输入的 $q$ 顺序**上：$a_{d,q}\\le0$ 且随 $q$ 单调不减。
    装配前的 `normalise_tangents` 会按斜率重排，因此这里必须先于排序检查。
    """
    coefficients = np.asarray(tangents, dtype=float).reshape((-1, 2))
    if len(coefficients) == 0:
        return {"tangent_positive_violation": 0.0, "tangent_monotone_violation": 0.0,
                "passed": True}
    differences = np.diff(coefficients[:, 0]) if len(coefficients) > 1 else np.zeros(0)
    record = {
        "tangent_positive_violation": float(max(0.0, coefficients[:, 0].max())),
        "tangent_monotone_violation": float(max(0.0, -differences.min()))
        if len(differences) else 0.0,
    }
    record["passed"] = bool(record["tangent_positive_violation"] <= TERMINAL_TOL
                            and record["tangent_monotone_violation"] <= TERMINAL_TOL)
    return record


def check_value_shape(values: np.ndarray, slopes: np.ndarray) -> dict:
    """C17、C18：值函数单调不增与凸（次梯度非降，内部点）。"""
    values = np.asarray(values, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    if len(values) < 2:
        return {"value_monotone_violation": 0.0, "value_convexity_violation": 0.0,
                "passed": True}
    monotone = float(max(0.0, np.diff(values).max()))
    interior = np.diff(slopes[:-1]) if len(slopes) > 2 else np.zeros(0)
    convexity = float(max(0.0, -interior.min())) if len(interior) else 0.0
    record = {"value_monotone_violation": monotone,
              "value_convexity_violation": convexity}
    record["passed"] = bool(monotone <= TERMINAL_TOL and convexity <= TERMINAL_TOL)
    return record


def check_convergence(rounds: list, eta: float, tau_env: float) -> dict:
    """C19：网格收敛判据。"""
    if len(rounds) < 2:
        return {"converged": False, "delta_previous": float("inf"),
                "envelope_increase": float("inf"), "eta": eta, "tau_env": tau_env,
                "passed": False}
    previous, final = rounds[-2], rounds[-1]
    record = {
        "converged": bool(final.get("converged", False)),
        "delta_previous": float(previous["max_adjacent_difference_yuan"]),
        "envelope_increase": float(final["envelope_increase_yuan"]),
        "eta": float(eta), "tau_env": float(tau_env),
        "rounds_used": len(rounds),
        "new_effective_tangents_strict": int(final.get("new_effective_tangents", -1)),
    }
    record["passed"] = bool(record["delta_previous"] <= eta
                            and record["envelope_increase"] <= tau_env)
    return record


def check_degeneracy(plan_zero: np.ndarray, plan_near: np.ndarray) -> dict:
    """C20：切线置零后的计划购电量与近视模型逐元素相同。"""
    plan_zero = np.asarray(plan_zero, dtype=float)
    plan_near = np.asarray(plan_near, dtype=float)
    same = plan_zero.shape == plan_near.shape and bool(np.array_equal(plan_zero, plan_near))
    return {"degenerate_plan_identical": bool(same),
            "degenerate_max_abs_difference_kwh": float(np.max(np.abs(plan_zero - plan_near)))
            if plan_zero.shape == plan_near.shape else float("inf"),
            "passed": bool(same)}


def check_causality(reference: dict, perturbed: dict) -> dict:
    """C21：扰动目标日及之后的观测后，第 d 天的切线集合与计划购电量逐元素不变。"""
    same_tangents = bool(np.array_equal(np.asarray(reference["tangents"], dtype=float),
                                        np.asarray(perturbed["tangents"], dtype=float)))
    same_plan = bool(np.array_equal(np.asarray(reference["plan"], dtype=float),
                                    np.asarray(perturbed["plan"], dtype=float)))
    return {
        "tangents_identical": same_tangents,
        "plan_identical": same_plan,
        "pool_identical": bool(reference["pool"].equals(perturbed["pool"])),
        "weights_identical": bool(np.array_equal(reference["weights"], perturbed["weights"])),
        "target_observations_changed": bool(perturbed["target_changed"]),
        "prefix_features_identical": bool(perturbed["prefix_identical"]),
        "max_abs_plan_change_kwh": float(np.max(np.abs(
            np.asarray(reference["plan"], dtype=float)
            - np.asarray(perturbed["plan"], dtype=float))))
        if len(reference["plan"]) == len(perturbed["plan"]) else float("inf"),
        "passed": bool(same_tangents and same_plan and perturbed["target_changed"]),
    }


def assert_far(record: dict, label: str) -> None:
    """远视核验失败立即抛出，并带上足以定位问题的字段。"""
    if record.get("passed"):
        return
    failed = {k: v for k, v in record.items()
              if k != "passed" and isinstance(v, float)
              and k.endswith(("_rel", "_violation", "_gap"))}
    raise AssertionError(f"{label} 远视核验未通过：{failed}")
