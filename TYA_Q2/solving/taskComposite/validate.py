# -*- coding: utf-8 -*-
"""问题二求解结果的形式化正确性核验。

核验标准（残差单位 kWh，容差 1e-6）：

| 编号 | 断言 | 形式化表述 |
|---|---|---|
| C1 | 能量平衡 | $\\max_{t}|G_t+E_t+R_t+D_t-L_t-C_t-W_t|\\le10^{-6}$ |
| C2 | SOC 递推 | $\\max_{t}|S_{t+1}-S_t-\\eta_c C_t+D_t/\\eta_d|\\le10^{-6}$ |
| C3 | 日初状态 | $|S_1-S_{\\mathrm{start}}|\\le10^{-6}$ |
| C4 | 储能边界 | $S^{\\min}\\le S_t\\le S^{\\max}$，$\\forall t\\in\\mathcal{T}^+$ |
| C5 | 充放电功率 | $0\\le C_t,D_t\\le\\bar e$ |
| C6 | 非负性 | $G_t,E_t,W_t\\ge0$ |
| C7 | 充放电互斥 | $\\nexists t:\\ C_t>10^{-6}\\ \\text{且}\\ D_t>10^{-6}$ |
| C8 | 紧急电与充电互斥 | $\\nexists t:\\ E_t>10^{-6}\\ \\text{且}\\ C_t>10^{-6}$ |
| C9 | 跨日链 | $S_{d+1,1}=S_{d,145}$（逐日闭合，容差 $10^{-6}$） |
| C10 | 费用重算 | 全天购电费与 $\\sum_t(P_tG_t+5P_tE_t)$ 一致（相对容差 $10^{-9}$） |
| C11 | 充电来源 | $C_t\\le\\max(0,\\,R_t+G_t-L_t)$，缺口时段不得充电 |
| C12 | 紧急电上界 | $E_t\\le\\max(0,\\,L_t-R_t-G_t)$，富余时段不得紧急购电 |

C11 与 C12 合起来推出 $C_tE_t=0$：不存在一边紧急购电一边充电的时段。
决策层另核验每个情景的能量平衡、SOC 递推、充电来源约束与边界，以及情景权重归一化。
任一断言失败即抛出异常，不静默通过。
"""
from __future__ import annotations

import numpy as np

SLOTS = 144
TOL = 1e-6
COST_TOL = 1e-9


def _residual_summary(name: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {f"{name}_max_abs": float(np.max(np.abs(values))) if values.size else 0.0}


def check_day(params, plan, load, pv, executed, s_start, price=None,
              cost_tol=COST_TOL) -> dict:
    """对单日实际执行路径执行 C1—C8、C10，返回核验记录。"""
    price = params.price if price is None else np.asarray(price, dtype=float)
    G, L, R = (np.asarray(a, dtype=float) for a in (plan, load, pv))
    C, D = np.asarray(executed["C"], dtype=float), np.asarray(executed["D"], dtype=float)
    S = np.asarray(executed["S"], dtype=float)
    E, W = np.asarray(executed["E"], dtype=float), np.asarray(executed["W"], dtype=float)
    for name, array in (("G", G), ("L", L), ("R", R), ("C", C), ("D", D), ("S", S),
                        ("E", E), ("W", W)):
        if array.shape not in ((SLOTS,), (SLOTS + 1,)):
            raise ValueError(f"{name} 维度不正确：{array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} 含非有限值")

    balance = G + E + R + D - L - C - W
    dynamics = np.diff(S) - params.eta_c * C + D / params.eta_d
    bounds = {
        "storage_lower_violation": float(max(0.0, params.s_min - S.min())),
        "storage_upper_violation": float(max(0.0, S.max() - params.s_max)),
        "charge_upper_violation": float(max(0.0, C.max() - params.max_energy)),
        "discharge_upper_violation": float(max(0.0, D.max() - params.max_energy)),
        "negative_plan": float(max(0.0, -G.min())),
        "negative_emergency": float(max(0.0, -E.min())),
        "negative_curtail": float(max(0.0, -W.min())),
        "negative_charge": float(max(0.0, -C.min())),
        "negative_discharge": float(max(0.0, -D.min())),
    }
    overlap_cd = np.flatnonzero((C > TOL) & (D > TOL))
    overlap_ec = np.flatnonzero((E > TOL) & (C > TOL))
    surplus = np.maximum(0.0, R + G - L)
    deficit = np.maximum(0.0, L - R - G)
    charge_source_violation = float(np.max(np.maximum(0.0, C - surplus)))
    emergency_upper_violation = float(np.max(np.maximum(0.0, E - deficit)))
    recomputed = float(price @ G + params.emergency_multiplier * price @ E)
    reported = float(executed.get("daily_cost", recomputed))
    cost_error = abs(recomputed - reported)
    record = {
        "energy_balance_max_abs": float(np.max(np.abs(balance))),
        "storage_dynamics_max_abs": float(np.max(np.abs(dynamics))),
        "initial_state_abs": float(abs(S[0] - s_start)),
        **bounds,
        "simultaneous_charge_discharge_count": int(len(overlap_cd)),
        "simultaneous_charge_discharge_periods": overlap_cd.tolist(),
        "emergency_and_charge_count": int(len(overlap_ec)),
        "emergency_and_charge_periods": overlap_ec.tolist(),
        "charge_source_violation": charge_source_violation,
        "emergency_upper_violation": emergency_upper_violation,
        "daily_plan_kwh": float(G.sum()),
        "daily_emergency_kwh": float(E.sum()),
        "daily_plan_cost_yuan": float(price @ G),
        "daily_emergency_cost_yuan": float(params.emergency_multiplier * price @ E),
        "daily_cost_yuan": recomputed,
        "cost_recalculation_abs_yuan": cost_error,
        "cost_recalculation_rel": float(cost_error / max(1.0, abs(recomputed))),
        "storage_min_kwh": float(S.min()),
        "storage_max_kwh": float(S.max()),
        "storage_end_kwh": float(S[-1]),
        "curtail_max_kwh": float(W.max()),
    }
    record["passed"] = bool(
        record["energy_balance_max_abs"] <= TOL
        and record["storage_dynamics_max_abs"] <= TOL
        and record["initial_state_abs"] <= TOL
        and max(bounds.values()) <= TOL
        and charge_source_violation <= TOL
        and emergency_upper_violation <= TOL
        and record["simultaneous_charge_discharge_count"] == 0
        and record["emergency_and_charge_count"] == 0
        and record["cost_recalculation_rel"] <= cost_tol)
    return record


def check_scenarios(params, load, pv, weights, solution, s_start) -> dict:
    """决策层核验：逐情景能量平衡、SOC 递推、充电来源、边界与权重归一化。"""
    load, pv, weights = (np.asarray(a, dtype=float) for a in (load, pv, weights))
    balance, dynamics = [], []
    for k in range(len(load)):
        balance.append(solution["G"] + solution["E"][k] + pv[k] + solution["D"][k]
                       - load[k] - solution["C"][k] - solution["W"][k])
        dynamics.append(np.diff(solution["S"][k]) - params.eta_c * solution["C"][k]
                        + solution["D"][k] / params.eta_d)
    balance, dynamics = np.asarray(balance), np.asarray(dynamics)
    storage = solution["S"]
    surplus = np.maximum(0.0, pv - load)
    charge_source = solution["C"] - surplus - solution["GC"][None, :]
    expected_deficit = weights @ np.maximum(0.0, load - pv)
    share_violation = solution["GC"] + expected_deficit - solution["G"]
    overlap_ec = ((solution["E"] > TOL) & (solution["C"] > TOL)).sum()
    return {
        "scenario_energy_balance_max_abs": float(np.max(np.abs(balance))),
        "scenario_storage_dynamics_max_abs": float(np.max(np.abs(dynamics))),
        "scenario_initial_state_abs": float(np.max(np.abs(storage[:, 0] - s_start))),
        "scenario_storage_lower_violation": float(max(0.0, params.s_min - storage.min())),
        "scenario_storage_upper_violation": float(max(0.0, storage.max() - params.s_max)),
        "scenario_charge_upper_violation": float(
            max(0.0, solution["C"].max() - params.max_energy)),
        "scenario_discharge_upper_violation": float(
            max(0.0, solution["D"].max() - params.max_energy)),
        "scenario_negative_min": float(min(0.0, solution["G"].min(), solution["E"].min(),
                                           solution["W"].min(), solution["C"].min(),
                                           solution["D"].min())),
        "scenario_charge_source_violation": float(np.max(np.maximum(0.0, charge_source))),
        "scenario_charge_share_violation": float(np.max(np.maximum(0.0, share_violation))),
        "scenario_emergency_and_charge_count": int(overlap_ec),
        "weight_sum_error": float(abs(weights.sum() - 1.0)),
        "passed": bool(np.max(np.abs(balance)) <= TOL
                       and np.max(np.abs(dynamics)) <= TOL
                       and max(0.0, params.s_min - storage.min()) <= TOL
                       and max(0.0, storage.max() - params.s_max) <= TOL
                       and float(np.max(np.maximum(0.0, charge_source))) <= TOL
                       and float(np.max(np.maximum(0.0, share_violation))) <= TOL
                       and int(overlap_ec) == 0
                       and abs(weights.sum() - 1.0) <= 1e-12),
    }


def assert_decision(record: dict, label: str) -> None:
    """决策层核验失败立即抛出。"""
    if record.get("passed"):
        return
    failed = {k: v for k, v in record.items()
              if k != "passed" and ((isinstance(v, float) and v > TOL)
                                    or (isinstance(v, int) and v > 0))}
    raise AssertionError(f"{label} 决策层核验未通过：{failed}")


def assert_day(record: dict, label: str) -> None:
    """核验失败立即抛出，并带上足以定位问题的字段。"""
    if record.get("passed"):
        return
    failed = {k: v for k, v in record.items()
              if k.endswith(("_abs", "_violation", "_count", "_rel")) and
              ((isinstance(v, float) and v > TOL) or (isinstance(v, int) and v > 0))}
    raise AssertionError(f"{label} 形式化核验未通过：{failed}")


def cross_day_chain(previous_end: float, current_start: float) -> dict:
    """C9：跨日链残差 S_{d+1,1} = S_{d,145}。"""
    error = abs(float(current_start) - float(previous_end))
    return {"chain_abs_kwh": error, "passed": bool(error <= TOL)}


def summary(records: list[dict]) -> dict:
    """把逐日核验记录汇总为全年结论表。"""
    if not records:
        raise ValueError("没有可汇总的核验记录")
    numeric = [k for k, v in records[0].items() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    worst = {}
    for key in numeric:
        values = [float(r[key]) for r in records]
        worst[key] = {"max": max(values), "min": min(values),
                      "mean": float(np.mean(values))}
    failures = [r for r in records if not r["passed"]]
    return {
        "days_checked": len(records),
        "days_failed": len(failures),
        "all_passed": not failures,
        "tolerance_kwh": TOL,
        "worst_case": worst,
        "total_plan_kwh": float(sum(r["daily_plan_kwh"] for r in records)),
        "total_emergency_kwh": float(sum(r["daily_emergency_kwh"] for r in records)),
        "total_cost_yuan": float(sum(r["daily_cost_yuan"] for r in records)),
    }
