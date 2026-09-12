"""问题1：固定初始储电量 S_1 的购电费用敏感性分析。

运行：python -B sensitivity_q1.py
输出：outputs/s1_sensitivity.csv

先在 7500--9500 kWh 内按 25 kWh 细扫，再补充 1200--10800 kWh
范围内按 200 kWh 粗扫。每个扫描点均重新求解原线性规划；本脚本不绘图，
也不修改主模型的调度结果文件。
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from solve_q1 import (N, SOLVER_OPTIONS, TOL, analyze_initial_energy, build_lp,
                      load_data, solve_model, unpack, validate_solution)


COARSE_VALUES = tuple(range(1200, 10801, 200))
FINE_VALUES = tuple(range(7500, 9501, 25))


def fixed_initial_model(model, initial_energy):
    """固定 S[0]=S[144]=initial_energy，其余模型内容保持不变。"""
    objective, A_eq, b_eq, bounds = model
    fixed_bounds = list(bounds)
    fixed_bounds[4 * N] = (initial_energy, initial_energy)
    fixed_bounds[5 * N] = (initial_energy, initial_energy)
    return objective, A_eq, b_eq, fixed_bounds


def solve_at(initial_energy, price, L, R, model, optimal_cost, cost_tolerance):
    fixed_model = fixed_initial_model(model, initial_energy)
    objective, A_eq, b_eq, bounds = fixed_model
    result = linprog(objective, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                     method="highs", options=SOLVER_OPTIONS)
    if not result.success or result.status != 0:
        return {
            "初始储电量S_1(kWh)": initial_energy,
            "求解状态": "不可行" if result.status == 2 else "求解失败",
            "求解器状态码": int(result.status),
            "求解器信息": result.message,
        }

    validation = validate_solution(result, price, L, R, fixed_model)
    G, C, D, _, S = unpack(result.x)
    extra_cost = float(result.fun - optimal_cost)
    if abs(extra_cost) <= cost_tolerance:
        extra_cost = 0.0
    return {
        "初始储电量S_1(kWh)": initial_energy,
        "最优购电费用J(s)(元)": float(result.fun),
        "相对全局最优额外费用E(s)(元)": extra_cost,
        "是否达到全局最优费用": "是" if extra_cost == 0.0 else "否",
        "求解状态": "最优",
        "求解器状态码": int(result.status),
        "求解器信息": result.message,
        "迭代次数": int(result.nit),
        "实际初始储电量(kWh)": float(S[0]),
        "实际日末储电量(kWh)": float(S[-1]),
        "全天购电量(kWh)": float(G.sum()),
        "全天充电量(kWh)": float(C.sum()),
        "全天放电量(kWh)": float(D.sum()),
        "逐时段能量平衡最大误差(kWh)": validation["energy_balance_max_abs_kwh"],
        "储电递推最大误差(kWh)": validation["storage_dynamics_max_abs_kwh"],
        "日循环误差(kWh)": validation["daily_cycle_abs_kwh"],
        "最大越界量(kWh)": validation["bound_max_violation_kwh"],
        "原始与对偶目标差(元)": validation["primal_dual_gap_abs_yuan"],
        "同时充放电时段数": validation["simultaneous_charge_discharge_count"],
    }


def write_results(path, rows):
    fieldnames = [
        "初始储电量S_1(kWh)", "扫描层级", "最优购电费用J(s)(元)",
        "相对全局最优额外费用E(s)(元)", "是否达到全局最优费用", "求解状态",
        "求解器状态码", "求解器信息", "迭代次数", "实际初始储电量(kWh)",
        "实际日末储电量(kWh)", "全天购电量(kWh)", "全天充电量(kWh)",
        "全天放电量(kWh)", "逐时段能量平衡最大误差(kWh)",
        "储电递推最大误差(kWh)", "日循环误差(kWh)", "最大越界量(kWh)",
        "原始与对偶目标差(元)", "同时充放电时段数",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, optimal_cost, exact_low, exact_high):
    feasible = [row for row in rows if row["求解状态"] == "最优"]
    infeasible = [row for row in rows if row["求解状态"] != "最优"]
    fine = [row for row in feasible if row["扫描层级"] in ("局部细扫", "粗扫与细扫重合")]
    sampled_optimal = [row["初始储电量S_1(kWh)"] for row in fine
                       if row["是否达到全局最优费用"] == "是"]
    extra = np.array([row["相对全局最优额外费用E(s)(元)"] for row in feasible])
    s_values = np.array([row["初始储电量S_1(kWh)"] for row in feasible])
    overlap_count = sum(row["同时充放电时段数"] for row in feasible)
    max_residual = max(max(row["逐时段能量平衡最大误差(kWh)"],
                           row["储电递推最大误差(kWh)"],
                           row["日循环误差(kWh)"], row["最大越界量(kWh)"])
                       for row in feasible)
    return {
        "扫描点总数": len(rows),
        "可行点数": len(feasible),
        "不可行或失败点数": len(infeasible),
        "全局最优购电费用J*(元)": optimal_cost,
        "辅助优化给出的精确最优区间(kWh)": [exact_low, exact_high],
        "细扫中观测到的零额外费用点范围(kWh)": (
            [min(sampled_optimal), max(sampled_optimal)] if sampled_optimal else None),
        "全范围扫描最大额外费用(元)": float(extra.max()),
        "最大额外费用对应S_1(kWh)": float(s_values[np.argmax(extra)]),
        "所有扫描解同时充放电时段总数": int(overlap_count),
        "所有扫描解最大物理约束误差(kWh)": float(max_residual),
    }


def main():
    base = Path(__file__).resolve().parent
    price, L, R, _ = load_data(base / "附件1.csv")
    model = build_lp(price, L, R)
    free_result = solve_model(model)
    validate_solution(free_result, price, L, R, model)
    optimal_cost = float(free_result.fun)
    cost_tolerance = max(1e-6, 1e-9 * abs(optimal_cost))
    recomputed_interval = analyze_initial_energy(free_result, model)

    summary = json.loads((base / "outputs" / "summary.json").read_text(encoding="utf-8"))
    reported_cost = summary["求解结果"]["最优购电费用(元)"]
    if abs(reported_cost - optimal_cost) > cost_tolerance:
        raise RuntimeError("现有summary.json与重新求得的主模型最优费用不一致")
    interval = summary["初始储电量最优区间"]
    exact_low = float(recomputed_interval["下端点(kWh)"])
    exact_high = float(recomputed_interval["上端点(kWh)"])
    if (abs(float(interval["下端点(kWh)"]) - exact_low) > TOL
            or abs(float(interval["上端点(kWh)"]) - exact_high) > TOL):
        raise RuntimeError("现有summary.json与重新求得的最优初始储电量区间不一致")

    fine_set, coarse_set = set(FINE_VALUES), set(COARSE_VALUES)
    results = {}
    for value in FINE_VALUES:
        results[value] = solve_at(value, price, L, R, model, optimal_cost, cost_tolerance)
    for value in COARSE_VALUES:
        if value not in results:
            results[value] = solve_at(value, price, L, R, model, optimal_cost, cost_tolerance)

    rows = []
    for value in sorted(results):
        row = results[value]
        if value in fine_set and value in coarse_set:
            row["扫描层级"] = "粗扫与细扫重合"
        elif value in fine_set:
            row["扫描层级"] = "局部细扫"
        else:
            row["扫描层级"] = "全范围粗扫"
        rows.append(row)

    output = base / "outputs" / "s1_sensitivity.csv"
    write_results(output, rows)
    report = summarize(rows, optimal_cost, exact_low, exact_high)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
