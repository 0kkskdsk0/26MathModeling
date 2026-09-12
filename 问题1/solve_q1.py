"""问题1：自由初始储电量、日循环边界下的主线性规划。

运行：python -B solve_q1.py
输入和输出均相对于本脚本定位；主解不添加互斥或次级目标。
另外分析最优费用下的初始电量区间。
仅输出完整调度dispatch.csv和中文汇总summary.json，不生成中间或写作文件。
"""

import csv
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack


N = 144
DELTA = 1 / 6
S_MIN, S_MAX = 1200.0, 10800.0
MAX_ENERGY = 5000 * DELTA
ETA_C = ETA_D = 0.9
TOL = 1e-6  # 电量、约束残差的绝对容差，单位kWh。
SOLVER_OPTIONS = {"primal_feasibility_tolerance": 1e-8,
                  "dual_feasibility_tolerance": 1e-8}


def time_label(boundary):
    """boundary=0和144分别表示当天00:00和24:00。"""
    minutes = boundary * 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def load_data(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    columns = ["时段序号", "时间", "电价", "小区负载", "光伏发电预测功率",
               "负载能量(kWh)", "光伏能量(kWh)"]
    if reader.fieldnames != columns:
        raise ValueError(f"CSV列名或顺序不符：{reader.fieldnames}")
    rows = list(reader)
    if len(rows) != N:
        raise ValueError(f"应有{N}个时段，实际为{len(rows)}")
    for t, row in enumerate(rows, start=1):
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"第{t}段存在缺失值或多余字段")
        allowed_times = {time_label(t)} if t < N else {"0:00+1", "00:00+1", "24:00"}
        if int(row["时段序号"]) != t or row["时间"].strip() not in allowed_times:
            raise ValueError(f"第{t}段序号或时间标签不符")

    values = np.array([[float(row[name]) for name in columns[2:]] for row in rows])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("数值列包含非有限值或负值")
    price, load_power, pv_power, stored_L, stored_R = values.T
    if (price <= 0).any():
        raise ValueError("当前问题1假设电价严格为正")

    # 功率乘时长得到电量；CSV已有电量列只核验，不重复换算。
    L, R = load_power * DELTA, pv_power * DELTA
    errors = {"load_energy_max_difference_kwh": float(np.max(np.abs(L - stored_L))),
              "pv_energy_max_difference_kwh": float(np.max(np.abs(R - stored_R)))}
    if max(errors.values()) > TOL:
        raise ValueError(f"功率与电量列不一致：{errors}")
    audit = {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
             "period_count": N, "energy_source": "power_columns_times_1_over_6",
             "price_min": float(price.min()), "price_max": float(price.max()),
             "load_total_kwh": float(L.sum()), "pv_total_kwh": float(R.sum()),
             **errors}
    return price, L, R, audit


def build_lp(price, L, R):
    """变量顺序为[G(144), C(144), D(144), W(144), S(145)]。"""
    for name, vector in (("price", price), ("L", L), ("R", R)):
        if np.shape(vector) != (N,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}必须为长度{N}的有限数值向量")
    if (price <= 0).any() or (L < 0).any() or (R < 0).any():
        raise ValueError("要求price>0且L、R非负")
    objective = np.zeros(5 * N + 1)
    objective[:N] = price
    A_eq = lil_matrix((2 * N + 1, 5 * N + 1))
    b_eq = np.zeros(2 * N + 1)
    for t in range(N):
        # 能量平衡：G - C + D - W = L - R。
        A_eq[t, [t, N + t, 2 * N + t, 3 * N + t]] = [1, -1, 1, -1]
        b_eq[t] = L[t] - R[t]
        # C、D均为母线侧电量；S[0]对应论文S_1。
        A_eq[N + t, [N + t, 2 * N + t, 4 * N + t, 4 * N + t + 1]] = [
            -ETA_C, 1 / ETA_D, -1, 1]
    A_eq[-1, 4 * N] = -1
    A_eq[-1, 5 * N] = 1  # 日循环；不额外固定S_1为6000。
    bounds = ([(0, None)] * N + [(0, MAX_ENERGY)] * (2 * N)
              + [(0, float(value)) for value in R] + [(S_MIN, S_MAX)] * (N + 1))
    return objective, A_eq.tocsr(), b_eq, bounds


def solve_model(model):
    objective, A_eq, b_eq, bounds = model
    result = linprog(objective, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                     method="highs", options=SOLVER_OPTIONS)
    if not result.success or result.status != 0:
        raise RuntimeError(f"LP未达到最优状态：{result.message}")
    return result


def unpack(x):
    return x[:N], x[N:2*N], x[2*N:3*N], x[3*N:4*N], x[4*N:]


def validate_solution(result, price, L, R, model):
    """直接从物理公式重算残差，不仅依赖求解器的可行性判断。"""
    G, C, D, W, S = unpack(result.x)
    if not np.isfinite(result.x).all():
        raise RuntimeError("求解结果包含非有限值")
    residuals = {
        "energy_balance_max_abs_kwh": float(np.max(np.abs(G + R + D - L - C - W))),
        "storage_dynamics_max_abs_kwh": float(np.max(np.abs(np.diff(S) - ETA_C*C + D/ETA_D))),
        "daily_cycle_abs_kwh": float(abs(S[-1] - S[0])),
        "bound_max_violation_kwh": float(max(0, -G.min(), -C.min(), -D.min(), -W.min(),
            C.max()-MAX_ENERGY, D.max()-MAX_ENERGY, (W-R).max(), S_MIN-S.min(), S.max()-S_MAX)),
        "daily_charge_discharge_abs_kwh": float(abs(D.sum() - ETA_C*ETA_D*C.sum())),
        "daily_energy_ledger_abs_kwh": float(abs(G.sum() - (
            L.sum()-R.sum()+W.sum()+(1-ETA_C*ETA_D)*C.sum()))),
    }
    objective, A_eq, b_eq, bounds = model
    recomputed_cost = float(price @ G)
    cost_error = abs(recomputed_cost - result.fun)
    # 对偶目标与原目标应一致；独立记录最优性证据。
    dual_value = float(b_eq @ result.eqlin.marginals)
    for i, (lower, upper) in enumerate(bounds):
        dual_value += lower * result.lower.marginals[i]
        if upper is not None:
            dual_value += upper * result.upper.marginals[i]
    dual_gap = abs(recomputed_cost - dual_value)
    stationarity = float(np.max(np.abs(objective - A_eq.T @ result.eqlin.marginals
                                       - result.lower.marginals - result.upper.marginals)))
    cost_tol = max(1e-6, 1e-9 * abs(recomputed_cost))
    passed = (max(residuals.values()) <= TOL and cost_error <= cost_tol
              and dual_gap <= cost_tol and stationarity <= TOL)
    overlap = np.flatnonzero((C > TOL) & (D > TOL))
    validation = {
        "lp_checks_passed": bool(passed), "energy_tolerance_kwh": TOL,
        "cost_tolerance_yuan": cost_tol, **residuals,
        "objective_recalculation_abs_yuan": float(cost_error),
        "dual_objective_yuan": dual_value, "primal_dual_gap_abs_yuan": float(dual_gap),
        "dual_stationarity_max_abs": stationarity,
        "mutual_exclusivity_passed": bool(len(overlap) == 0),
        "simultaneous_charge_discharge_count": int(len(overlap)),
        "simultaneous_charge_discharge_periods": [
            {"t": int(t+1), "start": time_label(t), "end": time_label(t+1),
             "C_kwh": float(C[t]), "D_kwh": float(D[t])} for t in overlap],
    }
    if not passed:
        raise RuntimeError("结果校验失败：" + json.dumps(validation, ensure_ascii=False))
    return validation


def analyze_solution(result, price, L, R):
    """只统计现有主解；无储能对照直接计算，不增加优化目标。"""
    G, C, D, W, S = unpack(result.x)
    baseline_G = np.maximum(L - R, 0)
    baseline_cost = float(price @ baseline_G)
    cost = float(price @ G)
    # 电源在母线上不可区分；以下按光伏优先供负载、富余再充电分摊。
    pv_charge = np.minimum(C, np.maximum(R - L, 0))
    grid_charge = C - pv_charge
    blocks = []
    for start in range(0, N, 24):
        end = start + 24
        section = slice(start, end)
        blocks.append({
            "时间段": f"{time_label(start)}—{time_label(end)}",
            "购电量(kWh)": float(G[section].sum()),
            "充电量(kWh)": float(C[section].sum()),
            "放电量(kWh)": float(D[section].sum()),
            "段首储电量(kWh)": float(S[start]),
            "段末储电量(kWh)": float(S[end]),
            "购电费用(元)": float(price[section] @ G[section]),
        })
    return {
        "无储能对照": {
            "说明": "同一负载、光伏和电价下，不使用储能，缺口购电、光伏富余弃置。",
            "购电费用(元)": baseline_cost,
            "购电量(kWh)": float(baseline_G.sum()),
            "弃光量(kWh)": float(np.maximum(R - L, 0).sum()),
            "主模型节省费用(元)": baseline_cost - cost,
            "主模型节费率(%)": 100 * (baseline_cost - cost) / baseline_cost if baseline_cost > 0 else None,
            "主模型减少购电量(kWh)": float(baseline_G.sum() - G.sum()),
        },
        "运行特征": {
            "充电时段数": int(np.count_nonzero(C > TOL)),
            "放电时段数": int(np.count_nonzero(D > TOL)),
            "静置时段数": int(np.count_nonzero((C <= TOL) & (D <= TOL))),
            "充电达到功率上限的时段数": int(np.count_nonzero(np.abs(C - MAX_ENERGY) <= TOL)),
            "放电达到功率上限的时段数": int(np.count_nonzero(np.abs(D - MAX_ENERGY) <= TOL)),
            "外网最大购电功率(kW)": float(G.max() / DELTA),
            "储能最大放电功率(kW)": float(D.max() / DELTA),
            "储电量达到下限的时刻": [time_label(int(t)) for t in np.flatnonzero(np.abs(S - S_MIN) <= TOL)],
            "储电量达到上限的时刻": [time_label(int(t)) for t in np.flatnonzero(np.abs(S - S_MAX) <= TOL)],
        },
        "充电来源分摊": {
            "口径": "按光伏优先直供负载、富余光伏用于充电分摊；不是新增的电能路由约束。",
            "光伏充电量(kWh)": float(pv_charge.sum()),
            "外网充电量(kWh)": float(grid_charge.sum()),
        },
        "每四小时汇总": blocks,
    }


def analyze_initial_energy(result, model):
    """在最优面上求S_1的上下端点；辅助优化不替换主调度。"""
    f, A_eq, b_eq, bounds = model
    cost = float(f @ result.x)
    face_A = vstack([A_eq, csr_matrix(f[None, :])], format="csr")
    face_b = np.append(b_eq, cost)  # 使用未舍入费用，不能用展示值35118.60。
    endpoints, solutions = [], []
    for direction, name in ((1, "最小初始电量"), (-1, "最大初始电量")):
        objective = np.zeros_like(f)
        objective[4*N] = direction
        endpoint = solve_model((objective, face_A, face_b, bounds))
        G, C, D, W, S = unpack(endpoint.x)
        residual = float(np.max(np.abs(A_eq @ endpoint.x - b_eq)))
        bound_error = max(0.0, max(lower - endpoint.x[i] for i, (lower, _) in enumerate(bounds)),
                          max(endpoint.x[i] - upper for i, (_, upper) in enumerate(bounds) if upper is not None))
        cost_error = float(abs(f @ endpoint.x - cost))
        dual = float(face_b @ endpoint.eqlin.marginals)
        for i, (lower, upper) in enumerate(bounds):
            dual += lower * endpoint.lower.marginals[i]
            if upper is not None:
                dual += upper * endpoint.upper.marginals[i]
        dual_gap = float(abs(objective @ endpoint.x - dual))
        stationarity = float(np.max(np.abs(objective - face_A.T @ endpoint.eqlin.marginals
                                          - endpoint.lower.marginals - endpoint.upper.marginals)))
        sign_error = max(0.0, -float(endpoint.lower.marginals.min()), float(endpoint.upper.marginals.max()))
        if max(residual, bound_error, cost_error, dual_gap, stationarity, sign_error) > TOL:
            raise RuntimeError(f"{name}的最优面或对偶校验未通过")
        overlap = int(np.count_nonzero((C > TOL) & (D > TOL)))
        endpoints.append({"端点": name, "初始储电量(kWh)": float(S[0]),
                          "购电费用(元)": float(f @ endpoint.x), "费用固定误差(元)": cost_error,
                          "物理等式最大误差(kWh)": residual, "最大越界量(kWh)": float(bound_error),
                          "辅助优化原始与对偶目标差(kWh)": dual_gap,
                          "辅助优化驻点条件最大误差": stationarity,
                          "辅助优化对偶符号最大误差": sign_error,
                          "同时充放电时段数": overlap})
        solutions.append(endpoint.x)
    low, high = (endpoint["初始储电量(kWh)"] for endpoint in endpoints)
    if high < low - TOL:
        raise RuntimeError("初始电量区间上下端点顺序异常")
    _, c_low, d_low, _, _ = unpack(solutions[0])
    _, c_high, d_high, _, _ = unpack(solutions[1])
    conflicting = ((c_low > TOL) & (d_high > TOL)) | ((d_low > TOL) & (c_high > TOL))
    whole_interval_physical = not conflicting.any() and all(e["同时充放电时段数"] == 0 for e in endpoints)
    return {
        "分析方法": "保留原约束并固定未舍入最优费用，分别最小化和最大化S_1。",
        "固定最优费用(元)": cost, "下端点(kWh)": low, "上端点(kWh)": high,
        "区间宽度(kWh)": max(0.0, high-low), "数值核验容差": TOL,
        "是否存在非退化最优区间": "是" if high-low > TOL else "否（数值精度范围内）",
        "端点校验": endpoints,
        "整个区间的互斥可行性": ("通过：两端点无相反运行模式，凸组合仍满足互斥。"
                                 if whole_interval_physical else "尚未证明，不能仅凭LP凸性认定互斥。"),
        "填表所用初始储电量(kWh)": float(result.x[4*N]),
        "填表策略": "沿用原主解，不用辅助优化的端点策略替换。",
    }


def export_results(directory, result, price, L, R, audit, validation, initial_energy):
    G, C, D, W, S = unpack(result.x)
    directory.mkdir(exist_ok=True)
    # 保存未舍入的求解结果，避免展示精度破坏能量账目。
    with (directory / "dispatch.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["时段序号", "开始时间", "结束时间", "电价P(元/kWh)", "负载电量L(kWh)", "光伏电量R(kWh)",
                         "购电量G(kWh)", "充电量C(kWh)", "放电量D(kWh)", "弃光量W(kWh)",
                         "段首储电量S_t(kWh)", "段末储电量S_t+1(kWh)", "购电费用(元)"])
        for t in range(N):
            writer.writerow([t+1, time_label(t), time_label(t+1), price[t], L[t], R[t],
                             G[t], C[t], D[t], W[t], S[t], S[t+1], price[t]*G[t]])
    audit_labels = {
        "filename": "数据文件", "sha256": "文件SHA256校验值", "period_count": "时段数量",
        "price_min": "最低电价(元/kWh)", "price_max": "最高电价(元/kWh)",
        "load_total_kwh": "全天负载电量(kWh)", "pv_total_kwh": "全天光伏电量(kWh)",
        "load_energy_max_difference_kwh": "负载电量列最大核验误差(kWh)",
        "pv_energy_max_difference_kwh": "光伏电量列最大核验误差(kWh)",
    }
    validation_labels = {
        "energy_tolerance_kwh": "电量校验容差(kWh)", "cost_tolerance_yuan": "费用校验容差(元)",
        "energy_balance_max_abs_kwh": "逐时段能量平衡最大误差(kWh)",
        "storage_dynamics_max_abs_kwh": "储电递推最大误差(kWh)",
        "daily_cycle_abs_kwh": "日循环误差(kWh)",
        "bound_max_violation_kwh": "变量边界最大越界量(kWh)",
        "daily_charge_discharge_abs_kwh": "全天充放电效率关系误差(kWh)",
        "daily_energy_ledger_abs_kwh": "全天能量账目误差(kWh)",
        "objective_recalculation_abs_yuan": "购电费用重算误差(元)",
        "dual_objective_yuan": "对偶目标值(元)",
        "primal_dual_gap_abs_yuan": "原始与对偶目标差(元)",
        "dual_stationarity_max_abs": "对偶驻点条件最大误差",
        "simultaneous_charge_discharge_count": "同时充放电时段数",
    }
    chinese_validation = {
        "LP数值校验": "通过" if validation["lp_checks_passed"] else "未通过",
        "充放电互斥检查": "通过" if validation["mutual_exclusivity_passed"] else "未通过",
        **{label: validation[key] for key, label in validation_labels.items()},
        "同时充放电明细": [
            {"时段序号": row["t"], "开始时间": row["start"], "结束时间": row["end"],
             "充电量(kWh)": row["C_kwh"], "放电量(kWh)": row["D_kwh"]}
            for row in validation["simultaneous_charge_discharge_periods"]],
    }
    summary = {
        "当前阶段": "主线性规划及初始储电量最优区间分析",
        "求解结果": {
            "最优购电费用(元)": float(price @ G), "全天购电量(kWh)": float(G.sum()),
            "全天充电量_母线侧(kWh)": float(C.sum()), "全天放电量_母线侧(kWh)": float(D.sum()),
            "全天弃光量(kWh)": float(W.sum()), "起始储电量S_1(kWh)": float(S[0]),
            "日末储电量S_145(kWh)": float(S[-1]), "实际最低储电量(kWh)": float(S.min()),
            "实际最高储电量(kWh)": float(S.max()),
            "全天储能损耗(kWh)": float((1-ETA_C)*C.sum()+(1/ETA_D-1)*D.sum()),
        },
        "结果校验": chinese_validation,
        "结果分析": analyze_solution(result, price, L, R),
        "初始储电量最优区间": initial_energy,
        "结果表口径": {"电量单位": "kWh", "费用单位": "元",
                       "数值精度": "调度CSV和汇总保留计算精度；正式Excel与论文表建议显示两位小数，汇总按未舍入值计算",
                       "时间标签修正": "官方模板首段为0:10-0:20、末段为次日0:00-0:10；输出副本修正为00:00-00:10至23:50-24:00。"},
        "输入数据核验": {**{label: audit[key] for key, label in audit_labels.items()},
                         "电量计算口径": "功率列乘以1/6小时，已有电量列仅作交叉核验"},
        "模型参数": {
            "时段长度(小时)": DELTA, "充电效率": ETA_C, "放电效率": ETA_D,
            "储电量下限(kWh)": S_MIN, "储电量上限(kWh)": S_MAX,
            "单时段充放电上限_母线侧(kWh)": MAX_ENERGY, "初始储电量": "由优化决定",
            "日循环约束": "启用", "二进制变量": "未使用", "次级目标": "未使用",
        },
        "求解器信息": {
            "求解方法": "SciPy线性规划接口 / HiGHS", "状态码": int(result.status),
            "求解状态": "已获得最优解", "迭代次数": int(result.nit),
            "原始可行性容差": SOLVER_OPTIONS["primal_feasibility_tolerance"],
            "对偶可行性容差": SOLVER_OPTIONS["dual_feasibility_tolerance"],
            "变量数量": len(result.x), "等式约束数量": 2*N+1,
            "Python版本": platform.python_version(), "NumPy版本": np.__version__, "SciPy版本": scipy.__version__,
        },
        "结果说明": ("主解通过互斥检查，无需互斥修复；初始电量的唯一性见最优区间分析，结果表仍沿用原主解。"
                     if validation["mutual_exclusivity_passed"] else
                     "LP通过数值校验，但存在同时充放电；需进入处理阶段，尚不能作为最终物理调度。"),
    }
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    base = Path(__file__).resolve().parent
    price, L, R, audit = load_data(base / "附件1.csv")
    model = build_lp(price, L, R)
    result = solve_model(model)
    validation = validate_solution(result, price, L, R, model)
    initial_energy = analyze_initial_energy(result, model)
    summary = export_results(base / "outputs", result, price, L, R, audit, validation, initial_energy)
    if not validation["mutual_exclusivity_passed"]:
        raise RuntimeError("存在同时充放电，已保存调度审查结果；尚不能作为最终物理调度")
    print(json.dumps({key: summary[key] for key in ("求解结果", "初始储电量最优区间", "结果说明")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
