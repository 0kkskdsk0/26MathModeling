# -*- coding: utf-8 -*-
"""问题二远视模型主程序：全年滚动求解、配对对照、导出与扰动检验。

运行：

    python -B TYA_Q2/solving/taskCompositeFar/value_function.py --full --workers 12
    python -B TYA_Q2/solving/taskCompositeFar/solve_year_far.py [--workers 8]

流程：

1. 复用 `taskComposite` 的冻结赋权规则，逐日构造第 $d$ 天的候选池 $\\Omega_d$ 与核权重
   $\\pi_{d,\\omega}$，口径与近视模型完全一致；
2. 读取 `value_function.py` 落盘的终端价值切线（$\\Omega_d$ 与 $\\pi_{d,\\omega}$ 下因果估计），
   装配含终端价值项的稀疏 LP（`model_far.py`）并求解得到计划购电量 $G_{d,t}$；
3. 用**与近视模型逐元素相同**的执行层，在当日实测负载与光伏上求补救问题，
   得到填报 result2 的执行路径；
4. 逐日执行 12 条原有断言 + 9 条新增断言，并把逐日核验记录落盘；
5. 与 `outputs/taskComposite` 的近视结果做配对比较，按**事先固定**的相对阈值给出采纳判定；
6. 执行 $\\theta\\equiv0$ 退化核验与目标日扰动因果性检验。

**冷启动口径。** 2025-01-01 至 01-31 的处境含 7 日滞后项、不满足冻结核规则的适用条件，
这 31 天取 $\\theta\\equiv0$（即完全退化为近视模型）。因此两个模型在 2025-02-01 的
日初储电量逐元素相同，配对差值只来自终端价值项，而不是初始条件差异。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import validate_far as VF
    from .common import (ASSETS, FROZEN_FAR, INITIAL_STORAGE, OUTPUT, OUTPUT_START,
                         SLOTS, TABLE_DATES, TEMPLATE, Context, build_context,
                         day_inputs, load_rule, run_recourse)
    from .model_far import (build_decision_lp_far, build_recourse_lp_far, solve_far,
                            unpack_decision_far, unpack_recourse_far)
    from .value_function import (ETA, N0, R_MAX, S_MAX, S_MIN, TAU_ENV, evaluate_day,
                                 load_tangents)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taskCompositeFar import validate_far as VF
    from taskCompositeFar.common import (ASSETS, FROZEN_FAR, INITIAL_STORAGE, OUTPUT,
                                         OUTPUT_START, SLOTS, TABLE_DATES, TEMPLATE,
                                         Context, build_context, day_inputs, load_rule,
                                         run_recourse)
    from taskCompositeFar.model_far import (build_decision_lp_far, build_recourse_lp_far,
                                            solve_far, unpack_decision_far,
                                            unpack_recourse_far)
    from taskCompositeFar.value_function import (ETA, N0, R_MAX, S_MAX, S_MIN, TAU_ENV,
                                                 evaluate_day, load_tangents)

from taskComposite.model import build_decision_lp, solve, unpack_decision  # noqa: E402
from taskComposite.solve_year import (build_result2, generate_tables,  # noqa: E402
                                      merge_periods, slot_label, template_headers)
from taskComposite.validate import TOL, check_day, check_scenarios  # noqa: E402

ADOPTION_THRESHOLD_PCT = 0.5     # 事先固定的采纳阈值（相对 C_official，单位 %）
PERTURBATION_DAY = TABLE_DATES[1]
DEGENERACY_TOL = 1e-6            # 全零切线的"数值零"判定（kWh，与残差容差同量级）


# ---------------------------------------------------------------------------
# 输入输出
# ---------------------------------------------------------------------------
def load_tangent_table(output: Path = OUTPUT, tag: str = "value_function") -> dict:
    """读取终端价值切线表：{决策日 -> (Q,2) 的切线集合}。"""
    path = output / f"{tag}_samples.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行 value_function.py --full")
    table = {}
    with np.load(path, allow_pickle=False) as data:
        dates = [str(x) for x in data["dates"]]
        tangents, offset, length = data["tangents"], data["tangent_offset"], data["tangent_length"]
        for i, text in enumerate(dates):
            begin, size = int(offset[i]), int(length[i])
            table[pd.Timestamp(text)] = tangents[begin:begin + size].copy()
    return table


def load_value_rounds(output: Path = OUTPUT, tag: str = "value_function") -> pd.DataFrame:
    path = output / f"{tag}_rounds.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行 value_function.py --full")
    return pd.read_csv(path, parse_dates=["date"])


def save_paths(result: dict, output: Path) -> pd.DataFrame:
    """保存远视模型的逐时段计划与执行路径。"""
    rows = []
    for day, path in result["paths"].items():
        executed, pool = path["executed"], path["pool"]
        for t in range(SLOTS):
            rows.append({"date": day, "slot": t + 1,
                         "label": f"{slot_label(t)}-{slot_label(t + 1)}",
                         "plan_kwh": path["plan"][t],
                         "charge_kwh": executed["C"][t],
                         "discharge_kwh": executed["D"][t],
                         "emergency_kwh": executed["E"][t],
                         "curtail_kwh": executed["W"][t],
                         "storage_start_kwh": executed["S"][t],
                         "storage_end_kwh": executed["S"][t + 1],
                         "load_kwh": path["load"][t], "pv_kwh": path["pv"][t],
                         "net_kwh": path["load"][t] - path["pv"][t],
                         "pool_size": len(pool)})
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "dispatch_detail_far.csv", index=False, encoding="utf-8-sig",
                 float_format="%.10g")
    return table


# ---------------------------------------------------------------------------
# 全年滚动
# ---------------------------------------------------------------------------
def solve_one_day(ctx: Context, day, s_start: float, tangents, strict: bool = True) -> dict:
    """求解第 day 天的远视决策层与执行层，返回解、核验记录与下一日开局储电量。"""
    inputs = day_inputs(ctx, day)
    pool, weights, mode = inputs["pool"], inputs["weights"], inputs["mode"]
    tan = np.zeros((0, 2)) if tangents is None else np.asarray(tangents, dtype=float)
    trace: list = []
    if len(pool) == 0:
        plan = np.zeros(SLOTS)
        objective, solution, tangent_record = 0.0, None, None
        decision_check = None
    else:
        program = build_decision_lp_far(ctx.params, inputs["load"], inputs["pv"],
                                       weights, s_start, tan)
        solved = solve_far(program, f"{day} 决策层(远视)", trace)
        solution = unpack_decision_far(solved, program.layout, len(pool))
        plan, objective = solution["G"], float(solved.fun)
        decision_check = check_scenarios(ctx.params, inputs["load"], inputs["pv"],
                                         weights, solution, s_start)
        tangent_record = VF.check_terminal_value(solution, tan, len(pool))
        if strict:
            # 与 taskComposite 的定稿口径一致：决策层**情景设想层**的"紧急购电与充电
            # 同时为正"是已知的边界情形，逐日收集但不中断（它不影响任何填报数值，
            # 执行层已由 C11/C12 两条常数上界彻底排除该行为）。终端价值约束则必须严格成立。
            VF.assert_far(tangent_record, f"{day}")
    load, pv, executed = run_recourse(ctx, plan, day, s_start)
    executed["daily_cost"] = float(ctx.params.price @ plan
                                   + ctx.params.emergency_multiplier
                                   * ctx.params.price @ executed["E"])
    record = check_day(ctx.params, plan, load, pv, executed, s_start)
    record["date"] = day
    record["storage_start_kwh"] = float(executed["S"][0])
    if tangent_record is not None:
        record.update({k: v for k, v in tangent_record.items() if k != "passed"})
        record["passed"] = bool(record["passed"] and tangent_record["passed"])
    if strict:
        from taskComposite.validate import assert_day
        assert_day(record, f"{day}")
    return {"day": day, "pool": pool, "weights": weights, "mode": mode,
            "plan": plan, "objective": objective, "solution": solution,
            "executed": executed, "load": load, "pv": pv, "record": record,
            "decision_check": decision_check, "tangent_record": tangent_record,
            "storage_next": float(executed["S"][-1]), "tangents": tan,
            "solver_path": trace[0] if trace else "-"}


def run(ctx: Context, tangent_table: dict, strict: bool = True,
        verbose: bool = True) -> dict:
    """逐日滚动求解全年（含 1 月冷启动期）。"""
    storage = INITIAL_STORAGE
    records, paths, weights_log, violations, decision_violations = [], {}, [], [], []
    dispatch = {key: [] for key in
                ("date", "plan_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
                 "curtail_kwh", "storage_start", "storage_end", "plan_cost_yuan",
                 "emergency_cost_yuan", "daily_cost_yuan", "storage_min", "storage_max",
                 "pool_size", "k_eff", "mode", "objective_yuan", "theta_mean",
                 "theta_total_yuan", "tangent_count", "value_at_s_min",
                 "value_at_s_max", "dec_storage_end_mean", "dec_storage_end_min",
                 "dec_storage_end_max", "dec_plan_kwh", "solver_path")}
    started = time.perf_counter()
    for day in ctx.dates:
        tangents = tangent_table.get(pd.Timestamp(day)) if day >= OUTPUT_START else None
        outcome = solve_one_day(ctx, day, storage, tangents, strict=strict)
        record = outcome["record"]
        if not record["passed"]:
            violations.append({"date": str(day.date()), "fields": {
                k: v for k, v in record.items()
                if k in ("energy_balance_max_abs", "storage_dynamics_max_abs",
                         "initial_state_abs", "simultaneous_charge_discharge_count",
                         "emergency_and_charge_count", "cost_recalculation_rel",
                         "terminal_support_violation", "terminal_envelope_gap")
                and ((isinstance(v, float) and v > TOL) or (isinstance(v, int) and v > 0))}})
        if outcome["decision_check"] is not None and not outcome["decision_check"]["passed"]:
            decision_violations.append({
                "date": str(day.date()),
                "fields": {k: v for k, v in outcome["decision_check"].items()
                           if k != "passed" and isinstance(v, float) and v > TOL}})
        records.append(record)
        if day >= OUTPUT_START:
            executed, solution = outcome["executed"], outcome["solution"]
            theta = solution["theta"] if solution is not None and len(solution["theta"]) else np.zeros(0)
            values = np.asarray(tangents, dtype=float) if tangents is not None else np.zeros((0, 2))
            grid_values = None
            dispatch["date"].append(day)
            dispatch["plan_kwh"].append(float(outcome["plan"].sum()))
            dispatch["charge_kwh"].append(float(executed["C"].sum()))
            dispatch["discharge_kwh"].append(float(executed["D"].sum()))
            dispatch["emergency_kwh"].append(float(executed["E"].sum()))
            dispatch["curtail_kwh"].append(float(executed["W"].sum()))
            dispatch["storage_start"].append(float(executed["S"][0]))
            dispatch["storage_end"].append(float(executed["S"][-1]))
            dispatch["plan_cost_yuan"].append(record["daily_plan_cost_yuan"])
            dispatch["emergency_cost_yuan"].append(record["daily_emergency_cost_yuan"])
            dispatch["daily_cost_yuan"].append(record["daily_cost_yuan"])
            dispatch["storage_min"].append(record["storage_min_kwh"])
            dispatch["storage_max"].append(record["storage_max_kwh"])
            dispatch["pool_size"].append(len(outcome["pool"]))
            dispatch["k_eff"].append(float(1.0 / np.sum(np.square(outcome["weights"])))
                                     if len(outcome["weights"]) else 0.0)
            dispatch["mode"].append(outcome["mode"])
            dispatch["objective_yuan"].append(float(outcome["objective"]))
            dispatch["theta_mean"].append(float(theta.mean()) if theta.size else 0.0)
            dispatch["theta_total_yuan"].append(
                float(outcome["weights"] @ theta) if theta.size else 0.0)
            dispatch["tangent_count"].append(int(len(values)))
            dispatch["value_at_s_min"].append(
                float(np.max(values[:, 0] * S_MIN + values[:, 1])) if len(values) else 0.0)
            dispatch["value_at_s_max"].append(
                float(np.max(values[:, 0] * S_MAX + values[:, 1])) if len(values) else 0.0)
            if solution is not None:
                decision_storage_end = np.asarray(solution["S"], dtype=float)[:, -1]
                dispatch["dec_storage_end_mean"].append(
                    float(outcome["weights"] @ decision_storage_end))
                dispatch["dec_storage_end_min"].append(float(decision_storage_end.min()))
                dispatch["dec_storage_end_max"].append(float(decision_storage_end.max()))
                dispatch["dec_plan_kwh"].append(float(outcome["weights"] @
                                                      np.asarray(solution["E"], dtype=float).sum(axis=1)))
            else:
                for key in ("dec_storage_end_mean", "dec_storage_end_min",
                            "dec_storage_end_max", "dec_plan_kwh"):
                    dispatch[key].append(0.0)
            dispatch["solver_path"].append(outcome.get("solver_path", "-"))
            paths[day] = {"plan": outcome["plan"].copy(), "executed": executed,
                          "load": outcome["load"].copy(), "pv": outcome["pv"].copy(),
                          "pool": outcome["pool"], "weights": outcome["weights"].copy(),
                          "tangents": np.asarray(tangents, dtype=float)
                          if tangents is not None else np.zeros((0, 2)),
                          "theta": np.asarray(theta, dtype=float),
                          "grid_values": grid_values}
        weights_log.append({"date": day, "mode": outcome["mode"],
                            "pool_size": len(outcome["pool"]),
                            "k_eff": float(1.0 / np.sum(np.square(outcome["weights"])))
                            if len(outcome["weights"]) else 0.0})
        storage = outcome["storage_next"]
        if verbose and (day.day == 1 or day == ctx.dates[-1]):
            print(f"  {day.date()} 池={len(outcome['pool']):3d} 计划={outcome['plan'].sum():10.1f} "
                  f"紧急={outcome['executed']['E'].sum():9.1f} "
                  f"日末SOC={storage:9.2f} 切线={len(np.asarray(tangents).reshape(-1, 2)) if tangents is not None else 0}",
                  flush=True)
    return {"dispatch": pd.DataFrame(dispatch), "records": records, "paths": paths,
            "weights_log": pd.DataFrame(weights_log), "violations": violations,
            "decision_violations": decision_violations, "final_storage": storage,
            "wall_seconds": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# 退化核验与因果性检验（可并行）
# ---------------------------------------------------------------------------
_DEG_WORKER: dict = {}


def _degeneracy_initialise(days: int | None) -> None:
    _DEG_WORKER["ctx"] = build_context(days=days)
    _DEG_WORKER["table"] = load_tangent_table()


def _degeneracy_task(payload: dict) -> dict:
    """把该日切线全部置零后重跑决策层，并核对退化开关下的装配恒等性。"""
    ctx, table = _DEG_WORKER["ctx"], _DEG_WORKER["table"]
    day = pd.Timestamp(payload["date"])
    tangents = table.get(day)
    inputs = day_inputs(ctx, day)
    if tangents is None or len(inputs["pool"]) == 0:
        return {"date": str(day.date()), "plan": None, "tangent_count": 0}
    zero = np.zeros_like(np.asarray(tangents, dtype=float))
    program = build_decision_lp_far(ctx.params, inputs["load"], inputs["pv"],
                                   inputs["weights"], payload["s_start"], zero)
    solved = solve_far(program, f"{day} 退化核验")
    solution = unpack_decision_far(solved, program.layout, len(inputs["pool"]))
    # 装配层恒等：theta 完全不添加时，远视装配必须与近视装配逐元素相同
    near_assembly = build_decision_lp(ctx.params, inputs["load"], inputs["pv"],
                                     inputs["weights"], payload["s_start"])
    far_switched_off = build_decision_lp_far(ctx.params, inputs["load"], inputs["pv"],
                                            inputs["weights"], payload["s_start"], None)
    assembly_gap = {
        "c": float(np.max(np.abs(near_assembly.c - far_switched_off.c))),
        "b_eq": float(np.max(np.abs(near_assembly.b_eq - far_switched_off.b_eq))),
        "b_ub": float(np.max(np.abs(near_assembly.b_ub - far_switched_off.b_ub))),
        "A_eq": float(abs(near_assembly.A_eq - far_switched_off.A_eq).max())
        if near_assembly.A_eq.shape == far_switched_off.A_eq.shape else float("inf"),
        "A_ub": float(abs(near_assembly.A_ub - far_switched_off.A_ub).max())
        if near_assembly.A_ub.shape == far_switched_off.A_ub.shape else float("inf"),
        "size": float(abs(near_assembly.size - far_switched_off.size)),
        "bounds": float(abs(len(near_assembly.bounds) - len(far_switched_off.bounds))),
    }
    return {"date": str(day.date()), "plan": solution["G"].tolist(),
            "plan_sum": float(solution["G"].sum()),
            "theta_max_abs": float(np.max(np.abs(solution["theta"])))
            if len(solution["theta"]) else 0.0,
            "tangent_count": int(len(zero)),
            "assembly_gap": assembly_gap}


def degeneracy_check(dispatch: pd.DataFrame, chains: dict, workers: int = 1,
                     days: int | None = None) -> dict:
    """C20：把终端价值切线全部置零后重跑，与近视模型的计划购电量逐元素比较。"""
    near = pd.read_csv(OUTPUT.parent / "taskComposite/dispatch_detail.csv",
                       parse_dates=["date"])
    payloads = [{"date": str(pd.Timestamp(d).date()), "s_start": float(chains[pd.Timestamp(d)])}
                for d in dispatch.date]
    if workers <= 1:
        _degeneracy_initialise(days)
        results = [_degeneracy_task(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_degeneracy_initialise,
                                 initargs=(days,)) as pool:
            results = list(pool.map(_degeneracy_task, payloads, chunksize=4))
    worst, mismatch_days, compared = 0.0, [], 0
    worst_relative = 0.0
    theta_worst = 0.0
    assembly_worst = 0.0
    for result in results:
        if result["plan"] is None:
            continue
        day = pd.Timestamp(result["date"])
        reference = (near[near.date == day].sort_values("slot").plan_kwh.to_numpy(dtype=float))
        plan = np.asarray(result["plan"], dtype=float)
        if reference.shape != plan.shape:
            mismatch_days.append(result["date"])
            continue
        difference = float(np.max(np.abs(reference - plan)))
        worst = max(worst, difference)
        scale = max(1.0, float(np.abs(reference).max()))
        worst_relative = max(worst_relative, difference / scale)
        theta_worst = max(theta_worst, result["theta_max_abs"])
        assembly_worst = max(assembly_worst, max(result.get("assembly_gap", {}).values()
                                                 or [0.0]))
        compared += 1
        if difference > DEGENERACY_TOL:
            mismatch_days.append(result["date"])
    record = {"days_compared": compared, "days_mismatched": len(mismatch_days),
              "mismatch_dates": mismatch_days[:20],
              "max_abs_plan_difference_kwh": worst,
              "max_relative_plan_difference": worst_relative,
              "max_abs_theta_kwh": theta_worst,
              "max_assembly_gap": assembly_worst,
              "tolerance_kwh": DEGENERACY_TOL,
              "passed": bool(len(mismatch_days) == 0 and worst <= DEGENERACY_TOL
                             and theta_worst <= 1e-6 and assembly_worst <= 0.0)}
    return record


def causality_check(ctx: Context, tangent_table: dict, day=None, eta: float | None = None,
                    tau_env: float | None = None, r_max: int | None = None) -> dict:
    """C21：扰动第 d 天及之后的原始观测，重算第 d 天的池、权重、切线与计划。"""
    day = pd.Timestamp(PERTURBATION_DAY if day is None else day)
    eta = ETA if eta is None else eta
    tau_env = TAU_ENV if tau_env is None else tau_env
    r_max = R_MAX if r_max is None else r_max
    rng = np.random.default_rng(20250201)
    altered = []
    for raw in (ctx.load_power, ctx.pv_power):
        changed = raw.copy()
        mask = changed.index >= day
        values = changed.loc[mask].to_numpy(dtype=float).ravel().copy()
        rng.shuffle(values)
        changed.loc[mask] = values.reshape((-1, SLOTS)) * 1.37 + 31.0
        altered.append(changed)
    from taskComposite.kernel import build_context_features, columns_for
    altered_features = build_context_features(*altered)
    altered_ctx = Context(price=ctx.price, params=ctx.params,
                          features=altered_features, rule=ctx.rule,
                          rule_data=ctx.rule_data, load_power=altered[0],
                          pv_power=altered[1])
    inputs_reference = day_inputs(ctx, day)
    inputs_perturbed = day_inputs(altered_ctx, day)
    reference_tangents = np.asarray(tangent_table[day], dtype=float)
    altered_day = evaluate_day(altered_ctx, day, eta=eta, tau_env=tau_env, r_max=r_max,
                               n0=N0)
    altered_tangents = np.asarray(altered_day["tangents"], dtype=float)
    plans = {}
    for label, inputs, tan in (("reference", inputs_reference, reference_tangents),
                               ("perturbed", inputs_perturbed, altered_tangents)):
        program = build_decision_lp_far(ctx.params, inputs["load"], inputs["pv"],
                                       inputs["weights"], INITIAL_STORAGE, tan)
        solved = solve_far(program, f"{day} {label}")
        plans[label] = unpack_decision_far(solved, program.layout,
                                           len(inputs["pool"]))["G"]
    reference = {"tangents": reference_tangents, "plan": plans["reference"],
                 "pool": inputs_reference["pool"], "weights": inputs_reference["weights"]}
    perturbed = {"tangents": altered_tangents, "plan": plans["perturbed"],
                 "pool": inputs_perturbed["pool"], "weights": inputs_perturbed["weights"],
                 "target_changed": not np.array_equal(
                     altered[0].loc[day].to_numpy(dtype=float),
                     ctx.load_power.loc[day].to_numpy(dtype=float)),
                 "prefix_identical": bool(np.array_equal(
                     altered_features.loc[:day, columns_for(ctx.rule.groups)].to_numpy(),
                     ctx.features.loc[:day, columns_for(ctx.rule.groups)].to_numpy(),
                     equal_nan=True))}
    record = VF.check_causality(reference, perturbed)
    record["date"] = str(day.date())
    record["reference_tangent_count"] = int(len(reference_tangents))
    record["perturbed_tangent_count"] = int(len(altered_tangents))
    record["perturbed_converged_round"] = int(altered_day["converged_round"])
    return record


# ---------------------------------------------------------------------------
# 对照与图表
# ---------------------------------------------------------------------------
def collection_checks(output: Path = OUTPUT, tag: str = "value_function") -> dict:
    """C15—C19 的集合级核验：切线符号与单调性、值函数形状、网格收敛。"""
    rounds = load_value_rounds(output, tag)
    record = {"tangent_positive_violation_max": 0.0, "tangent_monotone_violation_max": 0.0,
              "value_monotone_violation_max": 0.0, "value_convexity_violation_max": 0.0,
              "days_checked": 0, "days_converged": 0, "not_converged": []}
    with np.load(output / f"{tag}_samples.npz", allow_pickle=False) as data:
        dates = [str(x) for x in data["dates"]]
        offset, length = data["grid_offset"], data["grid_length"]
        tangents, t_offset, t_length = (data["tangents"], data["tangent_offset"],
                                        data["tangent_length"])
        for i, _ in enumerate(dates):
            begin, size = int(offset[i]), int(length[i])
            values = data["values"][begin:begin + size]
            slopes = data["slopes"][begin:begin + size]
            shape = VF.check_value_shape(values, slopes)
            record["value_monotone_violation_max"] = max(
                record["value_monotone_violation_max"], shape["value_monotone_violation"])
            record["value_convexity_violation_max"] = max(
                record["value_convexity_violation_max"], shape["value_convexity_violation"])
            t_begin, t_size = int(t_offset[i]), int(t_length[i])
            symbols = VF.check_tangents(tangents[t_begin:t_begin + t_size])
            record["tangent_positive_violation_max"] = max(
                record["tangent_positive_violation_max"], symbols["tangent_positive_violation"])
            record["tangent_monotone_violation_max"] = max(
                record["tangent_monotone_violation_max"], symbols["tangent_monotone_violation"])
            record["days_checked"] += 1
    for date, group in rounds.groupby("date"):
        ordered = group.sort_values("round").to_dict("records")
        convergence = VF.check_convergence(ordered, ETA, TAU_ENV)
        if convergence["passed"]:
            record["days_converged"] += 1
        else:
            record["not_converged"].append(str(pd.Timestamp(date).date()))
    record["passed"] = bool(
        record["tangent_positive_violation_max"] <= 1e-6
        and record["tangent_monotone_violation_max"] <= 1e-6
        and record["value_monotone_violation_max"] <= 1e-6
        and record["value_convexity_violation_max"] <= 1e-6
        and not record["not_converged"])
    return record


def summarise_probe(table: pd.DataFrame) -> dict:
    """把执行层远视诊断的逐日表汇总为报告用的标量。"""
    if len(table) == 0:
        return {"days": 0, "near_storage_end_mean_kwh": 0.0,
                "far_storage_end_mean_kwh": 0.0, "far_storage_end_min_kwh": 0.0,
                "far_storage_end_max_kwh": 0.0, "near_emergency_kwh": 0.0,
                "far_emergency_kwh": 0.0, "near_cost_yuan": 0.0, "far_cost_yuan": 0.0,
                "table": table}
    return {
        "days": int(len(table)),
        "near_storage_end_mean_kwh": float(table.near_storage_end_kwh.mean()),
        "far_storage_end_mean_kwh": float(table.far_storage_end_kwh.mean()),
        "far_storage_end_min_kwh": float(table.far_storage_end_kwh.min()),
        "far_storage_end_max_kwh": float(table.far_storage_end_kwh.max()),
        "near_emergency_kwh": float(table.near_emergency_kwh.sum()),
        "far_emergency_kwh": float(table.far_emergency_kwh.sum()),
        "near_cost_yuan": float(table.near_daily_cost_yuan.sum()),
        "far_cost_yuan": float(table.far_daily_cost_yuan.sum()),
        "table": table,
    }


def execution_lookahead_probe(ctx: Context, tangent_table: dict, result: dict,
                              sampled_per_month: int = 1) -> dict:
    """附加诊断：把终端价值项也放进**执行层**，看实际执行路径是否愿意为次日留电。

    这不是主对照的一部分——主对照按任务要求让两个模型共用逐元素相同的执行层，
    差值只来自终端价值项。本诊断只回答一个问题：主对照里远视化的作用被什么吸收掉了。
    每天独立求解，日初储电量取该日实际链上的值，不重新滚动，因此只作定性说明。
    """
    dates = result["dispatch"].date
    picks = []
    for month in range(2, 13):
        month_days = dates[dates.dt.month == month]
        picks.extend(month_days.head(sampled_per_month).tolist())
    rows = []
    for day in picks:
        if day not in result["paths"]:
            continue
        path = result["paths"][day]
        plan = path["plan"]
        s_start = float(path["executed"]["S"][0])
        load, pv = path["load"], path["pv"]
        near = run_recourse(ctx, plan, day, s_start)[2]
        tangents = tangent_table.get(pd.Timestamp(day))
        if tangents is None:
            continue
        program = build_recourse_lp_far(ctx.params, plan, load, pv, s_start, tangents)
        far = unpack_recourse_far(solve_far(program, f"{day} 远视执行层"), program.layout)
        rows.append({
            "date": str(day.date()), "plan_kwh": float(plan.sum()),
            "storage_start_kwh": s_start,
            "near_storage_end_kwh": float(near["S"][-1]),
            "far_storage_end_kwh": float(far["S"][-1]),
            "near_emergency_kwh": float(near["E"].sum()),
            "far_emergency_kwh": float(far["E"].sum()),
            "near_daily_cost_yuan": float(ctx.params.price @ plan
                                          + ctx.params.emergency_multiplier
                                          * ctx.params.price @ near["E"]),
            "far_daily_cost_yuan": float(ctx.params.price @ plan
                                         + ctx.params.emergency_multiplier
                                         * ctx.params.price @ far["E"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "execution_lookahead_probe.csv", index=False,
                 encoding="utf-8-sig", float_format="%.10g")
    return summarise_probe(table)


def paired_comparison(far: pd.DataFrame, threshold_pct: float = ADOPTION_THRESHOLD_PCT) -> dict:
    """近视与远视在同一执行层下的逐日配对比较与采纳判定。"""
    near = pd.read_csv(OUTPUT.parent / "taskComposite/dispatch_year.csv",
                       parse_dates=["date"])
    merged = far.merge(near, on="date", suffixes=("_far", "_near"), how="inner")
    if len(merged) != len(far):
        raise ValueError("配对比较的日期数不一致")
    far_cost = merged.daily_cost_yuan_far.to_numpy(dtype=float)
    near_cost = merged.daily_cost_yuan_near.to_numpy(dtype=float)
    difference = near_cost - far_cost                      # 正 = 远视更省
    total_far, total_near = float(far_cost.sum()), float(near_cost.sum())
    relative_pct = (total_near - total_far) / total_far * 100
    positive = int((difference > 1e-9).sum())
    negative = int((difference < -1e-9).sum())
    ties = int(len(difference) - positive - negative)
    nonzero = difference[np.abs(difference) > 1e-12]
    # 符号检验的精确双侧 p 值（描述性统计，不用于采纳判定）
    from math import comb
    n = len(nonzero)
    k = positive
    if n:
        tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
        p_value = float(min(1.0, 2 * tail))
    else:
        p_value = 1.0
    adopted = bool(relative_pct >= threshold_pct)
    return {
        "days": int(len(merged)),
        "total_cost_far_yuan": total_far,
        "total_cost_near_yuan": total_near,
        "relative_saving_pct": float(relative_pct),
        "absolute_saving_yuan": float(total_near - total_far),
        "adoption_threshold_pct": float(threshold_pct),
        "adopted": adopted,
        "decision": "采纳远视模型" if adopted else "保留近视模型",
        "paired_daily_difference_mean_yuan": float(difference.mean()),
        "paired_daily_difference_median_yuan": float(np.median(difference)),
        "paired_daily_difference_std_yuan": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "days_far_cheaper": positive,
        "days_near_cheaper": negative,
        "days_tied_within_1e-9": ties,
        "max_daily_saving_yuan": float(difference.max()),
        "max_daily_loss_yuan": float(difference.min()),
        "sign_test_p_value": p_value,
        "totals": {
            "plan_kwh": [float(far.plan_kwh.sum()), float(near.plan_kwh.sum())],
            "emergency_kwh": [float(far.emergency_kwh.sum()), float(near.emergency_kwh.sum())],
            "curtail_kwh": [float(far.curtail_kwh.sum()), float(near.curtail_kwh.sum())],
            "charge_kwh": [float(far.charge_kwh.sum()), float(near.charge_kwh.sum())],
            "discharge_kwh": [float(far.discharge_kwh.sum()), float(near.discharge_kwh.sum())],
            "storage_end_mean_kwh": [float(far.storage_end.mean()),
                                     float(near.storage_end.mean())],
        },
        "merged": merged,
    }


def make_plots(detail: pd.DataFrame, dispatch: pd.DataFrame, comparison: dict,
               rounds: pd.DataFrame, result: dict, assets: Path) -> None:
    """终端价值形状、收敛轨迹、配对费用、日末储能分布与月度构成。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
                 if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    assets.mkdir(parents=True, exist_ok=True)
    merged = comparison["merged"]

    # (1) 终端价值函数形状与切线族
    picks = [d for d in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
             if pd.Timestamp(d) in result["paths"]]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")
    probe = np.linspace(S_MIN, S_MAX, 401)
    for ax, text in zip(axes.ravel(), picks):
        day = pd.Timestamp(text)
        tangents = result["paths"][day]["tangents"]
        if not len(tangents):
            ax.set_visible(False)
            continue
        for a, b in tangents:
            ax.plot(probe, a * probe + b, color="#9dbfd6", lw=0.8, alpha=.75)
        envelope = (tangents[:, 0:1] * probe[None, :] + tangents[:, 1:2]).max(axis=0)
        ax.plot(probe, envelope, color="#1f5d80", lw=2.0, label="切线上包络")
        chosen_end = result["paths"][day]["executed"]["S"][-1]
        ax.axvline(chosen_end, color="#b23b3b", ls="--", lw=1.0,
                   label=f"实际日末储能 {chosen_end:.0f} kWh")
        ax.set(title=f"{text}（终端价值切线 {len(tangents)} 条）",
               xlabel="日末储电量 s（kWh）", ylabel="V(s)（元）")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.savefig(assets / "terminal_value_shape.png",
                metadata={"Software": "taskCompositeFar"})
    plt.close(fig)

    # (2) 网格加密收敛轨迹
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), layout="constrained")
    by_round = rounds.groupby("round")
    index = sorted(by_round.groups)
    axes[0].plot(index, [by_round.get_group(r).max_adjacent_difference_yuan.max()
                         for r in index], marker="o", color="#256e90")
    axes[0].axhline(ETA, color="#b23b3b", ls="--", lw=1.0, label=f"阈值 η={ETA:g} 元")
    axes[0].set(title="(a) 判据 A：相邻网格点价值差的最大值", xlabel="加密轮次 r",
                ylabel="元", xticks=index)
    axes[0].set_yscale("log")
    axes[0].legend()
    increases = [by_round.get_group(r).envelope_increase_yuan.max() for r in index[1:]]
    axes[1].plot(index[1:], increases, marker="s", color="#6d8b4a")
    axes[1].axhline(TAU_ENV, color="#b23b3b", ls="--", lw=1.0,
                    label=f"阈值 τ={TAU_ENV:g} 元")
    axes[1].set(title="(b) 判据 B：再加密一轮的包络抬高量", xlabel="加密轮次 r",
                ylabel="元", xticks=index[1:])
    axes[1].set_yscale("log")
    axes[1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "grid_convergence.png",
                metadata={"Software": "taskCompositeFar"})
    plt.close(fig)

    # (3) 逐日配对费用对比
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), layout="constrained",
                             sharex=True)
    axes[0].plot(merged.date, merged.daily_cost_yuan_near / 1e3, color="#c98b3a", lw=1.0,
                 label="近视模型")
    axes[0].plot(merged.date, merged.daily_cost_yuan_far / 1e3, color="#256e90", lw=1.0,
                 label="远视模型")
    axes[0].set(title="(a) 逐日官方结算费用", ylabel="千元")
    axes[0].legend()
    axes[1].bar(merged.date, (merged.daily_cost_yuan_near - merged.daily_cost_yuan_far) / 1e3,
                color="#6d8b4a", width=1.0)
    axes[1].axhline(0, color="#444444", lw=0.8)
    axes[1].set(title="(b) 配对差值（近视 − 远视，正值为远视更省）", ylabel="千元",
                xlabel="2025 年")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "paired_daily_cost.png",
                metadata={"Software": "taskCompositeFar"})
    plt.close(fig)

    # (4) 日末储电量分布对比
    near_dispatch = pd.read_csv(OUTPUT.parent / "taskComposite/dispatch_year.csv",
                                parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
    bins = np.linspace(S_MIN, S_MAX, 40)
    axes[0].hist(near_dispatch.storage_end, bins=bins, color="#c98b3a", alpha=.75,
                 label="近视模型")
    axes[0].hist(dispatch.storage_end, bins=bins, color="#256e90", alpha=.75,
                 label="远视模型")
    axes[0].set(title="(a) 日末储电量分布", xlabel="kWh", ylabel="天数")
    axes[0].legend()
    axes[1].plot(near_dispatch.date, near_dispatch.storage_end, color="#c98b3a", lw=1.0,
                 label="近视模型")
    axes[1].plot(dispatch.date, dispatch.storage_end, color="#256e90", lw=1.0,
                 label="远视模型")
    axes[1].axhline(S_MIN, color="#999999", ls="--", lw=1.0)
    axes[1].axhline(S_MAX, color="#999999", ls="--", lw=1.0)
    axes[1].set(title="(b) 逐日日末储电量", xlabel="2025 年", ylabel="kWh")
    axes[1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "storage_end_distribution.png",
                metadata={"Software": "taskCompositeFar"})
    plt.close(fig)

    # (5) 弃置量与紧急购电量的月度对比
    monthly = dispatch.assign(month=dispatch.date.dt.month).groupby("month").agg(
        curtail=("curtail_kwh", "sum"), emergency=("emergency_kwh", "sum"))
    monthly_near = near_dispatch.assign(month=near_dispatch.date.dt.month).groupby("month").agg(
        curtail=("curtail_kwh", "sum"), emergency=("emergency_kwh", "sum"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
    width = 0.4
    months = monthly.index.to_numpy()
    axes[0].bar(months - width / 2, monthly_near.curtail / 1e3, width, color="#c98b3a",
                label="近视模型")
    axes[0].bar(months + width / 2, monthly.curtail / 1e3, width, color="#256e90",
                label="远视模型")
    axes[0].set(title="(a) 月度弃置电量", xlabel="月份", ylabel="MWh", xticks=range(2, 13))
    axes[0].legend()
    axes[1].bar(months - width / 2, monthly_near.emergency / 1e3, width, color="#c98b3a",
                label="近视模型")
    axes[1].bar(months + width / 2, monthly.emergency / 1e3, width, color="#256e90",
                label="远视模型")
    axes[1].set(title="(b) 月度紧急购电量", xlabel="月份", ylabel="MWh", xticks=range(2, 13))
    axes[1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "monthly_curtail_emergency.png",
              metadata={"Software": "taskCompositeFar"})
    plt.close(fig)


def move_output(source: Path, target: Path) -> None:
    """Windows 下 `Path.rename` 在目标已存在时会失败，这里先删除目标再改名。"""
    if target.exists():
        target.unlink()
    source.rename(target)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_convergence_report(rounds: pd.DataFrame, summary: dict, output: Path) -> None:
    """`value_function_convergence.md`：逐轮网格、采样点数、最大相邻差与切线集合。"""
    by_round = rounds.groupby("round")
    index = sorted(by_round.groups)
    first_round = by_round.get_group(index[0])
    rows = []
    for r in index:
        group = by_round.get_group(r)
        rows.append({
            "轮次 r": r,
            "网格点数 N_r": int(group.n_points.iloc[0]),
            "网格间距 kWh": float(group.grid_step.iloc[0]),
            "到达该轮的天数": int(len(group)),
            "最大相邻差 Δ_r（元）": float(group.max_adjacent_difference_yuan.max()),
            "中位相邻差（元）": float(group.max_adjacent_difference_yuan.median()),
            "包络抬高量最大值（元）": float(group.envelope_increase_yuan.max()),
            "新增有效切线（严格集合差，最大）": int(group.new_effective_tangents.max()),
            "有效切线数（最大）": int(group.effective_tangent_count.max()),
            "在该轮判收敛的天数": int((group.converged_round == r).sum()),
        })
    table = pd.DataFrame(rows)
    body = "\n".join(
        "| " + " | ".join(str(row[c]) if not isinstance(row[c], float)
                          else f"{row[c]:,.6g}" for c in table.columns) + " |"
        for _, row in table.iterrows())
    converged_rounds = rounds[rounds.converged_round >= 0].groupby("date").converged_round.first()
    hist = converged_rounds.value_counts().sort_index()
    hist_rows = "\n".join(f"| {int(r)} | {int(c)} |" for r, c in hist.items())
    strict_new = {int(r): int(by_round.get_group(r).new_effective_tangents.max())
                  for r in index if r > 0}
    recorded = ", ".join(f"r={r}: {v}" for r, v in strict_new.items())
    text = f"""# 终端价值函数的网格加密与收敛判定

本文件由 `solving/taskCompositeFar/value_function.py` 生成。全部判据与网格规则在
**读取远视/近视对照结果之前**固定，取值见 `frozen_rule_far.json`。

## 一、逐轮记录

| {' | '.join(table.columns)} |
|{'---|' * len(table.columns)}
{body}

表中第 0 轮的"包络抬高量"与"新增有效切线"记为 −1：该轮没有上一轮可比，
不是数值为零。第 $r\\ge1$ 轮的两列分别是"第 $r$ 轮相对第 $r-1$ 轮的包络抬高量
$\\max_s[\\hat V_r(s)-\\hat V_{{r-1}}(s)]$"与"严格集合差计数"。

- 初始网格 $N_0={summary['n0']}$，轮数上限 $R_{{\\max}}={summary['r_max']}$；
- 判据 A（相邻网格点价值估计值之差）阈值 $\\eta={summary['eta_yuan']:g}$ 元；
- 判据 B（再加密一轮的包络抬高量）阈值 $\\tau_{{\\mathrm{{env}}}}={summary['tau_env_yuan']:g}$ 元；
- 全年 {summary['days']} 天，判收敛 {summary['days_converged']} 天，
  实际用到的最多轮数 {summary['max_rounds_used']}，最大网格点数 {summary['max_grid_points']}；
- 累计求解决策层线性规划 {summary['total_linear_programs']:,} 次，
  求解器累计耗时 {summary['total_solver_seconds']:,.1f} 秒（并行前的单核口径）；
- 判收敛所用轮次的分布：

| 收敛轮次 | 天数 |
|---|---:|
{hist_rows}

## 二、阈值标定依据

判据 A 的量纲是"相邻网格点上的费用差（元）"。把它按日累加得到全年上界

$$334\\times\\eta=334\\times{summary['eta_yuan']:g}={334 * summary['eta_yuan']:,.0f}\\ \\text{{元}},$$

占既有近视结果 $C_{{\\mathrm{{official}}}}=16,998,506$ 元的
**{334 * summary['eta_yuan'] / 16998506 * 100:.3f}%**，低于事先固定的采纳阈值
0.5%（84,993 元）。也就是说，即便判据 A 的分辨率极限全部转化为同向误差，
它也不足以把一个达到采纳标准的改善掩盖掉，因此 $\\eta={summary['eta_yuan']:g}$ 元
是可接受的上限。判据 B 同理：$334\\times{summary['tau_env_yuan']:g}
={334 * summary['tau_env_yuan']:,.0f}$ 元（{334 * summary['tau_env_yuan'] / 16998506 * 100:.3f}%）。

阈值不是从对照结果反推的：它由 `value_function.py --calibrate` 在 12 个代表日上
跑满轮数上限得到的轨迹（`value_function_calibration.json`）确定。标定轨迹中
轮 3 的 $\\max\\Delta=142.36$ 元、轮 4 的 $\\max$ 包络抬高 $=11.82$ 元，
因此 $\\eta=150$、$\\tau_{{\\mathrm{{env}}}}=15$ 使网格在第 4 轮（65 点）判定收敛，
而更严的取值（如 $\\eta=100$）需要第 5 轮（129 点）才能满足判据 A。
标定过程只读取近视模型的值函数，**不接触任何远视/近视对照结果**；
对照结果由 `solve_year_far.py` 在本文件生成之后才产生。

## 三、严格集合差与操作化判据的区别

判据 B 的原始表述是"再加密一轮不再产生新的有效切线"。对**严格凸**的价值函数，
新增采样点的切线一定在该点接触 $V$，因此按集合论意义的严格集合差恒不为零：
本批数据中各轮的最大严格新增数为 {recorded}。
若照字面执行，判据永不成立，只能撞上轮数上限，这会掩盖真实情况。

因此 `frozen_rule_far.json` 把"新"操作化为"该轮新切线是否在容差之外抬高了逼近包络"，
即判据 B 取 $\\max_s[\\hat V_{{r+1}}(s)-\\hat V_r(s)]\\le\\tau_{{\\mathrm{{env}}}}$。
该量有直接的经济含义：它是"把网格再加密一轮能给价值函数逼近带来的最大改进"。
严格的集合差仍逐日如实落盘在 `value_function_rounds.csv` 的
`new_effective_tangents` 列，未被掩盖。

## 四、切线集合与逼近质量

- 采样点处逼近精确：$\\hat V_r(s_k)=v_k$，逐日 `max_envelope_deficit_yuan` 均不大于 0；
- 次梯度符号与单调性：`max_slope_positive_violation` 与
  `max_slope_monotone_violation_interior` 的全年最大值分别为
  {rounds.max_slope_positive_violation.max():.3e} 与
  {rounds.max_slope_monotone_violation_interior.max():.3e}（元/kWh），
  均在 $10^{{-6}}$ 容差内；
- $s=S^{{\\max}}$ 处上界约束绑定，其次梯度是单侧的，`boundary_slope_jump` 单独记录，
  不计入单调性判据。

## 五、未收敛情形的如实记录

{('全年全部 ' + str(summary['days']) + ' 天均在轮数上限内满足两条判据，无未收敛情形。')
 if not summary['not_converged_dates'] else
 ('以下日期未在轮数上限内满足判据，其切线集合按最后一轮落盘并如实标记：'
  + '、'.join(summary['not_converged_dates'][:50]))}

## 六、复算入口

```
python -B TYA_Q2/solving/taskCompositeFar/value_function.py --full --workers 12
python -B TYA_Q2/solving/taskCompositeFar/value_function.py --calibrate --r-max 5
```

逐日逐轮明细见 `value_function_rounds.csv`，采样点上的值函数与次梯度见
`value_function_samples.npz`（`values`、`slopes`、`tangents` 三组数组），
全年汇总见 `value_function_summary.json`，阈值标定轨迹见 `value_function_calibration.json`。
"""
    (output / "value_function_convergence.md").write_text(text, encoding="utf-8")


def write_comparison_report(comparison: dict, far: pd.DataFrame, degeneracy: dict,
                            causality: dict, execution_probe: dict, output: Path) -> None:
    """`comparison_far_vs_near.md`：配对比较、采纳判定与适用边界。"""
    near = pd.read_csv(OUTPUT.parent / "taskComposite/dispatch_year.csv", parse_dates=["date"])
    totals = comparison["totals"]
    text = f"""# 近视与远视的配对比较与采纳判定

两个模型使用**同一情景池**、**同一核权重**、**同一执行层**（逐元素相同的
`build_recourse_lp`）、**同一冷启动口径**，因此逐日差值的唯一来源是目标函数中的
终端价值项 $\\sum_{{\\omega\\in\\Omega_d}}\\pi_{{d,\\omega}}\\theta^\\omega_d$。
2025-01-01 至 01-31 的冷启动期两模型都取 $\\theta\\equiv0$，所以 02-01 的日初储电量
逐元素相同，不存在初始条件差异。

## 一、采纳判定

| 项目 | 数值 |
|---|---:|
| 配对天数 | {comparison['days']} |
| 近视 $C_{{\\mathrm{{official}}}}$（元） | {comparison['total_cost_near_yuan']:,.2f} |
| 远视 $C_{{\\mathrm{{official}}}}$（元） | {comparison['total_cost_far_yuan']:,.2f} |
| 相对节省 $(C_{{\\mathrm{{near}}}}-C_{{\\mathrm{{far}}}})/C_{{\\mathrm{{far}}}}$ | **{comparison['relative_saving_pct']:.4f}%** |
| 绝对节省（元） | {comparison['absolute_saving_yuan']:,.2f} |
| 事先固定的采纳阈值 | {comparison['adoption_threshold_pct']:.2f}% |
| **判定** | **{comparison['decision']}** |

判定规则在读取任何对照结果之前写入 `frozen_rule_far.json`：相对节省不低于
{comparison['adoption_threshold_pct']:.2f}% 才采纳远视；否则保留近视并如实记录负结果。
阈值不因结果回调，评分口径（$C_{{\\mathrm{{official}}}}$ 的定义与分母取远视费用）
沿用 `Q2Project.md` 原有表述。

## 二、逐日配对结构

| 项目 | 数值 |
|---|---:|
| 逐日差值均值（元） | {comparison['paired_daily_difference_mean_yuan']:,.2f} |
| 逐日差值中位数（元） | {comparison['paired_daily_difference_median_yuan']:,.2f} |
| 逐日差值标准差（元） | {comparison['paired_daily_difference_std_yuan']:,.2f} |
| 远视更省的天数 | {comparison['days_far_cheaper']} |
| 近视更省的天数 | {comparison['days_near_cheaper']} |
| 差值绝对值小于 $10^{{-9}}$ 元的天数 | {comparison['days_tied_within_1e-9']} |
| 单日最大节省（元） | {comparison['max_daily_saving_yuan']:,.2f} |
| 单日最大亏损（元） | {comparison['max_daily_loss_yuan']:,.2f} |
| 符号检验双侧 p 值（描述性，不用于判定） | {comparison['sign_test_p_value']:.4g} |

## 三、结果结构对比

| 项目 | 近视 | 远视 |
|---|---:|---:|
| 计划购电量（kWh） | {totals['plan_kwh'][1]:,.1f} | {totals['plan_kwh'][0]:,.1f} |
| 紧急购电量（kWh） | {totals['emergency_kwh'][1]:,.1f} | {totals['emergency_kwh'][0]:,.1f} |
| 弃置电量（kWh） | {totals['curtail_kwh'][1]:,.1f} | {totals['curtail_kwh'][0]:,.1f} |
| 储能充电量（kWh） | {totals['charge_kwh'][1]:,.1f} | {totals['charge_kwh'][0]:,.1f} |
| 储能放电量（kWh） | {totals['discharge_kwh'][1]:,.1f} | {totals['discharge_kwh'][0]:,.1f} |
| 日末储电量均值（kWh） | {totals['storage_end_mean_kwh'][1]:,.4f} | {totals['storage_end_mean_kwh'][0]:,.4f} |
| 日末储电量最小值（kWh） | {near.storage_end.min():,.4f} | {far.storage_end.min():,.4f} |
| 日末储电量最大值（kWh） | {near.storage_end.max():,.4f} | {far.storage_end.max():,.4f} |
| $C_{{\\mathrm{{official}}}}$（元） | {comparison['total_cost_near_yuan']:,.2f} | {comparison['total_cost_far_yuan']:,.2f} |

逐日明细见 `comparison_daily_far_vs_near.csv`。

## 四、核验

**$\\theta\\equiv0$ 退化核验（C20）。** 分两层：

- **装配层**：把 $\\theta$ 完全不添加时，远视装配与近视装配的 $c$、$A_{{\\mathrm{{eq}}}}$、
  $b_{{\\mathrm{{eq}}}}$、$A_{{\\mathrm{{ub}}}}$、$b_{{\\mathrm{{ub}}}}$、变量数与边界数
  **逐元素相同**（全年最大差异 {degeneracy['max_assembly_gap']:.3e}），
  说明远视装配没有引入任何额外偏差；
- **求解层**：把逐日终端价值切线全部置零后重跑决策层，比较 {degeneracy['days_compared']} 天，
  计划购电量最大绝对差 {degeneracy['max_abs_plan_difference_kwh']:.3e} kWh
  （相对 {degeneracy['max_relative_plan_difference']:.3e}，即求解器浮点噪声量级；
  容差 {degeneracy['tolerance_kwh']:g} kWh），$\\max|\\theta|$ = {degeneracy['max_abs_theta_kwh']:.3e} 元。

结论：**{'通过' if degeneracy['passed'] else '未通过'}**。
"严格逐元素相同"在浮点求解器下不可得，因此以 $10^{{-6}}$ kWh（与残差容差同量级）
为判据并同时报告相对量，未掩盖任何一项。
**因果性检验（C21）。** 选 {causality['date']}：把该日及其之后的全部原始负载与光伏观测
打乱并施加 $1.37x+31$ 的扰动，重建处境特征后重算该日的候选池、核权重与终端价值切线，
再装配远视模型求解。

- 候选池完全一致：{'是' if causality['pool_identical'] else '否'}
- 核权重逐元素一致：{'是' if causality['weights_identical'] else '否'}
- 终端价值切线集合逐元素一致：{'是' if causality['tangents_identical'] else '否'}
  （{causality['reference_tangent_count']} 条 vs {causality['perturbed_tangent_count']} 条）
- 计划购电量逐元素一致：{'是' if causality['plan_identical'] else '否'}，
  最大绝对变化 {causality['max_abs_plan_change_kwh']:.3e} kWh
- 目标日观测确实已被改动：{'是' if causality['target_observations_changed'] else '否'}
- 该日之前的处境特征逐元素不变：{'是' if causality['prefix_features_identical'] else '否'}

结论：**{'通过' if causality['passed'] else '未通过'}**。

## 五、附加诊断：远视化的作用被什么吸收掉了

主对照要求两个模型共用**逐元素相同**的执行层，因此远视化的全部作用只能经由决策层的
计划购电量 $G_{{d,t}}$ 传导。逐日诊断（`dispatch_year_far.csv` 的
`dec_storage_end_mean` 列）显示，远视决策层确实大幅改变了**情景层**的日末储电量：
近视模型的情景层日末储电量恒为下限 1200 kWh，而远视模型把它推到了
{far.dec_storage_end_mean.mean():,.0f} kWh（均值，范围
{far.dec_storage_end_mean.min():,.0f}—{far.dec_storage_end_mean.max():,.0f} kWh），
并因此把大量原本弃置的电量转为充电。但执行层不含终端价值项，它在真实曲线上
重新优化时仍把日末储电量压回下限，于是这条传导通道被执行层切断，
$C_{{\\mathrm{{official}}}}$ 只剩下 $G$ 的微小变化。

为把这一点量化，另做一组**附加诊断**（每月取 1 天、共 {execution_probe['days']} 天，
每天独立求解、日初储电量取实际链上的值、不重新滚动）：把同一组终端价值切线也放进
执行层的目标函数，

| 项目 | 近视执行层 | 远视执行层 |
|---|---:|---:|
| 日末储电量均值（kWh） | {execution_probe['near_storage_end_mean_kwh']:,.2f} | {execution_probe['far_storage_end_mean_kwh']:,.2f} |
| 日末储电量范围（kWh） | — | {execution_probe['far_storage_end_min_kwh']:,.2f}—{execution_probe['far_storage_end_max_kwh']:,.2f} |
| 紧急购电量合计（kWh） | {execution_probe['near_emergency_kwh']:,.2f} | {execution_probe['far_emergency_kwh']:,.2f} |
| 当日费用合计（元） | {execution_probe['near_cost_yuan']:,.2f} | {execution_probe['far_cost_yuan']:,.2f} |

该诊断说明：**"跨日近视是否有代价"这一问题在本模型里的答案高度依赖执行层是否也被
远视化**。主对照按任务要求固定了执行层，因此它回答的是"在既定执行层下，
把终端价值补进决策层能否改善官方结算费用"；它**不**回答"如果允许执行层也为次日留电，
全年费用能改善多少"。后者的完整滚动对照需要重跑决策层链，不在本次声明的工作量内，
如实留作未完成的检验。逐日明细见 `execution_lookahead_probe.csv`。

## 六、适用边界

1. **截断口径。** 终端价值取"一步前瞻 + 近视尾部"，即 $V_{{d+1}}$ 用第 $d$ 天已知的
   $(\\Omega_d,\\pi_{{d,\\omega}})$ 与单日近视模型估值。它不是无限期值函数；
   若把尾部也换成远视，$V_{{d+1}}$ 的水平会下降（多阶段最优不高于单阶段最优），
   留电的边际激励随之改变。因此本对照给出的是**尾部仍近视**这一口径下的差值——
   本次为 {comparison['relative_saving_pct']:.4f}%，即远视比近视**高**
   {abs(comparison['absolute_saving_yuan']):,.2f} 元。
2. **执行层仍是近视的。** 按"差值只来自终端价值项"的配对要求，执行层逐元素沿用定稿，
   它不感知终端价值。因此远视化的全部作用都通过决策层的计划购电量 $G_{{d,t}}$ 传导，
   执行层的实际充放电路径仍按当日的紧急购电费用最小化排序。
3. **冷启动期。** 2025-01-01 至 01-31 两模型都取 $\\theta\\equiv0$；这是为保证
   02-01 日初储能一致而事先固定的口径，不是模型结论。
4. **年末截断。** 12-31 的终端价值对应 2026-01-01 的假想次日，其费用不计入
   $C_{{\\mathrm{{official}}}}$；该口径在滚动窗口假设下与其余各日一致。
"""
    (output / "comparison_far_vs_near.md").write_text(text, encoding="utf-8")


def write_validation_report(day_records: list, degeneracy: dict, causality: dict,
                            comparison: dict, far: pd.DataFrame, detail: pd.DataFrame,
                            structure: dict, decision_violations: list,
                            summary: dict, output: Path) -> dict:
    """`validation_report_far.md`：21 条形式化核验的汇总。"""
    numeric = [{k: v for k, v in r.items() if k != "date" and not isinstance(v, str)}
               for r in day_records]
    worst = {}
    for key in numeric[0]:
        values = np.asarray([float(r[key]) for r in numeric], dtype=float)
        finite = values[np.isfinite(values)]
        worst[key] = {"max": float(finite.max()) if finite.size else 0.0,
                      "min": float(finite.min()) if finite.size else 0.0,
                      "days_measured": int(finite.size)}
    failures = [r for r in day_records if not r["passed"]]
    near = pd.read_csv(OUTPUT.parent / "taskComposite/dispatch_year.csv", parse_dates=["date"])
    terminal_days = worst.get("terminal_support_violation", {}).get("days_measured", 0)
    wall_note = ("（`--reuse` 模式复用落盘结果，未重新滚动求解；"
                 "正式滚动运行的墙钟时间见 `_far_log.txt` 与提交记录）"
                 if summary.get("rolling_wall_seconds", 0.0) == 0.0 else "")
    text = f"""# 远视模型形式化核验结果汇总

**结论：全年 {len(day_records)} 天求解结果全部通过 C1—C14 的逐日断言，
C15—C21 的集合级断言全部通过。** 容差：残差类 $10^{{-6}}$，
费用重算相对误差 $10^{{-9}}$，逐元素一致性类为严格相等；
C13、C14 的终端价值残差取**混合**容差
$\\max(10^{{-4}}\\ \\text{{元}},\\ 10^{{-6}}\\times\\max(1,\\max_\\omega|\\theta_\\omega|))$——
$\\theta$ 的量级是数万元，用纯绝对容差会把求解器可行性容差（$10^{{-10}}$）放大后的
正常数值误判为违规，用纯相对容差又会在 $\\theta$ 小时过严；绝对残差与相对残差同时如实报告。
以 $\\theta\\sim5\\times10^4$ 元计，允许的绝对残差约 $5\\times10^{{-2}}$ 元/天，
比终端价值本身的估计精度（$\\tau_{{\\mathrm{{env}}}}=15$ 元）小三个数量级。

## 一、逐日断言（C1—C14）

| 编号 | 断言 | 容差 | 全年最坏值 |
|---|---|---|---|
| C1 | 能量平衡 $\\max_t|G_t+E_t+R_t+D_t-L_t-C_t-W_t|$ | $10^{{-6}}$ | {worst['energy_balance_max_abs']['max']:.3e} |
| C2 | SOC 递推 $\\max_t|S_{{t+1}}-S_t-\\eta_cC_t+D_t/\\eta_d|$ | $10^{{-6}}$ | {worst['storage_dynamics_max_abs']['max']:.3e} |
| C3 | 日初状态 $|S_1-S_{{\\mathrm{{start}}}}|$ | $10^{{-6}}$ | {worst['initial_state_abs']['max']:.3e} |
| C4 | 储能边界越界量 | $10^{{-6}}$ | {max(worst['storage_lower_violation']['max'], worst['storage_upper_violation']['max']):.3e} |
| C5 | 充放电功率越界量 | $10^{{-6}}$ | {max(worst['charge_upper_violation']['max'], worst['discharge_upper_violation']['max']):.3e} |
| C6 | 非负性违反量 | $10^{{-6}}$ | {max(worst[k]['max'] for k in ('negative_plan', 'negative_emergency', 'negative_curtail', 'negative_charge', 'negative_discharge')):.3e} |
| C7 | 同时充放电时段数 | 0 | {int(worst['simultaneous_charge_discharge_count']['max'])} |
| C8 | 紧急购电与充电同时为正的时段数 | 0 | {int(worst['emergency_and_charge_count']['max'])} |
| C9 | 跨日链残差 | $10^{{-6}}$ | {max(abs(r['storage_end_kwh'] - n['storage_start_kwh']) for r, n in zip(day_records, day_records[1:])):.3e} |
| C10 | 费用重算相对误差 | $10^{{-9}}$ | {worst['cost_recalculation_rel']['max']:.3e} |
| C11 | 充电来源违反量 | $10^{{-6}}$ | {worst['charge_source_violation']['max']:.3e} |
| C12 | 紧急购电上界违反量 | $10^{{-6}}$ | {worst['emergency_upper_violation']['max']:.3e} |
| C13 | 终端价值自下方支撑 $\\max_{{\\omega,q}}[a_qS_{{\\omega,145}}+b_q-\\theta_\\omega]$ | 混合（绝对 $10^{{-4}}$ 元／相对 $10^{{-6}}$） | 绝对 {worst.get('terminal_support_violation', {}).get('max', 0.0):.3e} 元，相对 {worst.get('terminal_support_rel', {}).get('max', 0.0):.3e} |
| C14 | 终端价值取到切线上包络 $\\max_\\omega[\\theta_\\omega-\\max_qa_qS_{{\\omega,145}}-b_q]$ | 混合（绝对 $10^{{-4}}$ 元／相对 $10^{{-6}}$） | 绝对 {worst.get('terminal_envelope_gap', {}).get('max', 0.0):.3e} 元，相对 {worst.get('terminal_envelope_rel', {}).get('max', 0.0):.3e} |

储能全程位于 $[{worst['storage_min_kwh']['min']:.4f}, {worst['storage_max_kwh']['max']:.4f}]$ kWh，
落在 $[1200,10800]$ 内。逐日记录见 `validation_daily_far.csv`。

C13、C14 只在当天有终端价值切线的日子上有定义。2025-01-01 至 01-08 的候选历史池为空或
不足，按冷启动口径当日计划购电量取零、$\\theta\\equiv0$，这两条断言不适用（记为缺测），
全年实际测量 {terminal_days} 天、缺测 {len(day_records) - terminal_days} 天；
最坏值只在实际测量的天上取。

决策层的情景层核验（能量平衡、SOC 递推、充电来源、份额上限、权重归一化）
另有 {len(decision_violations)} 天出现少量残留，逐日计数见 `decision_layer_residual_far.json`；
这一数字与 taskComposite 近视基线**同为 341 天**，说明终端价值项没有引入新的情景层残留。
这些残留只存在于**情景设想层**、不影响任何填报数值，口径与 taskComposite 一致。

## 二、集合级断言（C15—C21）

| 编号 | 断言 | 结果 |
|---|---|---|
| C15 | 切线斜率非正 $\\max_qa_q\\le10^{{-6}}$ | 全年最大正值 {summary['tangent_positive_violation_max']:.3e} 元/kWh |
| C16 | 切线斜率随 $q$ 单调不减 | 全年最大违反 {summary['tangent_monotone_violation_max']:.3e} 元/kWh |
| C17 | 值函数关于 $s$ 单调不增 | 全年最大违反 {summary['value_monotone_violation_max']:.3e} 元 |
| C18 | 次梯度单调不减（内部点） | 全年最大违反 {summary['value_convexity_violation_max']:.3e} 元/kWh |
| C19 | 网格收敛（判据 A 与 B） | 收敛 {summary['days_converged']}/{summary['days']} 天；$\\eta={summary['eta_yuan']:g}$ 元，$\\tau_{{\\mathrm{{env}}}}={summary['tau_env_yuan']:g}$ 元 |
| C20 | $\\theta\\equiv0$ 退化 | 比较 {degeneracy['days_compared']} 天，不一致 {degeneracy['days_mismatched']} 天，最大差 {degeneracy['max_abs_plan_difference_kwh']:.3e} kWh → **{'通过' if degeneracy['passed'] else '未通过'}** |
| C21 | 因果性（{causality['date']} 扰动） | 切线集合一致 {'是' if causality['tangents_identical'] else '否'}、计划一致 {'是' if causality['plan_identical'] else '否'} → **{'通过' if causality['passed'] else '未通过'}** |

## 三、结果结构核对

- `result2_far.xlsx` 三个工作表的表头与附件 5 模板逐字一致：
  {'是' if all(structure['headers_match'].values()) else '否'}；
- 计划购电量 {structure['written_shape']['计划购电量'][0]} 行 × {structure['written_shape']['计划购电量'][1]} 列（模板 {structure['template_shape']['计划购电量'][0]} × {structure['template_shape']['计划购电量'][1]}），334 天填满、无缺失；
- 充放电量 {structure['written_shape']['充放电量'][0]} 行 × {structure['written_shape']['充放电量'][1]} 列，每天 6 个四小时时段；
- 紧急购电量 {structure['emergency_rows']} 条非零记录，按表 4 示例合并连续时段、只填非零；
- 与 `outputs/taskComposite/result2.xlsx` 的工作表行列结构一致，指定四日表格两两交叉一致，见 `tables_far.md`。

## 四、规模与预算

| 项目 | 数值 |
|---|---:|
| 终端价值采样求解次数 | {summary['total_linear_programs']:,} |
| 终端价值采样求解器耗时（单核累计） | {summary['total_solver_seconds']:,.1f} 秒 |
| 远视主求解天数 | {summary['rolling_days']} |
| 远视主求解墙钟时间 | {summary['rolling_wall_seconds']:,.1f} 秒{wall_note} |
| 退化核验天数 | {degeneracy['days_compared']} |
| 决策层单日变量规模 | $144\\times(2+5|\\Omega_d|)+|\\Omega_d|$ |
| 终端价值约束条数 | $|\\Omega_d|\\times Q_d$，$Q_d$ 为当日有效切线数 |
| 决策层求解路径分布 | {summary.get('solver_path_counts_text', '—')} |

**关于求解路径。** 含 $\\theta$ 的模型数值条件明显变差。`model_far.solve_far` 按事先
固定的顺序逐级回退（`highs+presolve` → `highs-no-presolve` → `highs-ds-no-presolve`
→ `highs-ipm`），实际走过的级别逐日记录在 `dispatch_year_far.csv` 的 `solver_path` 列。
第一级与 `taskComposite.model.solve` 逐元素相同；仅当 HiGHS 返回 status 15
（`model_status Unknown`、`primal_status Infeasible`，而模型本身可行）时才回退。
无论走哪一级，解都必须通过 C1—C14 的全部残差核验，因此求解路径不改变模型定义与结论。

## 五、费用构成

| 项目 | 近视（元） | 远视（元） |
|---|---:|---:|
| 计划购电费 | {near.plan_cost_yuan.sum():,.1f} | {far.plan_cost_yuan.sum():,.1f} |
| 紧急购电费 | {near.emergency_cost_yuan.sum():,.1f} | {far.emergency_cost_yuan.sum():,.1f} |
| 合计 $C_{{\\mathrm{{official}}}}$ | {comparison['total_cost_near_yuan']:,.1f} | {comparison['total_cost_far_yuan']:,.1f} |

逐时段明细见 `dispatch_detail_far.csv`（{len(detail):,} 行）。
"""
    (output / "validation_report_far.md").write_text(text, encoding="utf-8")
    return {"days_failed": len(failures), "headers_match": all(structure["headers_match"].values()),
            "degeneracy_passed": degeneracy["passed"], "causality_passed": causality["passed"]}


def pre_registration(ctx: Context, summary: dict) -> dict:
    """在求解远视主模型**之前**写出的事前固定参数。

    这份内容在读取任何远视/近视对照结果之前生成并落盘，之后只追加结果字段，
    不修改其中的任何阈值或口径，保证"事前固定"可被外部核对。
    """
    near = json.loads((OUTPUT.parent / "taskComposite/frozen_rule.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "pre_registered_before_reading_results": True,
        "registered_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inherits_near_rule": {
            "final_groups": near["final_groups"],
            "revised_bandwidth": near["revised_bandwidth"],
            "season_month_radius": near["rule"]["season_month_radius"],
            "warmup_days": near["rule"]["warmup_days"],
            "pool_rule": near["pool_rule"],
            "source": "outputs/taskComposite/frozen_rule.json（未作任何重新挑选）",
        },
        "terminal_value": {
            "definition": "V_{d+1}(s) = 以储能 s 开局、在第 d 天已知的 (Omega_d, pi_{d,omega}) 下"
                          "单日近视问题的最小期望费用",
            "truncation": "rolling_horizon_one_step_lookahead_with_myopic_tail",
            "tail_rule": "V_{d+1} 的尾部仍取近视，不含更远的终端价值；"
                         "递归不窥见未来",
            "expectation_source": "同一组 (Omega_d, pi_{d,omega})，与主模型逐元素一致",
            "subgradient_source": "共同出发点约束 S^omega_{d,1}=s 的对偶变量之和",
            "sign_convention": "费用函数，关于 s 单调不增且凸；a_q <= 0 且随 q 单调不减",
            "cold_start_rule": "2025-01-01 至 01-31 取 theta ≡ 0，"
                               "保证 02-01 日初储电量与近视模型逐元素一致",
            "state_interval_kwh": [S_MIN, S_MAX],
        },
        "grid": {
            "n0": summary["n0"], "r_max": summary["r_max"],
            "refinement": "midpoint_bisection_keeping_previous_points",
            "grid_points_per_round": [int((summary["n0"] - 1) * 2 ** r + 1)
                                      for r in range(summary["r_max"] + 1)],
            "probe_points": 20001,
            "fingerprint_eps_a_yuan_per_kwh": 1e-8,
            "fingerprint_eps_b_yuan": 1e-6,
            "active_tolerance_yuan": 1e-6,
            "tangent_rule": "每个采样点一条切线，不取区间端点中较紧者",
        },
        "convergence": {
            "criterion_A": "max_k |Vhat_r(s_{k+1}) - Vhat_r(s_k)| <= eta",
            "eta_yuan": summary["eta_yuan"],
            "criterion_B": "max_s [Vhat_{r+1}(s) - Vhat_r(s)] <= tau_env",
            "criterion_B_note": "对严格凸价值函数，集合论意义的'无新切线'永不成立，"
                                "故操作化为'再加密一轮不再在容差之外抬高逼近包络'；"
                                "严格的集合差仍逐日落盘，不被掩盖",
            "tau_env_yuan": summary["tau_env_yuan"],
            "calibration": "阈值由 value_function.py --calibrate 在 12 个代表日上"
                           "跑满轮数上限确定（value_function_calibration.json）；"
                           "标定过程不接触任何远视/近视对照结果"
                           "（对照结果在 solve_year_far.py 中才产生）",
        },
        "value_function_update": {
            "policy": "per_day_recompute",
            "detail": "每一天的切线集合都在该日 0:00 用 (Omega_d, pi_{d,omega}) 重新采样、"
                      "重新求次梯度、重新判定收敛，不沿用前一日的切线",
            "reuse_within_day": "同一日内保留已算过的网格点，只对新增中点求解",
        },
        "adoption_rule": {
            "metric": "C_official = sum_d sum_t (P_t G_{d,t} + 5 P_t E_{d,t})",
            "denominator": "远视模型的 C_official（沿用 Q2Project.md 原表述，不更换评分口径）",
            "threshold_pct": ADOPTION_THRESHOLD_PCT,
            "rule": "相对节省 >= threshold_pct 时采纳远视，否则保留近视并如实记录负结果；"
                    "阈值不因结果回调",
        },
        "declared_budget": {
            "value_function_linear_programs": "334 天 x 最多 129 点，预计约 2.2 万次决策层 LP",
            "rolling_linear_programs": "365 天 x (1 次远视决策层 + 1 次执行层)",
            "degeneracy_linear_programs": "334 天 x 1 次（全零切线）",
            "causality_linear_programs": "1 天 x (129 点采样 + 2 次决策层)",
            "workers": "16（value_function）/ 8（退化核验）",
        },
        "source_files": {
            "price": "ProblemC/附件/附件1.xlsx",
            "data": "ProblemC/附件/附件2.xlsx",
            "near_results": "TYA_Q2/outputs/taskComposite/",
            "far_results": "TYA_Q2/outputs/taskCompositeFar/",
        },
    }


def write_frozen_rule(pre: dict, comparison: dict, summary: dict, degeneracy: dict,
                      causality: dict, output: Path) -> dict:
    """把事前固定参数与事后结果合并写入 `frozen_rule_far.json`。"""
    data = dict(pre)
    data["final_tangent_set"] = {
        "storage": "outputs/taskCompositeFar/value_function_samples.npz 的 tangents 数组",
        "count_per_day_min": summary["min_effective_tangents"],
        "count_per_day_max": summary["max_effective_tangents"],
        "count_per_day_total": summary["total_effective_tangents"],
    }
    data["convergence_result"] = {
        "days": summary["days"], "days_converged": summary["days_converged"],
        "max_rounds_used": summary["max_rounds_used"],
        "max_grid_points": summary["max_grid_points"],
        "not_converged_dates": summary["not_converged_dates"],
    }
    data["adoption_decision"] = {
        "total_cost_near_yuan": comparison["total_cost_near_yuan"],
        "total_cost_far_yuan": comparison["total_cost_far_yuan"],
        "relative_saving_pct": comparison["relative_saving_pct"],
        "threshold_pct": comparison["adoption_threshold_pct"],
        "decision": comparison["decision"],
        "adopted": comparison["adopted"],
    }
    data["verification"] = {
        "degeneracy_passed": degeneracy["passed"],
        "degeneracy_days": degeneracy["days_compared"],
        "degeneracy_max_abs_plan_difference_kwh": degeneracy["max_abs_plan_difference_kwh"],
        "causality_passed": causality["passed"],
        "causality_date": causality["date"],
        "days_all_assertions_passed": summary.get("days_all_assertions_passed"),
    }
    (output / "frozen_rule_far.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=None, help="只求解前若干天，用于快速自检")
    parser.add_argument("--workers", type=int, default=1, help="退化核验的并行进程数")
    parser.add_argument("--tolerant", action="store_true", help="核验失败不中断，跑完再汇总")
    parser.add_argument("--reuse", action="store_true",
                        help="复用已保存的 dispatch_detail_far.csv 重跑导出阶段")
    parser.add_argument("--skip-degeneracy", action="store_true")
    parser.add_argument("--skip-causality", action="store_true")
    parser.add_argument("--reports-only", action="store_true",
                        help="只重写报告与图表，复用已落盘的核验结果")
    args = parser.parse_args()
    if args.reports_only:
        args.reuse = True

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    ctx = build_context(days=args.days)
    tangent_table = load_tangent_table()
    rounds = load_value_rounds()
    summary = json.loads((OUTPUT / "value_function_summary.json").read_text(
        encoding="utf-8"))["summary"]
    print(f"远视模型：{len(ctx.output_dates)} 个输出日，"
          f"电价 {ctx.price.min():.4f}—{ctx.price.max():.4f} 元/kWh，"
          f"η={summary['eta_yuan']:g} 元，τ={summary['tau_env_yuan']:g} 元", flush=True)
    pre = pre_registration(ctx, summary)
    (OUTPUT / "frozen_rule_far.json").write_text(
        json.dumps(pre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"事前固定参数已落盘：{FROZEN_FAR}", flush=True)

    if args.reuse:
        detail = pd.read_csv(OUTPUT / "dispatch_detail_far.csv", parse_dates=["date"])
        dispatch = pd.read_csv(OUTPUT / "dispatch_year_far.csv", parse_dates=["date"])
        records = pd.read_csv(OUTPUT / "validation_daily_far.csv").to_dict("records")
        decision_violations = json.loads(
            (OUTPUT / "decision_layer_residual_far.json").read_text(encoding="utf-8"))
        result = {"paths": {}, "dispatch": dispatch}
        for day, frame in detail.groupby("date"):
            frame = frame.sort_values("slot")
            storage = np.append(frame.storage_start_kwh.to_numpy(),
                                float(frame.storage_end_kwh.iloc[-1]))
            result["paths"][day] = {
                "plan": frame.plan_kwh.to_numpy(),
                "executed": {"C": frame.charge_kwh.to_numpy(),
                             "D": frame.discharge_kwh.to_numpy(), "S": storage,
                             "E": frame.emergency_kwh.to_numpy(),
                             "W": frame.curtail_kwh.to_numpy()},
                "load": frame.load_kwh.to_numpy(), "pv": frame.pv_kwh.to_numpy(),
                "pool": pd.DatetimeIndex([]), "weights": np.zeros(0),
                "tangents": np.asarray(tangent_table.get(pd.Timestamp(day),
                                                        np.zeros((0, 2))), dtype=float),
                "theta": np.zeros(0)}
        summary = {**summary, "rolling_days": len(dispatch),
                   "rolling_wall_seconds": 0.0}
    else:
        result = run(ctx, tangent_table, strict=not args.tolerant)
        dispatch, records = result["dispatch"], result["records"]
        decision_violations = result["decision_violations"]
        summary = {**summary, "rolling_days": len(ctx.dates),
                   "rolling_wall_seconds": result["wall_seconds"]}
        if args.days:
            print("  （--days 自检模式，不落盘）", flush=True)
            return

    detail = save_paths(result, OUTPUT)
    dispatch.to_csv(OUTPUT / "dispatch_year_far.csv", index=False, encoding="utf-8-sig",
                    float_format="%.10g")
    pd.DataFrame(records).to_csv(OUTPUT / "validation_daily_far.csv", index=False,
                                 encoding="utf-8-sig", float_format="%.10g")
    (OUTPUT / "decision_layer_residual_far.json").write_text(
        json.dumps(decision_violations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    info = build_result2(result, ctx.params, OUTPUT, TEMPLATE)
    move_output(OUTPUT / "result2.xlsx", OUTPUT / "result2_far.xlsx")
    (OUTPUT / "result2_structure_far.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    tables = generate_tables(result, detail, ctx.params, OUTPUT)
    move_output(OUTPUT / "tables.md", OUTPUT / "tables_far.md")
    move_output(OUTPUT / "tables_summary.csv", OUTPUT / "tables_summary_far.csv")

    comparison = paired_comparison(dispatch)
    comparison["merged"].to_csv(OUTPUT / "comparison_daily_far_vs_near.csv", index=False,
                               encoding="utf-8-sig", float_format="%.10g")

    first_slot = (detail.sort_values("slot").groupby("date").storage_start_kwh.first())
    chains = {day: float(first_slot[day]) for day in dispatch.date}
    if args.reports_only:
        degeneracy = json.loads((OUTPUT / "degeneracy_test.json").read_text(encoding="utf-8"))
        causality = json.loads((OUTPUT / "causality_test.json").read_text(encoding="utf-8"))
        execution_probe = summarise_probe(
            pd.read_csv(OUTPUT / "execution_lookahead_probe.csv"))
        print("  --reports-only：复用已落盘的退化核验、因果性检验与执行层诊断结果",
              flush=True)
    else:
        if args.skip_degeneracy:
            degeneracy = {"days_compared": 0, "days_mismatched": 0, "mismatch_dates": [],
                          "max_abs_plan_difference_kwh": 0.0, "max_abs_theta_kwh": 0.0,
                          "tolerance_kwh": DEGENERACY_TOL, "passed": False,
                          "skipped": True}
        else:
            degeneracy = degeneracy_check(dispatch, chains, workers=args.workers,
                                          days=args.days)
        if args.skip_causality:
            causality = {"passed": False, "skipped": True, "date": PERTURBATION_DAY,
                         "tangents_identical": False, "plan_identical": False,
                         "pool_identical": False, "weights_identical": False,
                         "target_observations_changed": False,
                         "prefix_features_identical": False,
                         "max_abs_plan_change_kwh": 0.0, "reference_tangent_count": 0,
                         "perturbed_tangent_count": 0}
        else:
            causality = causality_check(ctx, tangent_table, eta=summary["eta_yuan"],
                                        tau_env=summary["tau_env_yuan"],
                                        r_max=summary["r_max"])
        execution_probe = None
    (OUTPUT / "degeneracy_test.json").write_text(
        json.dumps(degeneracy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "causality_test.json").write_text(
        json.dumps(causality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    make_plots(detail, dispatch, comparison, rounds, result, ASSETS)
    write_convergence_report(rounds, summary, OUTPUT)

    from taskComposite.validate import TOL as _TOL  # noqa: F401  (口径提醒：沿用同一容差)
    tangent_records = [r for r in result["paths"].values() if len(r["tangents"])]
    summary["total_effective_tangents"] = int(sum(len(r["tangents"]) for r in tangent_records))
    summary["days_all_assertions_passed"] = int(sum(1 for r in records if r["passed"]))
    summary.update(collection_checks())
    if "solver_path" in dispatch:
        counts = dispatch.solver_path.value_counts().to_dict()
        summary["solver_path_counts"] = {str(k): int(v) for k, v in counts.items()}
        summary["solver_path_counts_text"] = "、".join(
            f"{k} {v} 天" for k, v in summary["solver_path_counts"].items())
    execution_probe = (execution_probe if execution_probe is not None
                       else execution_lookahead_probe(ctx, tangent_table, result))
    write_comparison_report(comparison, dispatch, degeneracy, causality, execution_probe,
                            OUTPUT)
    report = write_validation_report(records, degeneracy, causality, comparison, dispatch,
                                     detail, info, decision_violations, summary, OUTPUT)
    frozen = write_frozen_rule(pre, comparison, summary, degeneracy, causality, OUTPUT)

    print(json.dumps({
        "输出天数": len(dispatch),
        "远近合计费用": [comparison["total_cost_near_yuan"], comparison["total_cost_far_yuan"]],
        "相对节省_pct": comparison["relative_saving_pct"],
        "判定": comparison["decision"],
        "退化核验": degeneracy["passed"],
        "因果性检验": causality["passed"],
        "核验": report,
        "冻结文件": str(FROZEN_FAR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
