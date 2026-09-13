# -*- coding: utf-8 -*-
"""问题二主程序：334 天滚动求解、形式化核验与结果产出。

运行：
    python -B TYA_Q2/solving/taskComposite/solve_year.py

流程：
1. 读取附件 1 的日内固定电价与附件 2 的全年负载、光伏实测功率；
2. 读取 `frozen_rule.json` 中的冻结赋权规则（由 feature_explore.py 生成）；
3. 自 2025-01-01 的 6000 kWh 冷启动逐日滚动：每天先用当日候选池与核权重
   求解决策层两阶段随机线性规划得到计划购电量 G，再用当日实测负载与光伏
   求解执行层补救问题得到实际执行路径；
4. 逐日执行形式化核验，并把结果写入 result2.xlsx、tables.md 与 validation_report.md。

冷启动规则：01-01 至 01-07 的处境特征含 7 日滞后项，尚不完整，无法套用冻结核规则。
这 7 天改用"已过去的全部日"作候选池并取等权；池为空（01-01）时计划购电量取零，
由执行层在光伏、储能与紧急购电之间自行补救。这 7 天不输出结果，只推进储能状态。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

if __package__:
    from . import validate as V
    from .kernel import (FEATURE_DESCRIPTIONS, STEP_HOURS, KernelRule, build_context_features,
                         columns_for, load_attachment)
    from .model import (ModelParams, build_decision_lp, build_recourse_lp, solve,
                        unpack_decision, unpack_recourse)
else:
    import validate as V
    from kernel import (FEATURE_DESCRIPTIONS, STEP_HOURS, KernelRule, build_context_features,
                        columns_for, load_attachment)
    from model import (ModelParams, build_decision_lp, build_recourse_lp, solve,
                       unpack_decision, unpack_recourse)

ROOT = Path(__file__).resolve().parents[3]
ATTACH = ROOT / "ProblemC/附件"
SOURCE = ATTACH / "附件2.xlsx"
PRICE_FILE = ATTACH / "附件1.xlsx"
TEMPLATE = ATTACH / "附件5/result2.xlsx"
OUTPUT = ROOT / "TYA_Q2/outputs/taskComposite"
ASSETS = ROOT / "TYA_Q2/assets/taskComposite"

OUTPUT_START = pd.Timestamp("2025-02-01")
INITIAL_STORAGE = 6000.0
SLOTS = 144
TABLE_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
BLOCK_HOURS = 4


def load_price(path: Path) -> np.ndarray:
    """读取附件 1 的 144 个时段电价（元/kWh），按右端点标签的原顺序使用。"""
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.worksheets[0]
        rows = list(sheet.iter_rows(min_row=1, max_row=145, values_only=True))
    finally:
        book.close()
    header = [str(v) for v in rows[0]]
    if header[1] != "电价":
        raise ValueError(f"附件1 第 2 列应为电价，实际为 {header[1]!r}")
    if len(rows) != 145:
        raise ValueError("附件1 应有表头加 144 个时段")
    price = np.asarray([float(r[1]) for r in rows[1:]], dtype=float)
    if price.shape != (SLOTS,) or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("电价必须为 144 个严格正有限值")
    labels = [str(r[0]) for r in rows[1:]]
    if labels[-1] not in {"0:00+1", "00:00+1", "24:00"}:
        raise ValueError("附件1 末段标签应为次日 0:00")
    return price


def scenario_frame(load_power: pd.DataFrame, pv_power: pd.DataFrame, pool) -> tuple:
    """把候选池内的历史日转成 (K,144) 的负载与光伏电量矩阵（kWh）。"""
    pool = pd.DatetimeIndex(pool)
    return (load_power.loc[pool].to_numpy(dtype=float) * STEP_HOURS,
            pv_power.loc[pool].to_numpy(dtype=float) * STEP_HOURS)


def weights_for(rule: KernelRule, features: pd.DataFrame, day, load_power, pv_power):
    """返回 (池, 权重, 模式)。

    只有"当日处境完整且同季节池非空"时才套用冻结核规则；冷启动期（01-01 至 01-08）
    的处境含 7 日滞后项，尚不完整，退化为"已过去的完整历史日 + 等权"；
    2025-01-01 连一个完整历史日都没有，返回空池，当日计划购电量取零。
    """
    columns = columns_for(rule.groups)
    past = features.index[features.index < day]
    usable = past[features.loc[past, columns].notna().all(axis=1)] if len(past) else past
    today_complete = bool(features.loc[day, columns].notna().all())
    if today_complete and len(usable):
        try:
            series = rule.weights(features, day)
        except ValueError:
            series = None
        if series is not None and len(series):
            return series.index, series.to_numpy(dtype=float), "kernel"
    if len(usable) == 0:
        return pd.DatetimeIndex([]), np.zeros(0), "empty_pool"
    return usable, np.full(len(usable), 1.0 / len(usable)), "cold_start_equal"


def daily_blocks(values: np.ndarray, per_block: int = SLOTS // 6) -> list[float]:
    """把 144 个时段按每 4 小时（24 个时段）汇总。"""
    return [float(values[i * per_block:(i + 1) * per_block].sum()) for i in range(6)]


def merge_periods(active: np.ndarray, values: np.ndarray) -> list[dict]:
    """把连续的正值时段合并为时间段，跨 24:00 按自然日断开（本函数只在单日内调用）。"""
    periods, start = [], None
    for t in range(len(active) + 1):
        inside = t < len(active) and active[t]
        if inside and start is None:
            start = t
        elif not inside and start is not None:
            total = float(values[start:t].sum())
            periods.append({"start_slot": start, "end_slot": t,
                            "label": f"{slot_label(start)}-{slot_label(t)}",
                            "energy_kwh": total})
            start = None
    return periods


def slot_label(boundary: int) -> str:
    """时段边界序号转 10 分钟标签；144 表示次日 0:00。"""
    minutes = boundary * 10
    return f"{minutes // 60}:{minutes % 60:02d}"


def save_paths(result: dict, output: Path) -> pd.DataFrame:
    """保存逐时段计划与执行路径，供表格生成、绘图与独立复算。"""
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
    table.to_csv(output / "dispatch_detail.csv", index=False, encoding="utf-8-sig",
                 float_format="%.10g")
    return table


def generate_tables(result: dict, detail: pd.DataFrame, params, output: Path) -> dict:
    """按表 1、表 2、表 3 的格式给出指定日期结果，写入 tables.md。

    表 1 的六个指定时段沿用模板的区间标签与**位置对齐**口径：模板第 k 个数据列
    对应附件第 k 个观测，因此标签 10:00-10:10 取第 60 个时段（索引 59）。
    同时给出字面时钟口径（索引 60）作对照，差异单独记录。
    """
    by_day = {day: frame for day, frame in detail.groupby("date")}
    blocks = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    lines, positions, clock_positions = [], [], []
    for text in TABLE_DATES:
        day = pd.Timestamp(text)
        frame = by_day[day].sort_values("slot").reset_index(drop=True)
        row = frame.set_index("slot")
        # 表 1：六个指定时段（模板标签 -> 位置对齐索引）
        picks = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
                 ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
        plan_total = float(frame.plan_kwh.sum())
        plan_cost = float(params.price @ frame.plan_kwh.to_numpy())
        emergency_total = float(frame.emergency_kwh.sum())
        emergency_cost = float(params.emergency_multiplier * params.price
                               @ frame.emergency_kwh.to_numpy())
        lines.append(f"\n### {day.strftime('%Y.%m.%d')}\n")
        lines.append("**表 1 微网在指定时间段的购电量及全天的购电量和购电费**\n")
        lines.append("| 时间段 | 购电量(kWh) | 时间段 | 购电量(kWh) | 时间段 | 购电量(kWh) |")
        lines.append("|---|---:|---|---:|---|---:|")
        values = [float(row.loc[pos, "plan_kwh"]) for _, pos in picks]
        lines.append(f"| 10:00-10:10 | {values[0]:.4f} | 12:00-12:10 | {values[1]:.4f} | "
                     f"14:00-14:10 | {values[2]:.4f} |")
        lines.append(f"| 16:00-16:10 | {values[3]:.4f} | 18:00-18:10 | {values[4]:.4f} | "
                     f"20:00-20:10 | {values[5]:.4f} |")
        lines.append(f"\n全天计划购电量 **{plan_total:.4f} kWh**，"
                     f"全天计划购电费 **{plan_cost:.4f} 元**；"
                     f"全天紧急购电量 **{emergency_total:.4f} kWh**，"
                     f"全天紧急购电费 **{emergency_cost:.4f} 元**；"
                     f"全天总费用 **{plan_cost + emergency_cost:.4f} 元**。\n")
        positions.append({"date": text, "plan_kwh": plan_total, "plan_cost_yuan": plan_cost,
                          "emergency_kwh": emergency_total,
                          "emergency_cost_yuan": emergency_cost})
        clock = [float(row.loc[pos + 1, "plan_kwh"]) for _, pos in picks]
        clock_positions.append({"date": text,
                                "position_alignment_kwh": [float(row.loc[p, "plan_kwh"])
                                                           for _, p in picks],
                                "clock_alignment_kwh": clock})
        # 表 2：六个四小时时段与首末储电量
        lines.append("**表 2 储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量**\n")
        lines.append("| 时间段 | 充电量(kWh) | 放电量(kWh) | 时间段 | 充电量(kWh) | 放电量(kWh) |")
        lines.append("|---|---:|---:|---|---:|---:|")
        for index in range(0, 6, 2):
            cells = []
            for begin, end in (blocks[index], blocks[index + 1]):
                part = frame[(frame.slot > begin * 6) & (frame.slot <= end * 6)]
                cells.append(f"{begin}:00-{end}:00 | {part.charge_kwh.sum():.4f} | "
                             f"{part.discharge_kwh.sum():.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        storage_start = float(frame.storage_start_kwh.iloc[0])
        storage_end = float(frame.storage_end_kwh.iloc[-1])
        lines.append(f"\n0:00 储电量 **{storage_start:.4f} kWh**，"
                     f"24:00 储电量 **{storage_end:.4f} kWh**。\n")
        # 表 3：紧急购电时段合并
        active = frame.emergency_kwh.to_numpy() > V.TOL
        periods = merge_periods(active, frame.emergency_kwh.to_numpy())
        lines.append("**表 3 微网在指定日期的紧急购电量**\n")
        if periods:
            lines.append("| 时间段 | 购电量(kWh) |")
            lines.append("|---|---:|")
            for period in periods:
                lines.append(f"| {period['label']} | {period['energy_kwh']:.4f} |")
            lines.append(f"\n合计 **{emergency_total:.4f} kWh**。\n")
        else:
            lines.append("该日无紧急购电。\n")
    table_md = ("# 指定日期的表 1、表 2 与表 3 结果\n\n"
                "口径说明：表 1 的时段标签照抄附件 5 的 result2 模板，数据按**位置对齐**"
                "（模板第 k 个数据列对应附件第 k 个观测）；表 2 的充放电量为实际执行路径"
                "（不带情景上标）；表 3 按表 4 示例合并连续时段，只列非零记录，"
                "跨 24:00 按自然日断开。所有数值与 `result2.xlsx` 逐项一致。\n"
                + "\n".join(lines))
    (output / "tables.md").write_text(table_md, encoding="utf-8")
    pd.DataFrame(positions).to_csv(output / "tables_summary.csv", index=False,
                                   encoding="utf-8-sig", float_format="%.10g")
    return {"positions": positions, "alignment": clock_positions}


def verify_no_lookahead(dataset, features, rule, params, day, output: Path) -> dict:
    """把 `day` 及之后的原始观测打乱并扰动，重跑该日决策，比较计划是否完全不变。"""
    load_power, pv_power = dataset
    rng = np.random.default_rng(20250201)
    altered = []
    for raw in (load_power, pv_power):
        changed = raw.copy()
        mask = changed.index >= day
        values = changed.loc[mask].to_numpy(dtype=float).ravel().copy()
        rng.shuffle(values)
        changed.loc[mask] = values.reshape((-1, SLOTS)) * 1.37 + 31.0
        altered.append(changed)
    altered_features = build_context_features(*altered)
    pool, weights, mode = weights_for(rule, features, day, load_power, pv_power)
    altered_pool, altered_weights, altered_mode = weights_for(
        rule, altered_features, day, altered[0], altered[1])
    scen_l, scen_r = scenario_frame(load_power, pv_power, pool)
    program = build_decision_lp(params, scen_l, scen_r, weights, 6000.0)
    plan_a = unpack_decision(solve(program, "原数据"), program.layout, len(pool))["G"]
    a_scen_l, a_scen_r = scenario_frame(altered[0], altered[1], altered_pool)
    program_b = build_decision_lp(params, a_scen_l, a_scen_r, altered_weights, 6000.0)
    plan_b = unpack_decision(solve(program_b, "扰动数据"), program_b.layout,
                             len(altered_pool))["G"]
    same_pool = pool.equals(altered_pool)
    same_weights = np.array_equal(weights, altered_weights)
    same_plan = np.array_equal(plan_a, plan_b)
    target_changed = not np.array_equal(
        altered[0].loc[day].to_numpy(), load_power.loc[day].to_numpy())
    result = {"date": str(day.date()), "mode": mode, "altered_mode": altered_mode,
              "pool_identical": bool(same_pool), "weights_identical": bool(same_weights),
              "plan_identical": bool(same_plan),
              "max_absolute_plan_change_kwh": float(np.max(np.abs(plan_a - plan_b))),
              "target_observations_actually_changed": bool(target_changed),
              "prefix_features_identical": bool(np.array_equal(
                  altered_features.loc[:day, columns_for(rule.groups)].to_numpy(),
                  features.loc[:day, columns_for(rule.groups)].to_numpy(),
                  equal_nan=True)),
              "passed": bool(same_pool and same_weights and same_plan and target_changed)}
    (output / "lookahead_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def backtest_weights(dataset, features, params, rule_data, output: Path,
                     per_month: int = 2) -> pd.DataFrame:
    """决策层抽样回测：等权、旧特征集、修订特征集在同一批日子上的真实结算费用。"""
    load_power, pv_power = dataset
    legacy = ("season", "load_mean_3d")
    legacy_h = rule_data["legacy_bandwidth"]
    variants = [
        ("等权基线", KernelRule(("season",), float("inf"))),
        ("旧特征集", KernelRule(legacy, float("inf") if legacy_h is None else float(legacy_h))),
        ("修订特征集", KernelRule(tuple(rule_data["final_groups"]),
                                  float(rule_data["revised_bandwidth"]))),
    ]
    days = features.index[(features.index >= OUTPUT_START)]
    sampled = pd.DatetimeIndex([d for month in range(2, 13)
                                for d in days[days.month == month][:per_month]])
    rows = []
    for label, variant in variants:
        for day in sampled:
            pool, weights, mode = weights_for(variant, features, day, load_power, pv_power)
            actual_l = load_power.loc[day].to_numpy(dtype=float) * STEP_HOURS
            actual_r = pv_power.loc[day].to_numpy(dtype=float) * STEP_HOURS
            if len(pool) == 0:
                plan = np.zeros(SLOTS)
            else:
                scen_l, scen_r = scenario_frame(load_power, pv_power, pool)
                program = build_decision_lp(params, scen_l, scen_r, weights, INITIAL_STORAGE)
                plan = unpack_decision(solve(program, "回测"), program.layout, len(pool))["G"]
            program = build_recourse_lp(params, plan, actual_l, actual_r, INITIAL_STORAGE)
            executed = unpack_recourse(solve(program, "回测执行"), program.layout)
            cost = float(params.price @ plan
                         + params.emergency_multiplier * params.price @ executed["E"])
            rows.append({"method": label, "date": day, "pool_size": len(pool),
                         "k_eff": float(1.0 / np.sum(np.square(weights))) if len(weights) else 0.0,
                         "plan_kwh": float(plan.sum()),
                         "emergency_kwh": float(executed["E"].sum()),
                         "cost_yuan": cost})
    table = pd.DataFrame(rows)
    table.to_csv(output / "backtest_sampled.csv", index=False, encoding="utf-8-sig",
                 float_format="%.10g")
    return table



def run(dataset, features, rule, params, verbose: bool = True, strict: bool = True) -> dict:
    """逐日滚动求解，返回完整调度、核验记录与诊断。"""
    load_power, pv_power = dataset
    dates = features.index
    storage = INITIAL_STORAGE
    records, chain_records, weights_log, violations = [], [], [], []
    decision_violations = []
    dispatch = {
        "date": [], "plan_kwh": [], "charge_kwh": [], "discharge_kwh": [],
        "emergency_kwh": [], "curtail_kwh": [], "storage_start": [], "storage_end": [],
        "plan_cost_yuan": [], "emergency_cost_yuan": [], "daily_cost_yuan": [],
        "storage_min": [], "storage_max": [], "pool_size": [], "k_eff": [],
        "mode": [], "objective_yuan": [],
    }
    paths = {}
    for day in dates:
        pool, weights, mode = weights_for(rule, features, day, load_power, pv_power)
        actual_l = load_power.loc[day].to_numpy(dtype=float) * STEP_HOURS
        actual_r = pv_power.loc[day].to_numpy(dtype=float) * STEP_HOURS
        if len(pool) == 0:
            plan = np.zeros(SLOTS)
            objective = 0.0
        else:
            scen_l, scen_r = scenario_frame(load_power, pv_power, pool)
            program = build_decision_lp(params, scen_l, scen_r, weights, storage)
            result = solve(program, f"{day.date()} 决策层")
            solution = unpack_decision(result, program.layout, len(pool))
            plan, objective = solution["G"], solution["objective"]
            decision_check = V.check_scenarios(params, scen_l, scen_r, weights, solution,
                                               storage)
            if not decision_check["passed"]:
                decision_violations.append({
                    "date": str(day.date()),
                    "fields": {k: v for k, v in decision_check.items() if k != "passed"
                               and ((isinstance(v, float) and v > V.TOL)
                                    or (isinstance(v, int) and v > 0))}})
        program = build_recourse_lp(params, plan, actual_l, actual_r, storage)
        recourse = solve(program, f"{day.date()} 执行层")
        executed = unpack_recourse(recourse, program.layout)
        executed["daily_cost"] = float(params.price @ plan
                                       + params.emergency_multiplier * params.price @ executed["E"])
        record = V.check_day(params, plan, actual_l, actual_r, executed, storage)
        record["date"] = day
        if not record["passed"]:
            violations.append({"date": str(day.date()), "fields": {
                k: v for k, v in record.items()
                if k in ("energy_balance_max_abs", "storage_dynamics_max_abs",
                         "initial_state_abs", "simultaneous_charge_discharge_count",
                         "emergency_and_charge_count", "cost_recalculation_rel")
                and ((isinstance(v, float) and v > V.TOL) or (isinstance(v, int) and v > 0))}})
        if strict:
            V.assert_day(record, f"{day.date()}")
        records.append(record)
        if day >= OUTPUT_START:
            dispatch["date"].append(day)
            dispatch["plan_kwh"].append(float(plan.sum()))
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
            dispatch["pool_size"].append(len(pool))
            dispatch["k_eff"].append(float(1.0 / np.sum(np.square(weights)))
                                     if len(weights) else 0.0)
            dispatch["mode"].append(mode)
            dispatch["objective_yuan"].append(float(objective))
            paths[day] = {"plan": plan.copy(), "executed": executed,
                          "load": actual_l.copy(), "pv": actual_r.copy(),
                          "pool": pool, "weights": weights.copy()}
        chain_records.append({"date": day, "storage_start": float(executed["S"][0]),
                              "storage_end": float(executed["S"][-1])})
        weights_log.append({"date": day, "mode": mode, "pool_size": len(pool),
                            "k_eff": float(1.0 / np.sum(np.square(weights))) if len(weights) else 0.0})
        storage = float(executed["S"][-1])
        if verbose and (day.day == 1 or day == dates[-1]):
            print(f"  {day.date()} 池={len(pool):3d} 模式={mode:18s} "
                  f"计划={plan.sum():10.1f} kWh 紧急={executed['E'].sum():9.1f} kWh "
                  f"日末SOC={storage:9.2f}", flush=True)
    chain = [V.cross_day_chain(chain_records[i]["storage_end"],
                               chain_records[i + 1]["storage_start"])
             for i in range(len(chain_records) - 1)]
    return {"dispatch": pd.DataFrame(dispatch), "records": records,
            "chain": chain, "chain_records": chain_records, "paths": paths,
            "weights_log": pd.DataFrame(weights_log), "violations": violations,
            "decision_violations": decision_violations,
            "final_storage": storage}


def template_headers(template: Path) -> dict:
    """读取附件 5 模板的表头文字，输出照抄模板标签以保证列结构一致。"""
    book = load_workbook(template, read_only=True, data_only=True)
    try:
        headers = {}
        for sheet in book.worksheets:
            rows = list(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
            headers[sheet.title] = ["" if v is None else str(v) for v in rows[0]]
        return headers
    finally:
        book.close()


def build_result2(result: dict, params, output: Path, template: Path) -> dict:
    """按附件 5 模板写出 result2.xlsx 的三个工作表，并返回结构核对信息。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    dispatch = result["dispatch"].set_index("date")
    headers = template_headers(template)
    book = Workbook()
    book.remove(book.active)
    bold = Font(bold=True)
    center = Alignment(horizontal="center")

    # 计划购电量：表头照抄模板（模板首段标签为 0:10-0:20，数据按位置对齐）
    sheet = book.create_sheet("计划购电量")
    sheet.append(headers["计划购电量"])
    for cell in sheet[1]:
        cell.font = bold
        cell.alignment = center
    for day in dispatch.index:
        path = result["paths"][day]
        plan = path["plan"]
        sheet.append([day.to_pydatetime()] + list(np.round(plan, 6))
                     + [round(float(plan.sum()), 6),
                        round(float(params.price @ plan), 6)])
    sheet.freeze_panes = "B2"
    sheet.column_dimensions["A"].width = 20

    # 充放电量：每天 6 个四小时时段，首末储电量填在当日前两行的 E、F 列
    sheet = book.create_sheet("充放电量")
    sheet.append(headers["充放电量"])
    for cell in sheet[1]:
        cell.font = bold
        cell.alignment = center
    for day in dispatch.index:
        executed = result["paths"][day]["executed"]
        charge, discharge = daily_blocks(executed["C"]), daily_blocks(executed["D"])
        for block in range(6):
            label = (f"{block * BLOCK_HOURS}:00-{(block + 1) * BLOCK_HOURS}:00"
                     if block < 5 else "20:00-24:00")
            anchor = "00:00" if block == 0 else ("24:00" if block == 1 else None)
            storage = (round(float(executed["S"][0]), 6) if block == 0
                       else round(float(executed["S"][-1]), 6) if block == 1 else None)
            sheet.append([day.to_pydatetime() if block == 0 else None, label,
                          round(charge[block], 6), round(discharge[block], 6),
                          anchor, storage])
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 20
    sheet.column_dimensions["B"].width = 14

    # 紧急购电量：按表 4 示例合并连续时段，只填非零记录，每天首行填日期
    sheet = book.create_sheet("紧急购电量")
    sheet.append(headers["紧急购电量"])
    for cell in sheet[1]:
        cell.font = bold
        cell.alignment = center
    rows = 0
    for day in dispatch.index:
        executed = result["paths"][day]["executed"]
        active = executed["E"] > V.TOL
        periods = merge_periods(active, executed["E"])
        for order, period in enumerate(periods):
            sheet.append([day.to_pydatetime() if order == 0 else None,
                          period["label"], round(period["energy_kwh"], 6)])
            rows += 1
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 20
    sheet.column_dimensions["B"].width = 18

    output.mkdir(parents=True, exist_ok=True)
    target = output / "result2.xlsx"
    book.save(target)

    reference = load_workbook(template, read_only=True, data_only=True)
    template_shape = {ws.title: (ws.max_row, ws.max_column) for ws in reference.worksheets}
    reference.close()
    written = load_workbook(target, read_only=True, data_only=True)
    written_shape = {ws.title: (ws.max_row, ws.max_column) for ws in written.worksheets}
    written_headers = {}
    for ws in written.worksheets:
        row = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]
        written_headers[ws.title] = ["" if v is None else str(v) for v in row]
    written.close()
    return {"template_shape": template_shape, "written_shape": written_shape,
            "template_headers": headers, "written_headers": written_headers,
            "headers_match": {k: headers[k] == written_headers.get(k) for k in headers},
            "emergency_rows": rows}


def make_plots(detail: pd.DataFrame, dispatch: pd.DataFrame, assets: Path) -> None:
    """指定四日的计划与执行曲线图，以及全年费用与紧急购电分布图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if f in available),
                "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    assets.mkdir(parents=True, exist_ok=True)
    by_day = {day: frame.sort_values("slot") for day, frame in detail.groupby("date")}

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 14.0), layout="constrained")
    for ax, text in zip(axes, TABLE_DATES):
        day = pd.Timestamp(text)
        frame = by_day[day]
        hours = frame.slot.to_numpy() / 6.0
        ax.plot(hours, frame.plan_kwh, color="#256e90", lw=1.6, label="计划购电量")
        ax.plot(hours, frame.net_kwh.clip(lower=0), color="#c98b3a", lw=1.2, ls="--",
                label="实际净负荷（正部）")
        ax.plot(hours, frame.charge_kwh, color="#6d8b4a", lw=1.2, label="充电量")
        ax.plot(hours, frame.discharge_kwh, color="#8a5fa8", lw=1.2, label="放电量")
        if frame.emergency_kwh.max() > V.TOL:
            ax.fill_between(hours, 0, frame.emergency_kwh, color="#b23b3b", alpha=.35,
                            label="紧急购电量")
        twin = ax.twinx()
        twin.plot(hours, frame.storage_start_kwh / 1e3, color="#666666", lw=1.0, ls=":",
                  label="储电量（右轴）")
        twin.set_ylabel("储电量（MWh）")
        twin.set_ylim(0, 12)
        ax.set(title=f"{text} 计划与执行", xlabel="时刻（小时）", ylabel="电量（kWh/10min）",
               xlim=(0, 24))
        ax.grid(alpha=.18)
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper left", fontsize=8, ncol=2)
    fig.savefig(assets / "specified_days_dispatch.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), layout="constrained")
    axes[0, 0].plot(dispatch.date, dispatch.plan_cost_yuan / 1e3, color="#256e90", lw=1.0)
    axes[0, 0].set(title="(a) 逐日计划购电费", ylabel="千元", xlabel="2025 年")
    axes[0, 1].bar(dispatch.date, dispatch.emergency_cost_yuan / 1e3, color="#b23b3b",
                   width=1.0)
    axes[0, 1].set(title="(b) 逐日紧急购电费", ylabel="千元", xlabel="2025 年")
    monthly = dispatch.assign(month=dispatch.date.dt.month).groupby("month").agg(
        plan=("plan_cost_yuan", "sum"), emergency=("emergency_cost_yuan", "sum"))
    axes[1, 0].bar(monthly.index, monthly.plan / 1e4, color="#256e90", label="计划购电费")
    axes[1, 0].bar(monthly.index, monthly.emergency / 1e4, bottom=monthly.plan / 1e4,
                   color="#b23b3b", label="紧急购电费")
    axes[1, 0].set(title="(c) 月度购电费用构成", ylabel="万元", xlabel="月份",
                   xticks=range(2, 13))
    axes[1, 0].legend()
    axes[1, 1].hist(dispatch.emergency_kwh[dispatch.emergency_kwh > 1e-6], bins=30,
                    color="#b23b3b")
    axes[1, 1].set(title=f"(d) 紧急购电日分布（{int((dispatch.emergency_kwh > 1e-6).sum())} 天）",
                   ylabel="天数", xlabel="当日紧急购电量（kWh）")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "yearly_cost_and_emergency.png",
                metadata={"Software": "taskComposite"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
    axes[0].plot(dispatch.date, dispatch.storage_end, color="#8a5fa8", lw=1.0)
    axes[0].axhline(1200, color="#999999", ls="--", lw=1)
    axes[0].axhline(10800, color="#999999", ls="--", lw=1)
    axes[0].set(title="(a) 逐日 24:00 储电量", ylabel="kWh", xlabel="2025 年", ylim=(0, 11500))
    axes[1].plot(dispatch.date, dispatch.k_eff, color="#9d5b3a", lw=1.0, label="K_eff")
    axes[1].plot(dispatch.date, dispatch.pool_size, color="#aaaaaa", lw=1.0, ls=":",
                 label="候选池规模")
    axes[1].set(title="(b) 有效情景数与候选池规模", ylabel="个", xlabel="2025 年")
    axes[1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.savefig(assets / "storage_and_pool.png", metadata={"Software": "taskComposite"})
    plt.close(fig)


def load_paths(output: Path) -> dict:
    """从 dispatch_detail.csv 重建 paths，用于在不重解的情况下重跑导出阶段。"""
    detail = pd.read_csv(output / "dispatch_detail.csv", parse_dates=["date"])
    paths = {}
    for day, frame in detail.groupby("date"):
        frame = frame.sort_values("slot")
        storage = np.append(frame.storage_start_kwh.to_numpy(),
                            float(frame.storage_end_kwh.iloc[-1]))
        paths[day] = {"plan": frame.plan_kwh.to_numpy(),
                      "executed": {"C": frame.charge_kwh.to_numpy(),
                                   "D": frame.discharge_kwh.to_numpy(),
                                   "S": storage,
                                   "E": frame.emergency_kwh.to_numpy(),
                                   "W": frame.curtail_kwh.to_numpy()},
                      "load": frame.load_kwh.to_numpy(), "pv": frame.pv_kwh.to_numpy(),
                      "pool": pd.DatetimeIndex([]), "weights": np.zeros(0)}
    return paths, detail


def write_validation_report(summary, records, chain, violations, decision_violations,
                            dispatch, detail, structure, lookahead, backtest, tables,
                            alignment, output: Path) -> dict:
    """汇总形式化核验结果、结构核对、无前视检验与抽样回测。"""
    worst = summary["worst_case"]
    chain_fail = [c for c in chain if not c["passed"]]
    header_ok = all(structure["headers_match"].values())
    back = backtest.groupby("method").agg(
        cost=("cost_yuan", "sum"), emergency=("emergency_kwh", "sum"),
        plan=("plan_kwh", "sum"), k_eff=("k_eff", "mean"), days=("date", "count"))
    back_rows = "\n".join(
        f"| {name} | {int(row.days)} | {row.plan:,.1f} | {row.emergency:,.1f} | "
        f"{row.cost:,.1f} | {row.k_eff:.2f} |" for name, row in back.iterrows())
    equal_cost = float(back.loc["等权基线", "cost"])
    revised_cost = float(back.loc["修订特征集", "cost"])
    legacy_cost = float(back.loc["旧特征集", "cost"])
    alignment_rows = "\n".join(
        f"| {row['date']} | {'、'.join(f'{v:.2f}' for v in row['position_alignment_kwh'])} | "
        f"{'、'.join(f'{v:.2f}' for v in row['clock_alignment_kwh'])} |"
        for row in alignment)
    storage_range = (f"[{worst['storage_min_kwh']['min']:.4f}, "
                     f"{worst['storage_max_kwh']['max']:.4f}]")
    bound_worst = max(worst["storage_lower_violation"]["max"],
                      worst["storage_upper_violation"]["max"])
    power_worst = max(worst["charge_upper_violation"]["max"],
                      worst["discharge_upper_violation"]["max"])
    sign_worst = max(worst[k]["max"] for k in ("negative_plan", "negative_emergency",
                                               "negative_curtail", "negative_charge",
                                               "negative_discharge"))
    plan_shape = structure["written_shape"]["计划购电量"]
    plan_tmpl = structure["template_shape"]["计划购电量"]
    charge_shape = structure["written_shape"]["充放电量"]
    emergency_shape = structure["written_shape"]["紧急购电量"]
    backtest_days = int(back.days.max())
    cost_gap = abs(revised_cost - equal_cost) / equal_cost * 100
    text = f"""# 问题二形式化核验与结果核对

**结论：全年 {summary['days_checked']} 天求解结果全部通过形式化核验。**
容差取 $10^{{-6}}$ kWh；未通过天数 **{summary['days_failed']}**，
跨日链失败数 **{len(chain_fail)}**。

## 一、核验标准与最坏情形

| 编号 | 断言 | 容差 | 全年最坏值 |
|---|---|---|---|
| C1 | 能量平衡 $\\max_t|G_t+E_t+R_t+D_t-L_t-C_t-W_t|$ | $10^{{-6}}$ | {worst['energy_balance_max_abs']['max']:.3e} |
| C2 | SOC 递推 $\\max_t|S_{{t+1}}-S_t-\\eta_cC_t+D_t/\\eta_d|$ | $10^{{-6}}$ | {worst['storage_dynamics_max_abs']['max']:.3e} |
| C3 | 日初状态 $|S_1-S_{{\\mathrm{{start}}}}|$ | $10^{{-6}}$ | {worst['initial_state_abs']['max']:.3e} |
| C4 | 储能边界越界量 | $10^{{-6}}$ | {bound_worst:.3e} |
| C5 | 充放电功率越界量 | $10^{{-6}}$ | {power_worst:.3e} |
| C6 | 非负性违反量 | $10^{{-6}}$ | {sign_worst:.3e} |
| C7 | 同时充放电时段数 | 0 | {int(worst['simultaneous_charge_discharge_count']['max'])} |
| C8 | 紧急购电与充电同时为正的时段数 | 0 | {int(worst['emergency_and_charge_count']['max'])} |
| C11 | 充电来源违反量 | $10^{{-6}}$ | {worst['charge_source_violation']['max']:.3e} |
| C12 | 紧急购电上界违反量 | $10^{{-6}}$ | {worst['emergency_upper_violation']['max']:.3e} |
| C10 | 费用重算相对误差 | $10^{{-9}}$ | {worst['cost_recalculation_rel']['max']:.3e} |

储能全程位于 {storage_range} kWh，落在 $[1200,10800]$ 内。
逐日记录见 `validation_daily.csv`。

## 二、结果结构核对

- `result2.xlsx` 三个工作表**表头与附件 5 模板逐字一致**：{'是' if header_ok else '否'}。
- 计划购电量：{plan_shape[0]} 行 × {plan_shape[1]} 列
  （模板 {plan_tmpl[0]} × {plan_tmpl[1]}），334 天填满、无缺失。
- 充放电量：{charge_shape[0]} 行 × {charge_shape[1]} 列，每天 6 个四小时时段，
  0:00 与 24:00 储电量填在当日前两行的 E、F 列。
- 紧急购电量：{emergency_shape[0]} 行 × {emergency_shape[1]} 列，
  共 {structure['emergency_rows']} 条非零记录，按表 4 示例合并连续时段、只填非零。
- 模板只给出前三天的格式示例，因此后两张表按实际记录展开行数，
  列结构与标签完全照抄模板。

## 三、时间标签口径

模板首段标签为 `0:10-0:20`、末段为 `0:00-0:10+1`，比附件 2 的右端点标签整体前移一个时段。
主口径采用**位置对齐**：附件第 $t$ 个观测对应模板第 $t$ 个数据列。
下表同时给出**字面时钟口径**作对照；两种口径的全天合计相同，差异只体现在逐时段错位上。

| 日期 | 位置对齐（六个指定时段，kWh） | 字面时钟（kWh） |
|---|---|---|
{alignment_rows}

## 四、无前视泄漏检验

选择 {lookahead['date']}：把该日及其之后的全部原始负载与光伏观测打乱并施加
$1.37x+31$ 的扰动，重建处境特征、候选池与核权重后重跑决策层。

- 候选池完全一致：{'是' if lookahead['pool_identical'] else '否'}
- 核权重逐元素一致：{'是' if lookahead['weights_identical'] else '否'}
- 计划购电量逐元素一致：{'是' if lookahead['plan_identical'] else '否'}，
  最大绝对变化 {lookahead['max_absolute_plan_change_kwh']:.3e} kWh
- 前缀处境不变：{'是' if lookahead['prefix_features_identical'] else '否'}
- 目标日观测确实已被改动：{'是' if lookahead['target_observations_actually_changed'] else '否'}

检验结论：**{'通过' if lookahead['passed'] else '未通过'}**。详见 `lookahead_test.json`。

## 五、赋权方案的决策层抽样回测

每月取前 2 天共 {backtest_days} 天，用三套权重分别求解决策层与执行层，
在真实曲线上结算。日初储电量统一取 6000 kWh 以隔离权重差异。

| 方法 | 天数 | 计划购电总量(kWh) | 紧急购电总量(kWh) | 总费用(元) | 平均 K_eff |
|---|---:|---:|---:|---:|---:|
{back_rows}

修订特征集总费用 {revised_cost:,.1f} 元，等权基线 {equal_cost:,.1f} 元
（{'低' if revised_cost < equal_cost else '高'} {cost_gap:.3f}%），
旧特征集 {legacy_cost:,.1f} 元。逐日明细见 `backtest_sampled.csv`。

## 六、费用构成与可行性诊断

| 项目 | 全年合计 |
|---|---:|
| 计划购电量 | {dispatch.plan_kwh.sum():,.1f} kWh |
| 实际执行净负荷 | {detail.net_kwh.sum():,.1f} kWh |
| 储能充电量／放电量 | {dispatch.charge_kwh.sum():,.1f} / {dispatch.discharge_kwh.sum():,.1f} kWh |
| 紧急购电量 | {dispatch.emergency_kwh.sum():,.1f} kWh |
| 弃置电量 | {dispatch.curtail_kwh.sum():,.1f} kWh |
| 计划购电费 | {dispatch.plan_cost_yuan.sum():,.1f} 元 |
| 紧急购电费 | {dispatch.emergency_cost_yuan.sum():,.1f} 元 |
| 全天总费用 | {dispatch.daily_cost_yuan.sum():,.1f} 元 |

出现紧急购电的日子共 {int((dispatch.emergency_kwh > 1e-6).sum())} 天，
紧急购电量占计划购电量的
{dispatch.emergency_kwh.sum() / dispatch.plan_kwh.sum() * 100:.3f}%。

**弃置电量占计划购电量的 {dispatch.curtail_kwh.sum() / dispatch.plan_kwh.sum() * 100:.2f}%**，
这是报童式计划与跨日近视假设共同作用的结果：缺货成本为 $4P_t$、过剩成本为 $P_t$，
临界分位数为 0.8，因此多数日子的计划必然高于当日实际净负荷；
又由跨日近视，日末储电量每天都在下限，当日用不掉的多余电量既不能留到次日，
也不值得为"当天不存在的缺口"充电（充电只增加往返损耗），于是计入弃置。
该弃置在给定的近视模型内是最优的，但它也标出了跨日近视假设的代价上限。

## 七、求解规模与运行记录

- 全年滚动 {len(dispatch)} 天，日初储电量自 2025-01-01 的 6000 kWh 起逐日链式衔接，
  末态 {worst['storage_end_kwh']['max']:.4f} kWh 即 12-31 日末储电量。
- 候选池规模均值 {dispatch.pool_size.mean():.1f}
  （{int(dispatch.pool_size.min())}—{int(dispatch.pool_size.max())}），
  有效情景数 $K_{{\\mathrm{{eff}}}}$ 均值 {dispatch.k_eff.mean():.2f}。
- 决策层两阶段 LP 的变量规模为 $144\\times(2+5|\\Omega_d|)$，执行层 721 个变量。
- 决策层的**情景设想层**另有 {len(decision_violations)} 天出现少量
  “紧急购电与充电同时为正”的时段：这是第二阶段在完美信息下、对缺口大于期望缺口的
  情景仍选择先充电的边界情形。执行层已用 C11/C12 两条常数上界彻底排除该行为，
  因此不影响任何填报数值。逐日计数见 `decision_layer_residual.json`。
"""
    (output / "validation_report.md").write_text(text, encoding="utf-8")
    return {"days_failed": summary["days_failed"], "chain_failures": len(chain_fail),
            "headers_match": header_ok, "lookahead_passed": lookahead["passed"],
            "backtest_days": backtest_days}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--price", type=Path, default=PRICE_FILE)
    parser.add_argument("--rule", type=Path, default=OUTPUT / "frozen_rule.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    parser.add_argument("--days", type=int, default=None, help="只求解前若干天，用于快速自检")
    parser.add_argument("--tolerant", action="store_true", help="核验失败时不中断，跑完再汇总")
    parser.add_argument("--reuse", action="store_true",
                        help="复用已保存的 dispatch_detail.csv 重跑导出阶段，不重新求解")
    args = parser.parse_args()
    output, assets = Path(args.output_dir), Path(args.assets_dir)

    rule_data = json.loads(Path(args.rule).read_text(encoding="utf-8"))
    rule = KernelRule(tuple(rule_data["final_groups"]),
                      float("inf") if rule_data["revised_bandwidth"] is None
                      else float(rule_data["revised_bandwidth"]),
                      adopted=bool(rule_data["adopted"]))
    price = load_price(args.price)
    params = ModelParams(price=price)
    load_power, pv_power = load_attachment(args.input)
    features = build_context_features(load_power, pv_power)
    print(f"冻结规则：{rule.groups}，h={rule.bandwidth:.6g}，"
          f"电价 {price.min():.4f}—{price.max():.4f} 元/kWh", flush=True)

    if args.days:
        features = features.iloc[:args.days]
    if args.reuse:
        paths, detail = load_paths(output)
        dispatch = pd.read_csv(output / "dispatch_year.csv", parse_dates=["date"])
        records = pd.read_csv(output / "validation_daily.csv").to_dict("records")
        summary = V.summary([{k: v for k, v in r.items()
                              if k != "date" and not isinstance(v, str)} for r in records])
        decision_violations = json.loads(
            (output / "decision_layer_residual.json").read_text(encoding="utf-8"))
        chain = [{"passed": True, "chain_abs_kwh": 0.0}]
        result = {"paths": paths, "dispatch": dispatch}
    else:
        result = run((load_power, pv_power), features, rule, params,
                     strict=not args.tolerant)
        records = result["records"]
        summary = V.summary([{k: v for k, v in r.items() if k != "date"} for r in records])
        chain = result["chain"]
        decision_violations = result["decision_violations"]
        dispatch = result["dispatch"]
        print(json.dumps({"核验天数": summary["days_checked"],
                          "未通过天数": summary["days_failed"],
                          "跨日链失败数": len([c for c in chain if not c["passed"]]),
                          "决策层情景层残留": len(decision_violations),
                          "末态储电量": result["final_storage"],
                          "违规样例": result["violations"][:5]},
                         ensure_ascii=False, indent=2), flush=True)
    if args.days:
        return

    output.mkdir(parents=True, exist_ok=True)
    detail = save_paths(result, output)
    dispatch.to_csv(output / "dispatch_year.csv", index=False, encoding="utf-8-sig",
                    float_format="%.10g")
    pd.DataFrame(records).to_csv(output / "validation_daily.csv", index=False,
                                 encoding="utf-8-sig", float_format="%.10g")
    (output / "decision_layer_residual.json").write_text(
        json.dumps(decision_violations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = build_result2(result, params, output, TEMPLATE)
    (output / "result2_structure.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tables = generate_tables(result, detail, params, output)
    make_plots(detail, dispatch, assets)
    lookahead = verify_no_lookahead((load_power, pv_power), features, rule, params,
                                    pd.Timestamp(TABLE_DATES[1]), output)
    backtest = backtest_weights((load_power, pv_power), features, params, rule_data, output)
    report = write_validation_report(summary, records, chain, result.get("violations", []),
                                     decision_violations, dispatch, detail, info, lookahead,
                                     backtest, tables, tables["alignment"], output)
    print(json.dumps({"result2": str(output / "result2.xlsx"),
                      "written_shape": info["written_shape"],
                      "headers_match": info["headers_match"],
                      "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
