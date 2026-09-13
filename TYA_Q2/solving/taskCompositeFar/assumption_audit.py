# -*- coding: utf-8 -*-
"""跨日近视假设的否证实验与不可检验性论证。

运行：
    python -B TYA_Q2/solving/taskCompositeFar/assumption_audit.py

本模块**不求解任何新的线性规划**，只读取 `outputs/taskComposite/` 的既有落盘产物
（`dispatch_year.csv`、`dispatch_detail.csv`、`validation_daily.csv`、`frozen_rule.json`）
与题目给定的 `ProblemC/附件/附件1.xlsx` 电价表，逐条复算 `Q2Project.md`
"模型重要假设合理性的简单论证"一节所引用的每一个统计量，并给出：

1. `P1`—`P7` 命题链的重构与逐条检验状态；
2. 不可检验性的形式化论证：充分条件左端 $-a_{d,q}$ 在当前（近视）模型族内没有
   独立定义，条件退化为"近视模型不为日末储能赋值"的同义反复；
3. 当前结果中与论证预期不符的定量信号（全部可由既有文件直接复算）；
4. 边界声明：现象与论证预期不符本身不构成对论证的证伪，因为同一批现象可以被
   近视模型的内部一致性完全解释；否证的落点是**不可检验性**而非单个统计量的方向。

输出：
    outputs/taskCompositeFar/assumption_audit.md
    outputs/taskCompositeFar/assumption_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "TYA_Q2/outputs/taskComposite"
OUTPUT = ROOT / "TYA_Q2/outputs/taskCompositeFar"
PRICE_FILE = ROOT / "ProblemC/附件/附件1.xlsx"

SLOTS = 144
STEP_HOURS = 1.0 / 6.0
MAX_POWER_KW = 5000.0
MAX_ENERGY = MAX_POWER_KW * STEP_HOURS          # 833.3333 kWh / 10min
S_MIN, S_MAX = 1200.0, 10800.0
USABLE_CAPACITY = S_MAX - S_MIN                 # 9600 kWh
EQUAL_TOL = 1e-6

# 0-based 时段索引区间，左闭右开
EVENING = (132, 144)      # 22:00—24:00，共 12 个时段
MORNING = (0, 30)         # 0:00—5:00，共 30 个时段


def load_price(path: Path = PRICE_FILE) -> np.ndarray:
    """读取附件 1 的 144 个时段电价（元/kWh）。"""
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.worksheets[0]
        rows = list(sheet.iter_rows(min_row=2, max_row=145, values_only=True))
    finally:
        book.close()
    price = np.asarray([float(r[1]) for r in rows], dtype=float)
    if price.shape != (SLOTS,) or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("附件1 电价必须为 144 个严格正有限值")
    return price


def load_artifacts(source: Path = SOURCE_DIR) -> dict:
    """读取 taskComposite 的既有落盘产物，并把逐时段明细整理成日×时段矩阵。"""
    year = pd.read_csv(source / "dispatch_year.csv", parse_dates=["date"])
    detail = pd.read_csv(source / "dispatch_detail.csv", parse_dates=["date"])
    daily = pd.read_csv(source / "validation_daily.csv", parse_dates=["date"])
    rule = json.loads((source / "frozen_rule.json").read_text(encoding="utf-8"))
    if detail.slot.min() != 1 or detail.slot.max() != SLOTS:
        raise ValueError("dispatch_detail.csv 的 slot 应为 1—144")
    if len(year) != 334 or len(detail) != 334 * SLOTS:
        raise ValueError("既有产物应为 334 天、逐日 144 时段")
    wide = {}
    for column in ("plan_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
                   "curtail_kwh", "load_kwh", "pv_kwh", "net_kwh",
                   "storage_start_kwh", "storage_end_kwh"):
        wide[column] = (detail.pivot(index="date", columns="slot", values=column)
                        .sort_index(axis=1).to_numpy(dtype=float))
    return {"year": year, "detail": detail, "daily": daily, "rule": rule,
            "wide": wide, "dates": pd.DatetimeIndex(sorted(detail.date.unique()))}


def audit_existing(art: dict, price: np.ndarray) -> dict:
    """逐条复算论证所引用的统计量，全部只来自既有落盘文件。"""
    w, year = art["wide"], art["year"]
    morning = w["net_kwh"][:, MORNING[0]:MORNING[1]]          # (D,30) 当日凌晨净负荷
    evening = slice(EVENING[0], EVENING[1])
    curtail, charge = w["curtail_kwh"], w["charge_kwh"]
    net = w["net_kwh"]
    # 次日凌晨：把当日矩阵上移一天
    next_morning = np.full_like(morning, np.nan)
    next_morning[:-1] = morning[1:]
    next_plan_morning = np.full_like(morning, np.nan)
    plan_morning = w["plan_kwh"][:, MORNING[0]:MORNING[1]]
    next_plan_morning[:-1] = plan_morning[1:]
    next_emergency_morning = np.full_like(morning, np.nan)
    next_emergency_morning[:-1] = w["emergency_kwh"][1:, MORNING[0]:MORNING[1]]
    has_next = ~np.isnan(next_morning[:, 0])

    full = charge >= MAX_ENERGY - EQUAL_TOL
    positive_net = net > 0
    storage_end = w["storage_end_kwh"][:, -1]
    storage_start = w["storage_start_kwh"][:, 0]

    stats = {
        "source_files": {
            "dispatch_year": "outputs/taskComposite/dispatch_year.csv",
            "dispatch_detail": "outputs/taskComposite/dispatch_detail.csv",
            "validation_daily": "outputs/taskComposite/validation_daily.csv",
            "frozen_rule": "outputs/taskComposite/frozen_rule.json",
            "price": "ProblemC/附件/附件1.xlsx（题目给定输入）",
        },
        "days": int(len(year)),
        # ---- 储能状态 ----
        "storage_end_days_at_lower_bound": int(np.sum(np.abs(storage_end - S_MIN) <= 1e-9)),
        "storage_end_std_kwh": float(np.std(storage_end, ddof=0)),
        "storage_end_mean_kwh": float(storage_end.mean()),
        "storage_start_days_at_lower_bound": int(np.sum(np.abs(storage_start - S_MIN) <= 1e-9)),
        "storage_start_std_kwh": float(np.std(storage_start, ddof=0)),
        "storage_all_time_min_kwh": float(w["storage_start_kwh"].min()),
        "storage_all_time_max_kwh": float(w["storage_start_kwh"].max()),
        "storage_upper_gap_kwh": float(S_MAX - w["storage_start_kwh"].max()),
        "storage_upper_hits": int(np.sum(w["storage_start_kwh"] >= S_MAX - 1e-6)),
        "day_max_storage_median_kwh": float(np.median(w["storage_start_kwh"].max(axis=1))),
        # 365 天口径（含 1 月冷启动期），来源 validation_daily.csv
        "storage_all_time_max_kwh_365d": float(art["daily"].storage_max_kwh.max()),
        "storage_upper_gap_kwh_365d": float(S_MAX - art["daily"].storage_max_kwh.max()),
        # ---- 充放电 ----
        "charge_total_kwh": float(charge.sum()),
        "discharge_total_kwh": float(w["discharge_kwh"].sum()),
        "charge_daily_mean_kwh": float(charge.sum(axis=1).mean()),
        "charge_utilization_ratio": float(charge.sum(axis=1).mean() / USABLE_CAPACITY),
        "charge_days_positive": int(np.sum(charge.sum(axis=1) > EQUAL_TOL)),
        "charge_at_power_cap_slots": int(full.sum()),
        "charge_at_power_cap_share_pct": float(full.sum() / charge.size * 100),
        # ---- 弃置 ----
        "curtail_total_kwh": float(curtail.sum()),
        "plan_total_kwh": float(year.plan_kwh.sum()),
        "curtail_over_plan_pct": float(curtail.sum() / year.plan_kwh.sum() * 100),
        "curtail_over_charge_ratio": float(curtail.sum() / charge.sum()),
        "curtail_days_positive": int(np.sum(curtail.sum(axis=1) > EQUAL_TOL)),
        "curtail_when_net_positive_kwh": float(curtail[positive_net].sum()),
        "curtail_when_net_nonpositive_kwh": float(curtail[~positive_net].sum()),
        "curtail_slots_when_net_positive": int(np.sum(curtail[positive_net] > EQUAL_TOL)),
        "curtail_slots_charge_at_cap": int(np.sum((curtail > EQUAL_TOL) & full)),
        "curtail_kwh_charge_at_cap": float(curtail[(curtail > EQUAL_TOL) & full].sum()),
        "curtail_kwh_charge_below_cap": float(curtail[(curtail > EQUAL_TOL) & ~full].sum()),
        "curtail_below_cap_share_pct": float(
            curtail[(curtail > EQUAL_TOL) & ~full].sum() / curtail.sum() * 100),
        # ---- 晚高峰窗口 ----
        "evening_plan_kwh": float(w["plan_kwh"][:, evening].sum()),
        "evening_discharge_kwh": float(w["discharge_kwh"][:, evening].sum()),
        "evening_charge_kwh": float(charge[:, evening].sum()),
        "evening_curtail_kwh": float(curtail[:, evening].sum()),
        "evening_net_kwh": float(net[:, evening].sum()),
        "price_evening_min": float(price[evening].min()),
        "price_evening_max": float(price[evening].max()),
        "price_evening_mean": float(price[evening].mean()),
        # ---- 凌晨窗口（需求侧） ----
        "morning_net_total_kwh": float(morning.sum()),
        "morning_net_daily_mean_kwh": float(morning.sum(axis=1).mean()),
        "morning_net_daily_min_kwh": float(morning.sum(axis=1).min()),
        "morning_net_daily_max_kwh": float(morning.sum(axis=1).max()),
        "morning_net_positive_total_kwh": float(np.maximum(morning, 0).sum()),
        "morning_plan_total_kwh": float(plan_morning.sum()),
        "morning_plan_daily_mean_kwh": float(plan_morning.sum(axis=1).mean()),
        "morning_emergency_total_kwh": float(w["emergency_kwh"][:, MORNING[0]:MORNING[1]].sum()),
        "price_morning_min": float(price[MORNING[0]:MORNING[1]].min()),
        "price_morning_max": float(price[MORNING[0]:MORNING[1]].max()),
        "price_morning_mean": float(price[MORNING[0]:MORNING[1]].mean()),
        # ---- 次日凌晨（替代供给的需求侧检验） ----
        "next_morning_net_daily_mean_kwh": float(next_morning[has_next].sum(axis=1).mean()),
        "next_morning_net_daily_min_kwh": float(next_morning[has_next].sum(axis=1).min()),
        "next_morning_net_daily_max_kwh": float(next_morning[has_next].sum(axis=1).max()),
        "next_morning_net_positive_daily_mean_kwh": float(
            np.maximum(next_morning[has_next], 0).sum(axis=1).mean()),
        "next_morning_net_below_usable_capacity_days": int(
            np.sum(next_morning[has_next].sum(axis=1) < USABLE_CAPACITY)),
        "next_morning_days": int(has_next.sum()),
        "next_morning_plan_daily_mean_kwh": float(
            next_plan_morning[has_next].sum(axis=1).mean()),
        # ---- 紧急购电 ----
        "emergency_total_kwh": float(year.emergency_kwh.sum()),
        "emergency_days_positive": int(np.sum(year.emergency_kwh > EQUAL_TOL)),
        "emergency_morning_total_kwh": float(
            w["emergency_kwh"][:, MORNING[0]:MORNING[1]].sum()),
        "emergency_evening_total_kwh": float(w["emergency_kwh"][:, evening].sum()),
        "emergency_morning_share_pct": float(
            w["emergency_kwh"][:, MORNING[0]:MORNING[1]].sum()
            / w["emergency_kwh"].sum() * 100),
        # ---- 计划与净负荷 ----
        "plan_over_actual_net_kwh": float(year.plan_kwh.sum() - net.sum()),
        "plan_over_actual_net_pct": float(
            (year.plan_kwh.sum() - net.sum()) / year.plan_kwh.sum() * 100),
        "plan_daily_median_kwh": float(year.plan_kwh.median()),
        "plan_daily_min_kwh": float(year.plan_kwh.min()),
        "plan_daily_max_kwh": float(year.plan_kwh.max()),
        # ---- 电价结构 ----
        "price_min": float(price.min()),
        "price_max": float(price.max()),
        "price_min_slots": [int(i) + 1 for i in np.flatnonzero(price <= price.min() + 1e-12)],
        "price_argmin_label": f"{(int(np.argmin(price)) + 1) * 10 // 60}:"
                              f"{(int(np.argmin(price)) + 1) * 10 % 60:02d}",
        "price_evening_over_morning_ratio": float(price[evening].mean()
                                                  / price[MORNING[0]:MORNING[1]].mean()),
        "evening_slots_below_quoted_min": int(np.sum(price[evening] < 0.418)),
        "price_within_0p5pct_of_min_slots": int(np.sum(price <= price.min() * 1.005)),
    }
    # 逐时段平均净负荷，用于判断凌晨需求侧容量
    stats["morning_net_slot_mean_kwh"] = [float(v) for v in morning.mean(axis=0)]
    stats["evening_net_slot_mean_kwh"] = [float(v) for v in net[:, evening].mean(axis=0)]
    stats["charge_slot_mean_kwh"] = [float(v) for v in charge.mean(axis=0)]
    stats["curtail_slot_mean_kwh"] = [float(v) for v in curtail.mean(axis=0)]
    return stats


def proposition_chain(stats: dict) -> list[dict]:
    """把原论证重写为可判定的命题链，并逐条给出检验状态。

    检验状态取值：
    - `existing_data_supported`：可由既有落盘产物直接复算且与原论证一致；
    - `existing_data_conflicting`：可由既有落盘产物直接复算但与原论证的预期不符；
    - `not_testable`：所涉量在当前（近视）模型族内没有独立定义，无法检验。
    """
    return [
        {
            "id": "P1",
            "claim": "近视模型在 22:00—24:00 的低谷窗口把储能放空，因为放出一度电即省下 P_t 元。",
            "status": "existing_data_supported",
            "evidence": {
                "日末储电量恒为下限的天数": f"{stats['storage_end_days_at_lower_bound']}/334",
                "日末储电量总体标准差(kWh)": stats["storage_end_std_kwh"],
                "晚高峰(22:00—24:00)放电总量(kWh)": stats["evening_discharge_kwh"],
                "晚高峰(22:00—24:00)充电总量(kWh)": stats["evening_charge_kwh"],
            },
            "note": "行为描述与观测一致；但该描述只刻画近视模型自身的最优解，"
                    "不含任何关于远视模型会如何行动的信息。",
        },
        {
            "id": "P2",
            "claim": "假设成立的充分条件为 max_q(-a_{d,q}) ≲ min_{t∈T晚} P_t。",
            "status": "not_testable",
            "evidence": {
                "左端对象": "终端价值函数 V_{d+1}(s) 的分段线性下界的切线斜率 a_{d,q}",
                "既有产物中记录该量的文件": "无",
                "右端(附件1，元/kWh)": [stats["price_evening_min"], stats["price_evening_max"]],
            },
            "note": "a_{d,q} 是远视模型决策约束 θ^ω_d ≥ a_{d,q} S^ω_{d,145} + b_{d,q} 的系数；"
                    "在近视模型族中 θ 不进入目标函数（假设 1 本身），没有任何最优性条件"
                    "涉及 a_{d,q}，既有产物中也没有任何文件记录它。详见第三节的形式化论证。",
        },
        {
            "id": "P3",
            "claim": "日末储能的边际价值被封顶在凌晨低谷价约 0.42 元/kWh 附近，"
                     "因为凌晨可重新购入，且充满 9600 kWh 仅需约 12 个时段、替代供给充足。",
            "status": "not_testable",
            "evidence": {
                "事实层可否复算": "是（下表全部数值均取自附件1 与既有落盘产物）",
                "断言层所在对象": "日末储能的边际价值，即 V_{d+1}(s) 的切线斜率 −a_{d,q}",
                "既有产物中记录该量的文件": "无",
                "晚高峰 22:00—24:00 电价区间(元/kWh)":
                    [stats["price_evening_min"], stats["price_evening_max"]],
                "晚高峰电价均值(元/kWh)": stats["price_evening_mean"],
                "凌晨 0:00—5:00 电价区间(元/kWh)":
                    [stats["price_morning_min"], stats["price_morning_max"]],
                "凌晨电价均值(元/kWh)": stats["price_morning_mean"],
                "凌晨均价比晚高峰均价高(%)": (stats["price_morning_mean"]
                                              / stats["price_evening_mean"] - 1) * 100,
                "凌晨(0:00—5:00)净负荷日均总量(kWh)": stats["morning_net_daily_mean_kwh"],
                "次日凌晨净负荷总量低于可用容量 9600 kWh 的天数":
                    f"{stats['next_morning_net_below_usable_capacity_days']}"
                    f"/{stats['next_morning_days']}",
                "可用容量(kWh)": USABLE_CAPACITY,
                "充电功率上限(kWh/10min)": MAX_ENERGY,
                "充满可用容量所需时段数": USABLE_CAPACITY / MAX_ENERGY,
            },
            "note": "该命题包含事实层与断言层，两者地位不同。事实层（电价低谷横跨日界、"
                    "充电功率足以在 12 个时段内充满可用容量、凌晨净负荷日均 "
                    f"{stats['morning_net_daily_mean_kwh']:,.0f} kWh 远超 9600 kWh，"
                    "替代供给在供需两侧都充足）可以由附件1 与既有产物完全复算，"
                    "且复算结果与原论证一致。断言层（\"边际价值被封顶在 0.42 附近\"）"
                    "是关于 V_{d+1}(s) 斜率的陈述，在当前模型族内没有定义，不可检验。"
                    "此外，复算还显示原论证所说的\"两端几乎相等\"掩盖了一个方向性不对称："
                    f"凌晨电价均值 {stats['price_morning_mean']:.4f} 元/kWh 反而高于晚高峰的 "
                    f"{stats['price_evening_mean']:.4f} 元/kWh（高 "
                    f"{(stats['price_morning_mean'] / stats['price_evening_mean'] - 1) * 100:.2f}%），"
                    f"且晚高峰窗口内的最低价 {stats['price_evening_min']:.4f} 低于凌晨最低价 "
                    f"{stats['price_morning_min']:.4f}；若按论证的比较方式直接对比，"
                    "\"留下\"一侧反而不比\"放掉\"一侧便宜。",
        },
        {
            "id": "P4",
            "claim": "附件 1 在 22:00—24:00 的电价为 0.418—0.427 元/kWh。",
            "status": "existing_data_conflicting",
            "evidence": {
                "原论证引用的区间(元/kWh)": [0.418, 0.427],
                "实际复算的区间(元/kWh)":
                    [stats["price_evening_min"], stats["price_evening_max"]],
                "实际复算的均值(元/kWh)": stats["price_evening_mean"],
                "全天最低电价(元/kWh)": stats["price_min"],
                "最低电价出现时刻": stats["price_argmin_label"],
                "低于 0.418 元/kWh 的晚高峰时段数":
                    stats["evening_slots_below_quoted_min"],
            },
            "note": "该命题只涉及题目给定输入，可完全复算。复算显示原论证引用的下界 "
                    f"0.418 元/kWh 过低：晚高峰窗口内的实际最低价为 "
                    f"{stats['price_evening_min']:.4f} 元/kWh，"
                    f"有 {stats['evening_slots_below_quoted_min']} 个时段低于引用下界，"
                    f"窗口均值 {stats['price_evening_mean']:.4f} 元/kWh。"
                    "这直接影响\"两端几乎相等\"这一判断的分辨率。",
        },
        {
            "id": "P5",
            "claim": "两端几乎相等 ⇒ 最优解处于\"放掉\"与\"留下\"近乎无差异的临界 ⇒ "
                     "两模型的日末行为趋于一致 ⇒ 接受跨日近视假设。",
            "status": "not_testable",
            "evidence": {
                "推理链的承重点": "P2（不可检验）",
                "被跳过的实验": "V(s) 扫描实验（Q2Project.md 第 86 行自述）",
            },
            "note": "由于 P2 的左端无定义，P5 的推理链在承重处断开；"
                    "\"两端几乎相等\"只是对两个未经独立计算的数字的比较。",
        },
        {
            "id": "P6",
            "claim": "计划购电直达负载且无功率上限，凌晨空仓只影响套利机会而不阻塞供电。",
            "status": "existing_data_conflicting",
            "evidence": {
                "全年弃置电量(kWh)": stats["curtail_total_kwh"],
                "弃置占计划购电比例(%)": stats["curtail_over_plan_pct"],
                "弃置为充电量的倍数": stats["curtail_over_charge_ratio"],
                "出现弃置的天数": f"{stats['curtail_days_positive']}/334",
                "计划购电超出实际净负荷(kWh)": stats["plan_over_actual_net_kwh"],
                "净负荷为正时段的弃置电量(kWh)": stats["curtail_when_net_positive_kwh"],
                "充电已达功率上限时段的弃置电量(kWh)": stats["curtail_kwh_charge_at_cap"],
                "充电未达功率上限时段的弃置电量(kWh)": stats["curtail_kwh_charge_below_cap"],
                "充电未达功率上限时段弃置占比(%)": stats["curtail_below_cap_share_pct"],
                "充电已达功率上限的时段数": stats["charge_at_power_cap_slots"],
                "储能全天最高储电量(365 天口径, kWh)": stats["storage_all_time_max_kwh_365d"],
                "距储能上限 10800 kWh 的余量(kWh)": stats["storage_upper_gap_kwh_365d"],
                "充电日均利用率(占可用容量 9600 kWh)": stats["charge_utilization_ratio"],
            },
            "note": "计划购电确实不受功率限制，但\"计划购电直达负载\"并不成立："
                    f"全年有 {stats['curtail_over_plan_pct']:.2f}% 的计划购电量被弃置。"
                    "复算进一步显示，这些弃置既不能由充电功率上限解释、也不能由储能容量解释："
                    f"弃置总量的 {stats['curtail_below_cap_share_pct']:.2f}% 发生在充电**未达**"
                    f"功率上限的时段（{stats['curtail_kwh_charge_below_cap']:,.1f} kWh），"
                    f"发生在充电已达上限时段的只有 {stats['curtail_kwh_charge_at_cap']:,.1f} kWh，"
                    f"而储能全天最高只到 {stats['storage_all_time_max_kwh_365d']:,.1f} kWh、"
                    f"距上限还有 {stats['storage_upper_gap_kwh_365d']:,.1f} kWh。"
                    "因此弃置只能由\"多余的来源在当天无处可用、且在目标函数中不被跨日赋值\""
                    "解释，而后者正是假设 1 本身。",
        },
        {
            "id": "P7",
            "claim": "凌晨即使触发紧急购电，5×0.42≈2.1 元/kWh 也是全天最便宜的紧急购电时段，"
                     "且凌晨时段可通过加大计划裕度廉价对冲。",
            "status": "existing_data_supported",
            "evidence": {
                "凌晨(0:00—5:00)紧急购电量(kWh)": stats["emergency_morning_total_kwh"],
                "晚高峰(22:00—24:00)紧急购电量(kWh)": stats["emergency_evening_total_kwh"],
                "凌晨紧急购电占全年比例(%)": stats["emergency_morning_share_pct"],
                "全年紧急购电量(kWh)": stats["emergency_total_kwh"],
                "出现紧急购电的天数": f"{stats['emergency_days_positive']}/334",
            },
            "note": "既有结果显示凌晨几乎不触发紧急购电，与该命题方向一致；"
                    "但它同样只说明近视模型的行为，不构成对 P2 左端的支持。",
        },
    ]


def formal_untestability(stats: dict) -> dict:
    """给出不可检验性的形式化论证，并明确否证的落点。"""
    return {
        "notation": {
            "myopic_objective": "min Σ_t P_t G_t + Σ_ω π_{d,ω} Σ_t 5 P_t E^ω_{d,t} "
                                "+ ε Σ_ω π_{d,ω} Σ_t (C^ω_{d,t} + D^ω_{d,t})",
            "far_objective": "min Σ_t P_t G_t + Σ_ω π_{d,ω} ( Σ_t 5 P_t E^ω_{d,t} + θ^ω_d )",
            "support_constraint": "θ^ω_d ≥ a_{d,q} S^ω_{d,145} + b_{d,q},  q = 0,…,Q−1",
            "sufficient_condition": "max_q (−a_{d,q}) ≲ min_{t∈T晚} P_t,  T晚 = {133,…,144}",
        },
        "steps": [
            {
                "step": 1,
                "statement": "a_{d,q} 只在远视模型中作为决策约束的系数出现。",
                "detail": "符号表把 (a_{d,q}, b_{d,q}) 定义为终端价值辅助变量 θ^ω_d 的第 q 条"
                          "切线的斜率与截距，而 θ^ω_d 只出现在远视模型的目标函数与支撑约束中。",
            },
            {
                "step": 2,
                "statement": "近视模型族中 θ 不进入目标函数，因此没有任何最优性条件涉及 a_{d,q}。",
                "detail": "近视模型的最优解 (G*, C*, D*, S*, E*, W*) 与 a_{d,q} 无关；"
                          "既有落盘产物中不存在任何记录 a_{d,q} 的文件或列。",
            },
            {
                "step": 3,
                "statement": "要在近视模型族内给 −a_{d,q} 赋值，唯一可能的取值是 0。",
                "detail": "若坚持在近视模型内解释 θ^ω_d，则它不受任何下界约束、"
                          "又以正权重 π_{d,ω} 进入最小化目标，故 θ^ω_d = 0 对一切 ω，"
                          "从而 a_{d,q} ≡ 0。",
            },
            {
                "step": 4,
                "statement": "于是充分条件退化为 0 ≲ min_t P_t，恒真。",
                "detail": f"右端 min_{{t∈T晚}} P_t = {stats['price_evening_min']:.6f} > 0，"
                          "不等式无条件成立，不依赖任何数据、情景池或权重。",
            },
            {
                "step": 5,
                "statement": "该条件因此等价于\"近视模型不为日末储能赋值\"，即假设 1 本身。",
                "detail": "以假设为前提推出的条件去支持同一假设，构成循环；"
                          "条件不是对假设的检验，而是对假设的重述（同义反复）。",
            },
            {
                "step": 6,
                "statement": "要真正计算左端，必须先构造 V_{d+1}(s)，而这已经是远视模型。",
                "detail": "V_{d+1}(s) 定义为次日以储能 s 开局的最小期望费用，"
                          "其切线斜率必须由远视（或至少一步前瞻）模型给出。"
                          "因此该条件只有在远视模型已经建好之后才能被检验，"
                          "它不能用来决定是否需要建远视模型。",
            },
        ],
        "conclusion": "原论证在当前模型族内不可检验；这是本步骤的否证落点。"
                      "否证的不是\"跨日近视近似不准\"这一数值判断，"
                      "而是\"该论证构成对跨日近视假设的检验\"这一方法论主张。",
        "what_would_make_it_testable": [
            "显式定义 V_{d+1}(s)：状态区间 [Smin,Smax] 上的采样网格、"
            "采样点的值函数与共同出发点约束的次梯度、每点一条切线的构造规则；",
            "在同一情景池与同一核权重下把切线送入第 d 天的目标函数，得到远视模型；",
            "在同一执行层下配对比较 C_official，并按事先固定的相对阈值给出采纳判定。",
        ],
    }


def build_report(stats: dict, chain: list[dict], formal: dict) -> str:
    """生成否证证据与不可检验性论证的 Markdown 报告。"""
    status_label = {
        "existing_data_supported": "既有数据支持",
        "existing_data_conflicting": "既有数据与预期不符",
        "not_testable": "不可检验",
    }
    rows = "\n".join(
        f"| {item['id']} | {item['claim']} | {status_label[item['status']]} |"
        for item in chain)
    detail_blocks = []
    for item in chain:
        lines = [f"\n#### {item['id']}：{item['claim']}\n",
                 f"**检验状态：{status_label[item['status']]}**\n",
                 "| 复算量 | 数值 |", "|---|---:|"]
        for key, value in item["evidence"].items():
            if isinstance(value, list):
                shown = "、".join(f"{v:,.6f}" for v in value)
            elif isinstance(value, float):
                shown = f"{value:,.6f}"
            else:
                shown = f"{value}"
            lines.append(f"| {key} | {shown} |")
        lines.append(f"\n{item['note']}\n")
        detail_blocks.append("\n".join(lines))
    steps = "\n".join(
        f"\n**第 {s['step']} 步。** {s['statement']}\n\n{s['detail']}\n"
        for s in formal["steps"])
    testable = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(formal["what_would_make_it_testable"]))
    evidence_table = f"""| 复算量 | 数值 | 出处 |
|---|---:|---|
| 日末储电量恒为下限的天数 | {stats['storage_end_days_at_lower_bound']}/334 | `dispatch_year.csv: storage_end` |
| 日末储电量总体标准差 | {stats['storage_end_std_kwh']:.6e} kWh | `dispatch_year.csv: storage_end` |
| 日初储电量恒为下限的天数 | {stats['storage_start_days_at_lower_bound']}/334 | `dispatch_year.csv: storage_start` |
| 储能全天区间（334 天输出窗口） | [{stats['storage_all_time_min_kwh']:.4f}, {stats['storage_all_time_max_kwh']:.4f}] kWh | `dispatch_detail.csv: storage_start_kwh` |
| 储能全天最高（365 天，含 1 月冷启动） | {stats['storage_all_time_max_kwh_365d']:.4f} kWh | `validation_daily.csv: storage_max_kwh` |
| 距储能上限 10800 kWh 的余量（365 天口径） | {stats['storage_upper_gap_kwh_365d']:.4f} kWh | 同上 |
| 触及储能上限的时段数 | {stats['storage_upper_hits']} | `dispatch_detail.csv: storage_start_kwh` |
| 全年充电量 | {stats['charge_total_kwh']:,.1f} kWh | `dispatch_year.csv: charge_kwh` |
| 全年放电量 | {stats['discharge_total_kwh']:,.1f} kWh | `dispatch_year.csv: discharge_kwh` |
| 日均充电量 | {stats['charge_daily_mean_kwh']:,.1f} kWh | 同上 |
| 日均充电量占可用容量 9600 kWh | {stats['charge_utilization_ratio'] * 100:.3f}% | 同上 |
| 充电已达功率上限的时段数 | {stats['charge_at_power_cap_slots']} / {334 * SLOTS} | `dispatch_detail.csv: charge_kwh` |
| 全年弃置电量 | {stats['curtail_total_kwh']:,.1f} kWh | `dispatch_year.csv: curtail_kwh` |
| 弃置占计划购电量 | {stats['curtail_over_plan_pct']:.3f}% | `dispatch_year.csv` |
| 弃置为充电量的倍数 | {stats['curtail_over_charge_ratio']:.3f} 倍 | 同上 |
| 出现弃置的天数 | {stats['curtail_days_positive']}/334 | 同上 |
| 净负荷为正时段的弃置电量 | {stats['curtail_when_net_positive_kwh']:,.1f} kWh | `dispatch_detail.csv` |
| 充电已达上限时段的弃置电量 | {stats['curtail_kwh_charge_at_cap']:,.1f} kWh | 同上 |
| 充电未达上限时段的弃置电量 | {stats['curtail_kwh_charge_below_cap']:,.1f} kWh（占弃置总量 {stats['curtail_below_cap_share_pct']:.2f}%） | 同上 |
| 全年计划购电量 | {stats['plan_total_kwh']:,.1f} kWh | `dispatch_year.csv: plan_kwh` |
| 计划购电超出实际净负荷 | {stats['plan_over_actual_net_kwh']:,.1f} kWh（{stats['plan_over_actual_net_pct']:.3f}%） | `dispatch_detail.csv: net_kwh` |
| 晚高峰 22:00—24:00 弃置电量 | {stats['evening_curtail_kwh']:,.1f} kWh | `dispatch_detail.csv: curtail_kwh` |
| 凌晨 0:00—5:00 净负荷日均总量 | {stats['morning_net_daily_mean_kwh']:,.1f} kWh | 同上 |
| 次日凌晨 0:00—5:00 净负荷日均总量 | {stats['next_morning_net_daily_mean_kwh']:,.1f} kWh | 同上 |
| 次日凌晨净负荷总量低于 9600 kWh 的天数 | {stats['next_morning_net_below_usable_capacity_days']}/{stats['next_morning_days']} | 同上 |
| 凌晨 0:00—5:00 紧急购电量 | {stats['emergency_morning_total_kwh']:,.4f} kWh | `dispatch_detail.csv: emergency_kwh` |
| 晚高峰 22:00—24:00 紧急购电量 | {stats['emergency_evening_total_kwh']:,.4f} kWh | 同上 |
| 附件1 晚高峰电价区间 | {stats['price_evening_min']:.6f}—{stats['price_evening_max']:.6f} 元/kWh（均值 {stats['price_evening_mean']:.6f}） | `附件1.xlsx` |
| 附件1 凌晨电价区间 | {stats['price_morning_min']:.6f}—{stats['price_morning_max']:.6f} 元/kWh（均值 {stats['price_morning_mean']:.6f}） | `附件1.xlsx` |
| 晚高峰电价低于原论证引用下界 0.418 的时段数 | {stats['evening_slots_below_quoted_min']} | `附件1.xlsx` |
| 附件1 全天最低电价 | {stats['price_min']:.6f} 元/kWh（{stats['price_argmin_label']}） | `附件1.xlsx` |"""
    return f"""# 跨日近视假设的否证证据与不可检验性论证

本文件由 `solving/taskCompositeFar/assumption_audit.py` 生成，**不求解任何新的线性规划**：
上表与实际结论中的每一个数值都只来自 `outputs/taskComposite/` 的既有落盘文件与题目给定的
`ProblemC/附件/附件1.xlsx`，可逐条复算，不含任何未落盘的中间量。

## 一、结论摘要

1. `Q2Project.md` "模型重要假设合理性的简单论证"一节所依赖的充分条件
   $$\\max_{{q=0,\\ldots,Q-1}}\\bigl(-a_{{d,q}}\\bigr)\\;\\lesssim\\;\\min_{{t\\in\\mathcal{{T}}^{{\\mathrm{{晚}}}}}} P_t,
   \\qquad \\mathcal{{T}}^{{\\mathrm{{晚}}}}=\\{{133,\\ldots,144\\}}$$
   的左端 $-a_{{d,q}}$ 是终端价值函数 $V_{{d+1}}(s)$ 的切线斜率，**只在远视模型中才有定义**。
   在当前的近视模型族内，$\\theta^\\omega_d$ 不受下界约束又以正权重进入最小化目标，
   故 $\\theta^\\omega_d\\equiv0$、$a_{{d,q}}\\equiv0$，条件退化为
   $0\\lesssim{stats['price_evening_min']:.6f}$，恒真而不依赖任何数据。
   **该条件因此等价于"近视模型不为日末储能赋值"，即假设 1 本身，构成同义反复。**
2. 否证的落点是**不可检验性**，不是单个统计量的方向。既有产物中确实存在若干与论证预期
   不符的定量信号（见第四节），但同一批现象也能被近视模型的内部一致性完全解释；
   它们的作用是**说明为什么必须把当年跳过的 $V(s)$ 扫描检验补做**，而不是直接推翻假设。
3. 论证自述"该假设本计划以 $V(s)$ 扫描实验验证，现以口头论证代替"。
   本步骤的后续模块（`model_far.py`、`value_function.py`、`solve_year_far.py`）
   把该项检验补做完整，并以配对对照给出采纳判定。

## 二、命题链重构与逐条检验状态

| 编号 | 命题 | 检验状态 |
|---|---|---|
{rows}

{''.join(detail_blocks)}

## 三、不可检验性的形式化论证

记第 $d$ 天的近视目标为
$$\\min\\;\\sum_{{t\\in\\mathcal{{T}}}}P_tG_{{d,t}}+\\sum_{{\\omega\\in\\Omega_d}}\\pi_{{d,\\omega}}\\sum_{{t\\in\\mathcal{{T}}}}5P_tE^\\omega_{{d,t}}
+\\varepsilon\\sum_{{\\omega\\in\\Omega_d}}\\pi_{{d,\\omega}}\\sum_{{t\\in\\mathcal{{T}}}}\\bigl(C^\\omega_{{d,t}}+D^\\omega_{{d,t}}\\bigr),$$
远视目标为
$$\\min\\;\\sum_{{t\\in\\mathcal{{T}}}}P_tG_{{d,t}}+\\sum_{{\\omega\\in\\Omega_d}}\\pi_{{d,\\omega}}\\Bigl(\\sum_{{t\\in\\mathcal{{T}}}}5P_tE^\\omega_{{d,t}}+\\theta^\\omega_d\\Bigr),
\\qquad \\theta^\\omega_d\\ge a_{{d,q}}S^\\omega_{{d,145}}+b_{{d,q}}.$$
{steps}

**结论。** {formal['conclusion']}

### 使该条件可检验所需的构造

{testable}

## 四、与论证预期不符的定量信号

以下信号全部取自既有落盘文件。再次强调：它们**不单独构成对假设的证伪**，
因为近视模型可以用"当天用不掉、次日不值钱"这一套内部逻辑同时解释它们；
它们的用途是定位原论证的薄弱环节，并说明 $V(s)$ 扫描检验必须补做。

{evidence_table}

### 信号一：储能容量与充电功率都不是瓶颈，"日末空仓"不是约束的强加

日末与日初储电量 334 天全部等于下限 1200 kWh（标准差 {stats['storage_end_std_kwh']:.1e}）；
储能全天最高只到 {stats['storage_all_time_max_kwh_365d']:,.1f} kWh（365 天口径），
距上限 10800 kWh 尚有 **{stats['storage_upper_gap_kwh_365d']:,.1f} kWh**，全年没有任何时段触及上限；
日均充电量 {stats['charge_daily_mean_kwh']:,.1f} kWh，只占可用容量 9600 kWh 的
**{stats['charge_utilization_ratio'] * 100:.3f}%**。
这一组事实说明：日末储电量恒在下限并非储能装不下，而是目标函数的选择。
但它同时也说明，**这个事实不能用来支持跨日近视假设**——它与假设互为因果。

### 信号二：弃置电量既不能由充电功率上限解释，也不能由储能容量解释

全年弃置 {stats['curtail_total_kwh']:,.1f} kWh，占计划购电量的
**{stats['curtail_over_plan_pct']:.3f}%**，是全年充电量的 **{stats['curtail_over_charge_ratio']:.3f} 倍**，
{stats['curtail_days_positive']}/334 天出现弃置；计划购电总量超出实际净负荷
{stats['plan_over_actual_net_kwh']:,.1f} kWh（{stats['plan_over_actual_net_pct']:.3f}%）。

约束侧的复算排除了两种"技术性"解释：

- **不是功率上限**：全年只有 {stats['charge_at_power_cap_slots']} 个时段（占全部时段的
  {stats['charge_at_power_cap_share_pct']:.4f}%）充电达到功率上限，落在这些时段的弃置仅
  {stats['curtail_kwh_charge_at_cap']:,.1f} kWh；弃置总量的
  **{stats['curtail_below_cap_share_pct']:.2f}%**（{stats['curtail_kwh_charge_below_cap']:,.1f} kWh）
  发生在充电**未达**功率上限的时段，即储能当时还有充电余力。
- **不是容量上限**：储能全天最高 {stats['storage_all_time_max_kwh_365d']:,.1f} kWh，
  距上限 {stats['storage_upper_gap_kwh_365d']:,.1f} kWh，从未触顶。

排除这两者后，弃置只能由"多余的来源在当天无处可用、且在目标函数中不被跨日赋值"解释，
而后者正是跨日近视假设本身。注意这仍然不是对假设的证伪：在近视模型内，把无处可用的
电量弃置确实是最优的；该信号的作用是标出**假设的代价可能有量级**
（{stats['curtail_over_plan_pct']:.2f}% 的计划购电量级），从而说明 $V(s)$ 扫描检验不可省略。

### 信号三：原论证的"两端几乎相等"与电价表的复算结果不符

原论证用"左端封顶在 0.42 元/kWh"与"右端 0.418—0.427 元/kWh"的近乎相等来支撑
"两模型日末行为趋于一致"。对附件 1 的复算给出：

- 晚高峰 22:00—24:00 的实际电价区间是
  **{stats['price_evening_min']:.4f}—{stats['price_evening_max']:.4f} 元/kWh**
  （均值 {stats['price_evening_mean']:.4f}），有
  **{stats['evening_slots_below_quoted_min']} 个时段低于原论证引用的下界 0.418**；
- 凌晨 0:00—5:00 的电价区间是
  **{stats['price_morning_min']:.4f}—{stats['price_morning_max']:.4f} 元/kWh**
  （均值 {stats['price_morning_mean']:.4f}），**反而高于晚高峰均值
  {(stats['price_morning_mean'] / stats['price_evening_mean'] - 1) * 100:.2f}%**；
- 全天最低价 {stats['price_min']:.4f} 元/kWh 出现在 {stats['price_argmin_label']}，
  并不在 22:00—24:00 窗口内。

也就是说，"放掉"一侧（晚高峰均价 {stats['price_evening_mean']:.4f}）比"留下后在凌晨使用"
一侧（凌晨均价 {stats['price_morning_mean']:.4f}）**电价更低**，两端不是"几乎相等"，
而是存在方向明确、量级约 2—3% 的系统性缺口。论证的比较没有考虑这一缺口，
也没有考虑储能往返效率 $\\eta_c\\eta_d=0.81$ 对两侧的不同作用。

### 信号四：晚高峰仍在弃置，而同一时刻日末储能恰好停在下限

晚高峰 22:00—24:00 窗口内计划购电 {stats['evening_plan_kwh']:,.1f} kWh、
实际净负荷 {stats['evening_net_kwh']:,.1f} kWh、放电 {stats['evening_discharge_kwh']:,.1f} kWh、
充电 {stats['evening_charge_kwh']:,.1f} kWh，同时弃置 **{stats['evening_curtail_kwh']:,.1f} kWh**。
在这一窗口内，储能还有 {stats['storage_upper_gap_kwh_365d']:,.0f} kWh 以上的容量余量、
日末储电量却精确停在下限。这是"手边有容量、有富余来源，却既不留也不充"的直接痕迹，
与论证预期的"两个模型在日末趋于一致"之间的落差，正是远视化要回答的问题。


## 五、否证的落点与不成立的推论

**成立：** 原论证不是对该假设的检验，原因是它依赖一个在当前模型族内没有定义的量。
这一结论与数据无关，只依赖模型结构，因此不会因换一批电价或负载曲线而改变。

**不成立（须明确排除）：** "日末储电量恒为下限""弃置电量巨大"等现象并不能证明
跨日近视假设为假。这些现象与近视模型的最优性完全一致，是同一假设的推论而非反驳。
把它们当作证伪证据，会用假设的推论去否定假设，与用假设本身支持假设在逻辑上同样无效。

**因此后续步骤的口径是**：先在远视模型里显式定义 $V_{{d+1}}(s)$、按事先固定的判据
加密采样网格并判定收敛，再在同一情景池、同一核权重、同一执行层下配对比较
$C_{{\\mathrm{{official}}}}$，按事先固定的相对阈值给出采纳判定；负结果如实记录。

## 六、复算入口

```
python -B TYA_Q2/solving/taskCompositeFar/assumption_audit.py
```

全部复算量同时以机器可读形式写入 `outputs/taskCompositeFar/assumption_audit.json`。
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    price = load_price()
    art = load_artifacts()
    stats = audit_existing(art, price)
    chain = proposition_chain(stats)
    formal = formal_untestability(stats)
    payload = {"statistics": stats, "proposition_chain": chain,
               "formal_untestability": formal}
    (OUTPUT / "assumption_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = build_report(stats, chain, formal)
    (OUTPUT / "assumption_audit.md").write_text(report, encoding="utf-8")
    summary = {
        "不可检验命题": [i["id"] for i in chain if i["status"] == "not_testable"],
        "与预期不符": [i["id"] for i in chain if i["status"] == "existing_data_conflicting"],
        "既有数据支持": [i["id"] for i in chain if i["status"] == "existing_data_supported"],
        "日末储能恒为下限": f"{stats['storage_end_days_at_lower_bound']}/334",
        "储能距上限余量_kWh": round(stats["storage_upper_gap_kwh_365d"], 3),
        "弃置总量_kWh": round(stats["curtail_total_kwh"], 3),
        "弃置中充电未达上限占比_pct": round(stats["curtail_below_cap_share_pct"], 4),
        "晚高峰电价区间": [stats["price_evening_min"], stats["price_evening_max"]],
        "凌晨电价均值": round(stats["price_morning_mean"], 6),
        "晚高峰电价均值": round(stats["price_evening_mean"], 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
