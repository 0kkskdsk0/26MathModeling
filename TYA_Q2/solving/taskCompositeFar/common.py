# -*- coding: utf-8 -*-
"""taskCompositeFar 的共享上下文：路径、数据加载、情景池与核权重的口径复用。

本模块**不改变** taskComposite 的任何口径：情景池、标尺、核权重与执行层全部直接
调用 `solving/taskComposite/` 的既有实现，只在此处集中加载与缓存，使近视与远视
两个模型能在同一情景池、同一核权重、同一执行层下配对比较。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOLVING = Path(__file__).resolve().parents[1]
if str(SOLVING) not in sys.path:
    sys.path.insert(0, str(SOLVING))

from taskComposite.kernel import (KernelRule, build_context_features, load_attachment)  # noqa: E402
from taskComposite.model import ModelParams  # noqa: E402
from taskComposite.solve_year import load_price as _load_price  # noqa: E402
from taskComposite.solve_year import scenario_frame as _scenario_frame  # noqa: E402
from taskComposite.solve_year import weights_for  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[3]
ATTACH = ROOT / "ProblemC/附件"
PRICE_FILE = ATTACH / "附件1.xlsx"
SOURCE_FILE = ATTACH / "附件2.xlsx"
TEMPLATE = ATTACH / "附件5/result2.xlsx"

NEAR_OUTPUT = ROOT / "TYA_Q2/outputs/taskComposite"
OUTPUT = ROOT / "TYA_Q2/outputs/taskCompositeFar"
ASSETS = ROOT / "TYA_Q2/assets/taskCompositeFar"
FROZEN_NEAR = NEAR_OUTPUT / "frozen_rule.json"
FROZEN_FAR = OUTPUT / "frozen_rule_far.json"

SLOTS = 144
STEP_HOURS = 1.0 / 6.0
OUTPUT_START = pd.Timestamp("2025-02-01")
INITIAL_STORAGE = 6000.0
TABLE_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


@dataclass(frozen=True)
class Context:
    """一次加载、多处复用的口径对象。"""

    price: np.ndarray
    params: ModelParams
    features: pd.DataFrame
    rule: KernelRule
    rule_data: dict
    load_power: pd.DataFrame
    pv_power: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.features.index

    @property
    def output_dates(self) -> pd.DatetimeIndex:
        return self.features.index[self.features.index >= OUTPUT_START]


def load_rule(path: Path = FROZEN_NEAR) -> tuple[KernelRule, dict]:
    """读取 taskComposite 冻结的赋权规则，不做任何重新挑选。"""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bandwidth = data["revised_bandwidth"]
    rule = KernelRule(tuple(data["final_groups"]),
                      float("inf") if bandwidth is None else float(bandwidth),
                      adopted=bool(data["adopted"]))
    return rule, data


def build_context(price_file: Path = PRICE_FILE, source_file: Path = SOURCE_FILE,
                  rule_file: Path = FROZEN_NEAR, days: int | None = None) -> Context:
    """加载电价、负载光伏、处境特征与冻结规则。"""
    price = _load_price(Path(price_file))
    params = ModelParams(price=price)
    load_power, pv_power = load_attachment(Path(source_file))
    features = build_context_features(load_power, pv_power)
    if days:
        features = features.iloc[:days]
    rule, rule_data = load_rule(rule_file)
    return Context(price=price, params=params, features=features, rule=rule,
                   rule_data=rule_data, load_power=load_power, pv_power=pv_power)


def day_inputs(ctx: Context, day) -> dict:
    """返回第 day 天的候选池、核权重与情景矩阵；口径与 taskComposite 完全一致。"""
    pool, weights, mode = weights_for(ctx.rule, ctx.features, day, ctx.load_power,
                                      ctx.pv_power)
    if len(pool):
        scen_l, scen_r = _scenario_frame(ctx.load_power, ctx.pv_power, pool)
    else:
        scen_l = np.zeros((0, SLOTS))
        scen_r = np.zeros((0, SLOTS))
    return {"date": pd.Timestamp(day), "pool": pool, "weights": weights, "mode": mode,
            "load": scen_l, "pv": scen_r}


def actual_day(ctx: Context, day) -> tuple[np.ndarray, np.ndarray]:
    """第 day 天的实测负载与光伏电量曲线（kWh）。"""
    day = pd.Timestamp(day)
    return (ctx.load_power.loc[day].to_numpy(dtype=float) * STEP_HOURS,
            ctx.pv_power.loc[day].to_numpy(dtype=float) * STEP_HOURS)


def run_recourse(ctx: Context, plan: np.ndarray, day, s_start: float):
    """执行层：固定计划购电量，在实测负载与光伏下求补救问题。

    与 taskComposite 执行层逐元素相同（同一装配函数、同一参数、同一求解器设置）。
    返回 (实测负载, 实测光伏, 执行层解)。
    """
    from taskComposite.model import build_recourse_lp, solve, unpack_recourse
    load, pv = actual_day(ctx, day)
    program = build_recourse_lp(ctx.params, plan, load, pv, s_start)
    executed = unpack_recourse(solve(program, f"{day} 执行层"), program.layout)
    return load, pv, executed
