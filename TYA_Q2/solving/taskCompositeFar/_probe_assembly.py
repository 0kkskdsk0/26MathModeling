# -*- coding: utf-8 -*-
"""临时探针二：分辨装配耗时与求解耗时，并验证复用自己的 A 矩阵只改 b_eq 是否等价。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "TYA_Q2/solving"))

from taskComposite.kernel import KernelRule, build_context_features, load_attachment
from taskComposite.model import (ModelParams, SOLVER_OPTIONS, build_decision_lp, solve,
                                 unpack_decision)
from taskComposite.solve_year import load_price, scenario_frame, weights_for  # noqa

ATTACH = ROOT / "ProblemC/附件"
price = load_price(ATTACH / "附件1.xlsx")
params = ModelParams(price=price)
load_power, pv_power = load_attachment(ATTACH / "附件2.xlsx")
features = build_context_features(load_power, pv_power)
rule = KernelRule(("season", "net_lag_7d", "pv_mean_3d", "load_lag_7d", "net_mean_7d",
                   "low_load_day"), 1.0438414641946645)

day = pd.Timestamp("2025-09-23")
pool, weights, mode = weights_for(rule, features, day, load_power, pv_power)
scen_l, scen_r = scenario_frame(load_power, pv_power, pool)
K = len(pool)
print("pool", K)

t0 = time.perf_counter()
program = build_decision_lp(params, scen_l, scen_r, weights, 1200.0)
t_build = time.perf_counter() - t0
print(f"装配一次 {t_build:.3f}s, 变量数 {program.size}, 等式约束 {program.A_eq.shape}")

# 复用 A，只改 b_eq
t0 = time.perf_counter()
res = solve(program, "基准")
t_solve = time.perf_counter() - t0
print(f"求解 {t_solve:.3f}s V={res.fun:.6f}")

start_offset = 2 * K * 144
b2 = program.b_eq.copy()
b2[start_offset:start_offset + K] = 6000.0
t0 = time.perf_counter()
res2 = linprog(program.c, A_ub=program.A_ub, b_ub=program.b_ub, A_eq=program.A_eq, b_eq=b2,
               bounds=program.bounds, method="highs", options=SOLVER_OPTIONS)
t_reuse = time.perf_counter() - t0
print(f"复用 A 求解 {t_reuse:.3f}s V={res2.fun:.6f} success={res2.success}")

# 对照：完整重建
t0 = time.perf_counter()
p3 = build_decision_lp(params, scen_l, scen_r, weights, 6000.0)
res3 = solve(p3, "重建")
t_full = time.perf_counter() - t0
print(f"重建+求解 {t_full:.3f}s V={res3.fun:.6f}")
print("b_eq 差异", float(np.abs(p3.b_eq - b2).max()),
      "A_eq 差异", float(abs(p3.A_eq - program.A_eq).max()))
