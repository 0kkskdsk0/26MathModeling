# -*- coding: utf-8 -*-
"""临时探针：测量单次决策层 LP 的耗时，并抽查 V(s) 与共同出发点对偶变量。"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "TYA_Q2/solving"))

from taskComposite.kernel import (KernelRule, build_context_features, load_attachment,
                                  columns_for)
from taskComposite.model import ModelParams, build_decision_lp, solve, unpack_decision
from taskComposite.solve_year import load_price, scenario_frame, weights_for  # noqa

ATTACH = ROOT / "ProblemC/附件"
price = load_price(ATTACH / "附件1.xlsx")
params = ModelParams(price=price)
load_power, pv_power = load_attachment(ATTACH / "附件2.xlsx")
features = build_context_features(load_power, pv_power)
rule = KernelRule(("season", "net_lag_7d", "pv_mean_3d", "load_lag_7d", "net_mean_7d",
                   "low_load_day"), 1.0438414641946645)

day = pd.Timestamp("2025-06-21")
pool, weights, mode = weights_for(rule, features, day, load_power, pv_power)
print("pool", len(pool), mode)
scen_l, scen_r = scenario_frame(load_power, pv_power, pool)
K = len(pool)
start_offset = 2 * K * 144

rows = []
for s in [1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0, 8400.0, 9600.0, 10800.0]:
    t0 = time.perf_counter()
    program = build_decision_lp(params, scen_l, scen_r, weights, s)
    result = solve(program, "probe")
    dt = time.perf_counter() - t0
    lam = np.asarray(result.eqlin.marginals)[start_offset:start_offset + K]
    g = float(lam.sum())
    sol = unpack_decision(result, program.layout, K)
    rows.append({"s": s, "V": float(result.fun), "g_sum_lambda": g,
                 "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                 "G_total": float(sol["G"].sum()),
                 "S_end_mean": float((weights @ sol["S"][:, -1])),
                 "S_end_min": float(sol["S"][:, -1].min()),
                 "S_end_max": float(sol["S"][:, -1].max()),
                 "E_exp": float(sum(weights[k] * sol["E"][k].sum() for k in range(K))),
                 "seconds": dt})
    print(rows[-1], flush=True)

table = pd.DataFrame(rows)
print(table.to_string())
print("mean seconds", table.seconds.mean())
