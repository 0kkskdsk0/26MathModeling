"""跨日终端价值（对齐 problem3_solving_final.md 式 24-26）。

17 个 SOC 网格点 + 确定性 24h 影子（次日点预测）+ 斜率投影凸分段线性。
"""
import numpy as np

import myh.src.config as config
from myh.src.dispatch import solve_dispatch


def _isotonic_nondecreasing(y):
    y = np.asarray(y, dtype=float).copy()
    if len(y) == 0:
        return y
    counts = np.ones(len(y), dtype=int)
    values = y
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
        else:
            total = values[i] * counts[i] + values[i + 1] * counts[i + 1]
            cnt = counts[i] + counts[i + 1]
            values[i] = total / cnt
            counts[i] = cnt
            values = np.delete(values, i + 1)
            counts = np.delete(counts, i + 1)
            if i > 0:
                i -= 1
    return np.repeat(values, counts)


def compute_terminal_value(load_next, pv_next, price_next,
                           grid=None, end_soc=None):
    """返回 (a, b)，V(s)=max(0, max_q(a_q s + b_q))。"""
    grid = np.asarray(config.TERM_GRID if grid is None else grid, float)
    end_soc = config.TERM_END_SOC if end_soc is None else end_soc

    Q = []
    for s_q in grid:
        res = solve_dispatch(load_next, pv_next, price_next, s0=s_q, s_end=end_soc)
        if res["status"] != "ok":
            raise RuntimeError(f"terminal shadow failed at SOC={s_q}: {res.get('message')}")
        Q.append(res["obj"])
    Q = np.asarray(Q)

    step = grid[1] - grid[0]
    slopes = np.minimum(_isotonic_nondecreasing(np.diff(Q) / step), 0.0)
    Q_tilde = np.r_[Q[0], Q[0] + step * np.cumsum(slopes)]
    return slopes, Q_tilde[:-1] - slopes * grid[:-1]
