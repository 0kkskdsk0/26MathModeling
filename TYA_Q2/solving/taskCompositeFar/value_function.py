# -*- coding: utf-8 -*-
"""终端价值函数 V_{d+1}(s) 的因果采样、次梯度提取、切线生成与网格自适应加密。

**建模口径（在读取任何对照结果之前固定）**

1. *终端价值的对象。* 第 $d$ 天使用的一步前瞻终端价值是
   $$V_{d+1}(s)=\\min\\Bigl\\{\\sum_{t\\in\\mathcal T}P_tG_t
   +\\sum_{\\omega\\in\\Omega_d}\\pi_{d,\\omega}\\sum_{t\\in\\mathcal T}5P_tE^\\omega_t
   \\;:\\;S^\\omega_{d,1}=s,\\ \\text{其余约束同第 }d\\text{ 天的近视模型}\\Bigr\\},$$
   即"以储能 $s$ 开局、在**第 $d$ 天已经可知的**情景池 $\\Omega_d$ 与核权重 $\\pi_{d,\\omega}$
   下的单日近视最优期望费用"。次日自己的池 $\\Omega_{d+1}$ 需要第 $d$ 天的观测，
   在第 $d$ 天 0:00 不可知，因此**不得**使用；用同一组 $(\\Omega_d,\\pi_{d,\\omega})$
   也保证了终端价值项与主模型的口径一致，两个模型才可配对。

2. *截断口径。* 滚动窗口：$V_{d+1}$ 的尾部仍取近视（不含更远的终端价值），
   即"一步前瞻 + 近视尾部"。递归不窥见未来，且与 taskComposite 的滚动架构同构。
   输出窗口首日之前的冷启动期（2025-01-01 至 01-31）取 $\\theta\\equiv0$，
   使两个模型在 02-01 的日初储电量逐元素相同，配对差值只来自终端价值项。

3. *采样与次梯度。* 在 $[S^{\\min},S^{\\max}]=[1200,10800]$ 上取网格点 $s_k$，
   对每个点解一次上述 LP，记最优值 $v_k=V_{d+1}(s_k)$。次梯度取**共同出发点约束**
   $S^\\omega_{d,1}=s$（$\\omega\\in\\Omega_d$，共 $|\\Omega_d|$ 条等式约束）的对偶变量之和
   $$g_k=\\sum_{\\omega\\in\\Omega_d}\\lambda^\\omega_k,\\qquad
   \\lambda^\\omega_k=\\frac{\\partial v_k}{\\partial b^\\omega},\\qquad b^\\omega=s_k .$$
   由线性规划值函数对右端项的次梯度性质，$g_k$ 是 $V_{d+1}$ 在 $s_k$ 处的次梯度，
   且因为 $V_{d+1}$ 关于 $s$ 单调不增、凸，必有 $g_k\\le0$ 且 $g$ 随 $s$ 单调不减。

4. *切线构造规则（每个采样点一条，不取区间端点中较紧者）。* 第 $k$ 条切线的斜率为
   $a_k=g_k$、截距为 $b_k=v_k-g_ks_k$，即
   $$\\theta\\ge a_k\\,s+b_k,\\qquad k=0,\\dots,N_r-1 .$$
   因为 $\\theta$ 以正权重 $\\pi_{d,\\omega}$ 进入最小化目标，模型会自动取
   $\\theta=\\max_k(a_ks+b_k)$，即切线的上包络；由次梯度性质，每条切线都在 $V_{d+1}$ 下方，
   所以包络自下方支撑 $V_{d+1}$，不会高估留电的价值。

5. *网格加密规则。* $N_0=5$ 的均匀网格起，每轮在相邻网格点之间插入中点
   （保留原点，$N_{r+1}=2N_r-1$，第 $r$ 轮网格是第 $r-1$ 轮的超集，已算过的点直接复用）。

6. *收敛判据（作用在相邻网格点的价值函数估计值之差上，不作用在区间宽度上）。*
   两条同时成立即判收敛：
   $$\\text{(A)}\\quad\\Delta_r=\\max_{k}\\bigl|\\hat V_r(s^{(r)}_{k+1})-\\hat V_r(s^{(r)}_k)\\bigr|\\le\\eta,
   \\qquad
   \\text{(B)}\\quad\\max_{s\\in\\mathcal S}\\bigl[\\hat V_{r+1}(s)-\\hat V_r(s)\\bigr]\\le\\tau_{\\mathrm{env}} .$$
   其中 $\\hat V_r$ 为第 $r$ 轮切线集合的上包络（在采样点上它与 $v_k$ 逐点相等）。
   (A) 是任务规定的判据：相邻网格点上价值函数估计值之差的最大值不超过阈值；
   (B) 是"再加密一轮不再产生新的有效切线"的可操作版本——对**严格凸**的价值函数，
   每个新采样点的切线都会在该点接触 $V$，因此按集合论意义"没有新切线"永不成立；
   把"新"操作化为"该轮新切线是否在容差之外抬高了逼近包络"，才是一个可达且
   有经济含义的判据。同时仍如实报告严格的集合差 `new_effective_tangents`
   （对严格凸函数它恒不为零），不掩盖事实。
   "有效切线"的集合身份由下述规则唯一确定：先按 $(a,b)$ 指纹（$\\epsilon_a=10^{-8}$ 元/kWh、
   $\\epsilon_b=10^{-6}$ 元）去重得到**规范切线**，再保留在检验网格
   $\\mathcal S$（$[1200,10800]$ 上 20001 点）上至少有一处满足
   $a_js+b_j\\ge\\hat V(s)-10^{-6}$ 的规范切线。

**运行**

    python -B TYA_Q2/solving/taskCompositeFar/value_function.py --calibrate
    python -B TYA_Q2/solving/taskCompositeFar/value_function.py --full

`--calibrate` 只在少数代表日上跑满 $R_{\\max}$ 轮，用于确定 $\\eta$；它不接触任何
远视/近视对照结果。`--full` 在全年 334 个输出日上执行同一套规则并落盘。
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
    from .common import (OUTPUT, SLOTS, Context, build_context, day_inputs)
else:  # 直接以脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taskCompositeFar.common import (OUTPUT, SLOTS, Context, build_context,
                                         day_inputs)

from taskComposite.model import build_decision_lp, solve  # noqa: E402

S_MIN, S_MAX = 1200.0, 10800.0
N0 = 5                  # 初始网格规模
R_MAX = 5               # 轮数上限（最终网格 65 点）
ETA = 100.0             # 判据 A：相邻网格点价值估计值之差的阈值（元）
TAU_ENV = 1.0           # 判据 B：再加密一轮允许的最大包络抬高量（元）
EPS_A = 1e-8            # 切线斜率指纹容差（元/kWh）
EPS_B = 1e-6            # 切线截距指纹容差（元）
TAU_ACTIVE = 1e-6       # 有效切线的活跃判据容差（元）
PROBE_POINTS = 20001    # 检验网格点数
ALLOWED_MONOTONE_VIOLATION = 1e-6   # 次梯度单调性容差（元/kWh）


def grid_for_round(r: int, n0: int = N0, s_min: float = S_MIN,
                   s_max: float = S_MAX) -> np.ndarray:
    """第 r 轮网格：在上一轮相邻点之间插入中点，保留原点。"""
    if r < 0:
        raise ValueError("轮次必须非负")
    n = (n0 - 1) * 2 ** r + 1
    return np.linspace(s_min, s_max, n)


def sample_value(params, scen_load: np.ndarray, scen_pv: np.ndarray,
                 weights: np.ndarray, s_start: float) -> tuple:
    """求解一次单日近视 LP，返回 (V(s), 共同出发点对偶之和)。"""
    program = build_decision_lp(params, scen_load, scen_pv, weights, s_start)
    result = solve(program, f"V({s_start:.4f})")
    scenarios = len(scen_load)
    if scenarios == 0:
        raise ValueError("空情景池没有值函数")
    offset = 2 * scenarios * SLOTS          # 与 model.VariableLayout 的等式块顺序一致
    marginals = np.asarray(result.eqlin.marginals, dtype=float)[offset:offset + scenarios]
    return float(result.fun), float(marginals.sum())


def tangents_from(values: np.ndarray, slopes: np.ndarray,
                  grid: np.ndarray) -> np.ndarray:
    """由采样点的值函数与次梯度生成切线 (a_k, b_k)，斜率 a_k = g_k。"""
    intercepts = values - slopes * grid
    return np.column_stack([slopes, intercepts])


def canonical_tangents(tangents: np.ndarray) -> np.ndarray:
    """按 (a,b) 指纹去重，得到规范切线集合。"""
    tangents = np.asarray(tangents, dtype=float)
    if tangents.size == 0:
        return tangents.reshape((0, 2))
    key = np.column_stack([np.round(tangents[:, 0] / EPS_A).astype(np.int64),
                           np.round(tangents[:, 1] / EPS_B).astype(np.int64)])
    _, index = np.unique(key, axis=0, return_index=True)
    return tangents[np.sort(index)]


def tangent_keys(tangents: np.ndarray) -> set:
    """切线的整数指纹集合，用于逐轮比较"是否产生新的有效切线"。"""
    tangents = np.asarray(tangents, dtype=float).reshape((-1, 2))
    return {(int(round(a / EPS_A)), int(round(b / EPS_B))) for a, b in tangents}


def envelope_values(tangents: np.ndarray, s_probe: np.ndarray) -> np.ndarray:
    """切线集合在给定点上的上包络。"""
    tangents = np.asarray(tangents, dtype=float).reshape((-1, 2))
    if len(tangents) == 0:
        return np.full(len(s_probe), -np.inf)
    return (tangents[:, 0:1] * s_probe[None, :] + tangents[:, 1:2]).max(axis=0)


def effective_tangents(tangents: np.ndarray, s_probe: np.ndarray) -> tuple:
    """返回 (有效切线, 每条有效切线的作用区间)。

    有效 = 规范切线中，在检验网格上至少有一处其值不低于上包络减 `TAU_ACTIVE`。
    作用区间按"该切线是唯一 argmax（在容差内）"的采样点范围给出。
    """
    canon = canonical_tangents(tangents)
    if len(canon) == 0:
        return canon, []
    values = canon[:, 0:1] * s_probe[None, :] + canon[:, 1:2]
    env = values.max(axis=0)
    active = values >= env[None, :] - TAU_ACTIVE
    keep = active.any(axis=1)
    kept, kept_active = canon[keep], active[keep]
    spans = []
    for row in kept_active:
        index = np.flatnonzero(row)
        spans.append([float(s_probe[index[0]]), float(s_probe[index[-1]])])
    order = np.argsort(kept[:, 0])
    return kept[order], [spans[i] for i in order]


def round_metrics(values: np.ndarray, slopes: np.ndarray, tangents: np.ndarray,
                  s_probe: np.ndarray) -> dict:
    """单轮诊断量：最大相邻差、次梯度符号与单调性违反、有效切线数。

    次梯度的单调性只在**内部网格点**上检查：$s=S^{\\max}$ 处上界约束绑定，
   该点的一阶信息是单侧的，把它计入会掩盖真实的单调性信息。
    """
    differences = np.abs(np.diff(values))
    interior = np.diff(slopes[:-1]) if len(slopes) > 2 else np.zeros(0)
    effective, spans = effective_tangents(tangents, s_probe)
    return {
        "max_adjacent_difference_yuan": float(differences.max()) if len(differences) else 0.0,
        "mean_adjacent_difference_yuan": float(differences.mean()) if len(differences) else 0.0,
        "min_adjacent_difference_yuan": float(differences.min()) if len(differences) else 0.0,
        "max_slope_positive_violation": float(max(0.0, slopes.max())),
        "max_slope_monotone_violation_interior": float(max(0.0, interior.max()))
        if len(interior) else 0.0,
        "boundary_slope_jump": float(slopes[-1] - slopes[-2]) if len(slopes) > 1 else 0.0,
        "canonical_tangent_count": int(len(canonical_tangents(tangents))),
        "effective_tangent_count": int(len(effective)),
        "value_at_s_min_yuan": float(values[0]),
        "value_at_s_max_yuan": float(values[-1]),
    }, effective, spans


def evaluate_day(ctx: Context, day, eta: float = ETA, tau_env: float = TAU_ENV,
                 r_max: int = R_MAX, n0: int = N0,
                 s_probe: np.ndarray | None = None) -> dict:
    """对第 day 天逐轮加密采样网格，直到满足收敛判据或触及轮数上限。"""
    inputs = day_inputs(ctx, day)
    if len(inputs["pool"]) == 0:
        return {"date": pd.Timestamp(day), "pool_size": 0, "mode": inputs["mode"],
                "converged": False, "converged_round": -1, "rounds": [],
                "grid": np.zeros(0), "values": np.zeros(0), "slopes": np.zeros(0),
                "tangents": np.zeros((0, 2)), "tag": "empty_pool"}
    if s_probe is None:
        s_probe = np.linspace(S_MIN, S_MAX, PROBE_POINTS)
    scen_l, scen_r, weights = inputs["load"], inputs["pv"], inputs["weights"]

    grid = np.zeros(0)
    values_prev = slopes_prev = np.zeros(0)
    values = slopes = np.zeros(0)
    previous_envelope = None
    rounds, solutions, elapsed = [], 0, 0.0
    converged, converged_round = False, -1
    for r in range(r_max + 1):
        target = grid_for_round(r, n0)
        fresh = target if r == 0 else target[1::2]   # 第 r 轮网格是第 r-1 轮的超集
        fresh_values = np.empty(len(fresh))
        fresh_slopes = np.empty(len(fresh))
        for i, s in enumerate(fresh):
            begin = time.perf_counter()
            value, slope = sample_value(ctx.params, scen_l, scen_r, weights, float(s))
            elapsed += time.perf_counter() - begin
            solutions += 1
            fresh_values[i], fresh_slopes[i] = value, slope
        grid = target
        values = np.empty(len(target))
        slopes = np.empty(len(target))
        if r == 0:
            values, slopes = fresh_values, fresh_slopes
        else:
            values[0::2], slopes[0::2] = values_prev, slopes_prev
            values[1::2], slopes[1::2] = fresh_values, fresh_slopes
        tangents = tangents_from(values, slopes, grid)
        metrics, effective, spans = round_metrics(values, slopes, tangents, s_probe)
        envelope = envelope_values(tangents, s_probe)
        metrics.update({
            "round": r, "n_points": int(len(grid)),
            "grid_step": float(grid[1] - grid[0]) if len(grid) > 1 else 0.0,
            "new_effective_tangents": -1,
            "envelope_increase_yuan": -1.0,
            "max_envelope_deficit_yuan": float(
                np.max(envelope_values(effective, grid) - values)),
        })
        if previous_envelope is not None:
            metrics["new_effective_tangents"] = len(
                tangent_keys(effective) - tangent_keys(previous_effective))
            metrics["envelope_increase_yuan"] = float(
                np.max(envelope - previous_envelope))
            if (rounds[-1]["max_adjacent_difference_yuan"] <= eta
                    and metrics["envelope_increase_yuan"] <= tau_env):
                converged, converged_round = True, r
        rounds.append(metrics)
        previous_effective, previous_envelope = effective, envelope
        values_prev, slopes_prev = values, slopes
        if converged:
            break

    effective, spans = effective_tangents(tangents_from(values, slopes, grid), s_probe)
    return {
        "date": pd.Timestamp(day), "pool_size": int(len(inputs["pool"])),
        "mode": inputs["mode"], "converged": bool(converged),
        "converged_round": int(converged_round), "rounds": rounds,
        "grid": grid, "values": values, "slopes": slopes,
        "tangents": effective, "spans": spans, "solutions": int(solutions),
        "seconds": float(elapsed), "tag": "ok",
    }


# ---------------------------------------------------------------------------
# 并行驱动
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _initialise(days: int | None) -> None:
    _WORKER["ctx"] = build_context(days=days)
    _WORKER["probe"] = np.linspace(S_MIN, S_MAX, PROBE_POINTS)


def _task(payload: dict) -> dict:
    ctx = _WORKER["ctx"]
    return evaluate_day(ctx, pd.Timestamp(payload["date"]), eta=payload["eta"],
                        tau_env=payload["tau_env"], r_max=payload["r_max"],
                        n0=payload["n0"], s_probe=_WORKER["probe"])


def evaluate_many(dates, eta: float = ETA, tau_env: float = TAU_ENV, r_max: int = R_MAX,
                  n0: int = N0, workers: int = 1, days: int | None = None,
                  verbose: bool = True) -> list:
    """对一组日期求值；workers>1 时按天并行。"""
    payloads = [{"date": str(pd.Timestamp(d)), "eta": eta, "tau_env": tau_env,
                 "r_max": r_max, "n0": n0} for d in dates]
    if workers <= 1:
        _initialise(days)
        results = []
        for i, payload in enumerate(payloads):
            results.append(_task(payload))
            if verbose and (i % 20 == 0 or i == len(payloads) - 1):
                print(f"  [{i + 1}/{len(payloads)}] {payload['date']} "
                      f"轮数={len(results[-1]['rounds'])} "
                      f"收敛={results[-1]['converged']}", flush=True)
        return results
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise,
                             initargs=(days,)) as pool:
        for i, result in enumerate(pool.map(_task, payloads, chunksize=1)):
            results.append(result)
            if verbose and (i % 20 == 0 or i == len(payloads) - 1):
                print(f"  [{i + 1}/{len(payloads)}] {result['date'].date()} "
                      f"轮数={len(result['rounds'])} 收敛={result['converged']}",
                      flush=True)
    return results


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def rounds_frame(results: list) -> pd.DataFrame:
    """把逐日逐轮诊断展开成长表。"""
    rows = []
    for result in results:
        for metrics in result["rounds"]:
            rows.append({"date": result["date"], "pool_size": result["pool_size"],
                         "converged": result["converged"],
                         "converged_round": result["converged_round"], **metrics})
    return pd.DataFrame(rows)


def save_results(results: list, output: Path = OUTPUT, tag: str = "value_function") -> dict:
    """落盘逐轮诊断、采样值、最终有效切线与全局收敛判定。"""
    output.mkdir(parents=True, exist_ok=True)
    table = rounds_frame(results)
    table.to_csv(output / f"{tag}_rounds.csv", index=False, encoding="utf-8-sig",
                 float_format="%.10g")

    grid_parts, value_parts, slope_parts, tangent_parts = [], [], [], []
    grid_index, tangent_index, dates, summaries = [], [], [], []
    grid_total = tangent_total = 0
    for result in results:
        dates.append(str(result["date"].date()))
        grid_index.append((grid_total, len(result["grid"])))
        grid_total += len(result["grid"])
        grid_parts.append(np.asarray(result["grid"], dtype=float))
        value_parts.append(np.asarray(result["values"], dtype=float))
        slope_parts.append(np.asarray(result["slopes"], dtype=float))
        tangent_index.append((tangent_total, len(result["tangents"])))
        tangent_total += len(result["tangents"])
        tangent_parts.append(np.asarray(result["tangents"], dtype=float).reshape((-1, 2)))
        summaries.append({
            "date": dates[-1], "pool_size": result["pool_size"], "mode": result["mode"],
            "converged": result["converged"], "converged_round": result["converged_round"],
            "n_grid_points": int(len(result["grid"])),
            "n_rounds": len(result["rounds"]),
            "n_solutions": result.get("solutions", 0),
            "solver_seconds": round(result.get("seconds", 0.0), 4),
            "n_effective_tangents": int(len(result["tangents"])),
            "max_adjacent_difference_yuan": (
                result["rounds"][-1]["max_adjacent_difference_yuan"]
                if result["rounds"] else 0.0),
            "status": result.get("tag", "ok"),
        })
    np.savez_compressed(
        output / f"{tag}_samples.npz",
        dates=np.asarray(dates),
        grid=np.concatenate(grid_parts) if grid_parts else np.zeros(0),
        grid_offset=np.asarray([o for o, _ in grid_index], dtype=np.int64),
        grid_length=np.asarray([n for _, n in grid_index], dtype=np.int64),
        values=np.concatenate(value_parts) if value_parts else np.zeros(0),
        slopes=np.concatenate(slope_parts) if slope_parts else np.zeros(0),
        tangents=np.concatenate(tangent_parts) if tangent_parts else np.zeros((0, 2)),
        tangent_offset=np.asarray([o for o, _ in tangent_index], dtype=np.int64),
        tangent_length=np.asarray([n for _, n in tangent_index], dtype=np.int64),
    )
    converged = [s for s in summaries if s["converged"]]
    summary = {
        "eta_yuan": ETA, "tau_env_yuan": TAU_ENV, "n0": N0, "r_max": R_MAX,
        "days": len(summaries), "days_converged": len(converged),
        "max_rounds_used": max(s["n_rounds"] for s in summaries) if summaries else 0,
        "max_grid_points": max(s["n_grid_points"] for s in summaries) if summaries else 0,
        "total_linear_programs": int(sum(s["n_solutions"] for s in summaries)),
        "total_solver_seconds": round(sum(s["solver_seconds"] for s in summaries), 2),
        "max_adjacent_difference_yuan": max(
            (s["max_adjacent_difference_yuan"] for s in summaries), default=0.0),
        "min_effective_tangents": min((s["n_effective_tangents"] for s in summaries),
                                      default=0),
        "max_effective_tangents": max((s["n_effective_tangents"] for s in summaries),
                                      default=0),
        "per_round_max_adjacent_difference_yuan": {
            int(r): float(g["max_adjacent_difference_yuan"].max())
            for r, g in table.groupby("round")},
        "per_round_max_envelope_increase_yuan": {
            int(r): float(g["envelope_increase_yuan"].max())
            for r, g in table.groupby("round")},
        "per_round_max_new_effective_tangents": {
            int(r): int(g["new_effective_tangents"].max())
            for r, g in table.groupby("round")},
        "per_round_days": {int(r): int(len(g)) for r, g in table.groupby("round")},
        "not_converged_dates": [s["date"] for s in summaries if not s["converged"]],
    }
    (output / f"{tag}_summary.json").write_text(
        json.dumps({"summary": summary, "days": summaries}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    return summary


def load_tangents(date, output: Path = OUTPUT, tag: str = "value_function") -> np.ndarray:
    """读取某一天的最终有效切线集合，供远视模型装配使用。"""
    with np.load(output / f"{tag}_samples.npz", allow_pickle=False) as data:
        dates = [str(x) for x in data["dates"]]
        index = dates.index(str(pd.Timestamp(date).date()))
        offset = int(data["tangent_offset"][index])
        length = int(data["tangent_length"][index])
        return data["tangents"][offset:offset + length].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="只在代表日上跑满轮数上限，用于确定阈值 eta 与 tau_env")
    parser.add_argument("--full", action="store_true", help="全年 334 天逐轮加密")
    parser.add_argument("--days", type=int, default=None, help="只取前若干天，快速自检")
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--tau-env", type=float, default=TAU_ENV)
    parser.add_argument("--r-max", type=int, default=R_MAX)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.calibrate:
        ctx = build_context(days=args.days)
        pool = ctx.output_dates
        picks = []
        for month in range(2, 13):
            month_days = pool[pool.month == month]
            if len(month_days):
                picks.append(month_days[len(month_days) // 2])
        picks.append(pool[-1])
        picks = pd.DatetimeIndex(sorted(set(picks)))
        print(f"标定日 {len(picks)} 天：{', '.join(str(d.date()) for d in picks)}",
              flush=True)
        started = time.perf_counter()
        results = evaluate_many(picks, eta=-np.inf, tau_env=-np.inf, r_max=args.r_max,
                                workers=args.workers, days=args.days)
        table = rounds_frame(results)
        summary = {int(r): {"days": int(len(g)),
                            "max_delta": float(g["max_adjacent_difference_yuan"].max()),
                            "median_delta": float(g["max_adjacent_difference_yuan"].median()),
                            "max_new_effective": int(g["new_effective_tangents"].max()),
                            "max_effective_count": int(g["effective_tangent_count"].max()),
                            "max_envelope_increase": float(
                                g["envelope_increase_yuan"].max()),
                            "median_envelope_increase": float(
                                g["envelope_increase_yuan"].median()),
                            "max_monotone_violation_interior": float(
                                g["max_slope_monotone_violation_interior"].max()),
                            "max_slope_positive_violation": float(
                                g["max_slope_positive_violation"].max())}
                   for r, g in table.groupby("round")}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "value_function_calibration.json").write_text(
            json.dumps({"days": [str(d.date()) for d in picks], "per_round": summary,
                        "r_max": args.r_max, "n0": N0,
                        "wall_seconds": round(time.perf_counter() - started, 1)},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    dates = build_context(days=args.days).output_dates
    started = time.perf_counter()
    results = evaluate_many(dates, eta=args.eta, tau_env=args.tau_env, r_max=args.r_max,
                            workers=args.workers, days=args.days)
    summary = save_results(results, tag="value_function")
    print(json.dumps({"wall_seconds": round(time.perf_counter() - started, 1),
                      **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
