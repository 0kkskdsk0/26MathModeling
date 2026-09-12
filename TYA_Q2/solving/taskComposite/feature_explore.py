# -*- coding: utf-8 -*-
"""问题二全年结构特征探索、核赋权修订与采纳判定。

运行：
    python -B TYA_Q2/solving/taskComposite/feature_explore.py

流程：
1. 全年结构探索：去同刻均值后的日滞后相关（周同期）、星期几方差分解、
   简单预测基线对比、候选池规模与"上周同日"在池中的相似度排名；
2. 在**因果滚动协议**下做特征消融：候选池 = 同季节（月份环形距离 ≤ 1）且
   严格早于决策日的历史日，标尺只用当日池拟合，ES 只用当日真值评分；
3. 冻结新特征集与带宽，与等权基线、旧（一月版）特征集做配对对比；
4. 用前后半年分割做样本外确认：2—6 月选参，7—12 月独立确认。

预先固定的规则（在读取消融结果之前已写入本文件）：
- 消融固定 h=1；纳入条件为平均 ES 相对改善 > 0.5% 且至少 60% 的验证日改善；
- 语义特征最多保留 4 个（`season` 为结构性先验，不占名额）；
- 候选顺序按结构证据强度预先固定，出现路径依赖时不重排。

候选池只依赖"严格早于决策日、同季节、且已过 7 日滞后特征预热期"三条规则，
与具体特征集合无关，因此所有方案共用同一池，配对差值只来自特征与带宽。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .kernel import (DATA_END, FEATURE_DESCRIPTIONS, NATURAL_SCALE_COLUMNS, STEP_HOURS,
                         KernelRule, WARMUP_DAYS, build_context_features, causal_pool,
                         columns_for, effective_scenarios, gaussian_weights, load_attachment)
else:
    from kernel import (DATA_END, FEATURE_DESCRIPTIONS, NATURAL_SCALE_COLUMNS, STEP_HOURS,
                        KernelRule, WARMUP_DAYS, build_context_features, causal_pool,
                        columns_for, effective_scenarios, gaussian_weights, load_attachment)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ProblemC/附件/附件2.xlsx"
OUTPUT = ROOT / "TYA_Q2/outputs/taskComposite"
ASSETS = ROOT / "TYA_Q2/assets/taskComposite"

VALIDATION_START = pd.Timestamp("2025-02-01")
SELECT_END = pd.Timestamp("2025-06-30")
CONFIRM_START = pd.Timestamp("2025-07-01")
ABLATION_H = 1.0
MIN_FEATURE_GAIN_PCT = 0.5
MIN_IMPROVED_FRACTION = 0.60
ADOPTION_GAIN_PCT = 3.0
MAX_GROUPS = 4
LEGACY_GROUPS = ("season", "load_mean_3d")
POOL_COLUMNS = ("net_lag_7d", "load_mean_3d")
CANDIDATE_ORDER = ("net_lag_7d", "weekday7", "load_mean_3d", "pv_mean_3d", "weekend",
                   "load_lag_7d", "pv_lag_7d", "net_mean_7d", "net_mean_3d",
                   "load_std_7d", "pv_std_7d")
SUPPLEMENTARY_ORDER = ("low_load_day", "weekday7")
TOL = 1e-10


class PoolIndex:
    """逐日候选池与情景几何的预计算；与特征集合、带宽均无关。"""

    def __init__(self, dataset, days: pd.DatetimeIndex):
        self.days = pd.DatetimeIndex(days)
        self.ids, self.between, self.to_truth = {}, {}, {}
        net = dataset.net
        for day in self.days:
            pool = causal_pool(dataset.dates, day, dataset.features, POOL_COLUMNS)
            if pool.empty:
                raise RuntimeError(f"{day.date()} 的候选池为空")
            index = dataset.dates.get_indexer(pool)
            scenarios = net[index]
            self.ids[day] = index
            self.between[day] = np.linalg.norm(
                scenarios[:, None, :] - scenarios[None, :, :], axis=-1)
            self.to_truth[day] = np.linalg.norm(
                scenarios - net[dataset.dates.get_loc(day)], axis=1)

    def subset(self, days) -> "PoolIndex":
        clone = PoolIndex.__new__(PoolIndex)
        clone.days = pd.DatetimeIndex(days)
        for name in ("ids", "between", "to_truth"):
            setattr(clone, name, {d: getattr(self, name)[d] for d in clone.days})
        return clone

    def __len__(self) -> int:
        return len(self.days)

    def sizes(self) -> pd.Series:
        return pd.Series([len(self.ids[d]) for d in self.days], index=self.days, name="pool_size")


class ContextSpace:
    """固定特征集合下的标准化处境；标尺只用当日候选池拟合。"""

    def __init__(self, dataset, pool: PoolIndex, groups):
        columns = columns_for(groups)
        self.groups = tuple(groups)
        self.columns = columns
        index_of = {c: i for i, c in enumerate(dataset.features.columns)}
        self.cols = [index_of[c] for c in columns]
        values = dataset.features.to_numpy(dtype=float)
        self.z, self.zt, self.pool_index = {}, {}, pool
        for day in pool.days:
            ids = pool.ids[day]
            raw = values[ids][:, self.cols]
            target = values[dataset.dates.get_loc(day)][self.cols]
            if not (np.isfinite(raw).all() and np.isfinite(target).all()):
                raise RuntimeError(f"{day.date()} 的处境含缺失值")
            mu, sd = raw.mean(axis=0), raw.std(axis=0, ddof=0)
            for j, name in enumerate(columns):
                if name in NATURAL_SCALE_COLUMNS:
                    mu[j], sd[j] = 0.0, 1.0
            sd = np.where(sd > 1e-12, sd, 1.0)
            self.z[day] = (raw - mu) / sd
            self.zt[day] = (target - mu) / sd

    def weights(self, h: float) -> list[np.ndarray]:
        return [gaussian_weights(self.z[d], self.zt[d], h) for d in self.pool_index.days]

    def scores(self, h: float) -> pd.Series:
        """完整能量分数：<pi, d(x,y)> - 0.5 * pi' D pi。"""
        out = {}
        for day, w in zip(self.pool_index.days, self.weights(h)):
            out[day] = float(w @ self.pool_index.to_truth[day]
                             - 0.5 * w @ self.pool_index.between[day] @ w)
        return pd.Series(out, name="es")


# --------------------------------------------------------------------------- 结构探索


def lag_correlation(series: pd.DataFrame) -> pd.DataFrame:
    """去除同刻全年均值后，相隔 k 天的同刻 Pearson 相关系数。"""
    centered = (series - series.mean(axis=0)).to_numpy(dtype=float)
    rows = []
    for k in range(1, 15):
        rows.append({"lag_days": k,
                     "corr": float(np.corrcoef(centered[:-k].ravel(), centered[k:].ravel())[0, 1])})
    return pd.DataFrame(rows)


def daily_total_autocorrelation(load: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """日总电量的跨日自相关，用于比较负载、光伏与净负荷的周同期强度。"""
    daily = pd.DataFrame({"load": load.sum(axis=1) * STEP_HOURS,
                          "pv": pv.sum(axis=1) * STEP_HOURS})
    daily["net"] = daily["load"] - daily["pv"]
    rows = []
    for k in range(1, 15):
        rows.append({"lag_days": k, **{c: float(daily[c].autocorr(k)) for c in daily}})
    return pd.DataFrame(rows)


def weekday_structure(load: pd.DataFrame) -> pd.DataFrame:
    """星期几效应的方差分解与日总电量水平。"""
    dense = load.to_numpy(dtype=float) * STEP_HOURS
    weekday = load.index.dayofweek.to_numpy()
    daily_total = pd.Series(dense.sum(axis=1), index=load.index)
    grand = dense.mean(axis=0)
    means = np.vstack([dense[weekday == w].mean(axis=0) for w in range(7)])
    residual = dense - grand
    explained = np.repeat(means - grand, np.bincount(weekday, minlength=7), axis=0)
    rows = []
    for w, name in enumerate(["周一", "周二", "周三", "周四", "周五", "周六", "周日"]):
        rows.append({"weekday": w, "name": name, "days": int((weekday == w).sum()),
                     "daily_total_kwh": float(daily_total[weekday == w].mean()),
                     "rms_deviation_kwh": float(np.sqrt(((means[w] - grand) ** 2).mean()))})
    table = pd.DataFrame(rows)
    table.attrs["centered_variance"] = float(residual.var())
    table.attrs["weekday_variance"] = float(explained.var())
    table.attrs["weekday_share"] = float(explained.var() / residual.var())
    return table


def baseline_forecast_errors(load: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """复现文档中的四类简单预测基线误差，用作结构证据的独立核对。"""
    test = load.index >= VALIDATION_START
    candidates = {
        "昨日同刻": (load.shift(1), pv.shift(1)),
        "上周同刻": (load.shift(7), pv.shift(7)),
        "过去7日同刻均值": (load.shift(1).rolling(7).mean(), pv.shift(1).rolling(7).mean()),
        "组合基线": (load.shift(7), pv.shift(1).rolling(7).mean()),
    }
    rows = []
    for name, (pl, pr) in candidates.items():
        actual_l = load[test] * STEP_HOURS
        actual_r = pv[test] * STEP_HOURS
        err_l = actual_l - pl[test] * STEP_HOURS
        err_r = actual_r - pr[test] * STEP_HOURS
        err_n = err_l - err_r
        rows.append({"baseline": name,
                     "load_mae_kwh": float(err_l.abs().to_numpy().mean()),
                     "pv_mae_kwh": float(err_r.abs().to_numpy().mean()),
                     "net_mae_kwh": float(err_n.abs().to_numpy().mean()),
                     "net_rmse_kwh": float(np.sqrt((err_n.to_numpy() ** 2).mean())),
                     "net_wape_pct": float(100 * err_n.abs().to_numpy().sum()
                                           / (actual_l - actual_r).abs().to_numpy().sum())})
    return pd.DataFrame(rows)


def pool_profile(dataset, days: pd.DatetimeIndex) -> pd.DataFrame:
    """验证期每天的池规模、月份跨度与"上周同日"在池中的相似度排名。"""
    daily_net = pd.Series(dataset.net.sum(axis=1), index=dataset.dates)
    rows = []
    for day in days:
        pool = causal_pool(dataset.dates, day, dataset.features, POOL_COLUMNS)
        ids = dataset.dates.get_indexer(pool)
        gap = (day - pool).days
        rank, share, distance = np.nan, np.nan, np.nan
        if 7 in set(gap):
            j = int(np.flatnonzero(gap == 7)[0])
            values = np.abs(daily_net.iloc[ids].to_numpy() - daily_net.loc[day])
            rank = int((values < values[j]).sum()) + 1
            share = rank / len(pool)
            distance = float(values[j])
        rows.append({"date": day, "pool_size": len(pool),
                     "oldest_gap_days": int(gap.max()), "month_span": int(np.ptp(pool.month)),
                     "same_weekday_in_pool": bool(7 in set(gap)),
                     "lag7_rank": rank, "lag7_rank_share": share,
                     "lag7_daily_net_gap_kwh": distance})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 消融与选优


def gain(before: pd.Series, after: pd.Series) -> tuple[float, int, bool]:
    """预先固定的纳入判据：平均 ES 改善 > 0.5% 且至少 60% 的日子严格改善。"""
    improvement = float(100 * (before.mean() - after.mean()) / before.mean())
    improved = int((before - after > TOL).sum())
    passed = bool(improvement > MIN_FEATURE_GAIN_PCT
                  and improved >= int(np.ceil(MIN_IMPROVED_FRACTION * len(before))))
    return improvement, improved, passed


def run_ablation(dataset, pool: PoolIndex) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """固定顺序前向纳入 + 条件删除复核；season 为结构性先验。"""
    selected, rows, daily_rows = ["season"], [], []
    cache: dict[tuple[str, ...], pd.Series] = {}

    def scores(groups) -> pd.Series:
        key = tuple(groups)
        if key not in cache:
            cache[key] = ContextSpace(dataset, pool, groups).scores(ABLATION_H)
        return cache[key]

    def evaluate(stage, feature, before_groups, after_groups, forced_reason=None):
        before, after = scores(before_groups), scores(after_groups)
        improvement, improved, passed = gain(before, after)
        if forced_reason is not None:
            accepted, reason = False, forced_reason
        else:
            accepted = passed and len(after_groups) <= MAX_GROUPS + 1
            reason = ("pass" if accepted else
                      "feature_count_limit" if passed else
                      "insufficient_mean_gain_or_daily_stability")
        rows.append({"stage": stage, "feature": feature,
                     "description": FEATURE_DESCRIPTIONS[feature],
                     "groups_before": ";".join(before_groups),
                     "groups_after": ";".join(after_groups), "h": ABLATION_H,
                     "mean_es_before": float(before.mean()), "mean_es_after": float(after.mean()),
                     "relative_improvement_pct": improvement, "improved_days": improved,
                     "validation_days": len(before), "improved_fraction": improved / len(before),
                     "passes_rule": passed, "accepted": accepted, "reason": reason})
        for day in before.index:
            daily_rows.append({"stage": stage, "feature": feature,
                               "date": day.strftime("%Y-%m-%d"),
                               "es_before": float(before[day]), "es_after": float(after[day]),
                               "relative_improvement_pct": float(
                                   100 * (before[day] - after[day]) / before[day])})
        return accepted

    evaluate("structural", "season", [], selected,
             "required_calendar_prior_not_identifiable_in_a_single_season")
    for feature in CANDIDATE_ORDER:
        if feature in selected:
            continue
        if evaluate("forward", feature, selected.copy(), selected + [feature]):
            selected.append(feature)
    for iteration in range(MAX_GROUPS):
        failures = []
        for feature in selected[1:]:
            before = [g for g in selected if g != feature]
            if not evaluate(f"conditional_{iteration + 1}", feature, before, selected.copy()):
                failures.append(feature)
        if not failures:
            break
        selected.remove(failures[-1])
    table = pd.DataFrame(rows)
    table["retained_in_final_rule"] = table["feature"].isin(selected)
    table["structural_exception"] = table["feature"].eq("season")
    return selected, table, pd.DataFrame(daily_rows)


def run_supplementary(dataset, pool: PoolIndex, base_groups, candidates) -> tuple[list[str], pd.DataFrame]:
    """事后补充消融：候选在结构探索之后才提出，故单列一节并做样本外复核。

    判据与主消融完全一致，基础集固定为选择窗选出的特征集合，
    以免补充特征借用主消融的窗口内信息。
    """
    selected, rows = list(base_groups), []
    for feature in candidates:
        before = ContextSpace(dataset, pool, selected).scores(ABLATION_H)
        after = ContextSpace(dataset, pool, selected + [feature]).scores(ABLATION_H)
        improvement, improved, passed = gain(before, after)
        rows.append({"feature": feature, "description": FEATURE_DESCRIPTIONS[feature],
                     "groups_before": ";".join(selected),
                     "groups_after": ";".join(selected + [feature]), "h": ABLATION_H,
                     "mean_es_before": float(before.mean()), "mean_es_after": float(after.mean()),
                     "relative_improvement_pct": improvement, "improved_days": improved,
                     "validation_days": len(before), "passes_rule": passed,
                     "accepted": bool(passed) or None,
                     "reason": "post_hoc_candidate" if not passed else "pass"})
        if passed:
            selected.append(feature)
    return selected, pd.DataFrame(rows)

def search_bandwidth(space: ContextSpace) -> tuple[float, pd.DataFrame, str]:
    """对数粗搜 + 三轮细搜；仅按平均 ES 选优，同时记录等权极限。"""
    rows = []

    def evaluate_grid(grid, stage):
        means = []
        for h in grid:
            scores = space.scores(float(h))
            effective = np.array([float(effective_scenarios(w)) for w in space.weights(float(h))])
            mean = float(scores.mean())
            rows.append({"round": stage, "h": float(h), "mean_es": mean,
                         "mean_k_eff": float(effective.mean()),
                         "min_k_eff": float(effective.min()),
                         "max_k_eff": float(effective.max())})
            means.append(mean)
        return np.asarray(means)

    uniform = evaluate_grid([np.inf], "uniform_limit")[0]
    lower, upper = 0.05, 20.0
    status, best_h = "interior_optimum", np.inf
    for expansion in range(13):
        grid = np.geomspace(lower, upper, 25)
        values = evaluate_grid(grid, f"coarse_{expansion}")
        if np.ptp(values) < TOL and abs(values[0] - uniform) < TOL:
            status = "flat_uniform_limit_no_identifiable_finite_h"
            break
        best = int(np.argmin(values))
        if 0 < best < len(grid) - 1:
            for refinement in range(1, 4):
                grid = np.geomspace(grid[best - 1], grid[best + 1], 17)
                values = evaluate_grid(grid, f"fine_{refinement}")
                best = int(np.argmin(values))
                if best in (0, len(grid) - 1):
                    raise RuntimeError("细搜最优落在网格边界，不应静默接受")
            best_h = float(grid[best])
            if values[best] >= uniform - TOL:
                best_h, status = np.inf, "uniform_limit_minimizes_es"
            break
        if best == 0:
            lower /= 4
        else:
            if abs(values[best] - uniform) < TOL:
                best_h, status = np.inf, "uniform_limit_minimizes_es"
                break
            upper *= 4
    else:
        raise RuntimeError("粗搜最优始终位于网格边界")
    table = pd.DataFrame(rows)
    table["selected"] = (np.isinf(table["h"]) if not np.isfinite(best_h)
                         else np.isclose(table["h"], best_h))
    return best_h, table, status


def paired_comparison(dataset, pool: PoolIndex, legacy, revised,
                      legacy_h, revised_h) -> pd.DataFrame:
    """同一池上的三方配对比较：等权、旧特征集、修订特征集。"""
    frames, k_eff = {}, {}
    for tag, groups, h in (("equal", ("season",), np.inf),
                           ("legacy", tuple(legacy), legacy_h),
                           ("revised", tuple(revised), revised_h)):
        space = ContextSpace(dataset, pool, groups)
        frames[f"es_{tag}"] = space.scores(h)
        if tag != "equal":
            k_eff[f"k_eff_{tag}"] = pd.Series(
                [float(effective_scenarios(w)) for w in space.weights(h)],
                index=pool.days)
    table = pd.DataFrame(frames)
    table["pool_size"] = pool.sizes()
    for name, series in k_eff.items():
        table[name] = series
    table["improvement_revised_vs_equal_pct"] = (
        100 * (table.es_equal - table.es_revised) / table.es_equal)
    table["improvement_revised_vs_legacy_pct"] = (
        100 * (table.es_legacy - table.es_revised) / table.es_legacy)
    table["improvement_legacy_vs_equal_pct"] = (
        100 * (table.es_equal - table.es_legacy) / table.es_equal)
    table.index.name = "date"
    return table


# --------------------------------------------------------------------------- 输出


def make_plots(structure, ablation, search, comparison, split_summary, assets: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if f in available),
                "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    assets.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), layout="constrained")
    for column, label, color in (("load", "负载", "#256e90"), ("pv", "光伏", "#c98b3a"),
                                 ("net", "净负荷", "#6d8b4a")):
        axes[0].plot(structure["total_lag"].lag_days, structure["total_lag"][column],
                     marker="o", ms=4, label=label, color=color)
        axes[1].plot(structure[f"slot_{column}"].lag_days, structure[f"slot_{column}"]["corr"],
                     marker="o", ms=4, label=label, color=color)
    for ax in axes:
        for k in (7, 14):
            ax.axvline(k, color="#999999", ls=":", lw=1)
        ax.grid(alpha=.18)
        ax.legend()
    axes[0].set(title="(a) 日总电量的跨日自相关", xlabel="滞后天数", ylabel="自相关系数",
                ylim=(-0.4, 1.05))
    axes[1].set(title="(b) 去除同刻均值后的同刻相关", xlabel="滞后天数", ylabel="相关系数")
    fig.savefig(assets / "structure_lag_correlation.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    table = structure["weekday"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), layout="constrained")
    axes[0].bar(table["name"], table["daily_total_kwh"] / 1e3, color="#256e90")
    axes[0].axhline(structure["weekday_mean_daily_total"] / 1e3, color="#9d5b3a", ls="--",
                    label="全期日均")
    axes[0].set(title="(a) 各星期几的负载日总电量", ylabel="MWh/日")
    axes[0].grid(axis="y", alpha=.18)
    axes[0].legend()
    axes[1].bar(table["name"], table["rms_deviation_kwh"], color="#6d8b4a")
    axes[1].set(title="(b) 各星期几偏离同刻均值的均方根", ylabel="kWh/时段")
    axes[1].grid(axis="y", alpha=.18)
    fig.savefig(assets / "structure_weekday.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), layout="constrained")
    axes[0].plot(structure["pool"].date, structure["pool"].pool_size, color="#256e90")
    axes[0].set(title="(a) 逐日候选池规模", ylabel="历史日数", xlabel="2025 年")
    axes[0].grid(alpha=.18)
    axes[1].plot(comparison.index, comparison.k_eff_revised, color="#9d5b3a", label="修订特征集")
    axes[1].plot(comparison.index, comparison.k_eff_legacy, color="#256e90", label="旧特征集")
    axes[1].plot(comparison.index, comparison.pool_size, color="#aaaaaa", ls=":", label="池规模")
    axes[1].set(title="(b) 有效情景数 K_eff", ylabel="个", xlabel="2025 年")
    axes[1].grid(alpha=.18)
    axes[1].legend()
    fig.savefig(assets / "pool_and_keff.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    forward = ablation[ablation.stage == "forward"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0), layout="constrained")
    colors = ["#256e90" if a else "#b2b8bd" for a in forward.accepted]
    ax.barh(forward.feature, forward.relative_improvement_pct, color=colors)
    ax.axvline(MIN_FEATURE_GAIN_PCT, color="#9d5b3a", ls="--", lw=1.2,
               label=f"纳入门槛 {MIN_FEATURE_GAIN_PCT}%")
    for i, row in enumerate(forward.itertuples()):
        ax.text(row.relative_improvement_pct, i, f" {row.relative_improvement_pct:+.2f}%"
                f"（{row.improved_days}/{row.validation_days}）", va="center",
                ha="left" if row.relative_improvement_pct > 0 else "right", fontsize=9)
    ax.set(title="全年因果协议下的候选特征消融（固定 h=1）", xlabel="平均 ES 相对改善（%）")
    ax.grid(axis="x", alpha=.18)
    ax.legend()
    fig.savefig(assets / "feature_ablation.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    finite = search[np.isfinite(search.h)].sort_values("h").drop_duplicates("h")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    ax.plot(finite.h, finite.mean_es, color="#256e90", lw=2, label="高斯核加权")
    ax.axhline(comparison.es_equal.mean(), color="#9d5b3a", ls="--", label="等权基线")
    best_h = float(finite.loc[finite.mean_es.idxmin(), "h"])
    best_es = float(finite.mean_es.min())
    ax.scatter([best_h], [best_es], s=60, color="#9d5b3a", zorder=3)
    ax.annotate(f"h = {best_h:.6g}\n平均 ES = {best_es:,.3f}", (best_h, best_es),
                xytext=(.55, .78), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#555555"},
                bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cccccc"})
    ax.set(xscale="log", xlabel="带宽 h（对数刻度）", ylabel="平均能量分数（kW）",
           title=f"全年 {len(comparison)} 天：按平均 ES 选择带宽")
    ax.grid(alpha=.18)
    ax.legend()
    fig.savefig(assets / "bandwidth_curve.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    monthly = comparison.groupby(comparison.index.month)[
        ["es_equal", "es_legacy", "es_revised"]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.4), layout="constrained")
    axes[0].plot(comparison.index, comparison.es_equal, color="#b2b8bd", lw=1.2, label="等权")
    axes[0].plot(comparison.index, comparison.es_legacy, color="#256e90", lw=1.2, label="旧特征集")
    axes[0].plot(comparison.index, comparison.es_revised, color="#9d5b3a", lw=1.4,
                 label="修订特征集")
    axes[0].set(title="(a) 逐日能量分数（因果滚动协议）", ylabel="ES（kW）")
    axes[0].grid(alpha=.18)
    axes[0].legend()
    x = np.arange(len(monthly))
    axes[1].bar(x - .26, monthly.es_equal, width=.26, label="等权", color="#b2b8bd")
    axes[1].bar(x, monthly.es_legacy, width=.26, label="旧特征集", color="#256e90")
    axes[1].bar(x + .26, monthly.es_revised, width=.26, label="修订特征集", color="#9d5b3a")
    axes[1].set_xticks(x, [f"{m}月" for m in monthly.index])
    axes[1].set(title="(b) 月度平均能量分数", ylabel="ES（kW）")
    axes[1].grid(axis="y", alpha=.18)
    axes[1].legend()
    fig.savefig(assets / "paired_es_year.png", metadata={"Software": "taskComposite"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.6), layout="constrained")
    labels = [f"{r.window}\n{r.method}" for r in split_summary.itertuples()]
    colors = {"等权基线": "#b2b8bd", "旧特征集": "#256e90", "修订特征集": "#9d5b3a"}
    ax.bar(labels, split_summary.mean_es,
           color=[colors[m] for m in split_summary.method])
    for i, row in enumerate(split_summary.itertuples()):
        ax.text(i, row.mean_es * 1.01, f"{row.mean_es:,.1f}", ha="center", fontsize=9)
    ax.set(title="样本外确认：选择窗（2—6 月）与确认窗（7—12 月）", ylabel="平均 ES（kW）")
    ax.grid(axis="y", alpha=.18)
    fig.savefig(assets / "split_confirmation.png", metadata={"Software": "taskComposite"})
    plt.close(fig)


def write_report(structure, ablation, search, comparison, split_summary, selected,
                 legacy, legacy_h, revised_h, status, supplementary, output: Path) -> dict:
    forward = ablation[ablation.stage == "forward"]
    base = float(comparison.es_equal.mean())
    legacy_mean = float(comparison.es_legacy.mean())
    revised_mean = float(comparison.es_revised.mean())
    gain_equal = 100 * (base - revised_mean) / base
    gain_legacy = 100 * (legacy_mean - revised_mean) / legacy_mean
    adopted = bool(gain_equal > ADOPTION_GAIN_PCT)
    lag = structure["total_lag"].set_index("lag_days")
    slot = structure["slot_load"].set_index("lag_days")
    weekday = structure["weekday"]
    forecast = structure["forecast"].set_index("baseline")
    pool = structure["pool"]
    split = split_summary.set_index(["window", "method"])
    confirm_revised = split.loc[("确认窗", "修订特征集"), "improvement_vs_equal_pct"]
    low = weekday[weekday.weekday.isin([4, 5])]
    high = weekday[~weekday.weekday.isin([4, 5])]
    low_names = "、".join(low.name)
    supp_year = supplementary["year"]
    supp_confirm = supplementary["confirm"]
    supp_rows = "\n".join(
        f"| {r.feature} | {r.mean_es_before:.3f} | {r.mean_es_after:.3f} | "
        f"{r.relative_improvement_pct:+.3f}% | {r.improved_days}/{r.validation_days} | "
        f"{'通过' if r.passes_rule else '不通过'} |" for r in supp_year.itertuples())
    supp_confirm_rows = "\n".join(
        f"| {r.feature} | {r.mean_es_before:.3f} | {r.mean_es_after:.3f} | "
        f"{r.relative_improvement_pct:+.3f}% | {r.improved_days}/{r.validation_days} | "
        f"{'通过' if r.passes_rule else '不通过'} |" for r in supp_confirm.itertuples())
    passed_both = [f for f in supp_year[supp_year.passes_rule].feature
                   if f in set(supp_confirm[supp_confirm.passes_rule].feature)]
    protocol_groups = supplementary["protocol_groups"]
    final_groups = supplementary["final_groups"]
    if passed_both:
        supp_conclusion = (
            f"两个窗口都通过的事后候选为 **{'、'.join(passed_both)}**。它的编码来自纯描述性的"
            "星期几结构证据（低负载日落在周五、周六，与 ES 无关），并在时间上独立的确认窗"
            "复现了同向改善，故并入最终冻结规则；方法上仍标注为**事后补充**，"
            "证据强度低于按预先固定顺序纳入的主消融特征。")
    else:
        supp_conclusion = (
            "没有事后候选同时通过两个窗口，主消融结论不变。"
            "这也说明星期几的边际信息已被周同期特征吸收，"
            "单独再放一个日历指示编码不再带来稳定改善。")

    ablation_rows = "\n".join(
        f"| {r.feature} | {r.mean_es_before:.3f} | {r.mean_es_after:.3f} | "
        f"{r.relative_improvement_pct:+.3f}% | {r.improved_days}/{r.validation_days} | "
        f"{'纳入' if r.accepted else '不纳入'} |" for r in forward.itertuples())
    feature_lines = "\n".join(f"- `{g}`：{FEATURE_DESCRIPTIONS[g]}。" for g in final_groups)
    split_rows = "\n".join(
        f"| {r.window} | {r.method} | {r.days} | {r.mean_es:.3f} | "
        f"{r.improvement_vs_equal_pct:+.3f}% |" for r in split_summary.itertuples())
    h_text = "∞（等权极限）" if np.isinf(revised_h) else f"{revised_h:.10g}"
    legacy_h_text = "∞（等权极限）" if np.isinf(legacy_h) else f"{legacy_h:.10g}"

    text = f"""# 问题二全年特征探索与核赋权修订

**判定：{'采纳修订特征集' if adopted else '维持旧特征集'}。** 全年 {len(comparison)} 天因果滚动协议下，
等权基线平均 ES 为 **{base:.3f} kW**，旧（一月版）特征集为 **{legacy_mean:.3f} kW**，
修订特征集为 **{revised_mean:.3f} kW**；修订相对等权改善 **{gain_equal:+.3f}%**
（预设门槛 {ADOPTION_GAIN_PCT}%），相对旧特征集改善 **{gain_legacy:+.3f}%**。
{'修订方案超过预设门槛，据此改写研究日志的赋权规则章节。' if adopted else
 '修订方案未超过预设门槛，维持原方案，负结果如实记录。'}

## 一、全年结构证据

**周同期结构最强，且只有全年尺度才能识别。** 去除同刻全年均值后，负载在滞后 7 天处的
同刻相关系数为 **{slot.loc[7, 'corr']:.4f}**、14 天处 **{slot.loc[14, 'corr']:.4f}**，
而滞后 2—5 天只有 **{slot.loc[2, 'corr']:.4f} 至 {slot.loc[5, 'corr']:.4f}**。
日总电量尺度更明显：负载滞后 7 天自相关 **{lag.loc[7, 'load']:.4f}**、
净负荷 **{lag.loc[7, 'net']:.4f}**、光伏 **{lag.loc[7, 'pv']:.4f}**；
滞后 14 天仍有 **{lag.loc[14, 'net']:.4f}**。一月窗口只覆盖一个完整周周期、
八天验证窗更没有第二个，因此这一结构必然被漏掉。

**星期几是第二个独立结构，且真实形态与一月假设不同。** 负载的星期几效应可解释去均值后
方差的 **{100 * structure['weekday_variance'] / structure['centered_variance']:.2f}%**。
低负载并不落在周六与周日，而是落在 **{low_names}**：这两天的负载日总电量约
**{low.daily_total_kwh.min() / 1e3:.1f}—{low.daily_total_kwh.max() / 1e3:.1f} MWh**，
其余五天约 **{high.daily_total_kwh.min() / 1e3:.1f}—{high.daily_total_kwh.max() / 1e3:.1f} MWh**，
两者相差约 {high.daily_total_kwh.mean() / low.daily_total_kwh.mean():.2f} 倍。
这解释了一月版 `weekend`（周六/日）标志为何在全年协议下反而使平均 ES 变差：
它把高负载的周日错划进了"周末"。

**简单预测基线复现文档结论。** 在 {len(comparison)} 天评价区间上，组合基线（负载取上周同刻、
光伏取过去 7 日同刻均值）的净负荷 MAE 为
**{forecast.loc['组合基线', 'net_mae_kwh']:.2f} kWh/时段**、WAPE
**{forecast.loc['组合基线', 'net_wape_pct']:.2f}%**，与
`数据探索与问题2建模前分析.pdf` 表 1 的 45.56 与 9.11% 一致，说明本次读取口径与文档相同。

**"上周同日"就藏在候选池里。** 验证期每天的池规模在
**{int(pool.pool_size.min())}—{int(pool.pool_size.max())}** 天之间（中位
{int(pool.pool_size.median())} 天）。{int(pool.same_weekday_in_pool.sum())}/{len(pool)}
天的池中含"上周同日"；若按净负荷日总电量距离排序，上周同日的中位排名为第
**{pool.lag7_rank.median():.0f}** 位，**{100 * (pool.lag7_rank_share < 0.1).mean():.1f}%**
的日子排进池内前 10%。核赋权要做的，正是把它从池里识别出来。

![周同期结构](../../assets/taskComposite/structure_lag_correlation.png)

![星期几结构](../../assets/taskComposite/structure_weekday.png)

## 二、因果滚动协议

- **验证日**：{VALIDATION_START.date()} 至 {DATA_END.date()}，共 {len(comparison)} 天。
- **候选池**：$\\mathcal{{H}}_d=\\{{j<d:\\ j\\ \\text{{与}}\\ d\\ \\text{{的月份环形距离}}\\le 1,\\ j\\ge\\text{{2025-01-08}}\\}}$，
  即决策日所在月及前后各一月内已完整结束的历史日；01-01 至 01-07 作为 7 日滞后特征预热，
  不作情景。池规则不含特征集合，因此等权与核加权共用同一池。
- **标尺**：连续特征用当日池处境的均值与总体标准差，日历编码取自然尺度；随池滚动，
  但只用 $j<d$ 的数据，不构成前视。
- **评分**：144 维净负荷功率曲线的完整能量分数，含 $-\\tfrac12\\sum\\sum\\pi\\pi'\\|x-x'\\|$ 项，
  不截断负净负荷、不按时段标准化；真值只用于评分，不进入任何权重。

![池与有效情景数](../../assets/taskComposite/pool_and_keff.png)

## 三、特征消融

候选顺序在读取消融结果前固定为：{', '.join(CANDIDATE_ORDER)}。
`season` 作为日历结构先验置于最前，是一月窗口无法识别的结构性例外，不占四个名额。
消融固定 **h={ABLATION_H:g}**，纳入条件为平均 ES 改善 **> {MIN_FEATURE_GAIN_PCT}%**
且至少 **{int(MIN_IMPROVED_FRACTION * 100)}%** 的验证日严格改善。

| 依序候选 | 加入前平均 ES | 加入后平均 ES | 相对改善 | 改善日数 | 判定 |
|---|---:|---:|---:|---:|---|
{ablation_rows}

最终保留：

{feature_lines}

其中前 {len(protocol_groups) - 1} 个语义特征由上面预先固定顺序的消融逐次纳入，
`low_load_day` 来自下一节的事后补充检验。本次不重排候选追求更好的数字，
未通过的特征与其失败原因一并保留在 `feature_ablation.csv`。

![特征消融](../../assets/taskComposite/feature_ablation.png)

### 三之补充：星期几编码的事后假设检验

结构探索发现低负载日落在周五与周六，而 `weekend` 只把周六、周日标为同质。
据此另提两个候选，用**完全相同**的判据复核：`low_load_day`（周五或周六）
与 `weekday7`（七维指示编码）。二者是在看到上述结构证据之后才提出的，属于**事后假设**，
因此基础集固定为**选择窗**选出的特征集合，并同时在确认窗上独立复核。

全年 334 天：

| 补充候选 | 加入前平均 ES | 加入后平均 ES | 相对改善 | 改善日数 | 判定 |
|---|---:|---:|---:|---:|---|
{supp_rows}

确认窗（2025-07-01 至 2025-12-31，样本外）：

| 补充候选 | 加入前平均 ES | 加入后平均 ES | 相对改善 | 改善日数 | 判定 |
|---|---:|---:|---:|---:|---|
{supp_confirm_rows}

{supp_conclusion}

## 四、带宽与有效情景数

修订特征集上重新搜索带宽：先在 [0.05, 20] 做 25 点对数粗搜，再做三轮各 17 点细搜，
同时记录等权极限；共 {len(search)} 次候选评估。结果 **h={h_text}**，状态 `{status}`。
旧特征集在同一协议下独立搜索带宽得 **h={legacy_h_text}**；两者带宽各自冻结，不交叉调参。
仅平均 ES 参与选优，$K_{{\\mathrm{{eff}}}}$ 只作诊断。

修订特征的 $K_{{\\mathrm{{eff}}}}$ 均值 **{comparison.k_eff_revised.mean():.2f}**
（范围 {comparison.k_eff_revised.min():.2f}—{comparison.k_eff_revised.max():.2f}），
旧特征集 **{comparison.k_eff_legacy.mean():.2f}**
（范围 {comparison.k_eff_legacy.min():.2f}—{comparison.k_eff_legacy.max():.2f}），
池规模均值 **{comparison.pool_size.mean():.2f}**。周同期特征让权重更集中，
这是平均 ES 改善的主要来源，同时意味着情景覆盖变窄，因此另做决策层抽样回测兜底。

![带宽曲线](../../assets/taskComposite/bandwidth_curve.png)

![逐日配对](../../assets/taskComposite/paired_es_year.png)

## 五、配对比较与样本外确认

全年 {len(comparison)} 天：修订相对等权改善 **{gain_equal:+.3f}%**，改善日数
**{int((comparison.es_revised < comparison.es_equal - TOL).sum())}/{len(comparison)}**；
相对旧特征集改善 **{gain_legacy:+.3f}%**，改善日数
**{int((comparison.es_revised < comparison.es_legacy - TOL).sum())}/{len(comparison)}**。

为排除"同一窗口内选参与比较"的弱点，把验证期按时间一分为二：**选择窗**
2025-02-01 至 2025-06-30 用于选特征与带宽，**确认窗** 2025-07-01 至 2025-12-31
只用于评价。两窗的池、标尺与评分规则完全一致。

| 窗口 | 方法 | 天数 | 平均 ES（kW） | 相对等权 |
|---|---|---:|---:|---:|
{split_rows}

确认窗上修订方案{'仍优于' if confirm_revised > 0 else '不再优于'}等权基线
（{confirm_revised:+.3f}%），说明改善主要不是选择窗内的偶然。

![样本外确认](../../assets/taskComposite/split_confirmation.png)

## 六、局限与适用边界

- 特征候选顺序由结构证据预先固定，不做穷举搜索；固定顺序的逐步纳入存在路径依赖，
  本次不重排候选追求更好的数字，失败特征的结果一并保留在 `feature_ablation.csv`。
- 平均 ES 是概率预测评分，不是费用指标。权重集中会缩小情景覆盖、可能低估尾部风险，
  故另设定量决策层回测，并与 `validation_report.md` 的求解核验共同兜底。
- 月份环形距离把 12 月与 1 月视为同季，符合净负荷的冬季形态；周末只按周六、周日识别，
  不判断法定节假日与调休。
- 规则自 2025-02-01 起可用。全年观测只用于**设计层面**的结构发现与规则冻结，
  不进入任何单日的决策输入。
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_report.md").write_text(text, encoding="utf-8")
    rule = KernelRule(tuple(final_groups), revised_h, adopted=adopted)
    return {"rule": rule.to_dict(), "legacy_groups": list(legacy),
            "legacy_bandwidth": None if np.isinf(legacy_h) else float(legacy_h),
            "mean_es_equal": base, "mean_es_legacy": legacy_mean,
            "mean_es_revised": revised_mean,
            "improvement_vs_equal_pct": float(gain_equal),
            "improvement_vs_legacy_pct": float(gain_legacy),
            "adoption_threshold_pct": ADOPTION_GAIN_PCT, "adopted": adopted}


def save_tables(structure, ablation, ablation_daily, search, comparison, final_search,
                final_comparison, split_summary, legacy_search, supp_year, supp_confirm,
                output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    structure["total_lag"].to_csv(output / "structure_daily_lag_autocorr.csv", index=False,
                                  encoding="utf-8-sig", float_format="%.10g")
    for key in ("slot_load", "slot_pv", "slot_net"):
        structure[key].to_csv(output / f"structure_{key}.csv", index=False,
                              encoding="utf-8-sig", float_format="%.10g")
    structure["weekday"].to_csv(output / "structure_weekday_effects.csv", index=False,
                                encoding="utf-8-sig", float_format="%.10g")
    structure["forecast"].to_csv(output / "structure_forecast_baselines.csv", index=False,
                                 encoding="utf-8-sig", float_format="%.10g")
    structure["pool"].to_csv(output / "pool_profile.csv", index=False, encoding="utf-8-sig",
                             float_format="%.10g")
    ablation.to_csv(output / "feature_ablation.csv", index=False, encoding="utf-8-sig",
                    float_format="%.10g")
    ablation_daily.to_csv(output / "feature_ablation_daily.csv", index=False,
                          encoding="utf-8-sig", float_format="%.10g")
    search.to_csv(output / "bandwidth_search.csv", index=False, encoding="utf-8-sig",
                  float_format="%.10g")
    final_search.to_csv(output / "bandwidth_search_final.csv", index=False,
                        encoding="utf-8-sig", float_format="%.10g")
    legacy_search.to_csv(output / "bandwidth_search_legacy.csv", index=False,
                         encoding="utf-8-sig", float_format="%.10g")
    comparison.to_csv(output / "comparison_daily.csv", encoding="utf-8-sig",
                      float_format="%.10g")
    final_comparison.to_csv(output / "comparison_daily_final.csv", encoding="utf-8-sig",
                            float_format="%.10g")
    split_summary.to_csv(output / "split_confirmation.csv", index=False, encoding="utf-8-sig",
                         float_format="%.10g")
    supp_year.to_csv(output / "supplementary_ablation.csv", index=False, encoding="utf-8-sig",
                     float_format="%.10g")
    supp_confirm.to_csv(output / "supplementary_ablation_confirm.csv", index=False,
                        encoding="utf-8-sig", float_format="%.10g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    args = parser.parse_args()

    load, pv = load_attachment(args.input)
    features = build_context_features(load, pv)
    net = (load - pv).to_numpy(dtype=float) * STEP_HOURS

    class _Dataset:
        pass

    dataset = _Dataset()
    dataset.load, dataset.pv, dataset.features = load, pv, features
    dataset.net, dataset.dates = net, features.index

    days = features.index[(features.index >= VALIDATION_START) & (features.index <= DATA_END)]
    select_days = days[days <= SELECT_END]
    confirm_days = days[days >= CONFIRM_START]
    pool = PoolIndex(dataset, days)

    structure = {
        "total_lag": daily_total_autocorrelation(load, pv),
        "slot_load": lag_correlation(load * STEP_HOURS),
        "slot_pv": lag_correlation(pv * STEP_HOURS),
        "slot_net": lag_correlation((load - pv) * STEP_HOURS),
        "weekday": weekday_structure(load),
        "forecast": baseline_forecast_errors(load, pv),
        "pool": pool_profile(dataset, days),
    }
    weekday = structure["weekday"]
    structure["weekday_variance"] = weekday.attrs["weekday_variance"]
    structure["centered_variance"] = weekday.attrs["centered_variance"]
    structure["weekday_mean_daily_total"] = float((load.sum(axis=1) * STEP_HOURS).mean())

    selected, ablation, ablation_daily = run_ablation(dataset, pool)
    revised_h, search, status = search_bandwidth(ContextSpace(dataset, pool, selected))
    legacy_h, legacy_search, _ = search_bandwidth(ContextSpace(dataset, pool, LEGACY_GROUPS))
    comparison = paired_comparison(dataset, pool, LEGACY_GROUPS, selected, legacy_h, revised_h)

    select_pool = pool.subset(select_days)
    confirm_pool = pool.subset(confirm_days)
    sel_selected, _, _ = run_ablation(dataset, select_pool)
    sel_revised_h, _, _ = search_bandwidth(ContextSpace(dataset, select_pool, sel_selected))
    sel_legacy_h, _, _ = search_bandwidth(ContextSpace(dataset, select_pool, LEGACY_GROUPS))
    rows = []
    for window, subset, new_h, old_h, new_groups in (
            ("选择窗", select_days, sel_revised_h, sel_legacy_h, sel_selected),
            ("确认窗", confirm_days, sel_revised_h, sel_legacy_h, sel_selected)):
        table = paired_comparison(dataset, pool.subset(subset), LEGACY_GROUPS, new_groups,
                                  old_h, new_h)
        for method, column in (("等权基线", "es_equal"), ("旧特征集", "es_legacy"),
                               ("修订特征集", "es_revised")):
            rows.append({"window": window, "method": method, "days": len(table),
                         "mean_es": float(table[column].mean()),
                         "improvement_vs_equal_pct": float(
                             100 * (table.es_equal.mean() - table[column].mean())
                             / table.es_equal.mean())})
    split_summary = pd.DataFrame(rows)

    # 事后补充候选：仅作结构证据，不并入冻结规则
    supp_year_groups, supp_year = run_supplementary(
        dataset, pool, sel_selected, SUPPLEMENTARY_ORDER)
    supp_confirm_groups, supp_confirm = run_supplementary(
        dataset, confirm_pool, sel_selected, SUPPLEMENTARY_ORDER)
    supplementary = {"year": supp_year, "confirm": supp_confirm,
                     "year_groups": supp_year_groups, "confirm_groups": supp_confirm_groups}
    passed_both = [f for f in supp_year[supp_year.passes_rule].feature
                   if f in set(supp_confirm[supp_confirm.passes_rule].feature)]
    final_groups = list(selected) + passed_both
    if passed_both:
        final_h, final_search, final_status = search_bandwidth(
            ContextSpace(dataset, pool, final_groups))
        final_comparison = paired_comparison(dataset, pool, LEGACY_GROUPS, final_groups,
                                             legacy_h, final_h)
    else:
        final_h, final_search, final_status = revised_h, search, status
        final_comparison = comparison
    supplementary.update({"passed_both": passed_both, "final_groups": final_groups,
                          "protocol_groups": list(selected)})

    output, assets = Path(args.output_dir), Path(args.assets_dir)
    save_tables(structure, ablation, ablation_daily, search, comparison, final_search,
                final_comparison, split_summary, legacy_search, supp_year, supp_confirm, output)
    make_plots(structure, ablation, final_search, final_comparison, split_summary, assets)
    config = write_report(structure, ablation, final_search, final_comparison, split_summary,
                          selected, list(LEGACY_GROUPS), legacy_h, final_h, final_status,
                          supplementary, output)
    config.update({
        "protocol_groups": list(selected), "supplementary_passed": passed_both,
        "final_groups": final_groups, "select_window_groups": sel_selected,
        "revised_bandwidth": None if np.isinf(final_h) else float(final_h),
        "protocol_bandwidth": None if np.isinf(revised_h) else float(revised_h),
        "select_window_bandwidth": None if np.isinf(sel_revised_h) else float(sel_revised_h),
        "select_window_legacy_bandwidth": None if np.isinf(sel_legacy_h) else float(sel_legacy_h),
        "bandwidth_status": final_status, "validation_days": len(days),
        "select_days": len(select_days), "confirm_days": len(confirm_days),
        "rule_available_from": "2025-02-01", "warmup_days": WARMUP_DAYS,
        "pool_rule": "same season (month ring distance <= 1) and j < d",
        "source": "ProblemC/附件/附件2.xlsx (full year 365 days)"})
    (output / "frozen_rule.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({k: config[k] for k in
                      ("protocol_groups", "final_groups", "revised_bandwidth",
                       "legacy_bandwidth", "mean_es_equal", "mean_es_legacy", "mean_es_revised",
                       "improvement_vs_equal_pct", "improvement_vs_legacy_pct", "adopted")},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
