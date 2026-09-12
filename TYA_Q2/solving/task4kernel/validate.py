"""Reproduce Q2 January feature ablations, bandwidth search and all deliverables.

Run from any working directory:
    python /path/to/TYA_Q2/solving/task4kernel/validate.py
Only January rows of attachment 2 are loaded. Validation is a tuning window,
not an independent test: feature membership and h become available on Feb 1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from .features import (FEATURE_GROUPS, FEATURE_DESCRIPTIONS, STEP_HOURS,
                           build_daily_features, load_january)
    from .kernel import (FrozenStandardizer, gaussian_weights, effective_scenarios,
                         pairwise_distances, energy_score, weights_for_day)
else:
    from features import (FEATURE_GROUPS, FEATURE_DESCRIPTIONS, STEP_HOURS,
                          build_daily_features, load_january)
    from kernel import (FrozenStandardizer, gaussian_weights, effective_scenarios,
                        pairwise_distances, energy_score, weights_for_day)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ProblemC/附件/附件2.xlsx"
OUTPUT = ROOT / "TYA_Q2/outputs/task4kernel"
ASSETS = ROOT / "TYA_Q2/assets/task4kernel"
HISTORY = pd.date_range("2025-01-04", "2025-01-23", name="date")
VALIDATION = pd.date_range("2025-01-24", "2025-01-31", name="date")
ABLATION_H = 1.0
MIN_FEATURE_GAIN_PCT = 0.5
MIN_IMPROVED_DAYS = 5
ADOPTION_GAIN_PCT = 3.0
MAX_GROUPS = 4
CANDIDATE_ORDER = ("weekend", "pv_mean_3d", "recency", "pv_lag_1d",
                   "pv_std_3d", "pv_trend_3d", "load_mean_3d")
FIXED_SCALE_COLUMNS = ("season_sin", "season_cos")
NUMERICAL_TOL = 1e-10


def columns_for(groups) -> list[str]:
    """Expand semantic features to their numerical coordinates."""
    return [c for group in groups for c in FEATURE_GROUPS[group]]


def feature_gain(before, after) -> tuple[float, int, bool]:
    """Predeclared ES-only inclusion rule: >0.5% mean gain and >=5/8 days."""
    before, after = np.asarray(before), np.asarray(after)
    gain = 100 * (before.mean() - after.mean()) / before.mean()
    improved = int(np.sum(before - after > NUMERICAL_TOL))
    return float(gain), improved, bool(gain > MIN_FEATURE_GAIN_PCT and
                                     improved >= MIN_IMPROVED_DAYS)


@dataclass
class Experiment:
    """Fixed January reference set and precomputed exact ES distances."""
    load: pd.DataFrame
    pv: pd.DataFrame

    def __post_init__(self):
        """Freeze pool/scaling using Jan 4-23 only, before every validation date."""
        self.features = build_daily_features(self.load, self.pv)
        all_columns = columns_for(("season",) + CANDIDATE_ORDER)
        self.scaler = FrozenStandardizer.fit(
            self.features.loc[HISTORY, all_columns], FIXED_SCALE_COLUMNS)
        self.scaled = self.scaler.transform(self.features.loc[HISTORY.union(VALIDATION)])
        # Scenarios remain complete observed days; no net-load clipping.
        net = self.load - self.pv
        self.scenarios = net.loc[HISTORY].to_numpy()
        self.truth = net.loc[VALIDATION].to_numpy()
        self.between = pairwise_distances(self.scenarios)
        self.to_truth = pairwise_distances(self.truth, self.scenarios)

    def weights(self, groups, h) -> np.ndarray:
        """Return (8,20) weights with exactly the same history for all methods."""
        cols = columns_for(groups)
        return gaussian_weights(self.scaled.loc[HISTORY, cols],
                                self.scaled.loc[VALIDATION, cols], h)

    def scores(self, groups, h) -> np.ndarray:
        """Calculate the full ES for each validation truth, with cached distances."""
        w = self.weights(groups, h)
        return (w * self.to_truth).sum(axis=1) - 0.5 * np.einsum(
            "di,ij,dj->d", w, self.between, w)

    def scaler_for(self, groups) -> FrozenStandardizer:
        """Select already frozen coordinates without re-estimating parameters."""
        ids = [self.scaler.columns.index(c) for c in columns_for(groups)]
        return FrozenStandardizer(tuple(self.scaler.columns[i] for i in ids),
                                  tuple(self.scaler.mean[i] for i in ids),
                                  tuple(self.scaler.scale[i] for i in ids))


def run_ablation(experiment: Experiment) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Fixed-order forward inclusion, followed by conditional removal audits.

    Season is the explicit structural exception. An accepted empirical feature
    must also pass removal from the final set at the same predeclared h=1.
    No reordering or threshold adjustment is based on observed performance.
    """
    selected, rows, daily_rows = ["season"], [], []

    def evaluate(stage, feature, before_groups, after_groups, force_reason=None):
        before = experiment.scores(before_groups, ABLATION_H)
        after = experiment.scores(after_groups, ABLATION_H)
        gain, improved, passed = feature_gain(before, after)
        if force_reason is not None:
            accepted, reason = False, force_reason
        else:
            accepted = passed and len(after_groups) <= MAX_GROUPS
            reason = ("pass" if accepted else "feature_count_limit" if passed
                      else "insufficient_mean_gain_or_daily_stability")
        row = {"stage": stage, "feature": feature,
               "description": FEATURE_DESCRIPTIONS[feature],
               "features_before": ";".join(before_groups),
               "features_after": ";".join(after_groups), "h": ABLATION_H,
               "mean_es_before": float(before.mean()), "mean_es_after": float(after.mean()),
               "relative_improvement_pct": gain, "improved_days": improved,
               "validation_days": len(VALIDATION), "passes_es_rule": passed,
               "accepted": accepted, "reason": reason}
        rows.append(row)
        for day, b, a in zip(VALIDATION, before, after):
            daily_rows.append({"stage": stage, "feature": feature,
                               "date": day.strftime("%Y-%m-%d"),
                               "es_before": b, "es_after": a,
                               "relative_improvement_pct": 100 * (b - a) / b})
        return row

    evaluate("structural", "season", [], selected,
             "required_prior_not_identifiable_in_january")
    for feature in CANDIDATE_ORDER:
        row = evaluate("forward", feature, selected.copy(), selected + [feature])
        if row["accepted"]:
            selected.append(feature)
    # If interactions invalidate a prior inclusion, remove and recheck the set.
    for iteration in range(MAX_GROUPS):
        failures = []
        for feature in selected[1:]:
            row = evaluate(f"conditional_{iteration + 1}", feature,
                           [g for g in selected if g != feature], selected.copy())
            if not row["passes_es_rule"]:
                failures.append(row)
        if not failures:
            break
        worst = min(failures, key=lambda r: (r["relative_improvement_pct"], r["feature"]))
        selected.remove(worst["feature"])
        worst["reason"] = "removed_after_conditional_audit"
    table = pd.DataFrame(rows)
    table["retained_in_final_rule"] = table["feature"].isin(selected)
    table["structural_exception"] = table["feature"].eq("season")
    return selected, table, pd.DataFrame(daily_rows)


def search_bandwidth(experiment: Experiment, groups) -> tuple[float, pd.DataFrame, str]:
    """Expand coarse edges, then perform three log-grid refinements around the best.

    ES is the only objective. Uniform weights are also recorded as h=inf.
    Flat objectives are reported as unidentifiable, not as an arbitrary edge h.
    """
    rows = []

    def evaluate_grid(grid, stage):
        means = []
        for h in grid:
            scores = experiment.scores(groups, float(h))
            effective = effective_scenarios(experiment.weights(groups, float(h)))
            mean = float(scores.mean())
            rows.append({"round": stage, "h": float(h), "mean_es": mean,
                         "mean_k_eff": float(effective.mean()),
                         "min_k_eff": float(effective.min()),
                         "max_k_eff": float(effective.max())})
            means.append(mean)
        return np.asarray(means)

    uniform = evaluate_grid([np.inf], "uniform_limit")[0]
    lower, upper = 0.05, 20.0
    status = "interior_optimum"
    for expansion in range(13):
        grid = np.geomspace(lower, upper, 25)
        values = evaluate_grid(grid, f"coarse_{expansion}")
        if np.ptp(values) < NUMERICAL_TOL and abs(values[0] - uniform) < NUMERICAL_TOL:
            best_h, status = np.inf, "flat_uniform_limit_no_identifiable_finite_h"
            break
        best = int(np.argmin(values))
        if 0 < best < len(grid) - 1:
            for refinement in range(1, 4):
                grid = np.geomspace(grid[best - 1], grid[best + 1], 17)
                values = evaluate_grid(grid, f"fine_{refinement}")
                best = int(np.argmin(values))
                if best in (0, len(grid) - 1):
                    raise RuntimeError("Refined minimum on boundary; do not accept it silently")
            best_h = float(grid[best])
            if values[best] >= uniform - NUMERICAL_TOL:
                best_h, status = np.inf, "uniform_limit_minimizes_es"
            break
        if best == 0:
            lower /= 4
        else:
            if abs(values[best] - uniform) < NUMERICAL_TOL and values[best] <= values.min():
                best_h, status = np.inf, "uniform_limit_minimizes_es"
                break
            upper *= 4
    else:
        raise RuntimeError("Coarse minimum still on grid boundary after 12 expansions")
    table = pd.DataFrame(rows)
    table["selected"] = table["h"].eq(best_h)
    return best_h, table, status


def compare(experiment: Experiment, groups, h) -> pd.DataFrame:
    """Produce paired ES values; aggregate improvement uses the ratio of mean ES."""
    baseline = experiment.scores([], np.inf)
    weighted = experiment.scores(groups, h)
    weights = experiment.weights(groups, h)
    return pd.DataFrame({"date": VALIDATION.strftime("%Y-%m-%d"),
                         "es_equal": baseline, "es_kernel": weighted,
                         "relative_improvement_pct": 100 * (baseline - weighted) / baseline,
                         "k_eff_kernel": effective_scenarios(weights),
                         "k_eff_equal": len(HISTORY),
                         "weight_sum": weights.sum(axis=1),
                         "history_size": len(HISTORY)})


def smoke_tests(experiment: Experiment, groups, h, ablations, search) -> pd.DataFrame:
    """Verify probability, full ES and no outcome leakage into fixed-rule predictions.

    For each d, shuffle and perturb ALL raw observations on/after d. Rebuild
    contexts and reference scaling, and check exact equality of the pool,
    context, parameters and weights for every ablation set and searched h.
    Score against the separately saved original y_d. Re-selecting features/h
    using shuffled validation labels is deliberately NOT called a causal test.
    """
    assert energy_score([[0.0], [2.0]], [1.0], [.5, .5]) == .5
    assert energy_score([[0.0], [2.0]], [0.0], [.25, .75]) == 1.125
    assert energy_score([[0.0], [0.0], [2.0]], [1.0], [.25, .25, .5]) == .5
    assert energy_score([[0.0], [2.0]], [2.0], [0, 1]) == 0
    assert np.array_equal(gaussian_weights([[0.0], [2.0]], [0.0], 1e-300), [1, 0])
    assert np.array_equal(gaussian_weights([[0.0], [2.0]], [0.0], np.inf), [.5, .5])
    specs = {(tuple(groups), float(v)) for v in search.h.unique()}
    for column in ("features_before", "features_after"):
        specs.update((tuple(filter(None, value.split(";"))), ABLATION_H)
                     for value in ablations[column])
    rows = []
    for i, day in enumerate(VALIDATION):
        rng = np.random.default_rng(4100 + i)
        altered = []
        for raw in (experiment.load, experiment.pv):
            changed = raw.copy()
            mask = changed.index >= day
            values = changed.loc[mask].to_numpy().copy().ravel()
            rng.shuffle(values)
            changed.loc[mask] = values.reshape((-1, raw.shape[1])) * 1.37 + 31.0
            altered.append(changed)
        changed_features = build_daily_features(*altered)
        # Independent prefix-only reconstruction also excludes target observations.
        prefix_features = build_daily_features(
            experiment.load.loc[experiment.load.index < day],
            experiment.pv.loc[experiment.pv.index < day], target_dates=[day])
        np.testing.assert_array_equal(changed_features.loc[day], experiment.features.loc[day])
        np.testing.assert_array_equal(prefix_features.loc[day], experiment.features.loc[day])
        np.testing.assert_array_equal(changed_features.loc[HISTORY], experiment.features.loc[HISTORY])
        changed_scaler = FrozenStandardizer.fit(
            changed_features.loc[HISTORY, list(experiment.scaler.columns)], FIXED_SCALE_COLUMNS)
        assert changed_scaler.to_dict() == experiment.scaler.to_dict()
        changed_scenarios = (altered[0] - altered[1]).loc[HISTORY].to_numpy()
        np.testing.assert_array_equal(changed_scenarios, experiment.scenarios)
        scaled = changed_scaler.transform(changed_features.loc[HISTORY.union([day])])
        for spec_groups, spec_h in sorted(specs, key=str):
            cols = columns_for(spec_groups)
            before = experiment.weights(spec_groups, spec_h)[i]
            after = gaussian_weights(scaled.loc[HISTORY, cols], scaled.loc[day, cols], spec_h)
            np.testing.assert_array_equal(before, after)
            assert abs(after.sum() - 1) < 1e-12 and (after >= 0).all()
            assert energy_score(experiment.scenarios, experiment.truth[i], before) == energy_score(
                changed_scenarios, experiment.truth[i], after)
        final_weights = experiment.weights(groups, h)[i]
        api_weights = weights_for_day(experiment.features, day, HISTORY,
                                      experiment.scaler_for(groups), h).to_numpy()
        np.testing.assert_array_equal(final_weights, api_weights)
        cached_score = experiment.scores(groups, h)[i]
        direct_score = energy_score(experiment.scenarios, experiment.truth[i], final_weights)
        np.testing.assert_allclose(cached_score, direct_score, rtol=1e-13, atol=1e-10)
        altered_truth = (altered[0] - altered[1]).loc[day].to_numpy()
        assert not np.array_equal(altered_truth, experiment.truth[i])
        rows.append({"date": day.strftime("%Y-%m-%d"), "weight_sum": final_weights.sum(),
                     "checked_rule_bandwidth_pairs": len(specs),
                     "history_unchanged": True, "features_exactly_unchanged": True,
                     "prefix_only_features_identical": True, "scaler_exactly_unchanged": True,
                     "weights_exactly_unchanged": True, "es_with_saved_truth_exactly_unchanged": True,
                     "target_observations_actually_changed": True,
                     "max_absolute_weight_change": 0.0, "passed": True})
    # The public API rejects the target day in the historical pool.
    try:
        weights_for_day(experiment.features, VALIDATION[0], [VALIDATION[0]],
                        experiment.scaler_for(groups), h)
    except ValueError:
        pass
    else:
        raise AssertionError("Date leakage guard did not reject a same-day scenario")
    return pd.DataFrame(rows)


def make_plots(search: pd.DataFrame, comparison: pd.DataFrame, h, assets: Path):
    """Render two ES figures using standard scientific plotting tools."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    finite = search[np.isfinite(search.h)].sort_values("h").drop_duplicates("h")
    baseline = comparison.es_equal.mean()
    best_es, best_keff = comparison.es_kernel.mean(), comparison.k_eff_kernel.mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")
    ax.plot(finite.h, finite.mean_es, color="#256e90", lw=2, label="高斯核加权")
    ax.axhline(baseline, color="#9d5b3a", ls="--", label="等权基线")
    annotation = (f"h = {h:.6g}\n平均 ES = {best_es:,.3f}\n平均 K_eff = {best_keff:.2f}"
                  if np.isfinite(h) else f"等权极限／平坦最优区间\n平均 ES = {best_es:,.3f}\n平均 K_eff = {best_keff:.2f}")
    point = h if np.isfinite(h) else finite.h.iloc[len(finite) // 2]
    ax.scatter([point], [best_es], s=60, color="#9d5b3a", zorder=3)
    ax.annotate(annotation, (point, best_es), xytext=(.56, .82), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#555555"},
                bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cccccc"})
    ax.set(xscale="log", xlabel="带宽 h（对数刻度）", ylabel="平均能量分数（kW）",
           title="1 月 24–31 日：按平均 ES 选择带宽")
    ax.grid(alpha=.18)
    ax.legend(loc="upper left")
    fig.savefig(assets / "bandwidth_curve.png", metadata={"Software": "task4kernel"})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")
    x = np.arange(len(comparison))
    ax.bar(x - .19, comparison.es_equal, width=.38, label="等权基线", color="#b2b8bd")
    ax.bar(x + .19, comparison.es_kernel, width=.38, label="高斯核加权", color="#256e90")
    for i, row in comparison.iterrows():
        ax.text(i, max(row.es_equal, row.es_kernel) * 1.025,
                f"{row.relative_improvement_pct:+.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, [d[5:] for d in comparison.date])
    ax.set(xlabel="验证日期（2025 年）", ylabel="能量分数（kW）",
           title="逐日 ES 配对比较：正百分比表示改善")
    ax.set_ylim(0, max(comparison.es_equal.max(), comparison.es_kernel.max()) * 1.2)
    ax.grid(axis="y", alpha=.18)
    ax.legend()
    fig.savefig(assets / "paired_es.png", metadata={"Software": "task4kernel"})
    plt.close(fig)


def write_report(groups, h, status, ablation, search, comparison, smoke,
                 config, output: Path):
    """Write the actual adoption decision, protocol, limitations and reuse contract."""
    base, weighted = comparison.es_equal.mean(), comparison.es_kernel.mean()
    gain = 100 * (base - weighted) / base
    decision = "采纳高斯核加权" if gain > ADOPTION_GAIN_PCT else "采纳等权"
    h_text = f"{h:.10g}" if np.isfinite(h) else "∞（等权极限；无可识别的有限最优值）"
    feature_lines = "\n".join(f"- `{g}`：{FEATURE_DESCRIPTIONS[g]}。" for g in groups)
    forward = ablation[ablation.stage == "forward"]
    ablation_rows = "\n".join(
        f"| {r.feature} | {r.mean_es_before:.3f} | {r.mean_es_after:.3f} | {r.relative_improvement_pct:+.3f}% | {r.improved_days}/8 | {'纳入' if r.accepted else '不纳入'} |"
        for r in forward.itertuples())
    comparison_rows = "\n".join(
        f"| {r.date} | {r.es_equal:.3f} | {r.es_kernel:.3f} | {r.relative_improvement_pct:+.3f}% | {r.k_eff_kernel:.2f} |"
        for r in comparison.itertuples())
    conditional = ablation[ablation.stage.str.startswith("conditional")]
    conditional_text = "；".join(
        f"{r.feature}：加入相对均值改善 {r.relative_improvement_pct:+.3f}%，{r.improved_days}/8 日改善"
        for r in conditional.itertuples()) or "无季内特征保留，无需条件删除检验。"
    negative = ("核加权超过预先规定的 3% 门槛，因此在本任务范围内采纳。"
                if gain > ADOPTION_GAIN_PCT else
                "核加权未超过预先规定的 3% 门槛，执行接口采用等权。"
                "带宽与候选特征的失败结果均保留，不能改门槛或换评分定义追求胜出。")
    text = f"""# 问题二：一月冻结相似日核赋权验证

**判定：{decision}。** 等权平均 ES 为 **{base:.6f} kW**，核加权为 **{weighted:.6f} kW**，相对改善 **{gain:.4f}%**。{negative}

## 数据与因果协议

来源为 `ProblemC/附件/附件2.xlsx` 的小区负载、光伏实测功率两张表。读取器只请求表头与 2025-01-01 至 01-31 的 31 行，不读取二月及之后的观测。每天 144 点，原始标签从 00:10 至 00:00+1，按完整日原顺序使用，不平移或拼接翌日数据。

- 历史池固定为 **1 月 4–23 日，共 20 日**；1–3 日只作为滞后特征预热。即本实验的可用池为项目同季节历史池与固定参考窗的交集，季节内条件一月自动满足。
- 所有方法均使用这个完整池，$\\Omega_d=\\mathcal H_d^{{\\mathrm{{ref}}}}$，不再随机抽样，避免额外蒙特卡洛误差。等权方法也不额外纳入 1–3 日。
- 验证窗固定为 **1 月 24–31 日，共 8 日**；不把已验证日加入历史池，也不滚动拟合标准化参数。查询日近三天的已知实测值允许变化，这是输入变化，不是重估规则。
- 原始特征 $f_j$ 对其自身 0:00 因果。连续特征的均值与总体标准差只用 1 月 4–23 日的历史处境拟合，随后固定。历史处境的标准化是在验证前用参考池统一完成，不声称在历史日当时已拥有整个参考池的标尺。
- 周末是周六/日标志，不含春节、法定节假日或调休判断；季节位置用月份圆周编码，不采用全年观测统计。

## 特征与消融

预先固定候选顺序：{', '.join(CANDIDATE_ORDER)}。所有加入与条件删除检验均用 **h={ABLATION_H:g}**；纳入条件为平均 ES 相对改善 **>{MIN_FEATURE_GAIN_PCT}%**，且至少 **{MIN_IMPROVED_DAYS}/8** 日严格改善。最多保留四个语义特征。固定顺序的逐步检验会有路径依赖，本次不重排候选追求更好的结果。

最终核规则保留：

{feature_lines}

月份的 sin/cos 是同一个季节位置特征的两维编码，按自然尺度使用。一月所有值均为 (0,1)，所以季节核距离严格为零；其加入消融改善为 0%，不满足季内纳入条件。依任务明确要求将其作为**结构性先验例外**保留，不能声称“每个保留特征都通过了一月消融”。全年月度统计证据仍需另行提供，本任务没有读取或检验该证据。它也不能据此获得跨季节效用保证。

| 依序候选 | 加入前平均 ES | 加入后平均 ES | 相对改善 | 改善日数 | 判定 |
|---|---:|---:|---:|---:|---|
{ablation_rows}

对最终组合的条件删除检查（仍固定 h=1）：{conditional_text}。完整步骤见 `feature_ablation.csv`，逐日方向见 `feature_ablation_daily.csv`。季内特征只在这个固定 h 下通过筛选；带宽选定后不再次挑特征。

## 核函数、带宽与情景数

令 $z$ 为冻结标准化后的处境，

$$\\pi_{{d,\\omega}} = \\frac{{\\exp[-\\|z_d-z_\\omega\\|_2^2/(2h^2)]}}{{\\sum_{{j\\in\\mathcal H_d^{{\\mathrm{{ref}}}}}}\\exp[-\\|z_d-z_j\\|_2^2/(2h^2)]}}.$$

选定 **h={h_text}**，搜索状态 `{status}`。先在 [0.05,20] 做 25 点对数粗搜，粗搜边缘最优时按四倍扩展；得到内部最优后，在相邻点夹定区间内做三轮 17 点细搜，同时记录 h=∞ 的等权极限。仅平均 ES 用于筛选 h，不用有效情景数调参。共记录 {len(search)} 次候选评估、{search.h.nunique()} 个不同带宽。

最优核的 $K_{{\\mathrm{{eff}}}}=1/\\sum\\pi^2$：均值 **{comparison.k_eff_kernel.mean():.4f}**，范围 **[{comparison.k_eff_kernel.min():.4f}, {comparison.k_eff_kernel.max():.4f}]**；等权恒为 20。有效情景数仅描述集中程度，不是额外评分指标。

## ES 定义与配对结果

$x_\\omega=(P^L_{{\\omega,t}}-P^R_{{\\omega,t}})_{{t=1}}^{{144}}$、$y_d$ 为同定义的验证日完整净负荷功率曲线；欧氏范数不按时段标准化、不截断负净负荷、不改成均值预测。两项都完整保留：

$$ES_d=\\sum_\\omega\\pi_{{d,\\omega}}\\|x_\\omega-y_d\\|_2-\\tfrac12\\sum_\\omega\\sum_{{\\omega'}}\\pi_{{d,\\omega}}\\pi_{{d,\\omega'}}\\|x_\\omega-x_{{\\omega'}}\\|_2.$$

ES 单位为 kW。若将所有坐标统一乘 $\\Delta=1/6$ 改用时段电量，则全部 ES 同比例缩小而排序、相对改善和最优 h 不变。研究日志将净负荷“功率”行标作 kWh，与其除以 $\\Delta$ 的定义不一致；本实现明确使用 kW，没有修改既有符号文件。

| 验证日 | 等权 ES | 核 ES | 相对改善 | 核 K_eff |
|---|---:|---:|---:|---:|
{comparison_rows}
| **平均** | **{base:.3f}** | **{weighted:.3f}** | **{gain:+.3f}%** | **{comparison.k_eff_kernel.mean():.2f}** |

表中最后一行改善为“两个平均 ES 的比”，不取逐日相对改善的均值。核在 {int((comparison.es_kernel < comparison.es_equal - NUMERICAL_TOL).sum())}/8 日改善。图见 `../../assets/task4kernel/bandwidth_curve.png` 与 `../../assets/task4kernel/paired_es.png`。

## 解释与适用边界

仅有 20 个参考日和 8 个调参日，核压低不相似历史的权重能否带来收益，取决于这些处境是否预测整日净负荷分布；无效特征会减少有效情景数并牺牲覆盖。消融表直接展示这些假设的支持或失败，不将未检验的节假日、天气变化当作已证实的原因。若核未胜出，可能原因包括样本少、处境信号弱或固定候选池覆盖不足，均属于事后假说。

**这里是同一调参窗口内的配对结果，不是独立样本外检验。** 1 月 24–31 日的真值用于选特征与 h，因此最终参数最早于 2 月 1 日可用；不能说这些最终参数在 1 月 24 日已经通过因果学习获得。每天的输入与权重在给定冻结规则时没有当日观测泄漏，和超参数选择的样本复用是两个问题。本次按任务在一月完成规则选择，不使用二月之后的数据作验证或修正。

固定参考池只是本任务检验的协议。未来若更换候选池，均须由调用者按 $j<d$、同季节等约束提供，且不得自动重拟合此标尺与 h；这种更换的表现未被本实验验证。全年的同季节池冷启动也未在本任务中解决。

## 最小验证与复跑

- `smoke_tests.csv`：8/8 日通过；权重和与 1 的最大偏差 {float(np.abs(comparison.weight_sum - 1).max()):.3g}。
- 对每个 d，将当日及之后的所有原始负载/光伏数值打乱并扰动；每个 d 重建特征、标尺、候选曲线，核验 {int(smoke.checked_rule_bandwidth_pairs.iloc[0])} 种特征/带宽组合。历史、当日处境、标准化参数、权重逐元素完全不变；仅用 d 之前的前缀重新构造处境也完全相同。
- 将原始 $y_d$ 作为独立评估真值保存后，重跑 ES 完全不变。若连评分真值也替换，ES 理应变化；若重新用打乱后的调参标签选 h/特征，规则也可能变化，不能用此测试宣称整个调参过程不依赖调参标签。
- 独立的手算一维 ES、重复情景拆分、极小带宽、等权极限与公开接口日期拒绝检查全部通过；批量 ES 与直接双重求和实现一致。

在仓库根目录，安装本目录的 `requirements.txt` 后执行：

```powershell
python TYA_Q2/solving/task4kernel/validate.py
```

可从任意工作目录使用脚本绝对路径执行，默认数据/输出路径由脚本位置解析；`--output-dir` 与 `--assets-dir` 可指定空输出目录。脚本不依赖 notebook、缓存或已有输出。`--verify-reproducibility` 会再启动独立进程写入临时空目录并比较数值表、冻结参数与图片的 SHA-256。实际记录见 `reproducibility.json`。

可执行研究叙事保存在 `TYA_Q2/solving/task4kernel/analysis.ipynb`。代码按任务指定位置组织，未另设重复的 utils 实现。

## 主模型复用

`frozen_rule.json` 保存参考日期、保留特征、精确 h、均值/标准差、采纳方法与规则最早可用日期。若 `adopted_method=equal`，调用 h=∞；否则使用 `bandwidth_h`。`weights.csv` 同时保存核候选权重与实际采纳权重。

```python
import json
from TYA_Q2.solving.task4kernel.features import build_daily_features
from TYA_Q2.solving.task4kernel.kernel import FrozenStandardizer, weights_for_day

rule = json.load(open('TYA_Q2/outputs/task4kernel/frozen_rule.json', encoding='utf-8'))
scaler = FrozenStandardizer.from_dict(rule['standardizer'])
# historical_load/pv 只需包含已知完整日；无需传入目标日真值。
contexts = build_daily_features(historical_load, historical_pv, target_dates=[target_day, *history_days])
h = float('inf') if rule['adopted_method'] == 'equal' else rule['bandwidth_h']
pi = weights_for_day(contexts, target_day, history_days, scaler, h)
```

`weights_for_day` 返回以历史日为索引的权重 Series，强制每个历史日早于目标日。`gaussian_weights` 也可直接接收已标准化矩阵，支持单日或批量目标。
"""
    (output / "protocol.md").write_text(text, encoding="utf-8")
    concise = f"""# 一月相似日核赋权：实验结论

**{decision}。** 固定一月验证窗的平均 ES：等权 **{base:.6f} kW**，高斯核 **{weighted:.6f} kW**，改善 **{gain:.4f}%**，{'超过' if gain > ADOPTION_GAIN_PCT else '未超过'}预先规定的 **3%** 采纳门槛。

## 冻结规则

- **参考池**：1 月 4–23 日，固定 20 个完整日。1–3 日用于三日滞后预热。验证日为 1 月 24–31 日，不扩充候选池，不更新标准化参数。两方法共用同一完整池，不额外随机抽样。
- **特征**：{'; '.join(FEATURE_DESCRIPTIONS[g] for g in groups)}。共 {len(groups)} 个语义特征、{len(columns_for(groups))} 个数值坐标。
- **标尺**：连续特征采用参考池处境的均值、总体标准差；季节圆周编码采用自然尺度。具体数值与精确 h 见 [frozen_rule.json](frozen_rule.json)。
- **高斯核**：$\\pi_{{d,\\omega}}\\propto\\exp[-\\|z_d-z_\\omega\\|^2/(2h^2)]$，选定 **h={h_text}**，随后冻结。所有处境仅用各日 0:00 可知信息。
- **有效情景数**：最优核 $K_{{\\mathrm{{eff}}}}$ 均值 **{comparison.k_eff_kernel.mean():.4f}**，范围 **[{comparison.k_eff_kernel.min():.4f}, {comparison.k_eff_kernel.max():.4f}]**；等权为 20。

## 特征消融与带宽

所有消融预先固定 h=1，按下表顺序逐个加入。平均 ES 改善须 **>0.5%** 且至少 **5/8 日**改善，最多四个特征。

| 候选 | 加入前平均 ES | 加入后平均 ES | 改善 | 改善日数 | 判定 |
|---|---:|---:|---:|---:|---|
{ablation_rows}

条件删除检查：{conditional_text}。逐日消融见 [feature_ablation_daily.csv](feature_ablation_daily.csv)。

**季节特征是先验例外**：一月月份编码均为 (0,1)，消融改善为 0%，未通过季内有效性验证；按任务要求保留。全年月度统计依据另行提供，本次没有使用全年观测来确定其编码或尺度。

带宽从 [0.05,20] 的 25 点对数粗网格开始，再做三轮各 17 点细搜，并比较等权极限；共 {len(search)} 次评估、{search.h.nunique()} 个不同 h。本次状态为 `{status}`。只有平均 ES 参与选择，K_eff 不参与调参。

## 逐日配对结果

ES 对 **144 维完整净负荷功率曲线（负载减光伏，kW）**计算欧氏距离，完整保留 $-\\frac12\\sum\\sum\\pi\\pi'\\|x-x'\\|$ 项，不截断负净负荷，不对评分坐标再标准化。

| 验证日 | 等权 ES | 核 ES | 改善 | 核 K_eff |
|---|---:|---:|---:|---:|
{comparison_rows}
| **平均** | **{base:.3f}** | **{weighted:.3f}** | **{gain:+.3f}%** | **{comparison.k_eff_kernel.mean():.2f}** |

总改善按两个平均 ES 计算；核在 **{int((comparison.es_kernel < comparison.es_equal - NUMERICAL_TOL).sum())}/8 日**改善，所有变差日照实保留。{negative}

![带宽与平均 ES](../../assets/task4kernel/bandwidth_curve.png)

![逐日配对 ES](../../assets/task4kernel/paired_es.png)

## 验证与适用边界

8/8 日通过权重归一化、历史日期约束和未来扰动检查。对每个 d，打乱并扰动当天及之后的原始数据，重建特征、标准化参数与权重；{int(smoke.checked_rule_bandwidth_pairs.iloc[0])} 种特征/带宽组合全部逐元素不变。ES 使用另外保存的原始真值评分，也完全不变。**替换评分真值或重新用被打乱的验证标签调参，本来就可能改变结果**；该检查证明的是给定规则的预测输入无前视泄漏。

**这八天同时用于选特征、选 h 和比较，结果属于调参窗口内表现，不是独立样本外证据。** 最终规则最早在 2 月 1 日可用。季节跨月效果及未来换池表现未经本实验检验；失败特征不等于在其他 h 或其他月份必定无效。

执行 `python TYA_Q2/solving/task4kernel/validate.py --verify-reproducibility` 可从附件一键重建数值表、参数和两张图；依赖见代码目录 `requirements.txt`。空目录独立进程复跑记录见 [reproducibility.json](reproducibility.json)，新建隔离环境验证见 [clean_environment.json](clean_environment.json)。[protocol.md](protocol.md) 给出完整协议、ES 定义及主模型接口用法。
"""
    (output / "report.md").write_text(concise, encoding="utf-8")


def save_results(experiment, groups, h, status, ablation, ablation_daily, search,
                 output=OUTPUT, assets=ASSETS):
    """Generate all requested results and diagnostics from evaluated experiments."""
    output, assets = Path(output), Path(assets)
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    comparison = compare(experiment, groups, h)
    smoke = smoke_tests(experiment, groups, h, ablation, search)
    gain = 100 * (comparison.es_equal.mean() - comparison.es_kernel.mean()) / comparison.es_equal.mean()
    adopted = "kernel" if gain > ADOPTION_GAIN_PCT else "equal"
    source_digest = hashlib.sha256(
        experiment.load.to_csv(float_format="%.17g").encode("utf-8") +
        experiment.pv.to_csv(float_format="%.17g").encode("utf-8")).hexdigest()
    config = {"schema_version": 1, "adopted_method": adopted,
              "selected_feature_groups": groups, "bandwidth_h": h if np.isfinite(h) else None,
              "bandwidth_limit": "finite" if np.isfinite(h) else "uniform",
              "bandwidth_search_status": status, "ablation_h": ABLATION_H,
              "feature_min_gain_pct": MIN_FEATURE_GAIN_PCT, "feature_min_improved_days": MIN_IMPROVED_DAYS,
              "adoption_threshold_pct": ADOPTION_GAIN_PCT, "mean_es_improvement_pct": gain,
              "history_dates": HISTORY.strftime("%Y-%m-%d").tolist(),
              "validation_dates": VALIDATION.strftime("%Y-%m-%d").tolist(),
              "rule_available_from": "2025-02-01", "pool_policy": "fixed_jan04_jan23",
              "standardizer": experiment.scaler_for(groups).to_dict(),
              "source": "ProblemC/附件/附件2.xlsx (January rows only)",
              "january_values_sha256": source_digest,
              "scenario": "144-dimensional load minus PV power, kW, Euclidean norm",
              "structural_exception": "season: mandatory calendar prior; not validated in January"}
    (output / "frozen_rule.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, table in {"feature_ablation": ablation, "feature_ablation_daily": ablation_daily,
                        "bandwidth_search": search, "comparison": comparison, "smoke_tests": smoke}.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    experiment.features.to_csv(output / "daily_features.csv", encoding="utf-8-sig", float_format="%.17g")
    w = experiment.weights(groups, h)
    weights = pd.DataFrame({"date": np.repeat(VALIDATION.strftime("%Y-%m-%d"), len(HISTORY)),
                            "history_date": np.tile(HISTORY.strftime("%Y-%m-%d"), len(VALIDATION)),
                            "kernel_weight": w.ravel(), "equal_weight": 1 / len(HISTORY),
                            "adopted_weight": w.ravel() if adopted == "kernel" else 1 / len(HISTORY)})
    weights.to_csv(output / "weights.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    make_plots(search, comparison, h, assets)
    write_report(groups, h, status, ablation, search, comparison, smoke, config, output)
    return config, comparison


def verify_reproducibility(source: Path, output: Path, assets: Path,
                           python_executable: Path | None = None) -> dict:
    """Rerun in a new process and empty output directories; compare binary digests."""
    import subprocess
    import tempfile
    filenames = ["feature_ablation.csv", "feature_ablation_daily.csv", "bandwidth_search.csv",
                 "comparison.csv", "smoke_tests.csv", "daily_features.csv", "weights.csv", "frozen_rule.json", "report.md", "protocol.md"]
    records = {}
    with tempfile.TemporaryDirectory(prefix="task4kernel_reproduce_") as directory:
        fresh = Path(directory)
        interpreter = str(python_executable) if python_executable else sys.executable
        command = [interpreter, "-B", str(Path(__file__).resolve()), "--input", str(source.resolve()),
                   "--output-dir", str(fresh / "outputs"), "--assets-dir", str(fresh / "assets")]
        subprocess.run(command, cwd=fresh, text=True, capture_output=True, check=True)
        environment_info = json.loads(subprocess.check_output(
            [interpreter, "-c", "import sys,site,json,importlib.metadata as m; "
             "print(json.dumps({'python':sys.version.split()[0], 'is_venv':sys.prefix!=sys.base_prefix, "
             "'user_site_enabled':site.ENABLE_USER_SITE, 'packages':{p:m.version(p) for p in "
             "['numpy','pandas','openpyxl','matplotlib','pillow']}}))"], text=True))
        for name in filenames + ["bandwidth_curve.png", "paired_es.png"]:
            is_image = name.endswith(".png")
            original = (assets if is_image else output) / name
            repeated = fresh / ("assets" if is_image else "outputs") / name
            original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            repeated_hash = hashlib.sha256(repeated.read_bytes()).hexdigest()
            records[name] = {"sha256": original_hash, "identical": original_hash == repeated_hash}
        if not all(r["identical"] for r in records.values()):
            raise AssertionError(f"Non-reproducible output: {records}")
    if python_executable and not (environment_info["is_venv"] and not environment_info["user_site_enabled"]):
        raise AssertionError("Clean-environment verification requires an isolated virtual environment")
    result = {"passed": True, "fresh_output_directory": True, "new_python_process": True,
              "environment": environment_info,
              "dependency_environment": "isolated virtual environment" if python_executable else "same installed dependencies",
              "files": records}
    filename = "clean_environment.json" if python_executable else "reproducibility.json"
    (output / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    """Command-line entry point; does not require previously generated artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    parser.add_argument("--verify-reproducibility", action="store_true")
    parser.add_argument("--verify-clean-python", type=Path,
                        help="Also rerun with an isolated venv interpreter and compare all artifacts")
    args = parser.parse_args()
    experiment = Experiment(*load_january(args.input))
    groups, ablation, daily = run_ablation(experiment)
    h, search, status = search_bandwidth(experiment, groups)
    config, comparison = save_results(experiment, groups, h, status, ablation, daily, search,
                                      args.output_dir, args.assets_dir)
    if args.verify_reproducibility:
        verify_reproducibility(args.input, args.output_dir, args.assets_dir)
    if args.verify_clean_python:
        verify_reproducibility(args.input, args.output_dir, args.assets_dir, args.verify_clean_python)
    print(json.dumps({"features": groups, "h": config["bandwidth_h"],
                      "adopted_method": config["adopted_method"],
                      "mean_es_equal": comparison.es_equal.mean(),
                      "mean_es_kernel": comparison.es_kernel.mean(),
                      "gain_pct": config["mean_es_improvement_pct"],
                      "mean_k_eff": comparison.k_eff_kernel.mean(), "smoke_days_passed": 8},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
