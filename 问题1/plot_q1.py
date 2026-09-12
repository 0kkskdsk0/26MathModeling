"""问题1第二代论文图主体。

运行：python -B plot_q1_v2.py
输入：附件1.csv、outputs/dispatch.csv、outputs/summary.json、
      outputs/s1_sensitivity.csv
输出：/tmp/q1_figures_v2_base/ 下四张待标注 SVG。

本脚本仅绘制图形主体，不写需要人工避让的说明性标注；最终文字和箭头
在矢量图层中手工排布。本脚本不调用优化器。
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, StrMethodFormatter
import numpy as np


BASE = Path(__file__).resolve().parent
OUTPUT = Path("/tmp/q1_figures_v2_base")
DELTA = 1 / 6
N = 144

# 低饱和论文配色；颜色之外同时使用方向、线型和填充区分。
INK = "#334A5E"
BLUE = "#5B7FA3"
BLUE_LIGHT = "#CBD9E5"
GOLD = "#D9A441"
GOLD_LIGHT = "#F2DEAA"
GREEN = "#4F9078"
GREEN_LIGHT = "#CEE2D9"
RED = "#B85C5C"
RED_LIGHT = "#E7CACA"
PURPLE = "#756A9E"
GRAY = "#68737D"
GRID = "#E3E7EA"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows, name):
    result = np.array([float(row[name]) for row in rows], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}含有非有限值")
    return result


def require_close(actual, expected, name, atol=1e-6):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise ValueError(f"{name}核验失败，最大误差为{error}")


def load_plot_data():
    source_path = BASE / "附件1.csv"
    dispatch_path = BASE / "outputs" / "dispatch.csv"
    summary_path = BASE / "outputs" / "summary.json"
    sensitivity_path = BASE / "outputs" / "s1_sensitivity.csv"
    source = read_csv(source_path)
    dispatch = read_csv(dispatch_path)
    sensitivity = read_csv(sensitivity_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(source) != N or len(dispatch) != N:
        raise ValueError("附件1和调度结果均须包含144个时段")
    expected_hash = summary["输入数据核验"]["文件SHA256校验值"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("附件1与主模型汇总记录不一致")

    index = np.arange(1, N + 1)
    require_close(values(source, "时段序号"), index, "输入时段序号")
    require_close(values(dispatch, "时段序号"), index, "调度时段序号")
    price = values(source, "电价")
    load_power = values(source, "小区负载")
    pv_power = values(source, "光伏发电预测功率")
    load_energy = load_power * DELTA
    pv_energy = pv_power * DELTA
    require_close(values(source, "负载能量(kWh)"), load_energy,
                  "附件负载功率与电量", atol=5e-7)
    require_close(values(source, "光伏能量(kWh)"), pv_energy,
                  "附件光伏功率与电量", atol=5e-7)

    G = values(dispatch, "购电量G(kWh)")
    C = values(dispatch, "充电量C(kWh)")
    D = values(dispatch, "放电量D(kWh)")
    W = values(dispatch, "弃光量W(kWh)")
    start = values(dispatch, "段首储电量S_t(kWh)")
    end = values(dispatch, "段末储电量S_t+1(kWh)")
    require_close(values(dispatch, "电价P(元/kWh)"), price, "电价")
    require_close(values(dispatch, "负载电量L(kWh)"), load_energy, "负载电量")
    require_close(values(dispatch, "光伏电量R(kWh)"), pv_energy, "光伏电量")
    require_close(G + pv_energy + D, load_energy + C + W, "逐时段能量平衡")
    require_close(start[1:], end[:-1], "储电量轨迹连续性")
    state = np.r_[start[0], end]
    require_close(state[0], state[-1], "日循环")
    require_close(np.diff(state), 0.9 * C - D / 0.9, "储电量递推")

    edges = np.arange(N + 1) * DELTA
    centers = (edges[:-1] + edges[1:]) / 2
    baseline_G = np.maximum(load_energy - pv_energy, 0)
    optimal_cost = float(price @ G)
    baseline_cost = float(price @ baseline_G)
    require_close(optimal_cost, summary["求解结果"]["最优购电费用(元)"], "最优费用")
    require_close(baseline_cost,
                  summary["结果分析"]["无储能对照"]["购电费用(元)"], "无储能费用")
    cumulative_baseline = np.r_[0, np.cumsum(price * baseline_G)]
    cumulative_optimal = np.r_[0, np.cumsum(price * G)]
    block_savings = np.array([
        np.sum(price[k * 12:(k + 1) * 12]
               * (baseline_G[k * 12:(k + 1) * 12] - G[k * 12:(k + 1) * 12]))
        for k in range(12)
    ])
    total_saving = baseline_cost - optimal_cost
    require_close(block_savings.sum(), total_saving, "两小时节费贡献汇总")

    local = [row for row in sensitivity
             if row["求解状态"] == "最优"
             and row["扫描层级"] in ("局部细扫", "粗扫与细扫重合")
             and 7500 <= float(row["初始储电量S_1(kWh)"]) <= 9500]
    local.sort(key=lambda row: float(row["初始储电量S_1(kWh)"]))
    s1 = values(local, "初始储电量S_1(kWh)")
    extra_cost = values(local, "相对全局最优额外费用E(s)(元)")
    if len(local) != 81 or not np.array_equal(s1, np.arange(7500, 9501, 25)):
        raise ValueError("局部敏感性结果应覆盖7500--9500 kWh，步长25 kWh")
    extra_cost[np.abs(extra_cost) < 1e-6] = 0
    interval = summary["初始储电量最优区间"]
    interval_low = round(float(interval["下端点(kWh)"]))
    interval_high = round(float(interval["上端点(kWh)"]))
    zero_s1 = s1[extra_cost == 0]
    require_close([zero_s1.min(), zero_s1.max()],
                  [interval_low, interval_high], "敏感性最优平台")

    return {
        "edges": edges, "centers": centers, "price": price,
        "load_power": load_power, "pv_power": pv_power,
        "net_power": load_power - pv_power,
        "grid_power": G / DELTA, "charge_power": C / DELTA,
        "discharge_power": D / DELTA, "state": state,
        "cumulative_baseline": cumulative_baseline,
        "cumulative_optimal": cumulative_optimal,
        "block_savings": block_savings, "baseline_cost": baseline_cost,
        "optimal_cost": optimal_cost, "total_saving": total_saving,
        "saving_rate": total_saving / baseline_cost * 100,
        "s1": s1, "extra_cost": extra_cost,
        "interval_low": interval_low, "interval_high": interval_high,
    }


def configure_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    required = {"Songti SC", "Times New Roman"}
    missing = required - available
    if missing:
        raise RuntimeError("缺少论文字体：" + "、".join(sorted(missing)))
    plt.rcParams.update({
        "font.family": ["Times New Roman", "Songti SC", "STIXGeneral"],
        "font.size": 9.5,
        "font.weight": "normal",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "axes.linewidth": 0.7,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.facecolor": "white",
        "legend.edgecolor": "#B7BDC2",
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_title(ax, label, title):
    ax.set_title(f"{label}  {title}", loc="left", pad=10)


def style_axis(ax, ylabel, show_x=False, zero_line=False):
    ax.set_xlim(0, 24)
    ticks = np.arange(0, 25, 4)
    ax.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which="major", length=3.2, width=0.65)
    ax.tick_params(which="minor", length=1.8, width=0.45)
    ax.grid(axis="y")
    ax.set_ylabel(ylabel, labelpad=7)
    if show_x:
        ax.set_xlabel("时间", labelpad=6)
    if zero_line:
        ax.axhline(0, color="#58636D", linewidth=0.8, zorder=2)


def save_figure(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": stem, "Creator": "plot_q1_v2.py"}
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.07,
                facecolor="white", metadata=metadata)
    plt.close(fig)


def plot_data_conditions(data):
    edges, centers = data["edges"], data["centers"]
    price = data["price"]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.15), sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.15, 1.05], "hspace": 0.26},
        layout="constrained")

    ax = axes[0]
    ax.stairs(price, edges, baseline=0.30, fill=True, facecolor=RED_LIGHT,
              alpha=0.55, edgecolor="none")
    ax.stairs(price, edges, baseline=None, color=RED, linewidth=1.25)
    imin, imax = int(np.argmin(price)), int(np.argmax(price))
    ax.scatter(centers[[imin, imax]], price[[imin, imax]], s=20,
               facecolor="white", edgecolor=RED, linewidth=1, zorder=4)
    ax.set_ylim(0.30, 1.52)
    ax.yaxis.set_major_locator(MultipleLocator(0.3))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    style_axis(ax, "电价\n(元/kWh)")
    panel_title(ax, "(a)", "实时购电价格")

    ax = axes[1]
    load, pv = data["load_power"], data["pv_power"]
    ax.fill_between(centers, 0, pv, color=GOLD_LIGHT, alpha=0.66, zorder=1)
    surplus = pv > load
    ax.fill_between(centers, load, pv, where=surplus, interpolate=True,
                    color=GOLD, alpha=0.48, zorder=2)
    ax.plot(centers, load, color=INK, linewidth=1.55, label="小区负载功率", zorder=4)
    ax.plot(centers, pv, color=GOLD, linewidth=1.55, label="光伏预测功率", zorder=4)
    peak = int(np.argmax(pv))
    ax.scatter(centers[peak], pv[peak], s=18, color=GOLD, zorder=5)
    ax.set_ylim(0, 8500)
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    style_axis(ax, "功率 (kW)")
    panel_title(ax, "(b)", "负载与光伏预测功率")
    ax.legend(loc="upper left", ncol=2, handlelength=2.5, borderpad=0.5)

    ax = axes[2]
    net = data["net_power"]
    ax.fill_between(centers, 0, net, where=net >= 0, interpolate=True,
                    color=BLUE_LIGHT, alpha=0.82, zorder=1)
    ax.fill_between(centers, 0, net, where=net < 0, interpolate=True,
                    color=GOLD_LIGHT, alpha=0.9, zorder=1)
    ax.plot(centers, net, color=INK, linewidth=1.45,
            label="净负荷", zorder=3)
    ax.set_ylim(-2600, 6500)
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    style_axis(ax, "净负荷功率\n(kW)", show_x=True, zero_line=True)
    panel_title(ax, "(c)", "净负荷及光伏盈余区间")
    ax.legend(loc="upper left", handlelength=2.5, borderpad=0.5)
    save_figure(fig, "q1_data_conditions")


def plot_optimal_dispatch(data):
    edges = data["edges"]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.25), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.06, 1.12], "hspace": 0.27},
        layout="constrained")

    ax = axes[0]
    grid = data["grid_power"]
    ax.stairs(grid, edges, baseline=0, fill=True, facecolor=BLUE_LIGHT,
              alpha=0.9, edgecolor="none", zorder=1)
    ax.stairs(grid, edges, baseline=None, color=BLUE, linewidth=1.25,
              label="外网购电功率", zorder=3)
    ax.set_ylim(0, 9000)
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    style_axis(ax, "购电功率\n(kW)")
    panel_title(ax, "(a)", "外网购电与价格响应")
    ax_price = ax.twinx()
    ax_price.stairs(data["price"], edges, baseline=0.30, fill=True,
                    facecolor=RED_LIGHT, edgecolor="none", alpha=0.12, zorder=0)
    ax_price.stairs(data["price"], edges, baseline=None, color=RED,
                    linewidth=1.0, alpha=0.78, label="实时电价", zorder=4)
    ax_price.set_ylim(0.30, 1.52)
    ax_price.set_ylabel("电价 (元/kWh)", color="black", labelpad=7)
    ax_price.tick_params(axis="y", colors="black", labelsize=8.3, length=3)
    ax_price.spines["right"].set_visible(True)
    ax_price.spines["right"].set_color("black")
    handles = [Line2D([0], [0], color=BLUE, lw=1.5),
               Line2D([0], [0], color=RED, lw=1.2)]
    ax.legend(handles, ["外网购电功率", "实时电价"], loc="upper right",
              ncol=2, handlelength=2.2, borderpad=0.5)

    ax = axes[1]
    charge, discharge = data["charge_power"], data["discharge_power"]
    ax.bar(edges[:-1], charge, width=DELTA, align="edge", color=GREEN,
           edgecolor="none", label="充电功率", zorder=3)
    ax.bar(edges[:-1], -discharge, width=DELTA, align="edge", color=RED,
           edgecolor="none", label="放电功率", zorder=3)
    ax.axhline(5000, color=GREEN, linestyle=(0, (5, 4)), linewidth=0.75, alpha=0.7)
    ax.axhline(-5000, color=RED, linestyle=(0, (5, 4)), linewidth=0.75, alpha=0.7)
    ax.set_ylim(-5600, 5600)
    ax.yaxis.set_major_locator(MultipleLocator(2500))
    style_axis(ax, "储能功率\n(kW)", zero_line=True)
    panel_title(ax, "(b)", "储能充放电功率")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2,
              borderpad=0.5)

    ax = axes[2]
    state = data["state"]
    lower, upper = 1200, 10800
    ax.axhspan(lower, upper, color="#EDF1F4", alpha=0.9, zorder=0)
    ax.fill_between(edges, lower, state, color="#D9D4E8", alpha=0.72, zorder=1)
    ax.plot(edges, state, color=PURPLE, linewidth=1.65, zorder=3,
            label="储电量")
    ax.axhline(lower, color=RED, linestyle=(0, (5, 4)), linewidth=0.8)
    ax.axhline(upper, color=RED, linestyle=(0, (5, 4)), linewidth=0.8)
    ax.scatter([0, 24], state[[0, -1]], s=24, facecolor="white",
               edgecolor=PURPLE, linewidth=1.1, zorder=4, clip_on=False)
    ax.set_ylim(0, 12000)
    ax.yaxis.set_major_locator(MultipleLocator(3000))
    style_axis(ax, "储电量\n(kWh)", show_x=True)
    panel_title(ax, "(c)", "储电量轨迹与日循环约束")
    save_figure(fig, "q1_optimal_dispatch")


def plot_cost_savings(data):
    edges = data["edges"]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 5.85),
        gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.22},
        layout="constrained")

    ax = axes[0]
    base = data["cumulative_baseline"]
    opt = data["cumulative_optimal"]
    ax.plot(edges, base, color=RED, linewidth=1.55, linestyle=(0, (5, 2)),
            label="无储能")
    ax.plot(edges, opt, color=BLUE, linewidth=1.8, label="储能优化")
    ax.fill_between(edges, opt, base, where=base >= opt, interpolate=True,
                    color=GREEN_LIGHT, alpha=0.72, label="累计净节省")
    ax.fill_between(edges, opt, base, where=base < opt, interpolate=True,
                    color=RED_LIGHT, alpha=0.62, label="前期充电投入")
    ax.scatter([24, 24], [base[-1], opt[-1]], s=22,
               color=[RED, BLUE], zorder=4, clip_on=False)
    ax.set_ylim(0, 52500)
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    style_axis(ax, "累计购电费用\n(元)", show_x=True)
    panel_title(ax, "(a)", "有无储能的累计购电费用")
    ax.legend(loc="upper left", ncol=4, columnspacing=1.2,
              handlelength=2.0, borderpad=0.5)

    ax = axes[1]
    contribution = data["block_savings"]
    cumulative = np.r_[0, np.cumsum(contribution)]
    x = np.arange(12)
    colors = np.where(contribution > 1e-6, GREEN,
                      np.where(contribution < -1e-6, RED, "#B6BDC3"))
    ax.bar(x, contribution, bottom=cumulative[:-1], width=0.72,
           color=colors, edgecolor="none", zorder=3)
    for i in range(11):
        ax.plot([x[i] + 0.36, x[i + 1] - 0.36],
                [cumulative[i + 1], cumulative[i + 1]],
                color="#A8AFB5", linewidth=0.7, zorder=2)
    total_x = 13
    ax.bar(total_x, data["total_saving"], width=0.8, color=BLUE,
           edgecolor="none", zorder=3)
    labels = [f"{2 * i:02d}:00-{2 * i + 2:02d}:00" for i in range(12)] + ["全天"]
    ticks = list(x) + [total_x]
    ax.set_xticks(ticks, labels, rotation=38, ha="right")
    ax.set_xlim(-0.7, 13.75)
    ax.set_ylim(-2600, 18000)
    ax.yaxis.set_major_locator(MultipleLocator(4000))
    ax.axhline(0, color="#58636D", linewidth=0.8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("累计节费 (元)", labelpad=7)
    ax.set_xlabel("时段（以每两小时划分）", labelpad=5)
    panel_title(ax, "(b)", "分时段购电成本节省贡献")
    legend = [Patch(facecolor=GREEN, label="正贡献"),
              Patch(facecolor=RED, label="负贡献"),
              Patch(facecolor=BLUE, label="全天总节费")]
    ax.legend(handles=legend, loc="upper left", ncol=3, borderpad=0.5)
    save_figure(fig, "q1_cost_savings")


def plot_s1_sensitivity(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.55), layout="constrained")
    s1, extra = data["s1"], data["extra_cost"]
    low, high = data["interval_low"], data["interval_high"]
    ax.axvspan(low, high, color=GREEN_LIGHT, alpha=0.85, zorder=0)
    ax.plot(s1, extra, color=PURPLE, linewidth=1.65, marker="o",
            markersize=2.3, markerfacecolor="white", markeredgewidth=0.65,
            label="额外购电费用 J(s) − J*", zorder=3)
    ax.axhline(0, color="#58636D", linewidth=0.75, zorder=2)
    ax.axvline(low, color=GREEN, linestyle=(0, (4, 3)), linewidth=0.8)
    ax.axvline(high, color=GREEN, linestyle=(0, (4, 3)), linewidth=0.8)
    selected = int(np.where(s1 == 8550)[0][0])
    ax.scatter(s1[selected], extra[selected], marker="*", s=70, color=GOLD,
               edgecolor="white", linewidth=0.55, zorder=5,
               label="本文采用的代表方案 S1 = 8550 kWh")
    ax.set_xlim(7500, 9500)
    ax.set_ylim(-0.08, 3.25)
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("固定初始储电量 (kWh)", labelpad=6)
    ax.set_ylabel("相对最优解的额外费用 (元)", labelpad=7)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), ncol=1,
              handlelength=2.3, borderpad=0.55)
    save_figure(fig, "q1_s1_sensitivity")


def main():
    data = load_plot_data()
    configure_style()
    plot_data_conditions(data)
    plot_optimal_dispatch(data)
    plot_cost_savings(data)
    plot_s1_sensitivity(data)
    print(json.dumps({
        "主体图临时目录": str(OUTPUT),
        "主体图数量": 4,
        "格式": "SVG（待手工标注）",
        "字体": {"中文": "Songti SC", "英文数字": "Times New Roman"},
        "图3节费汇总校验(元)": float(data["block_savings"].sum()),
        "图4最优区间(kWh)": [data["interval_low"], data["interval_high"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
