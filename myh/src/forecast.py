"""因果预测层：负载、光伏、电价（对齐 problem3/problem4_final 式 5-9、式1-6）。"""
import numpy as np
from scipy.interpolate import PchipInterpolator

import myh.src.config as config

_EPS = 1e-6
_EPS_PRICE = 1e-4
_scale_bounds_cache = {}
_price_cache = {}


def reset_forecast_cache():
    _scale_bounds_cache.clear()
    _price_cache.clear()


def scale_bounds(load_actual, d):
    """式 6 的 a_min/a_max：日期 d 以前负载缩放比例样本的 5%/95% 分位。"""
    if d in _scale_bounds_cache:
        return _scale_bounds_cache[d]
    ratios = []
    for j in range(7, d):
        base = load_actual[j - 7]
        for k in (1, 2, 3):
            t0 = config.STAGE_START[k]
            lo = max(0, t0 - 36)
            if t0 > lo:
                r = load_actual[j, lo:t0] / np.maximum(base[lo:t0], _EPS)
                ratios.append(float(np.median(r)))
    if len(ratios) >= 10:
        a_min, a_max = np.percentile(ratios, [5, 95])
    else:
        a_min, a_max = 0.5, 2.0
    _scale_bounds_cache[d] = (float(a_min), float(a_max))
    return float(a_min), float(a_max)


def load_forecast(load_actual, load_mean, d, t0, a=None):
    """式 5-7：周同期基线 + 同阶段缩放，返回区间 [t0,144) 的负载电量预测（kWh）。"""
    base = load_actual[d - 7] if d >= 7 else load_mean
    if a is None:
        if t0 > 0:
            ratios = load_actual[d, :t0] / np.maximum(base[:t0], _EPS)
            a = float(np.median(ratios))
            a_min, a_max = scale_bounds(load_actual, d)
            a = float(np.clip(a, a_min, a_max))
        else:
            a = 1.0
    L = np.zeros(config.T, dtype=float)
    L[t0:] = config.DT * np.maximum(0.0, a * base[t0:])
    return L


def pv_forecast(pv_actual, pv_forecast_arr, d, k):
    """式 8-9：附件 3 整点预报 + 发布时刻实测作 PCHIP 节点，逐十分钟积分。"""
    t0 = config.STAGE_START[k]
    nodes = np.concatenate([[pv_actual[d, t0]], pv_forecast_arr[d, k]])
    hours = np.arange(25, dtype=float)
    pchip = PchipInterpolator(hours, nodes, extrapolate=False)
    n_sub = 6
    fine = np.arange(0.0, 24.0 + 1e-9, config.DT / n_sub)
    vals = np.nan_to_num(np.clip(pchip(fine), 0.0, None), nan=0.0)
    R = np.zeros(config.T, dtype=float)
    for m in range(t0, config.T):
        R[m] = np.trapezoid(vals[m * n_sub:(m + 1) * n_sub + 1], dx=config.DT / n_sub)
    return R


def price_forecast(price_varying, price_fixed, season, d, t0):
    """电价三尺度因果预测（problem4_final 式 1-6），长度 144。

    t<t0 填真实价；t>=t0 填预测 = 周同期基线 + 同星期日内时段效应 + 水平修正。
    三尺度：昼夜=周基线的日内形状+时段效应；周内=周基线(d-7)+同星期效应；季节=季节基线+水平修正。
    """
    key = (d, t0)
    if key in _price_cache:
        return _price_cache[key].copy()

    T = config.T
    phat = np.empty(T)
    if t0 > 0:
        phat[:t0] = price_varying[d, :t0]

    # 季节基线（同季节历史日均值，冷启动与季节水平参考）
    seas = np.flatnonzero(season[:d] == season[d])
    bar_seas = price_varying[seas].mean(axis=0) if len(seas) >= 5 else price_fixed

    # 周同期基线
    m = price_varying[d - 7] if d >= 7 else bar_seas.copy()

    # 同星期日内时段效应 gamma（零和）
    gamma = np.zeros(T)
    if d >= 7:
        wd = d % 7
        same = [j for j in range(7, d) if j % 7 == wd]
        if len(same) >= 4:
            resid = np.zeros((len(same), T))
            for i, j in enumerate(same):
                e = price_varying[j] - price_varying[j - 7]
                resid[i] = e - np.median(e)
            gamma = np.median(resid, axis=0)
            gamma -= gamma.mean()

    # 水平修正 delta（最近至多 36 个已实现价格）
    delta = 0.0
    if t0 > 0:
        lo = max(0, t0 - 36)
        z = price_varying[d, lo:t0] - m[lo:t0] - gamma[lo:t0]
        delta = float(np.median(z))
        if d >= 7:
            z_all = np.concatenate([
                price_varying[j] - price_varying[j - 7] for j in range(7, d)
            ])
            zlo, zhi = np.percentile(z_all, [5, 95])
            delta = float(np.clip(delta, zlo, zhi))

    phat[t0:] = m[t0:] + gamma[t0:] + delta
    phat[t0:] = np.maximum(phat[t0:], _EPS_PRICE)
    _price_cache[key] = phat
    return phat.copy()
