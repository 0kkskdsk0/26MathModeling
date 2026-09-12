"""联合情景生成（对齐 problem3/problem4_final 式 10-12、式 7-8）。

- 问题三（fixed）：负载—光伏二元联合残差整段抽样。
- 问题四（varying）：负载—光伏—电价三元联合残差整段抽样，支持因果/完美价格两种口径。
"""
import numpy as np

import myh.src.config as config
from myh.src.forecast import load_forecast, pv_forecast, price_forecast


class ScenarioEngine:
    def __init__(self, data, price_type="fixed"):
        self.data = data
        self.price_type = price_type
        self.varying = price_type == "varying"
        self.load_actual = data.load_actual
        self.pv_actual = data.pv_actual
        self.pv_fcst = data.pv_forecast
        self.load_mean = data.load_mean
        self.price_fixed = data.price_fixed
        self.price_varying = data.price_varying
        self.season = data.season
        self.eps_L = {}
        self.eps_R = {}
        self.eps_p = {}

    def _residuals(self, d, k):
        """历史日 [0,d) 在第 k 阶段的负载/光伏/电价整段残差，增量缓存。"""
        t0 = config.STAGE_START[k]
        if k not in self.eps_L:
            self.eps_L[k] = []
            self.eps_R[k] = []
            self.eps_p[k] = []
        cur = len(self.eps_L[k])
        for j in range(cur, d):
            L_hat = load_forecast(self.load_actual, self.load_mean, j, t0)
            R_hat = pv_forecast(self.pv_actual, self.pv_fcst, j, k)
            eL = np.zeros(config.T)
            eR = np.zeros(config.T)
            eL[t0:] = config.DT * self.load_actual[j, t0:] - L_hat[t0:]
            eR[t0:] = config.DT * self.pv_actual[j, t0:] - R_hat[t0:]
            self.eps_L[k].append(eL)
            self.eps_R[k].append(eR)
            if self.varying:
                p_hat = price_forecast(self.price_varying, self.price_fixed,
                                       self.season, j, t0)
                eP = np.zeros(config.T)
                eP[t0:] = self.price_varying[j, t0:] - p_hat[t0:]
                self.eps_p[k].append(eP)

    def generate(self, d, k, rng, n_scen=None, r=None, observed_current=False,
                 price_info="causal"):
        """式 11-12 / 式 7-8。r 用于执行层：r 之前（含当前区间，若 observed_current）取实测。

        price_info：'causal' 用预测+残差生成电价情景；'perfect' 用当天真实价（信息基准）。
        """
        n_scen = n_scen or config.N_SCENARIOS
        t0 = config.STAGE_START[k]
        r = t0 if r is None else r
        self._residuals(d, k)
        eps_L = np.array(self.eps_L[k]) if d > 0 else np.zeros((0, config.T))
        eps_R = np.array(self.eps_R[k]) if d > 0 else np.zeros((0, config.T))
        eps_p = (np.array(self.eps_p[k]) if d > 0 else np.zeros((0, config.T))) \
            if self.varying else None

        L_hat = load_forecast(self.load_actual, self.load_mean, d, t0)
        R_hat = pv_forecast(self.pv_actual, self.pv_fcst, d, k)
        p_hat = price_forecast(self.price_varying, self.price_fixed, self.season, d, t0) \
            if self.varying else self.price_fixed

        if d == 0:
            L = np.repeat(L_hat[None, :], n_scen, axis=0)
            R = np.repeat(R_hat[None, :], n_scen, axis=0)
            src = np.zeros(n_scen, dtype=int)
        else:
            # 同季节条件化（与第二问一致）
            pool = np.flatnonzero(self.season[:d] == self.season[d])
            if pool.size >= 5:
                src = rng.choice(pool, size=n_scen, replace=True)
            else:
                src = rng.choice(d, size=n_scen, replace=True)
            L = np.maximum(0.0, L_hat[None, :] + eps_L[src])
            R = np.maximum(0.0, R_hat[None, :] + eps_R[src])

        # 已观测区间（含当前区间 r）固定为实测值
        stop = r + 1 if observed_current else r
        if stop > 0:
            L[:, :stop] = config.DT * self.load_actual[d, :stop]
            R[:, :stop] = config.DT * self.pv_actual[d, :stop]

        p = None
        if self.varying:
            if d == 0:
                p = np.repeat(p_hat[None, :], n_scen, axis=0)
            elif price_info == "perfect":
                p = np.broadcast_to(self.price_varying[d], (n_scen, config.T)).copy()
            else:
                p = np.maximum(1e-4, p_hat[None, :] + eps_p[src])
            if r >= 0:
                p[:, :r + 1] = self.price_varying[d, :r + 1]   # 已实现价格固定

        probs = np.full(n_scen, 1.0 / n_scen)
        return {"L": L, "R": R, "p": p, "probs": probs,
                "src_days": src if d > 0 else None,
                "L_hat": L_hat, "R_hat": R_hat, "p_hat": p_hat}
