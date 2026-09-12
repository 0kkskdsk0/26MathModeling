"""等权联合情景生成（对齐 problem3_solving_final.md 式 10-12）。"""
import numpy as np

import myh.src.config as config
from myh.src.forecast import load_forecast, pv_forecast


class ScenarioEngine:
    def __init__(self, data):
        self.data = data
        self.load_actual = data.load_actual
        self.pv_actual = data.pv_actual
        self.pv_fcst = data.pv_forecast
        self.load_mean = data.load_mean
        self.eps_L = {}
        self.eps_R = {}

    def _residuals(self, d, k):
        """历史日 [0,d) 在第 k 阶段的负载/光伏整段残差，增量缓存。"""
        t0 = config.STAGE_START[k]
        if k not in self.eps_L:
            self.eps_L[k] = []
            self.eps_R[k] = []
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

    def generate(self, d, k, rng, n_scen=None, r=None, observed_current=False):
        """式 11-12。r 用于执行层：r 之前（含当前区间，若 observed_current）取实测。"""
        n_scen = n_scen or config.N_SCENARIOS
        t0 = config.STAGE_START[k]
        r = t0 if r is None else r
        self._residuals(d, k)
        eps_L = np.array(self.eps_L[k]) if d > 0 else np.zeros((0, config.T))
        eps_R = np.array(self.eps_R[k]) if d > 0 else np.zeros((0, config.T))

        L_hat = load_forecast(self.load_actual, self.load_mean, d, t0)
        R_hat = pv_forecast(self.pv_actual, self.pv_fcst, d, k)

        if d == 0:
            L = np.repeat(L_hat[None, :], n_scen, axis=0)
            R = np.repeat(R_hat[None, :], n_scen, axis=0)
            src = np.zeros(n_scen, dtype=int)
        else:
            # 同季节条件化（与第二问一致）：只从与 d 同季节的历史日抽样
            pool = np.flatnonzero(self.data.season[:d] == self.data.season[d])
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

        probs = np.full(n_scen, 1.0 / n_scen)
        return {"L": L, "R": R, "probs": probs,
                "src_days": src if d > 0 else None,
                "L_hat": L_hat, "R_hat": R_hat}
