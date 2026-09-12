"""滚动控制器（对齐 problem3_solving_final.md 第 6-8 部分）。

四阶段滚动购电计划 + 确定性 MPC 执行；支持 M0/M1/M2/M3 策略对比与公共暖启动。
"""
import time
import numpy as np

import myh.src.config as config
from myh.src.scenarios import ScenarioEngine
from myh.src.forecast import load_forecast, pv_forecast, price_forecast
from myh.src.dispatch import solve_planning, solve_mpc
from myh.src.terminal_value import compute_terminal_value


class RollingController:
    def __init__(self, data, update_set, seed=0, price_type="fixed", price_info="causal"):
        if 0 not in update_set or not set(update_set) <= set(range(4)):
            raise ValueError("update_set 必须包含 0 且仅含 0/1/2/3")
        if price_type not in ("fixed", "varying"):
            raise ValueError("price_type 须为 fixed/varying")
        if price_info not in ("causal", "perfect"):
            raise ValueError("price_info 须为 causal/perfect")
        self.data = data
        self.update_set = sorted(set(update_set))
        self.seed = int(seed)
        self.price_type = price_type
        self.varying = price_type == "varying"
        self.price_info = price_info
        self.engine = ScenarioEngine(data, price_type=price_type)
        self.rng = np.random.default_rng(seed)

        self.load_actual = data.load_actual
        self.pv_actual = data.pv_actual
        self.pv_fcst = data.pv_forecast
        self.load_mean = data.load_mean
        self.price_fixed = data.price_fixed
        self.price_varying = data.price_varying
        self.season = data.season

        for name in ("GF", "G0", "C", "D", "E", "W"):
            setattr(self, name, np.zeros((config.N_DAYS, config.T)))
        self.S = np.zeros((config.N_DAYS, config.T + 1))
        self.completed_days = 0
        self.timings = {"planning": [], "mpc": [], "terminal": []}
        self._term_week = -1
        self._term_cache = None

    # ------------------------------------------------------------------
    # 终端价值（式 24-26，确定性 24h 影子，按周缓存）
    # ------------------------------------------------------------------
    def terminal_value(self, d):
        week = d // 7
        if week == self._term_week:
            return self._term_cache
        started = time.perf_counter()
        load_next = load_forecast(self.load_actual, self.load_mean, d + 1, 0)
        pv_mean = config.DT * self._rolling_mean_pv(d)
        if self.varying:
            price_next = price_forecast(self.price_varying, self.price_fixed,
                                        self.season, d + 1, 0)
        else:
            price_next = self.price_fixed.copy()
        self._term_cache = compute_terminal_value(load_next, pv_mean, price_next)
        self._term_week = week
        self.timings["terminal"].append(time.perf_counter() - started)
        return self._term_cache

    def _rolling_mean_pv(self, d):
        lo = max(0, d - 28)
        if d == 0:
            return self.data.pv_typical
        return self.pv_actual[lo:d].mean(axis=0)

    # ------------------------------------------------------------------
    # 规划层（两阶段随机 LP）
    # ------------------------------------------------------------------
    def planning_solve(self, d, k, s0, G0_ref, term_a, term_b):
        r0 = config.STAGE_START[k]
        scen = self.engine.generate(d, k, self.rng, price_info=self.price_info)
        H = config.T - r0
        if self.varying:
            p = scen["p"][:, r0:]
        else:
            p = np.broadcast_to(self.price_fixed[r0:], (scen["L"].shape[0], H)).copy()
        started = time.perf_counter()
        res = solve_planning(scen["L"][:, r0:], scen["R"][:, r0:], p,
                             scen["probs"], s0, G0=G0_ref,
                             term_a=term_a, term_b=term_b,
                             use_cvar=config.USE_CVAR, alpha=config.ALPHA_CVAR,
                             lam=config.LAMBDA_RISK,
                             g_floor_q=config.G_FLOOR_Q)
        self.timings["planning"].append(time.perf_counter() - started)
        if res["status"] != "ok":
            raise RuntimeError(f"planning day={d} stage={k}: {res.get('message')}")
        return res["G"]

    # ------------------------------------------------------------------
    # 执行层（确定性 MPC，式 22-23，LP 优先违规转 MILP）
    # ------------------------------------------------------------------
    def execute_block(self, d, t0, block_end, GF, term_a, term_b,
                      S, C, D, E, W):
        for r in range(t0, block_end):
            k = r // 36
            # 情景 MPC：当前区间 here-and-now，未来 wait-and-see（式 22-23 多情景版）
            scen = self.engine.generate(d, k, self.rng,
                                        n_scen=config.MPC_N_SCENARIOS,
                                        r=r, observed_current=True,
                                        price_info=self.price_info)
            L = scen["L"][:, r:]
            R = scen["R"][:, r:]
            if self.varying:
                p = scen["p"][:, r:]
            else:
                p = np.broadcast_to(self.price_fixed[r:],
                                    (L.shape[0], config.T - r)).copy()

            started = time.perf_counter()
            # LP 优先：先解无二元互斥的 LP，出现同时充放电或紧急购电充电才转 MILP（式 23）
            res = solve_mpc(L, R, p, scen["probs"], S[r], GF[r:],
                            term_a=term_a, term_b=term_b, mutex=False)
            if res["status"] == "ok" and (
                    res["C"][0] * res["D"][0] > 1e-6 or
                    res["C"][0] * res["E"][0] > 1e-6):
                res = solve_mpc(L, R, p, scen["probs"], S[r], GF[r:],
                                term_a=term_a, term_b=term_b, mutex=True)
            self.timings["mpc"].append(time.perf_counter() - started)
            if res["status"] != "ok":
                raise RuntimeError(f"MPC day={d} interval={r}: {res.get('message')}")
            C[r] = res["C"][0]
            D[r] = res["D"][0]
            E[r] = res["E"][0]
            W[r] = res["W"][0]
            S[r + 1] = S[r] + config.ETA_C * C[r] - D[r] / config.ETA_D

    # ------------------------------------------------------------------
    # 单日 / 全年
    # ------------------------------------------------------------------
    def run_day(self, d, s_start):
        GF = np.zeros(config.T)
        G0 = None
        S = np.zeros(config.T + 1)
        S[0] = s_start
        C = np.zeros(config.T)
        D = np.zeros(config.T)
        E = np.zeros(config.T)
        W = np.zeros(config.T)
        term_a = term_b = None

        for stage in range(4):
            r0 = config.STAGE_START[stage]
            if stage in self.update_set:
                if config.USE_TERMINAL_VALUE:
                    term_a, term_b = self.terminal_value(d)
                else:
                    term_a = term_b = None     # 跨日近视：与第二问一致，不含 θ
                G0_ref = None if G0 is None else G0[r0:]
                plan = self.planning_solve(d, stage, S[r0], G0_ref, term_a, term_b)
                GF[r0:] = plan
                if stage == 0:
                    G0 = GF.copy()
            block_end = config.STAGE_START[stage + 1] if stage < 3 else config.T
            self.execute_block(d, r0, block_end, GF, term_a, term_b, S, C, D, E, W)

        return GF, G0, C, D, E, W, S

    def run_year(self, n_days=None, start_day=0, start_soc=config.S0, verbose=True):
        n_days = config.N_DAYS if n_days is None else n_days
        if self.completed_days:
            raise RuntimeError("controller already run; create a fresh controller")
        s = start_soc
        for d in range(start_day, n_days):
            GF, G0, C, D, E, W, S = self.run_day(d, s)
            self.GF[d] = GF
            self.G0[d] = G0
            self.C[d] = C
            self.D[d] = D
            self.E[d] = E
            self.W[d] = W
            self.S[d] = S
            s = S[config.T]
            self.completed_days = d + 1
            if verbose and (d % 10 == 0 or d == n_days - 1):
                print(f"  day {d + 1}/{n_days} SOC_end={s:.1f} kWh", flush=True)
        return self
