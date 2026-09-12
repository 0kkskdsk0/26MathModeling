"""官方结算与验算（对齐 problem3_solving_final.md 式 31-37）。"""
import numpy as np

import myh.src.config as config


def price_matrix(data):
    return np.broadcast_to(data.price_fixed, (config.N_DAYS, config.T)).copy()


def official_cost(GF, G0, E, P, out_slice=None):
    if out_slice is None:
        out_slice = slice(config.OUTPUT_START_IDX, config.OUTPUT_END_IDX + 1)
    GFs, G0s, Es, Ps = GF[out_slice], G0[out_slice], E[out_slice], P[out_slice]
    normal = float(np.sum(Ps * GFs))
    adjust = float(np.sum(0.5 * Ps * np.abs(GFs - G0s)))
    emergency = float(np.sum(config.EMERGENCY_MULT * Ps * Es))
    return {"normal": normal, "adjust": adjust,
            "emergency": emergency, "total": normal + adjust + emergency}


def verify(ctrl, data, out_slice=None):
    if out_slice is None:
        out_slice = slice(config.OUTPUT_START_IDX, config.OUTPUT_END_IDX + 1)
    P = price_matrix(data)
    L = config.DT * data.load_actual[out_slice]
    R = config.DT * data.pv_actual[out_slice]
    GF, G0, E = ctrl.GF[out_slice], ctrl.G0[out_slice], ctrl.E[out_slice]
    C, D, W, S = ctrl.C[out_slice], ctrl.D[out_slice], ctrl.W[out_slice], ctrl.S[out_slice]

    bal = GF + E + R + D - L - C - W
    soc = S[:, 1:] - S[:, :-1] - config.ETA_C * C + D / config.ETA_D
    cost_recalc = float(np.sum(P[out_slice] * GF + 0.5 * P[out_slice] * np.abs(GF - G0)
                               + config.EMERGENCY_MULT * P[out_slice] * E))
    cost = official_cost(ctrl.GF, ctrl.G0, ctrl.E, P, out_slice)["total"]

    return {
        "max_bal": float(np.max(np.abs(bal))),
        "max_soc": float(np.max(np.abs(soc))),
        "cost_res": float(cost - cost_recalc),
        "soc_viol": int(np.sum((S < config.S_MIN - 1e-6) | (S > config.S_MAX + 1e-6))),
        "rate_viol": int(np.sum((C > config.E_BAR + 1e-6) | (D > config.E_BAR + 1e-6))),
        "simultaneous_cd": int(np.sum(C * D > 1e-6)),
        "emergency_charge": int(np.sum(C * E > 1e-6)),
    }
