"""调度优化层（对齐 problem3_solving_final.md 式 16-29）。

- 规划层 solve_planning：两阶段 wait-and-see 随机 LP，单 W 能量平衡。
- 执行层 solve_dispatch：确定性单情景 LP/MILP，式 23 充放电/紧急购电互斥。
"""
import numpy as np

import myh.src.config as config
from myh.src.lp import Model

INF = np.inf


def solve_dispatch(load, pv, price, s0, G_fixed=None, s_end=None,
                   term_a=None, term_b=None, mutex=False):
    """执行层确定性调度（式 22-23）。单 W 平衡，可选当前区间互斥二元 z,y。"""
    H = len(load)
    m = Model()
    free_G = G_fixed is None

    G = [m.var(f"G_{i}", lb=0.0, obj=0.0 if not free_G else price[i])
         for i in range(H)]
    C = [m.var(f"C_{i}", lb=0.0, ub=config.E_BAR) for i in range(H)]
    D = [m.var(f"D_{i}", lb=0.0, ub=config.E_BAR) for i in range(H)]
    E = [m.var(f"E_{i}", lb=0.0, obj=config.EMERGENCY_MULT * price[i]) for i in range(H)]
    W = [m.var(f"W_{i}", lb=0.0) for i in range(H)]
    S = [m.var(f"S_{i}", lb=config.S_MIN, ub=config.S_MAX) for i in range(H + 1)]
    theta = m.var("theta", lb=0.0, obj=1.0) if term_a is not None else None

    if mutex and H > 0:
        z = m.var("z", lb=0.0, ub=1.0, integer=True)
        y = m.var("y", lb=0.0, ub=1.0, integer=True)
        m.add({C[0]: 1.0, z: -config.E_BAR}, 'le', 0.0)
        m.add({D[0]: 1.0, z: config.E_BAR}, 'le', config.E_BAR)
        m.add({E[0]: 1.0, y: -load[0]}, 'le', 0.0)
        m.add({C[0]: 1.0, y: config.E_BAR}, 'le', config.E_BAR)

    m.add({S[0]: 1.0}, 'eq', s0)
    if s_end is not None:
        m.add({S[H]: 1.0}, 'eq', s_end)
    if term_a is not None:
        for q in range(len(term_a)):
            m.add({theta: 1.0, S[H]: -term_a[q]}, 'ge', term_b[q])

    for i in range(H):
        terms = {E[i]: 1.0, D[i]: 1.0, C[i]: -1.0, W[i]: -1.0}
        rhs = load[i] - pv[i]
        if free_G:
            terms[G[i]] = 1.0
        else:
            rhs -= G_fixed[i]
        m.add(terms, 'eq', rhs)
        m.add({S[i + 1]: 1.0, S[i]: -1.0, C[i]: -config.ETA_C,
               D[i]: 1.0 / config.ETA_D}, 'eq', 0.0)

    status, msg, x = m.solve()
    if status != 'ok':
        return {"status": status, "message": msg}
    Cv = np.array([x[c] for c in C])
    Dv = np.array([x[d] for d in D])
    Ev = np.array([x[e] for e in E])
    Wv = np.array([x[w] for w in W])
    Sv = np.array([x[s] for s in S])
    Gv = np.array([x[g] for g in G]) if free_G else np.asarray(G_fixed, float)
    obj = float(np.sum(price * Gv + config.EMERGENCY_MULT * price * Ev))
    if theta is not None:
        obj += float(x[theta])
    return {"status": "ok", "C": Cv, "D": Dv, "E": Ev, "W": Wv, "S": Sv,
            "G": Gv, "obj": obj}


def solve_planning(L, R, p, probs, s0, G0=None, term_a=None, term_b=None,
                   use_cvar=False, alpha=0.95, lam=0.0, g_floor_q=None):
    """两阶段随机 LP（式 16/21/27/29）。

    L/R/p: (n_scen, H) 情景负载/光伏/电价；probs: (n_scen,) 等权。
    第一阶段 G（here-and-now）、A（调整量）全情景共用；第二阶段 C/D/E/W/S/θ 按情景。
    """
    L, R, p = [np.asarray(x, float) for x in (L, R, p)]
    probs = np.asarray(probs, float)
    N, H = L.shape

    # 合并完全重复情景（同一历史块副本）：概率相加，分布不变。
    if N > 1:
        stacked = np.concatenate([L, R, p], axis=1)
        uniq, first_idx, inverse = np.unique(stacked, axis=0,
                                             return_index=True, return_inverse=True)
        if uniq.shape[0] < N:
            L, R, p = L[first_idx], R[first_idx], p[first_idx]
            probs = np.bincount(inverse, weights=probs)
            N, H = L.shape

    pbar = probs @ p
    has_adj = G0 is not None
    scale = (1.0 - lam) if use_cvar else 1.0

    # 报童下界：G 不得低于情景净负荷的 g_floor_q 分位（对冲 wait-and-see 乐观偏差）
    net_floor = None
    if g_floor_q is not None:
        net_floor = np.maximum(0.0, np.quantile(L - R, g_floor_q, axis=0))

    m = Model()
    G = [m.var(f"G_{i}", lb=0.0 if net_floor is None else net_floor[i],
               obj=scale * pbar[i]) for i in range(H)]
    A = [m.var(f"A_{i}", lb=0.0,
               obj=scale * 0.5 * pbar[i] if has_adj else 0.0) for i in range(H)]

    C = [[None] * H for _ in range(N)]
    D = [[None] * H for _ in range(N)]
    E = [[None] * H for _ in range(N)]
    W = [[None] * H for _ in range(N)]
    S = [[None] * (H + 1) for _ in range(N)]
    theta = [None] * N

    for w in range(N):
        pw = probs[w]
        for i in range(H):
            C[w][i] = m.var(f"C_{w}_{i}", lb=0.0, ub=config.E_BAR)
            D[w][i] = m.var(f"D_{w}_{i}", lb=0.0, ub=config.E_BAR)
            E[w][i] = m.var(f"E_{w}_{i}", lb=0.0,
                            obj=scale * pw * config.EMERGENCY_MULT * p[w, i])
            W[w][i] = m.var(f"W_{w}_{i}", lb=0.0)
        for i in range(H + 1):
            S[w][i] = m.var(f"S_{w}_{i}", lb=config.S_MIN, ub=config.S_MAX)
        theta[w] = m.var(f"theta_{w}", lb=0.0, obj=scale * pw)

        m.add({S[w][0]: 1.0}, 'eq', s0)
        if term_a is not None:
            for q in range(len(term_a)):
                m.add({theta[w]: 1.0, S[w][H]: -term_a[q]}, 'ge', term_b[q])
        for i in range(H):
            # 式 16 单 W 能量平衡
            m.add({G[i]: 1.0, E[w][i]: 1.0, D[w][i]: 1.0,
                   C[w][i]: -1.0, W[w][i]: -1.0}, 'eq', L[w, i] - R[w, i])
            m.add({S[w][i + 1]: 1.0, S[w][i]: -1.0,
                   C[w][i]: -config.ETA_C, D[w][i]: 1.0 / config.ETA_D}, 'eq', 0.0)

    if has_adj:
        for i in range(H):
            m.add({A[i]: 1.0, G[i]: -1.0}, 'ge', -G0[i])
            m.add({A[i]: 1.0, G[i]: 1.0}, 'ge', G0[i])

    if use_cvar:
        zeta = m.var("zeta", lb=-INF, obj=lam)
        u = [m.var(f"u_{w}", lb=0.0, obj=lam * probs[w] / (1.0 - alpha))
             for w in range(N)]
        for w in range(N):
            terms = {u[w]: 1.0, zeta: 1.0, theta[w]: -1.0}
            for i in range(H):
                terms[f"G_{i}"] = -p[w, i]
                if has_adj:
                    terms[f"A_{i}"] = -0.5 * p[w, i]
                terms[f"E_{w}_{i}"] = -config.EMERGENCY_MULT * p[w, i]
            m.add(terms, 'ge', 0.0)

    status, msg, x = m.solve()
    if status != 'ok':
        return {"status": status, "message": msg}

    Gv = np.array([x[g] for g in G])
    Av = np.array([x[a] for a in A])
    Eemg = sum(probs[w] * config.EMERGENCY_MULT *
               float(np.sum(p[w] * np.array([x[e] for e in E[w]])))
               for w in range(N))
    Eadj = 0.5 * float(np.sum(pbar * Av)) if has_adj else 0.0
    obj = float(np.sum(pbar * Gv)) + Eemg + Eadj
    return {"status": "ok", "G": Gv, "A": Av, "obj": obj}


def solve_mpc(L, R, p, probs, s0, G_fixed, term_a=None, term_b=None, mutex=True):
    """执行层情景 MPC：当前区间 here-and-now（全情景共用），未来 wait-and-see。

    L/R/p: (n_scen, H)。当前区间 t=0 的 C/D/E/W 全情景共用（可实施动作）；
    t>0 按情景自由补救。G_fixed 为已提交、不可改的购电计划。mutex 施加式 23 互斥。
    """
    L, R, p = [np.asarray(x, float) for x in (L, R, p)]
    probs = np.asarray(probs, float)
    N, H = L.shape

    # 合并完全重复情景
    if N > 1:
        stacked = np.concatenate([L, R, p], axis=1)
        uniq, first_idx, inverse = np.unique(stacked, axis=0,
                                             return_index=True, return_inverse=True)
        if uniq.shape[0] < N:
            L, R, p = L[first_idx], R[first_idx], p[first_idx]
            probs = np.bincount(inverse, weights=probs)
            N, H = L.shape

    m = Model()

    # 当前区间（here-and-now，全情景共用）
    C0 = m.var("C0", lb=0.0, ub=config.E_BAR)
    D0 = m.var("D0", lb=0.0, ub=config.E_BAR)
    E0 = m.var("E0", lb=0.0,
               obj=config.EMERGENCY_MULT * float(np.sum(probs * p[:, 0])))
    W0 = m.var("W0", lb=0.0)

    # 未来区间（wait-and-see）
    C = [[None] * H for _ in range(N)]
    D = [[None] * H for _ in range(N)]
    E = [[None] * H for _ in range(N)]
    W = [[None] * H for _ in range(N)]
    S = [[None] * (H + 1) for _ in range(N)]
    theta = [None] * N

    for w in range(N):
        pw = probs[w]
        for t in range(1, H):
            C[w][t] = m.var(f"C_{w}_{t}", lb=0.0, ub=config.E_BAR)
            D[w][t] = m.var(f"D_{w}_{t}", lb=0.0, ub=config.E_BAR)
            E[w][t] = m.var(f"E_{w}_{t}", lb=0.0,
                            obj=pw * config.EMERGENCY_MULT * p[w, t])
            W[w][t] = m.var(f"W_{w}_{t}", lb=0.0)
        for t in range(H + 1):
            S[w][t] = m.var(f"S_{w}_{t}", lb=config.S_MIN, ub=config.S_MAX)
        theta[w] = m.var(f"theta_{w}", lb=0.0, obj=pw)

        m.add({S[w][0]: 1.0}, 'eq', s0)
        if term_a is not None:
            for q in range(len(term_a)):
                m.add({theta[w]: 1.0, S[w][H]: -term_a[q]}, 'ge', term_b[q])

        # t=0 平衡（当前区间 L/R 为实测，全情景相同，用情景 0 代表）
        m.add({E0: 1.0, D0: 1.0, C0: -1.0, W0: -1.0}, 'eq',
              L[0, 0] - R[0, 0] - G_fixed[0])
        m.add({S[w][1]: 1.0, S[w][0]: -1.0, C0: -config.ETA_C,
               D0: 1.0 / config.ETA_D}, 'eq', 0.0)

        for t in range(1, H):
            m.add({E[w][t]: 1.0, D[w][t]: 1.0, C[w][t]: -1.0, W[w][t]: -1.0},
                  'eq', L[w, t] - R[w, t] - G_fixed[t])
            m.add({S[w][t + 1]: 1.0, S[w][t]: -1.0, C[w][t]: -config.ETA_C,
                   D[w][t]: 1.0 / config.ETA_D}, 'eq', 0.0)

    # 式 23：当前区间互斥（z 充放电、y 紧急购电不充电）
    if mutex:
        z = m.var("z", lb=0.0, ub=1.0, integer=True)
        y = m.var("y", lb=0.0, ub=1.0, integer=True)
        m.add({C0: 1.0, z: -config.E_BAR}, 'le', 0.0)
        m.add({D0: 1.0, z: config.E_BAR}, 'le', config.E_BAR)
        m.add({E0: 1.0, y: -L[0, 0]}, 'le', 0.0)
        m.add({C0: 1.0, y: config.E_BAR}, 'le', config.E_BAR)

    status, msg, x = m.solve()
    if status != 'ok':
        return {"status": status, "message": msg}

    Cv = np.empty(H)
    Dv = np.empty(H)
    Ev = np.empty(H)
    Wv = np.empty(H)
    Cv[0], Dv[0], Ev[0], Wv[0] = x[C0], x[D0], x[E0], x[W0]
    for t in range(1, H):
        Cv[t] = x[C[0][t]]
        Dv[t] = x[D[0][t]]
        Ev[t] = x[E[0][t]]
        Wv[t] = x[W[0][t]]
    obj = float(np.sum(probs * np.array([
        config.EMERGENCY_MULT * (p[w, 0] * x[E0] +
                                 np.sum(p[w, 1:] * np.array([x[E[w][t]] for t in range(1, H)])))
        + x[theta[w]] for w in range(N)])))
    return {"status": "ok", "C": Cv, "D": Dv, "E": Ev, "W": Wv,
            "S": np.array([x[s] for s in S[0]]), "obj": obj}
