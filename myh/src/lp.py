"""轻量 LP/MILP 建模与求解封装（scipy HiGHS）。"""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack

INF = np.inf


class Model:
    def __init__(self):
        self._names = []
        self._idx = {}
        self._lb = []
        self._ub = []
        self._obj = []
        self._integer = []
        self._rows = []
        self._rhs = []
        self._sense = []

    def var(self, name, lb=0.0, ub=INF, obj=0.0, integer=False):
        if name in self._idx:
            raise ValueError(f"重复变量: {name}")
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._lb.append(lb)
        self._ub.append(ub)
        self._obj.append(obj)
        self._integer.append(integer)
        return name

    def add(self, terms, sense, rhs):
        idxs, coefs = [], []
        for name, coef in terms.items():
            if coef == 0:
                continue
            idxs.append(self._idx[name])
            coefs.append(coef)
        self._rows.append((idxs, coefs))
        self._sense.append(sense)
        self._rhs.append(rhs)
        return len(self._rows) - 1

    def _build_matrix(self):
        n = len(self._names)
        m = len(self._rows)
        rows, cols, vals = [], [], []
        lb = np.empty(m)
        ub = np.empty(m)
        for i, sense in enumerate(self._sense):
            idxs, coefs = self._rows[i]
            for j, cf in zip(idxs, coefs):
                rows.append(i)
                cols.append(j)
                vals.append(cf)
            rhs = self._rhs[i]
            if sense == 'eq':
                lb[i] = rhs
                ub[i] = rhs
            elif sense == 'le':
                lb[i] = -INF
                ub[i] = rhs
            else:
                lb[i] = rhs
                ub[i] = INF
        return coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr(), lb, ub

    def solve(self, maximize=False):
        n = len(self._names)
        c = np.array(self._obj, float)
        if maximize:
            c = -c
        A, lb, ub = self._build_matrix()
        bounds = Bounds(np.array(self._lb, float), np.array(self._ub, float))
        if any(self._integer):
            integrality = np.array([1 if b else 0 for b in self._integer])
            res = milp(c, integrality=integrality, bounds=bounds,
                       constraints=LinearConstraint(A, lb, ub))
            if not res.success:
                return res.status, res.message, None
            x = res.x
        else:
            eq = np.isclose(lb, ub)
            A_eq = A[eq]
            b_eq = ub[eq]
            le = (lb == -INF) & (ub != INF)
            ge = (lb != -INF) & (ub == INF)
            A_ub = vstack([A[le], -A[ge]]).tocsr()
            b_ub = np.concatenate([ub[le], -lb[ge]])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=list(zip(self._lb, self._ub)), method='highs')
            if not res.success:
                return res.status, res.message, None
            x = res.x
        return 'ok', '', {name: x[i] for i, name in enumerate(self._names)}
