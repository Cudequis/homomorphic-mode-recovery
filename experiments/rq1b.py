"""RQ1/RQ2 in exact arithmetic (no FHE): certified degree against the smallest working degree.

Usage: python rq1b.py   (writes rq1b.json; needs only numpy)
"""
import json
import math
import os

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = 10.0
CAP = 1024


def ds_rq1(f_c, n=64, f_m=16):
    cluster = list(np.linspace(19.75, 20.25, f_c))
    b = n - f_m - f_c
    bg = list(np.linspace(28.0, 40.0, b)) if b > 0 else []
    return np.round(np.array([MODE] * f_m + cluster + bg), 6)


def cell(dados, sigma):
    n = len(dados)
    x = dados / sigma
    R = float((x.max() - x.min()) ** 2)
    U = (x[:, None] - x[None, :]) ** 2
    S = (1.0 / (1.0 + U)).sum(1)
    modal = dados == MODE
    by = {}
    for v, s in zip(dados, S):
        by.setdefault(v, s)
    dS = by[MODE] - max(s for v, s in by.items() if v != MODE)
    if dS <= 0:
        return dict(kernel_ok=False, dS=float(dS))
    s0 = 1 + 2 / R
    rho = s0 + math.sqrt(s0 * s0 - 1)
    arg = 8 * n / (math.sqrt(R + 1) * (rho - 1) * dS)
    d_cert = max(1, math.ceil(math.log(arg) / math.log(rho))) if arg > 1 else 1

    def ok(d):
        P = Chebyshev.interpolate(lambda u: 1.0 / (1.0 + u), d, domain=[0.0, R])
        return bool(modal[int(np.argmax(P(U).sum(1)))])

    # smallest degree with a correct argmax: doubling, then bisection
    hi = 1
    while hi <= CAP and not ok(hi):
        hi *= 2
    if hi > CAP:
        d_emp = None
    else:
        lo = hi // 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        d_emp = hi
    return dict(kernel_ok=True, dS=float(dS), d_cert=d_cert, d_emp=d_emp)


if __name__ == "__main__":
    sigmas = list(np.round(np.logspace(-2, 1, 31), 5))
    fcs = list(range(8, 49, 4))
    rows = []
    for f_c in fcs:
        dados = ds_rq1(f_c)
        for s in sigmas:
            r = cell(dados, s)
            r.update(f_c=f_c, sigma=float(s))
            rows.append(r)
    with open(os.path.join(HERE, "rq1b.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("gravado rq1b.json")
