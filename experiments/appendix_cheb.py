"""Numerical checks for Appendix A (Chebyshev approximation).

Usage: python appendix_cheb.py   (writes appendix_cheb.json and figures/bernstein_ellipse.pdf)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as C


HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})

DELTA_MAX = 20.11                       # reference dataset (Table 4.1)
SIGMAS = [5, 3, 2, 1]
EPS = [1e-2, 1e-4, 1e-6, 1e-8]
GRID_T = np.cos(np.linspace(0, np.pi, 200001))   # includes t = -1 and t = 1


def rate(R):
    s0 = 1 + 2 / R
    return s0, s0 + np.sqrt(s0 ** 2 - 1)


def cauchy(R):
    s0, _ = rate(R)
    return lambda t: (2 / R) / (t + s0)


def gauss(R):
    return lambda t: np.exp(-R * (t + 1) / 2)    # F(u) = exp(-u), K(t) = exp(-t^2)


def sup_err(f, coef):
    return float(np.max(np.abs(f(GRID_T) - C.chebval(GRID_T, coef))))


def min_degree(f, eps, dmax=400):
    for d in range(1, dmax + 1):
        if sup_err(f, C.chebinterpolate(f, d)) < eps:
            return d
    return None


def check_coefficients(R, kmax=40):
    s0, rho = rate(R)
    a = C.chebinterpolate(cauchy(R), 600)
    k = np.arange(1, kmax + 1)
    exact = (-1.0) ** k * 2 * rho ** (-k) / np.sqrt(R + 1)
    mask = np.abs(exact) > 1e-12                  # above rounding noise
    return float(np.max(np.abs(a[1:kmax + 1] - exact)[mask] / np.abs(exact[mask])))


def check_errors(R, d):
    s0, rho = rate(R)
    f = cauchy(R)
    a = C.chebinterpolate(f, 600)
    trunc = sup_err(f, a[:d + 1])
    interp = sup_err(f, C.chebinterpolate(f, d))
    pred = 2 * rho ** (-d) / (np.sqrt(R + 1) * (rho - 1))
    return {"trunc": trunc, "pred": pred, "interp": interp, "ratio": interp / trunc}


def check_asymptotics(ratio):
    R = ratio ** 2
    _, rho = rate(R)
    delta = 1 / ratio
    return {
        "ln_rho": float(np.log(rho)),
        "first": 2 * delta,
        "second": 2 * delta * (1 - delta ** 2 / 6),
        # exact certified threshold against d ~ (ratio/2) ln(4n/dS), with 4n/dS = 1e4
        "d_exact": float(np.log(2 * 1e4 / (np.sqrt(R + 1) * (rho - 1))) / np.log(rho)),
        "d_approx": ratio / 2 * np.log(1e4),
        "prod": float(np.sqrt(R + 1) * (rho - 1)),
        "prod_approx": 2 + 2 * delta,
    }


def ive(k, z, m=20000):
    """e^{-z} I_k(z) = (1/pi) int_0^pi e^{z(cos th - 1)} cos(k th) dth (trapezoidal rule)."""
    th = np.linspace(0, np.pi, m + 1)
    w = np.full(m + 1, np.pi / m)
    w[[0, -1]] /= 2
    vals = np.exp(z * (np.cos(th) - 1))[None, :] * np.cos(np.outer(k, th))
    return vals @ w / np.pi


def check_gauss_coefficients(R, kmax=40):
    a = C.chebinterpolate(gauss(R), 600)
    k = np.arange(1, kmax + 1)
    exact = (-1.0) ** k * 2 * ive(k, R / 2)       # 2 e^{-R/2} I_k(R/2)
    mask = np.abs(exact) > 1e-12
    return float(np.max(np.abs(a[1:kmax + 1] - exact)[mask] / np.abs(exact[mask])))


def fig_ellipse():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot([-1, 1], [0, 0], color=INK, lw=3, solid_capstyle="butt", zorder=3)
    for R, col, ls in [(4, BLUE, "-"), (25, ORANGE, "--")]:
        s0, rho = rate(R)
        a, b = (rho + 1 / rho) / 2, (rho - 1 / rho) / 2
        ax.plot(a * np.cos(th), b * np.sin(th), color=col, ls=ls,
                label=rf"$R={R}$, $\rho={rho:.2f}$")
        ax.plot([-s0], [0], marker="x", color=col, ms=8, mew=2, ls="none", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$\mathrm{Re}\,t$")
    ax.set_ylabel(r"$\mathrm{Im}\,t$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.savefig(os.path.join(FIG, "bernstein_ellipse.pdf"))
    plt.close(fig)


def main():
    out = {"coef_relerr": {}, "errors": [], "asym": [], "degrees": [], "gauss_coef_relerr": {}}
    for R in [4, 25, 100, 400]:
        out["coef_relerr"][R] = check_coefficients(R)
        out["gauss_coef_relerr"][R] = check_gauss_coefficients(R)
        for d in [10, 20]:
            out["errors"].append({"R": R, "d": d, **check_errors(R, d)})
    for s in SIGMAS:
        ratio = DELTA_MAX / s
        out["asym"].append({"sigma": s, "ratio": ratio, **check_asymptotics(ratio)})
        R = ratio ** 2
        row = {"sigma": s, "ratio": ratio, "cauchy": [], "gauss": []}
        for e in EPS:
            row["cauchy"].append(min_degree(cauchy(R), e))
            row["gauss"].append(min_degree(gauss(R), e))
        out["degrees"].append(row)
    with open(os.path.join(HERE, "appendix_cheb.json"), "w") as f:
        json.dump(out, f, indent=1)
    fig_ellipse()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
