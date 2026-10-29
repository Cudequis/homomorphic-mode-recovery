"""Figures of Chapter 5. Usage: python plots.py   (reads the .json files, writes figures/*.pdf)"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.bbox": "tight",
})


def load(name):
    with open(os.path.join(HERE, name + ".json")) as f:
        return json.load(f)


def fig_rq1b():
    rows = load("rq1b")
    rq1 = load("rq1")
    checks, bounds = rq1["fhe_checks"], rq1["bounds"]
    sig = sorted({r["sigma"] for r in rows})
    fcs = sorted({r["f_c"] for r in rows})
    edges = np.sqrt(np.array(sig[:-1]) * np.array(sig[1:]))
    s_edges = np.concatenate([[sig[0] ** 2 / edges[0]], edges, [sig[-1] ** 2 / edges[-1]]])
    rr = [f / 16 for f in fcs]
    r_edges = np.concatenate([[rr[0] - 0.125], np.array(rr[:-1]) + 0.125, [rr[-1] + 0.125]])
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
    norm = LogNorm(vmin=1, vmax=1024)
    for ax, key, title in ((axes[0], "d_cert", "certified degree"),
                           (axes[1], "d_emp", "smallest degree that recovers the mode")):
        Z = np.full((len(fcs), len(sig)), np.nan)
        W = np.zeros((len(fcs), len(sig)))
        for r in rows:
            i, j = fcs.index(r["f_c"]), sig.index(r["sigma"])
            if not r["kernel_ok"]:
                W[i, j] = 1
            else:
                v = r.get(key)
                Z[i, j] = 1024 if v is None else min(v, 1024)
        ax.grid(False)
        wrong = np.ma.masked_where(W == 0, W)
        ax.pcolormesh(s_edges, r_edges, wrong, cmap=ListedColormap([NEUTRAL]),
                      edgecolors="white", linewidth=0.4)
        ax.pcolor(s_edges, r_edges, wrong, hatch="////", alpha=0, edgecolor="#a9a8a4")
        mesh = ax.pcolormesh(s_edges, r_edges, np.ma.masked_invalid(Z), cmap="Blues", norm=norm,
                             edgecolors="white", linewidth=0.4)
        ax.plot([bounds[str(f)] for f in fcs], rr, color=INK, linestyle="--", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_xlabel(r"bandwidth $\sigma$")
        ax.set_title(title, fontsize=9, color=INK)
    for c in checks:
        if c["fhe_ok"]:
            axes[1].plot(c["sigma"], c["f_c"] / 16, marker="o", markersize=6, markerfacecolor=INK,
                         markeredgecolor="white", markeredgewidth=1.0, linestyle="none")
        else:
            axes[1].plot(c["sigma"], c["f_c"] / 16, marker="X", markersize=7, markerfacecolor=ORANGE,
                         markeredgecolor="white", markeredgewidth=0.8, linestyle="none")
    axes[0].set_ylabel("cluster size / mode frequency")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("polynomial degree (blue cells only)")
    handles = [Patch(facecolor=NEUTRAL, hatch="////", edgecolor="#a9a8a4",
                     label="mode missed: kernel prefers the cluster"),
               Line2D([], [], color=INK, linestyle="--", linewidth=1.1,
                      label="a priori separation bound"),
               Line2D([], [], marker="o", color="white", markerfacecolor=INK, markersize=6,
                      linestyle="none", label="CKKS run: mode recovered"),
               Line2D([], [], marker="X", color="white", markerfacecolor=ORANGE, markersize=7,
                      linestyle="none", label="CKKS run: mode missed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.45, -0.16))
    fig.savefig(os.path.join(FIG, "rq1_degrees.pdf"))
    plt.close(fig)


def fig_rq3():
    d = load("rq3")
    rows = [r for r in d["rows"] if "error" not in r]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot([r["scale"] for r in rows], [r["eta"] for r in rows], color=BLUE, marker="o",
            markersize=5)
    ax.axhline(d["dS"] / 2, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.text(rows[0]["scale"], d["dS"] / 2 * 1.6, r"$\Delta S/2$", color=INK, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("scaling prime size (bits)")
    ax.set_ylabel(r"observed CKKS error $\eta$")
    fig.savefig(os.path.join(FIG, "rq3_noise.pdf"))
    plt.close(fig)


def fig_rq4n():
    rows = load("rq4n")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for layout, color, marker, label in (("candidate", BLUE, "o", "one row per ciphertext"),
                                         ("block", ORANGE, "s", "block-packed")):
        rs = sorted([r for r in rows if r["layout"] == layout], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rs], [r["server_s"] for r in rs], color=color, marker=marker,
                markersize=5, label=label)
        if layout == "block":
            for r in rs:
                ax.annotate(str(r["pipelines"]), (r["n"], r["server_s"]),
                            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7,
                            color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"dataset size $n$")
    ax.set_ylabel("server time (s)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(FIG, "rq4_scaling.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_rq1b, fig_rq3, fig_rq4n):
        f()
        print("ok", f.__name__)
