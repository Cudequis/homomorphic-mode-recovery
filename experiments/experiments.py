"""Experiments of Chapter 5 (Evaluation).

Usage: python experiments.py <rq1|rq2|rq3|overflow|rq4d|rq4n|all>
Each experiment writes <name>.json to this folder.
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import openfhe
from numpy.polynomial.chebyshev import Chebyshev

OUT = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
MODE_VALUE = 10.0


# ----------------------------------------------------------------- datasets
def ds_reference():
    return np.array([10.0] * 20
                    + [20.03, 19.97, 20.08, 19.92, 20.12, 19.88, 20.05, 19.95, 20.10, 19.90, 20.02]
                    + [29.94, 30.11, 29.89, 30.04, 29.96])


def ds_rq1(f_c, n=64, f_m=16):
    """Mode: f_m copies of 10. Rival: a cluster of f_c distinct values in [19.75, 20.25]."""
    cluster = list(np.linspace(19.75, 20.25, f_c))
    b = n - f_m - f_c
    background = list(np.linspace(28.0, 40.0, b)) if b > 0 else []
    return np.round(np.array([MODE_VALUE] * f_m + cluster + background), 6)


def ds_scaled(n):
    """Same proportions as the reference dataset (20:11:5)."""
    f_m = round(20 * n / 36)
    f_c = round(11 * n / 36)
    f_r = n - f_m - f_c
    return np.round(np.array([MODE_VALUE] * f_m + list(np.linspace(19.88, 20.12, f_c))
                             + list(np.linspace(29.89, 30.11, f_r))), 6)


# ------------------------------------------------------------- plaintext
def analyse(dados, sigma, d):
    n = len(dados)
    x = dados / sigma
    R = float((x.max() - x.min()) ** 2)
    U = (x[:, None] - x[None, :]) ** 2
    S = (1.0 / (1.0 + U)).sum(1)
    P = Chebyshev.interpolate(lambda u: 1.0 / (1.0 + u), d, domain=[0.0, R])
    Sp = P(U).sum(1)
    us = np.linspace(0.0, R, 20001)
    eps = float(np.abs(1.0 / (1.0 + us) - P(us)).max())
    modal = dados == MODE_VALUE
    by_value = {}
    for v, s in zip(dados, S):
        by_value.setdefault(v, s)
    dS = by_value[MODE_VALUE] - max(s for v, s in by_value.items() if v != MODE_VALUE)
    s0 = 1.0 + 2.0 / R
    rho = s0 + math.sqrt(s0 * s0 - 1.0)
    d_cert = None
    if dS > 0:
        arg = 8 * n / (math.sqrt(R + 1) * (rho - 1) * dS)
        d_cert = max(1, math.ceil(math.log(arg) / math.log(rho))) if arg > 1 else 1
    return dict(n=n, R=R, rho=rho, eps=eps, dS=float(dS), d_cert=d_cert,
                kernel_ok=bool(modal[int(np.argmax(S))]),
                poly_ok=bool(modal[int(np.argmax(Sp))]),
                certified=bool(dS > 0 and n * eps < dS / 2),
                S_max=float(S.max()), S_poly=Sp, x=x)


def separation_bound(dados):
    """Theorem 4.6 solved for sigma (sufficient condition, a priori)."""
    vals, counts = np.unique(dados, return_counts=True)
    f_m = counts[vals == MODE_VALUE][0]
    f_y = counts[vals != MODE_VALUE].max()
    dmin = np.diff(np.sort(vals)).min()
    n = len(dados)
    return float(dmin * math.sqrt((f_m - f_y) / (n - f_m)))


# ------------------------------------------------------------------- FHE
def kfun(u):
    return 1.0 / (1.0 + max(0.0, u))


def context(depth, batch, scale, rot):
    p = openfhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(scale)
    p.SetBatchSize(batch)
    cc = openfhe.GenCryptoContext(p)
    for f in (openfhe.PKESchemeFeature.PKE, openfhe.PKESchemeFeature.LEVELEDSHE,
              openfhe.PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    k = cc.KeyGen()
    cc.EvalMultKeyGen(k.secretKey)
    cc.EvalRotateKeyGen(k.secretKey, rot)
    return cc, k


def rot_key_MB(cc, k):
    """Size of the rotation keys of this secret key (with the tag; without it OpenFHE
    serializes the keys of every context created in the process)."""
    f = os.path.join(TMP, "r")
    cc.SerializeEvalAutomorphismKey(f, openfhe.BINARY, k.secretKey.GetKeyTag())
    mb = os.path.getsize(f) / 1e6
    os.remove(f)
    return mb


def cheb_depth(d):
    """Levels consumed by EvalChebyshevFunction on [0, R], measured (Remark 4.22)."""
    return math.ceil(math.log2(d + 1)) + 2


def run_block(dados, sigma, d, scale=50, measure_keys=False):
    a = analyse(dados, sigma, d)
    x, R, n = a["x"], a["R"], a["n"]
    N = 2 ** math.ceil(math.log2(n))
    pad = float(x[0])
    rot = [-(2 ** k) for k in range(int(math.log2(N)))]
    depth = 1 + cheb_depth(d) + 1
    probe = openfhe.CCParamsCKKSRNS()
    probe.SetMultiplicativeDepth(depth)
    probe.SetScalingModSize(scale)
    s = openfhe.GenCryptoContext(probe).GetRingDimension() // 2
    batch = min(s, 2 ** math.ceil(math.log2(n * N)))
    per_ct = batch // N
    groups = math.ceil(n / per_ct)
    cc, k = context(depth, batch, scale, rot)
    # the dataset ciphertext is the same for every group, so it is encrypted once
    D = (list(x) + [pad] * (N - n)) * per_ct
    eD = cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(D))
    fhe = np.empty(n)
    t_server = 0.0
    for g in range(groups):
        cand = list(range(g * per_ct, min(n, (g + 1) * per_ct)))
        C, M = [], []
        for b in range(per_ct):
            if b < len(cand):
                C += [float(x[cand[b]])] * N
                M += [1.0] * n + [0.0] * (N - n)
            else:
                C += [pad] * N
                M += [0.0] * N
        eC = cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(C))
        t0 = time.perf_counter()
        diff = cc.EvalSub(eD, eC)
        kv = cc.EvalChebyshevFunction(kfun, cc.EvalMult(diff, diff), 0.0, R, d)
        y = cc.EvalMult(kv, cc.MakeCKKSPackedPlaintext(M, 1, kv.GetLevel(), None, batch))
        st = 1
        while st < N:
            y = cc.EvalAdd(y, cc.EvalRotate(y, -st))
            st *= 2
        t_server += time.perf_counter() - t0
        q = cc.Decrypt(k.secretKey, y)
        q.SetLength(batch)
        v = np.array(q.GetRealPackedValue())
        for b, i in enumerate(cand):
            fhe[i] = v[b * N + N - 1]
    out = dict(layout="block", n=n, N=N, sigma=sigma, d=d, scale=scale, depth=depth,
               ringdim=cc.GetRingDimension(), batch=batch, pipelines=groups,
               server_s=t_server, levels=y.GetLevel(),
               eta=float(np.abs(fhe - a["S_poly"]).max()),
               fhe_ok=bool((dados == MODE_VALUE)[int(np.argmax(fhe))]),
               poly_ok=a["poly_ok"], dS=a["dS"], S_max=a["S_max"])
    if measure_keys:
        out["rotkeys_MB"] = rot_key_MB(cc, k)
        out["n_rotkeys"] = len(rot)
    return out


def run_candidate(dados, sigma, d, scale=50):
    a = analyse(dados, sigma, d)
    x, R, n = a["x"], a["R"], a["n"]
    N = 2 ** math.ceil(math.log2(n))
    pad = float(x[0])
    rot = [-(2 ** k) for k in range(int(math.log2(N)))]
    depth = 1 + 1 + cheb_depth(d) + 1
    cc, k = context(depth, N, scale, rot)
    enc = cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(list(x) + [pad] * (N - n)))

    def pt(vals, ct):
        return cc.MakeCKKSPackedPlaintext(vals, 1, ct.GetLevel(), None, N)

    def rsum(c):
        st = 1
        while st < N:
            c = cc.EvalAdd(c, cc.EvalRotate(c, -st))
            st *= 2
        return c

    t0 = time.perf_counter()
    outs = []
    for i in range(n):
        m = [0.0] * N
        m[i] = 1.0
        c = rsum(cc.EvalMult(enc, pt(m, enc)))
        diff = cc.EvalSub(enc, c)
        kv = cc.EvalChebyshevFunction(kfun, cc.EvalMult(diff, diff), 0.0, R, d)
        outs.append(rsum(cc.EvalMult(kv, pt([1.0] * n + [0.0] * (N - n), kv))))
    t = time.perf_counter() - t0
    fhe = np.empty(n)
    for i, c in enumerate(outs):
        q = cc.Decrypt(k.secretKey, c)
        q.SetLength(1)
        fhe[i] = q.GetRealPackedValue()[0]
    return dict(layout="candidate", n=n, N=N, sigma=sigma, d=d, scale=scale, depth=depth,
                ringdim=cc.GetRingDimension(), server_s=t, levels=outs[0].GetLevel(),
                eta=float(np.abs(fhe - a["S_poly"]).max()),
                fhe_ok=bool((dados == MODE_VALUE)[int(np.argmax(fhe))]))


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[gravado {name}.json]", flush=True)


# ------------------------------------------------------------- experiments
def rq1():
    """CKKS check of configurations covering every region of the grid."""
    fcs = list(range(8, 49, 4))
    bounds = {str(f_c): separation_bound(ds_rq1(f_c)) for f_c in fcs}
    checks = []
    picks = [(8, 0.01), (8, 0.1), (8, 1.0), (8, 5.0), (16, 0.3), (16, 3.0), (24, 1.0),
             (24, 3.0), (32, 0.5), (32, 2.0), (40, 1.0), (48, 0.2)]
    for f_c, s in picks:
        r = run_block(ds_rq1(f_c), s, 31)
        r.update(f_c=f_c)
        checks.append(r)
        print(f"  fhe f_c={f_c} sigma={s}: poly_ok={r['poly_ok']} fhe_ok={r['fhe_ok']} "
              f"eta={r['eta']:.1e}", flush=True)
    save("rq1", dict(bounds=bounds, fhe_checks=checks))


def rq2():
    rows = []
    sets = [("reference", ds_reference())] + [(f"rq1_fc{f}", ds_rq1(f)) for f in (8, 16, 24)]
    for name, dados in sets:
        for s in (0.5, 1.0, 2.0, 3.0, 5.0):
            base = analyse(dados, s, 31)
            if base["dS"] <= 0:
                rows.append(dict(dataset=name, sigma=s, dS=base["dS"], kernel_ok=False))
                continue
            d_cond = d_emp = None
            for d in range(1, 301):
                a = analyse(dados, s, d)
                if d_emp is None and a["poly_ok"]:
                    d_emp = d
                if d_cond is None and a["n"] * a["eps"] < a["dS"] / 2:
                    d_cond = d
                if d_emp is not None and d_cond is not None:
                    break
            rows.append(dict(dataset=name, sigma=s, dS=base["dS"], kernel_ok=True,
                             d_cert=base["d_cert"], d_cond=d_cond, d_emp=d_emp))
            print(f"  {name} sigma={s}: cert={base['d_cert']} cond={d_cond} emp={d_emp}", flush=True)
    save("rq2", rows)


def rq3():
    rows = []
    for scale in (20, 25, 30, 35, 40, 45, 50):
        try:
            ts = []
            for _ in range(3):
                r = run_block(ds_reference(), 3.0, 31, scale=scale)
                ts.append(r["server_s"])
            r["server_s"] = float(np.median(ts))
            rows.append(r)
            print(f"  scale={scale}: ring=2^{r['ringdim'].bit_length() - 1} eta={r['eta']:.2e} "
                  f"t={r['server_s']:.2f}s ok={r['fhe_ok']}", flush=True)
        except Exception as e:
            rows.append(dict(scale=scale, error=f"{type(e).__name__}: {str(e)[:160]}"))
            print(f"  scale={scale}: FALHOU", flush=True)
    save("rq3", dict(dS=analyse(ds_reference(), 3.0, 31)["dS"], rows=rows))


def overflow():
    """n = 1024: the largest score exceeds 2^(59 - scale) with 50-bit primes."""
    rows = []
    for scale in (50, 45, 40):
        r = run_block(ds_scaled(1024), 3.0, 31, scale=scale)
        rows.append(r)
        print(f"  n=1024 scale={scale}: S_max={r['S_max']:.0f} eta={r['eta']:.2e} "
              f"ok={r['fhe_ok']} t={r['server_s']:.1f}s", flush=True)
    save("overflow", rows)


def rq4d():
    rows = []
    for d in (7, 15, 31, 63, 127):
        ts = []
        for _ in range(3):
            r = run_block(ds_reference(), 3.0, d)
            ts.append(r["server_s"])
        r["server_s"] = float(np.median(ts))
        r["eps"] = analyse(ds_reference(), 3.0, d)["eps"]
        rows.append(r)
        print(f"  d={d}: depth={r['depth']} ring=2^{r['ringdim'].bit_length() - 1} "
              f"t={r['server_s']:.2f}s ok={r['fhe_ok']}", flush=True)
    save("rq4d", rows)


def rq4n():
    rows = []
    for n in (16, 32, 64, 128, 256, 512, 1024):
        # n = 1024 uses 45 bits: with 50 the largest score does not fit (see overflow())
        r = run_block(ds_scaled(n), 3.0, 31, scale=45 if n >= 1024 else 50,
                      measure_keys=(n in (16, 1024)))
        rows.append(r)
        print(f"  bloco n={n}: pipelines={r['pipelines']} t={r['server_s']:.2f}s "
              f"eta={r['eta']:.1e} ok={r['fhe_ok']}", flush=True)
        save("rq4n", rows)
    for n in (16, 32, 64, 128, 256):
        r = run_candidate(ds_scaled(n), 3.0, 31)
        rows.append(r)
        print(f"  candidato n={n}: t={r['server_s']:.1f}s ok={r['fhe_ok']}", flush=True)
        save("rq4n", rows)


if __name__ == "__main__":
    which = sys.argv[1]
    names = ["rq1", "rq2", "rq3", "overflow", "rq4d", "rq4n"] if which == "all" else [which]
    for name in names:
        print(f"=== {name} ===", flush=True)
        t = time.perf_counter()
        globals()[name]()
        print(f"=== {name} terminou em {time.perf_counter() - t:.0f}s ===", flush=True)
