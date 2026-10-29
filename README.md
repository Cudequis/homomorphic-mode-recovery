# Homomorphic Mode Recovery via Kernel

Code accompanying the MSc dissertation *Homomorphic Mode Recovery via Kernel* (Instituto Superior
Técnico). A client outsources the statistical mode of a private dataset to an untrusted server under
CKKS. The server evaluates a kernelized score (Cauchy kernel approximated by a Chebyshev polynomial)
using only additions, multiplications and rotations, and the client takes the argmax in the clear.

## Contents

| Path | What it is |
|---|---|
| `moda_ckks_por_candidato.ipynb` | Reference implementation of the protocol (Chapter 4), per-candidate layout and block packing |
| `moda_ckks_simd.ipynb` | Alternative layout: the whole dataset in one ciphertext, one rotation per shift `k`, a single rotation key |
| `experiments/experiments.py` | Experiments of Chapter 5: `rq1`, `rq2`, `rq3`, `overflow`, `rq4d`, `rq4n`, or `all` |
| `experiments/rq1b.py` | Certified vs empirical degree in exact arithmetic (no FHE, numpy only) |
| `experiments/plots.py` | Figures of Chapter 5, written to `experiments/figures/` |
| `experiments/appendix_cheb.py` | Numerical checks and Bernstein-ellipse figure of Appendix A |
| `experiments/inspect_params.py`, `inspect_moduli.py` | Read the CKKS parameters OpenFHE actually selects |
| `experiments/*.json` | Results used in the dissertation |

## Requirements

Python 3 with OpenFHE 1.5.1 (Python bindings), numpy and matplotlib:

```bash
pip install -r requirements.txt
```

The OpenFHE experiments were run on Linux (WSL). The plotting and approximation scripts need only
numpy and matplotlib.

## Reproducing

```bash
cd experiments
./run_all.sh                 # all FHE experiments -> *.json
python rq1b.py               # exact-arithmetic degree study -> rq1b.json
python plots.py              # figures of Chapter 5
python appendix_cheb.py      # Appendix A checks and figure
```

Timings in the dissertation are medians of three runs on a 16-core AMD Ryzen 9 8940HX.
