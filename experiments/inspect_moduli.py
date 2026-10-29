"""Lists the paths in the context JSON that hold large integers (candidate moduli)."""
import json
import os
import sys
import tempfile

import openfhe

depth, batch = int(sys.argv[1]), int(sys.argv[2])
scale = int(sys.argv[3]) if len(sys.argv) > 3 else 50
p = openfhe.CCParamsCKKSRNS()
p.SetMultiplicativeDepth(depth)
p.SetScalingModSize(scale)
p.SetBatchSize(batch)
cc = openfhe.GenCryptoContext(p)
f = os.path.join(tempfile.mkdtemp(), "cc.json")
openfhe.SerializeToFile(f, cc, openfhe.JSON)
doc = json.load(open(f))


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, o


import re

keys = set()
for path, val in walk(doc):
    for seg in re.findall(r"/([A-Za-z_]+)", path):
        keys.add(seg)
import math

primes = []
for path, val in walk(doc):
    if re.search(r"/elp/ptr_wrapper/data/p\[\d+\]/ptr_wrapper/data/value0/cm/v$", path):
        primes.append(int(val))
print(f"ANEL 2^{cc.GetRingDimension().bit_length() - 1}  primos de Q: {len(primes)}  "
      f"bits: {[q.bit_length() for q in primes]}  log2 Q = {sum(math.log2(q) for q in primes):.1f}")
sys.exit(0)
print("chaves:", sorted(keys))
for path, val in walk(doc):
    if "/cm/" in path or path.endswith("/cm/v") or re.search(r"/cm/v\[\d+\]$", path):
        short = re.sub(r"/ptr_wrapper|/data|/value0", "", path)
        try:
            print(short, int(val).bit_length())
        except (TypeError, ValueError):
            pass
