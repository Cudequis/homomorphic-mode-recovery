"""Reads the parameters OpenFHE actually selects (Q and P moduli, key-switching digits).

Usage: python inspect_params.py <depth> <batch> [scale]
"""
import json
import math
import os
import re
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

out = {"depth": depth, "batch": batch, "scale": scale, "ringdim": cc.GetRingDimension()}
for name in ("GetFirstModSize", "GetScalingModSize", "GetNumLargeDigits", "GetKeySwitchTechnique",
             "GetSecretKeyDist", "GetSecurityLevel", "GetScalingTechnique", "GetStandardDeviation"):
    try:
        out[name] = str(getattr(p, name)())
    except Exception as e:
        out[name] = f"n/a ({type(e).__name__})"

f = os.path.join(tempfile.mkdtemp(), "cc.json")
openfhe.SerializeToFile(f, cc, openfhe.JSON)
txt = open(f).read()
out["json_bytes"] = len(txt)
# moduli appear as decimal strings; collect large integers and their bit lengths
nums = [int(x) for x in re.findall(r'"(\d{9,})"', txt)] or [int(x) for x in re.findall(r'(?<![\d.])(\d{9,})(?![\d.])', txt)]
out["moduli_bits_raw"] = [n.bit_length() for n in nums][:80]
for key in ("numPartQ", "numLargeDigits", "dnum", "auxBits", "ksTech", "numPerPartQ"):
    m = re.search(rf'"{key}"\s*:\s*(\d+)', txt)
    if m:
        out[key] = int(m.group(1))
print(json.dumps(out, indent=1))
