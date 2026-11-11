# -*- coding: utf-8 -*-
"""Cluster-robust envelope test (replaces the i.i.d. DKW reading).

Two things the i.i.d. band gets wrong, both raised in review:
  * requests are nested in prompt instances, so n0 overstates the sample size;
  * the load arm and the baseline are two independent samples, so their
    instance-to-instance shifts add rather than cancel.

Here both arms are bootstrapped over prompt instances and combined. The test is
the standard one-sided bootstrap: replicate D* = A*(tau) - F0*(m*tau); reject
the envelope when the delta-quantile of D* is already above zero, i.e. when the
data would rarely have produced A <= F0(m*tau) had the envelope held.
"""
import glob
import io
import json

import numpy as np
import pandas as pd

TAUS = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]
DELTA = 0.05
B = 800
rng = np.random.default_rng(20261107)


def loadjsonl(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in io.open(f, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    d = pd.DataFrame(rows)
    d["ttft_ms"] = pd.to_numeric(d["ttft_ms"], errors="coerce")
    return d[(d["status"] == 200) & d["ttft_ms"].notna()]


def boot_weights(cl, B, rng):
    ids = np.unique(cl)
    pos = {k: i for i, k in enumerate(ids)}
    ci = np.array([pos[v] for v in cl])
    W = rng.integers(0, len(ids), (B, len(ids)))
    M = np.zeros((B, len(ids)))
    for b in range(B):
        M[b] = np.bincount(W[b], minlength=len(ids))
    return M[:, ci]


e7 = loadjsonl("out/e7_final/*.jsonl")
e14 = loadjsonl("out/e14/*.jsonl")
e14 = e14[e14["arm"] == "solo"]
reb = json.load(io.open("out/rebuild.json", encoding="utf-8"))
probe = reb["probe"]["by_stack"]
load = e7[e7["arm"] == "load"]
solo7 = e7[e7["arm"] == "solo"]

out = {"meta": {"taus": TAUS, "delta": DELTA, "draws": B,
                "baseline": "e14 solo sweep when available, else the e7 solo arm",
                "cluster": "prompt instance id, bootstrapped independently per arm",
                "test": "one-sided; refuted when the delta-quantile of D* exceeds zero"},
       "by_stack": {}, "cells": []}

for tag, p in probe.items():
    ld = load[load["tag"] == tag]
    base = e14[e14["tag"] == tag]
    bname = "e14"
    if len(base) < 200:
        base, bname = solo7[solo7["tag"] == tag], "e7"
    if len(ld) < 50 or len(base) < 20:
        continue
    L = ld["ttft_ms"].values
    Sl = base["ttft_ms"].values
    WL = boot_weights(ld["id"].values, B, rng)
    WS = boot_weights(base["id"].values, B, rng)
    m95 = max(1.0, p["m95"])
    rec = {"n_load": int(len(L)), "n_solo": int(len(Sl)), "baseline": bname,
           "clusters_load": int(ld["id"].nunique()),
           "clusters_solo": int(base["id"].nunique()),
           "m95": round(m95, 3), "cells": []}
    for mname, m in (("m1", 1.0), ("m95", m95)):
        for tau in TAUS:
            a = float((L <= tau).mean())
            f = float((Sl <= m * tau).mean())
            a_b = (WL * (L <= tau)).sum(1) / WL.sum(1)
            f_b = (WS * (Sl <= m * tau)).sum(1) / WS.sum(1)
            D = a_b - f_b
            q05 = float(np.quantile(D, DELTA))
            q95 = float(np.quantile(D, 1 - DELTA))
            cell = {"m": mname, "tau": tau, "A": round(a, 4), "F0": round(f, 4),
                    "excess": round(a - f, 4),
                    "d_lo": round(q05, 4), "d_hi": round(q95, 4),
                    "refuted": bool(q05 > 0.0),
                    "judgeable": bool(f < 0.98)}
            rec["cells"].append(cell)
            out["cells"].append(dict(cell, stack=tag))
    out["by_stack"][tag] = rec

d = pd.DataFrame(out["cells"])
out["summary"] = {}
for mname in ("m1", "m95"):
    g = d[(d["m"] == mname) & d["judgeable"]]
    gg = d[d["m"] == mname]
    out["summary"][mname] = {
        "cells": int(len(gg)), "judgeable": int(len(g)),
        "refuted": int(g["refuted"].sum()),
        "worst_excess": round(float(gg["excess"].max()), 4),
        "worst_lower": round(float(g["d_lo"].max()), 4) if len(g) else None,
        "min_p_like": round(float((g["d_lo"] > 0).mean()), 4) if len(g) else None,
    }
io.open("out/e18_cluster.json", "w", encoding="utf-8").write(
    json.dumps(out, indent=1, ensure_ascii=False))

print(json.dumps(out["summary"], indent=1))
for mname in ("m1", "m95"):
    print("=== " + mname)
    for tag, v in out["by_stack"].items():
        cs = [c for c in v["cells"] if c["m"] == mname and c["judgeable"]]
        if not cs:
            continue
        print("  %-20s base=%-4s judge %d refuted %d worst %+0.3f d_lo[%+0.3f]"
              % (tag, v["baseline"], len(cs), sum(c["refuted"] for c in cs),
                 max(c["excess"] for c in cs), max(c["d_lo"] for c in cs)))
