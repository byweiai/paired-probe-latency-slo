# -*- coding: utf-8 -*-
"""Cluster-robust envelope test on the cache-free control (E16), plus BH.

E16 draws the two arms from disjoint, freshly worded prompts, so neither arm can
warm the other's prefix. The arms therefore share no cluster, and the two-sample
bootstrap over prompt instances is the honest band.
"""
import glob
import io
import json
import re

import numpy as np
import pandas as pd

TAUS = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]
DELTA = 0.05
B = 800
rng = np.random.default_rng(20261108)

rows = []
for f in glob.glob("out/e16/*.jsonl"):
    for line in io.open(f, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
d = pd.DataFrame(rows)
d["ttft_ms"] = pd.to_numeric(d["ttft_ms"], errors="coerce")
d = d[(d["status"] == 200) & d["ttft_ms"].notna()]
# cluster = the prompt instance inside an arm: ids look like solo-r3-B7
d["cl"] = d["arm"] + "-" + d["id"].str.split("-").str[-1]
pfast = json.load(io.open("out/rebuild.json", encoding="utf-8"))["probe"]["by_stack"]


def boot(cl, B, rng):
    ids = np.unique(cl)
    pos = {k: i for i, k in enumerate(ids)}
    ci = np.array([pos[v] for v in cl])
    M = np.zeros((B, len(ids)))
    for b in range(B):
        M[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    return M[:, ci]


out = {"meta": {"taus": TAUS, "delta": DELTA, "draws": B, "source": "out/e16/*.jsonl",
                "cluster": "prompt instance within arm"}, "by_stack": {}, "cells": []}
for tag, g in d.groupby("tag"):
    L = g[g["arm"] == "load"]
    S = g[g["arm"] == "solo"]
    if len(L) < 40 or len(S) < 40:
        print("skip", tag, len(L), len(S))
        continue
    m95 = max(1.0, pfast.get(tag, {}).get("m95", 1.0))
    WL = boot(L["cl"].values, B, rng)
    WS = boot(S["cl"].values, B, rng)
    Lv, Sv = L["ttft_ms"].values, S["ttft_ms"].values
    rec = {"n_load": len(Lv), "n_solo": len(Sv), "m95": round(m95, 3),
           "clusters": int(g["cl"].nunique()), "cells": []}
    for mname, m in (("m1", 1.0), ("m95", m95)):
        for tau in TAUS:
            a = float((Lv <= tau).mean())
            f = float((Sv <= m * tau).mean())
            a_b = (WL * (Lv <= tau)).sum(1) / WL.sum(1)
            f_b = (WS * (Sv <= m * tau)).sum(1) / WS.sum(1)
            D = a_b - f_b
            cell = {"m": mname, "tau": tau, "A": round(a, 4), "F0": round(f, 4),
                    "excess": round(a - f, 4),
                    "d_lo": round(float(np.quantile(D, DELTA)), 4),
                    "refuted": bool(np.quantile(D, DELTA) > 0),
                    "judgeable": bool(f < 0.98),
                    "p": max(float((D <= 0).mean()), 1.0 / B)}
            rec["cells"].append(cell)
            out["cells"].append(dict(cell, stack=tag))
    out["by_stack"][tag] = rec

df = pd.DataFrame(out["cells"])
out["summary"] = {}
for mname in ("m1", "m95"):
    sel = df[(df["m"] == mname) & df["judgeable"]].sort_values("p").reset_index(drop=True)
    n = len(sel)
    q = (sel["p"] * n / (sel.index + 1)).to_numpy(copy=True)
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    sel["q"] = q
    # write back
    for _, r in sel.iterrows():
        df.loc[(df["stack"] == r["stack"]) & (df["m"] == mname) & (df["tau"] == r["tau"]), "q"] = r["q"]
    sig = sel[sel["q"] <= DELTA]
    out["summary"][mname] = {
        "cells": int(len(df[df["m"] == mname])), "judgeable": int(n),
        "raw_sig": int((sel["p"] <= DELTA).sum()), "bh_sig": int(len(sig)),
        "stacks": int(sig["stack"].nunique()),
        "worst_excess": round(float(df[df["m"] == mname]["excess"].max()), 4),
        "min_p": round(float(sel["p"].min()), 4),
    }
    print("=== %s cells %d judgeable %d raw %d BH %d stacks %d worst %+0.3f"
          % (mname, len(df[df["m"] == mname]), n, int((sel["p"] <= DELTA).sum()),
             len(sig), sig["stack"].nunique(),
             df[df["m"] == mname]["excess"].max()))
    if len(sig):
        print("   ", ", ".join("%s@%d" % (r["stack"], r["tau"]) for _, r in sig.iterrows()))

out["cells"] = df.where(pd.notna(df), None).to_dict(orient="records")
io.open("out/e19_cachefree.json", "w", encoding="utf-8").write(
    json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out["summary"], indent=1))
