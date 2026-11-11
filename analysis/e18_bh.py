# -*- coding: utf-8 -*-
"""Add bootstrap p-values and Benjamini-Hochberg q-values to the cluster test."""
import io
import json

import numpy as np
import pandas as pd

P = "out/e18_cluster.json"
d = json.load(io.open(P, encoding="utf-8"))
cells = pd.DataFrame(d["cells"])

# recompute p from the stored quantiles is not possible; rerun the bootstrap part
import glob
import pandas as pd  # noqa: F401


def loadjsonl(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in io.open(f, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["ttft_ms"] = pd.to_numeric(df["ttft_ms"], errors="coerce")
    return df[(df["status"] == 200) & df["ttft_ms"].notna()]


TAUS = d["meta"]["taus"]
DELTA = d["meta"]["delta"]
B = d["meta"]["draws"]
rng = np.random.default_rng(20261107)


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
load = e7[e7["arm"] == "load"]
solo7 = e7[e7["arm"] == "solo"]

by = {v["baseline"]: None for v in d["by_stack"].values()}
rows = []
for tag, v in d["by_stack"].items():
    ld = load[load["tag"] == tag]
    base = e14[e14["tag"] == tag] if v["baseline"] == "e14" else solo7[solo7["tag"] == tag]
    L, S = ld["ttft_ms"].values, base["ttft_ms"].values
    WL = boot_weights(ld["id"].values, B, rng)
    WS = boot_weights(base["id"].values, B, rng)
    for c in v["cells"]:
        m = 1.0 if c["m"] == "m1" else v["m95"]
        tau = c["tau"]
        a_b = (WL * (L <= tau)).sum(1) / WL.sum(1)
        f_b = (WS * (S <= m * tau)).sum(1) / WS.sum(1)
        D = a_b - f_b
        p = float((D <= 0).mean())
        rows.append({"stack": tag, "m": c["m"], "tau": tau, "p": max(p, 1.0 / B),
                     "judgeable": c["judgeable"]})

r = pd.DataFrame(rows)
for mname in ("m1", "m95"):
    sel = r[(r["m"] == mname) & r["judgeable"]].copy().sort_values("p").reset_index(drop=True)
    n = len(sel)
    sel["q"] = sel["p"] * n / (sel.index + 1)
    # enforce monotonicity from the largest p down
    q = sel["q"].values
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    sel["q"] = q
    r.loc[sel.index if False else sel.index, "q"] = np.nan
    for i, row_ in sel.iterrows():
        r.loc[(r["stack"] == row_["stack"]) & (r["m"] == mname) & (r["tau"] == row_["tau"]), "q"] = row_["q"]
    sig = sel[sel["q"] <= DELTA]
    print("=== %s: judgeable %d, p<=.05 %d, BH q<=.05 %d on %d stacks"
          % (mname, n, int((sel["p"] <= DELTA).sum()), len(sig), sig["stack"].nunique()))
    if len(sig):
        print("   stacks:", ", ".join(sorted(sig["stack"].unique())))
        print("   min p %.4f  max q among rejected %.4f" % (sig["p"].min(), sig["q"].max()))
    d["summary"][mname]["bh_sig"] = int(len(sig))
    d["summary"][mname]["bh_stacks"] = int(sig["stack"].nunique())
    d["summary"][mname]["p_min"] = round(float(sel["p"].min()), 4)
    d["summary"][mname]["worst_excess_sig"] = (
        round(float(cells[(cells["m"] == mname) & cells["refuted"]]["excess"].max()), 4)
        if cells["refuted"].any() else None)

d["pvalues"] = r.where(pd.notna(r), None).to_dict(orient="records")
io.open(P, "w", encoding="utf-8").write(json.dumps(d, indent=1, ensure_ascii=False))
print(json.dumps(d["summary"], indent=1))
