"""Envelope scoreboard with the pooled solo baseline, computed for m=1 and m=m95."""
import glob
import io
import json
import math

import numpy as np
import pandas as pd

DELTA = 0.05
TAUS = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]


def dkw(n, d=DELTA):
    return math.sqrt(0.5 / n * math.log(2.0 / d)) if n else 1.0


reb = json.load(io.open("out/rebuild.json", encoding="utf-8"))
probe = reb["probe"]["by_stack"]
e14 = json.load(io.open("out/e14.json", encoding="utf-8"))["stacks"]


def loadjsonl(pattern, skip_smoke=False):
    rows = []
    for f in glob.glob(pattern):
        if skip_smoke and "smoke" in f:
            continue
        for line in io.open(f, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    d = pd.DataFrame(rows)
    d["ttft_ms"] = pd.to_numeric(d["ttft_ms"], errors="coerce")
    return d[(d["status"] == 200) & d["ttft_ms"].notna()]


e7 = loadjsonl("out/e7_final/*.jsonl")
e14d = loadjsonl("out/e14/*.jsonl", skip_smoke=True)
e14d = e14d[e14d["arm"] == "solo"]
load = e7[e7["arm"] == "load"]
solo7 = e7[e7["arm"] == "solo"]

cells, diag = [], {}
for tag, p in probe.items():
    v7 = np.array(sorted(solo7[solo7["tag"] == tag]["ttft_ms"].values))
    v14 = np.array(sorted(e14d[e14d["tag"] == tag]["ttft_ms"].values))
    if len(v7) < 20:
        continue
    pooled = np.array(sorted(np.concatenate([v7, v14]))) if len(v14) >= 200 else v7
    n0, eps = int(len(pooled)), dkw(int(len(pooled)))
    m95 = max(1.0, p["m95"])
    shift = (max(abs(float((v7 <= m95 * t).mean()) - float((v14 <= m95 * t).mean()))
                 for t in TAUS) if len(v14) >= 200 else float("nan"))
    diag[tag] = {"n_e7": int(len(v7)), "n_e14": int(len(v14)), "n0": n0,
                 "eps": round(eps, 3), "shift": round(shift, 3)}
    L = load[load["tag"] == tag]["ttft_ms"].values
    if len(L) < 50:
        continue
    for tau in TAUS:
        a = float((L <= tau).mean())
        for name, m in (("m1", 1.0), ("m95", m95)):
            thr = min(1.0, float((pooled <= m * tau).mean()) + eps)
            cells.append({"stack": tag, "tau": tau, "m": name, "A": round(a, 4),
                          "thr": round(thr, 4), "margin": round(thr - a, 4),
                          "disc": bool(thr < 1.0 - 1e-9), "eps": round(eps, 3), "n0": n0})

df = pd.DataFrame(cells)


def block(m):
    g = df[df["m"] == m]
    d = g[g["disc"]]
    return {"cells": int(len(g)), "discriminating": int(len(d)),
            "viol": int((d["margin"] < 0).sum()),
            "max_excess": round(float(max(0.0, -g["margin"].min())), 4),
            "noinfo": int((~g["disc"]).sum())}


out = {"meta": {"delta": DELTA, "taus": TAUS}, "m1": block("m1"), "m95": block("m95"),
       "by_stack": {}, "diag": diag, "cells": cells,
       "summary": {"eps_min": round(float(df["eps"].min()), 3),
                   "eps_max": round(float(df["eps"].max()), 3),
                   "n0_min": int(df["n0"].min()), "n0_max": int(df["n0"].max()),
                   "stacks_full": int(sum(1 for d in diag.values() if d["n_e14"] >= 1800)),
                   "shift_max": round(float(np.nanmax([d["shift"] for d in diag.values()])), 3)}}
for tag, g in df.groupby("stack"):
    g1, g9 = g[g["m"] == "m1"], g[g["m"] == "m95"]
    d9 = g9[g9["disc"]]
    out["by_stack"][tag] = {
        "n0": int(g["n0"].iloc[0]), "eps": round(float(g["eps"].iloc[0]), 3),
        "pfast": round(float(probe[tag]["pfast"]), 4), "m95": round(float(probe[tag]["m95"]), 3),
        "m1_viol": int((g1[g1["disc"]]["margin"] < 0).sum()),
        "m1_max_excess": round(float(max(0.0, -g1["margin"].min())), 4),
        "disc": int(len(d9)), "viol": int((d9["margin"] < 0).sum()),
        "worst": round(float(d9["margin"].min()) if len(d9) else float("nan"), 4)}

io.open("out/rebuild2.json", "w", encoding="utf-8").write(json.dumps(out, indent=1))

print("%-18s %5s %6s %6s %6s %6s %6s %6s" %
      ("stack", "n0", "eps", "m95", "m1vio", "m1exc", "disc", "worst95"))
for tag, v in sorted(out["by_stack"].items(), key=lambda kv: kv[1]["m95"]):
    print("%-18s %5d %6.3f %6.2f %6d %6.3f %6d %6.3f"
          % (tag, v["n0"], v["eps"], v["m95"], v["m1_viol"], v["m1_max_excess"], v["disc"], v["worst"]))
print("\nm=1 : cells %d disc %d viol %d max_excess %.3f"
      % (out["m1"]["cells"], out["m1"]["discriminating"], out["m1"]["viol"], out["m1"]["max_excess"]))
print("m95 : cells %d disc %d viol %d max_excess %.3f"
      % (out["m95"]["cells"], out["m95"]["discriminating"], out["m95"]["viol"], out["m95"]["max_excess"]))
print("eps %.3f-%.3f  n0 %d-%d  stacks_full %d  shift<=%.3f"
      % (out["summary"]["eps_min"], out["summary"]["eps_max"], out["summary"]["n0_min"],
         out["summary"]["n0_max"], out["summary"]["stacks_full"], out["summary"]["shift_max"]))
