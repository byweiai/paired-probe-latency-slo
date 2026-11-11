# -*- coding: utf-8 -*-
"""Single source of truth after the review round: every quantity the evaluation
reports, recomputed from raw files with the corrected pairing and the corrected
rate definitions. Writes out/rebuild.json."""
import glob
import io
import json
import math

import numpy as np
import pandas as pd

DELTA = 0.05
TAUS = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]
AS = [0.90, 0.95, 0.99]
MAXC = 8.0  # anything above this is not a usable latency objective


def dkw(n, delta=DELTA):
    return math.sqrt(0.5 / n * math.log(2.0 / delta)) if n else 1.0


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


out = {"meta": {"delta": DELTA, "source": "e7_final snapshot, per_request.csv, e5_governor.json"}}

# ------------------------------------------------------------------ paired probe
rows = []
for f in sorted(glob.glob("out/e7_final/*.jsonl")):
    if "pilot" in f:
        continue
    for line in io.open(f, encoding="utf-8"):
        rows.append(json.loads(line))
p = pd.DataFrame(rows)
p["ttft_ms"] = pd.to_numeric(p["ttft_ms"], errors="coerce")
solo = p[p["arm"] == "solo"]
load = p[p["arm"] == "load"]
m = solo.merge(load, on=["tag", "round", "id"], suffixes=("_s", "_l"))
m = m[(m["status_s"] == 200) & (m["status_l"] == 200)
      & m["ttft_ms_s"].notna() & m["ttft_ms_l"].notna()]
m["ratio"] = m["ttft_ms_s"] / m["ttft_ms_l"]
m["gain"] = 1.0 - m["ttft_ms_l"] / m["ttft_ms_s"]          # fractional speed-up

probe = {"tags": int(p["tag"].nunique()), "pairs": int(len(m)), "stacks": int(m["tag"].nunique())}
probe["rounds_per_stack"] = {t: int(g["round"].nunique()) for t, g in m.groupby("tag")}
probe["rounds_min"] = min(probe["rounds_per_stack"].values())
probe["rounds_max"] = max(probe["rounds_per_stack"].values())
probe["last_round_complete"] = sum(1 for v in probe["rounds_per_stack"].values() if v == probe["rounds_max"])

by = {}
for tag, g in m.groupby("tag"):
    r = g["ratio"].values
    k = int((r > 1).sum())
    lo, hi = wilson(k, len(r))
    sv = solo.loc[(solo["tag"] == tag) & (solo["status"] == 200) & solo["ttft_ms"].notna(), "ttft_ms"].values
    by[tag] = {"n": int(len(r)), "pfast": k / len(r), "pfast_lo": lo, "pfast_hi": hi,
               "m50": float(np.percentile(r, 50)), "m95": float(np.percentile(r, 95)),
               "m99": float(np.percentile(r, 99)), "n0": int(len(sv))}
probe["by_stack"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in by.items()}
probe["pfast_min"] = min(v["pfast"] for v in by.values())
probe["pfast_max"] = max(v["pfast"] for v in by.values())
probe["m95_min"] = min(v["m95"] for v in by.values())
probe["m95_max"] = max(v["m95"] for v in by.values())
probe["m95_min_stack"] = min(by, key=lambda t: by[t]["m95"])
probe["m95_max_stack"] = max(by, key=lambda t: by[t]["m95"])
probe["pfast_min_stack"] = min(by, key=lambda t: by[t]["pfast"])
probe["pfast_max_stack"] = max(by, key=lambda t: by[t]["pfast"])

# within-stack movement of m-hat across held-out rounds
spread = {}
for tag, g in m.groupby("tag"):
    per = [np.percentile(gg["ratio"].values, 95) for _, gg in g.groupby("round") if len(gg) >= 8]
    if len(per) >= 2:
        spread[tag] = float(max(per) / min(per))
probe["m95_round_spread"] = {k: round(v, 3) for k, v in spread.items()}
probe["m95_spread_min"] = min(spread.values())
probe["m95_spread_max"] = max(spread.values())

# per-class speed-up share
cls = {}
for c, g in m.groupby("cls_s"):
    cls[c] = {"n": int(len(g)),
              "pfast": float((g["ratio"] > 1).mean()),
              "gain10": float((g["gain"] >= 0.10).mean())}
probe["by_class"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cls.items()}
probe["gain10_min"] = min(v["gain10"] for v in cls.values())
probe["gain10_max"] = max(v["gain10"] for v in cls.values())
probe["gain10_min_cls"] = min(cls, key=lambda c: cls[c]["gain10"])
probe["gain10_max_cls"] = max(cls, key=lambda c: cls[c]["gain10"])
out["probe"] = probe

# ------------------------------------------------------------------ envelope cells
cells = []
for tag, e in by.items():
    sv = solo.loc[(solo["tag"] == tag) & (solo["status"] == 200) & solo["ttft_ms"].notna(), "ttft_ms"].values
    lg = load[load["tag"] == tag]
    eps = dkw(len(sv))
    for t in TAUS:
        A = float(((lg["status"] == 200) & (lg["ttft_ms"] <= t)).mean())
        row = {"tag": tag, "tau": t, "A": A, "F0": float(np.mean(sv <= t)), "n0": len(sv), "eps": eps}
        for key in ("m95", "m99"):
            F0m = float(np.mean(sv <= t * e[key]))
            row[key] = F0m
            row["exc_" + key] = A - F0m
            row["sig_" + key] = bool(A > F0m + eps)
            row["pinned_" + key] = bool(F0m >= 1.0)
        row["exc_m1"] = A - row["F0"]
        row["sig_m1"] = bool(A > row["F0"] + eps)
        row["pinned_m1"] = bool(row["F0"] >= 1.0)
        cells.append(row)
out["cells"] = cells
env = {
    "cells": len(cells),
    "sig_m1": sum(c["sig_m1"] for c in cells),
    "point_m1": sum(1 for c in cells if c["exc_m1"] > 0),
    "max_exc_m1": max(c["exc_m1"] for c in cells),
    "sig_m95": sum(c["sig_m95"] for c in cells),
    "point_m95": sum(1 for c in cells if c["exc_m95"] > 0),
    "max_exc_m95": max(c["exc_m95"] for c in cells),
    "pinned_m95": sum(c["pinned_m95"] for c in cells),
    "discriminating_m95": sum(1 for c in cells if not c["pinned_m95"]),
    "pinned_m1": sum(c["pinned_m1"] for c in cells),
    "eps_min": min(c["eps"] for c in cells),
    "eps_max": max(c["eps"] for c in cells),
    "n0_min": min(c["n0"] for c in cells),
    "n0_max": max(c["n0"] for c in cells),
    "n_required": int(math.ceil(0.5 * math.log(2 / DELTA) / 0.05 ** 2)),
}
out["envelope"] = env

# ------------------------------------------------------------------ metric comparison
df = pd.read_csv("out/per_request.csv", low_memory=False)
df["c"] = df["concurrency"].astype(str).str.strip("[]").astype(int)
df = df[df["status"].isin([200, 429])]
rounds = []
for (f, c), g in df.groupby(["file", "c"]):
    n = len(g)
    ok = g[g["status"] == 200]
    if len(ok) < 5:
        continue
    rounds.append({"stack": f, "c": int(c), "n": n, "theta": float((g["status"] == 429).sum() / n),
                   "att": {t: float(((g["status"] == 200) & (g["ttft_ms"] <= t)).sum() / n) for t in TAUS},
                   "adm": {t: float((ok["ttft_ms"] <= t).mean()) for t in TAUS}})
met = {"rounds": len(rounds), "cells": []}
for t in TAUS:
    for A in AS:
        att = sum(1 for r in rounds if r["att"][t] >= A)
        adm = sum(1 for r in rounds if r["adm"][t] >= A)
        dis = sum(1 for r in rounds if (r["adm"][t] >= A) != (r["att"][t] >= A))
        met["cells"].append({"tau": t, "A": A, "att": att, "adm": adm, "disagree": dis,
                             "disagree_rate": dis / len(rounds)})
met["max_disagree"] = max(c["disagree_rate"] for c in met["cells"])
met["max_disagree_cell"] = max(met["cells"], key=lambda c: c["disagree_rate"])
hi = [r for r in rounds if r["theta"] >= 0.2]
met["high_denial_rounds"] = len(hi)
if hi:
    worst = max(hi, key=lambda r: r["adm"][3000] - r["att"][3000])
    met["high_denial_example"] = {"stack": worst["stack"], "theta": worst["theta"],
                                  "att3": worst["att"][3000], "adm3": worst["adm"][3000]}
met["theta_zero_rounds"] = sum(1 for r in rounds if r["theta"] == 0)
out["metric"] = met

# ------------------------------------------------------------------ span and frame
span = {"platform": lambda: None}
r = pd.read_csv("out/rounds.csv")
span = {
    "combos_in_rounds": int(r.groupby(["file"]).ngroups),
    "levels_per_combo": {k: sorted(g["concurrency"].unique().tolist()) for k, g in r.groupby("file")},
    "rounds_total": int(len(r)),
}
lv = [len(v) for v in span["levels_per_combo"].values()]
span["levels_max"] = max(lv)
span["combos_at_max"] = sum(1 for x in lv if x == span["levels_max"])
span["vendors"] = sorted(r["vendor"].unique().tolist())
span["models"] = int(r["model"].nunique()) if "model" in r else None
out["frame"] = span

io.open("out/rebuild.json", "w", encoding="utf-8").write(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ("probe", "envelope", "frame")},
                 indent=1, default=float)[:2600])
print("\nmetric:", json.dumps(out["metric"]["cells"][:3]), met["max_disagree"], met["high_denial_example"])
