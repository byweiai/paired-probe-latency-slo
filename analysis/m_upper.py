"""Conservative upper bound for the acceleration quantile, and the envelope test run with it.

The certificate needs an estimate that is unlikely to sit *below* the true bound.  A point
quantile from the same pairs is not that: it is the statistic the test then uses, and across
rounds it moved by up to a factor of six.  Here we take, per stack, the largest of three
estimates -- the pooled 95th percentile, the upper end of a paired bootstrap for it, and the
largest per-round 95th percentile -- and re-run the cell test with that value.
"""
import glob
import io
import json
import math

import numpy as np
import pandas as pd

DELTA = 0.05
LEVEL = 0.95
BOOT = 2000
TAUS = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]
RNG = np.random.default_rng(20261106)


def dkw(n, d=DELTA):
    return math.sqrt(0.5 / n * math.log(2.0 / d)) if n else 1.0


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
solo7, load = e7[e7["arm"] == "solo"], e7[e7["arm"] == "load"]

# pairs, keyed on the stack as well as the round and the prompt
s = solo7[["tag", "round", "id", "ttft_ms"]].rename(columns={"ttft_ms": "t0"})
l = load[["tag", "round", "id", "ttft_ms"]].rename(columns={"ttft_ms": "t1"})
m = s.merge(l, on=["tag", "round", "id"])
assert not m.duplicated(subset=["tag", "round", "id"]).any()
m["R"] = m["t0"] / m["t1"]

rows = []
for tag, g in m.groupby("tag"):
    R = g["R"].values
    if len(R) < 30:
        continue
    point = float(np.quantile(R, LEVEL))
    boot = np.quantile(RNG.choice(R, size=(BOOT, len(R))), LEVEL, axis=1)
    boot_hi = float(np.quantile(boot, 0.975))
    per_round = g.groupby("round")["R"].apply(lambda v: float(np.quantile(v, LEVEL)) if len(v) >= 10 else np.nan)
    round_max = float(np.nanmax(per_round.values)) if np.isfinite(per_round.values).any() else float("nan")
    m_up = max(point, boot_hi, round_max)
    rows.append({"stack": tag, "pairs": int(len(R)), "P_fast": float((R > 1).mean()),
                 "m_point": point, "m_boot_hi": boot_hi, "m_round_max": round_max,
                 "m_up": m_up, "rounds": int(g["round"].nunique())})

up = pd.DataFrame(rows).sort_values("m_up").reset_index(drop=True)
print(up.to_string(index=False, float_format=lambda v: "%.3f" % v))

# ---- cells, run with the conservative bound --------------------------------
cells = []
for _, r in up.iterrows():
    tag = r["stack"]
    v7 = np.array(sorted(solo7[solo7["tag"] == tag]["ttft_ms"].values))
    v14 = np.array(sorted(e14d[e14d["tag"] == tag]["ttft_ms"].values))
    if len(v7) < 20:
        continue
    pooled = np.array(sorted(np.concatenate([v7, v14]))) if len(v14) >= 200 else v7
    eps = dkw(int(len(pooled)))
    L = load[load["tag"] == tag]["ttft_ms"].values
    if len(L) < 50:
        continue
    for tau in TAUS:
        a = float((L <= tau).mean())
        lo = min(1.0, float((pooled <= tau).mean()) + eps)              # m = 1
        hi = min(1.0, float((pooled <= r["m_up"] * tau).mean()) + eps)  # m = m_up
        cells.append({"stack": tag, "tau": tau, "A": round(a, 4),
                      "margin_m1": round(lo - a, 4), "margin_up": round(hi - a, 4),
                      "disc_m1": bool(lo < 1 - 1e-9), "disc_up": bool(hi < 1 - 1e-9),
                      "eps": round(eps, 3), "m_up": round(float(r["m_up"]), 3)})

c = pd.DataFrame(cells)
d1, du = c[c["disc_m1"]], c[c["disc_up"]]
summary = {
    "stacks": int(len(up)), "cells": int(len(c)),
    "disc_m1": int(len(d1)), "viol_m1": int((d1["margin_m1"] < 0).sum()),
    "max_exc_m1": round(float(max(0.0, -c["margin_m1"].min())), 4),
    "disc_up": int(len(du)), "viol_up": int((du["margin_up"] < 0).sum()),
    "min_margin_up": round(float(c["margin_up"].min()), 4),
    "m_up_max": round(float(up["m_up"].max()), 3),
    "m_up_min": round(float(up["m_up"].min()), 3),
    "m_point_max": round(float(up["m_point"].max()), 3),
    "ratio": round(float((up["m_up"] / up["m_point"]).max()), 3),
}
print()
for k, v in summary.items():
    print("%-12s %s" % (k, v))

json.dump({"level": LEVEL, "boot": BOOT, "delta": DELTA, "taus": TAUS,
           "stacks": up.to_dict("records"), "cells": cells, "summary": summary},
          io.open("out/m_upper.json", "w", encoding="utf-8"), indent=1)
print("\nwrote out/m_upper.json")
