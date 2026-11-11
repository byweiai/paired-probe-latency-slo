"""E14 - the certificate's free parameter taken at an upper confidence limit.

E13 uses the point estimate of the alpha-quantile of R = T0/T. A certificate needs
an upper estimate instead: if the quantile is under-estimated the bound is
anti-conservative. Here the same leave-one-round-out protocol is repeated with
m_hat replaced by the 90% upper limit of a 400-draw bootstrap of the alpha-quantile,
so the reported violation count answers the question a referee will ask: does the
envelope survive when the free parameter is estimated pessimistically?
"""
import glob
import json
import math
import os

import numpy as np

D = "out/e7_final"
TAU_Q = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
ALPHAS = [0.90, 0.95, 0.99]
GAMMA = 0.90          # nominal coverage of the bootstrap upper limit
DRAWS = 400
DELTA = 0.05
RNG = np.random.default_rng(20261107)


def load(tag):
    rows = []
    with open(os.path.join(D, tag), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pairs_by_round(rows):
    d = {}
    for r in rows:
        if r.get("status") != 200 or r.get("ttft_ms") is None:
            continue
        key = (r["round"], r["id"])
        d.setdefault(key, {})[r["arm"]] = float(r["ttft_ms"])
    return {k: v for k, v in d.items() if "solo" in v and "load" in v}


def ucb(ratio, a, gamma=GAMMA, draws=DRAWS):
    """gamma upper limit of the a-quantile, percentile bootstrap."""
    ratio = np.asarray(ratio, dtype=float)
    n = ratio.size
    idx = RNG.integers(0, n, size=(draws, n))
    reps = np.quantile(ratio[idx], a, axis=1)
    return float(np.quantile(reps, gamma))


out = {"gamma": GAMMA, "draws": DRAWS, "delta": DELTA, "alphas": ALPHAS,
       "stacks": [], "summary": {}}
agg = {a: {"cells": 0, "viol": 0, "max_excess": -9.0, "m_ucb_max": 0.0} for a in ALPHAS}
rounds_total = 0

for path in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
    tag = os.path.splitext(os.path.basename(path))[0]
    rows = load(os.path.basename(path))
    pr = pairs_by_round(rows)
    rounds = sorted({k[0] for k in pr})
    if len(rounds) < 2 or len(pr) < 20:
        continue
    rec = {"tag": tag, "n_pairs": len(pr), "rounds": rounds, "cells": []}
    for r in rounds:
        train = [v for k, v in pr.items() if k[0] != r]
        test = [v for k, v in pr.items() if k[0] == r]
        if len(train) < 15 or len(test) < 3:
            continue
        rounds_total += 1
        ratio = [v["solo"] / v["load"] for v in train if v["load"] > 0]
        solo_train = [v["solo"] for v in train]
        n0 = len(solo_train)
        eps = math.sqrt(math.log(2.0 / DELTA) / (2.0 * n0))
        load_test = [v["load"] for v in test]
        mu = {a: max(1.0, ucb(ratio, a)) for a in ALPHAS}
        for tq in TAU_Q:
            tau = float(np.quantile(solo_train, tq))
            A = float(np.mean(np.asarray(load_test) <= tau))
            cell = {"round": r, "tau_q": tq, "tau_ms": tau, "A": A, "eps": eps}
            for a in ALPHAS:
                m = mu[a]
                slack = 1.0 - a
                f0 = float(np.mean(np.asarray(solo_train) <= m * tau))
                ex = A - f0 - slack - eps
                cell[f"margin_{int(a*1000)}"] = ex
                cell[f"mucb_{int(a*1000)}"] = m
                agg[a]["cells"] += 1
                agg[a]["viol"] += int(ex > 0)
                agg[a]["max_excess"] = max(agg[a]["max_excess"], ex)
                agg[a]["m_ucb_max"] = max(agg[a]["m_ucb_max"], m)
            rec["cells"].append(cell)
    out["stacks"].append(rec)

out["summary"] = {"rounds": rounds_total, "stacks": len(out["stacks"]),
                  "by_rule": {str(k): v for k, v in agg.items()}}
with open("out/e14_ucb.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)

print("stacks", out["summary"]["stacks"], "rounds", rounds_total)
for k, v in out["summary"]["by_rule"].items():
    print(f"alpha={k}  cells={v['cells']:>4}  viol={v['viol']:>3}  "
          f"max_excess={v['max_excess']:+.4f}  max_m_ucb={v['m_ucb_max']:.2f}")
