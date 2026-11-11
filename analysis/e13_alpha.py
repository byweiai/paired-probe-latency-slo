"""E13 - the quantile-relaxed envelope with an analytic tail allowance.

Theorem (quantile form). For alpha in (0,1) let m_a be the alpha-quantile of the
acceleration ratio R = T0/T. Then
    A(tau) = Pr(T <= tau) <= F0(m_a * tau) + (1 - alpha),
because {T <= tau} is contained in {T0 <= m_a tau} union {R > m_a}.

Unlike the essential-supremum form, every measured stack satisfies the premise by
construction, and the slack (1 - alpha) is fixed before the test, not fitted to it.
This script re-runs the leave-one-round-out protocol of e12 with that bound, for a
pre-declared grid of alpha, and reports how many out-of-sample cells violate it.
"""
import glob
import json
import math
import os

import numpy as np

D = "out/e7_final"
TAU_Q = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
ALPHAS = [0.90, 0.95, 0.99, 0.995]
DELTA = 0.05


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


def q(emp, p):
    if not emp:
        return float("inf")
    return float(np.quantile(np.asarray(emp), p))


out = {"delta": DELTA, "tau_quantiles": TAU_Q, "alphas": ALPHAS, "stacks": [], "summary": {}}
agg = {a: {"cells": 0, "viol": 0, "max_excess": -9.0} for a in ALPHAS}
agg["m1"] = {"cells": 0, "viol": 0, "max_excess": -9.0}
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
        ratio = np.asarray([v["solo"] / v["load"] for v in train if v["load"] > 0])
        solo_train = [v["solo"] for v in train]
        n0 = len(solo_train)
        eps = math.sqrt(math.log(2.0 / DELTA) / (2.0 * n0))
        load_test = [v["load"] for v in test]
        m_a = {a: float(np.quantile(ratio, a)) for a in ALPHAS}
        for tq in TAU_Q:
            tau = q(solo_train, tq)
            A = float(np.mean(np.asarray(load_test) <= tau))
            cell = {"round": r, "tau_q": tq, "tau_ms": tau, "A": A, "eps": eps}
            # plain monotone envelope (m = 1)
            f0_m1 = float(np.mean(np.asarray(solo_train) <= tau))
            ex1 = A - f0_m1 - eps
            cell["margin_m1"] = ex1
            cell["viol_m1"] = bool(ex1 > 0)
            agg["m1"]["cells"] += 1
            agg["m1"]["viol"] += int(ex1 > 0)
            agg["m1"]["max_excess"] = max(agg["m1"]["max_excess"], ex1)
            for a in ALPHAS:
                ma = max(1.0, m_a[a])  # m is an acceleration bound: m >= 1 by definition
                slack = 1.0 - a
                f0 = float(np.mean(np.asarray(solo_train) <= ma * tau))
                ex = A - f0 - slack - eps
                cell[f"margin_{int(a*1000)}"] = ex
                cell[f"viol_{int(a*1000)}"] = bool(ex > 0)
                cell[f"m_{int(a*1000)}"] = ma
                agg[a]["cells"] += 1
                agg[a]["viol"] += int(ex > 0)
                agg[a]["max_excess"] = max(agg[a]["max_excess"], ex)
            rec["cells"].append(cell)
    out["stacks"].append(rec)

out["summary"] = {
    "rounds": rounds_total,
    "stacks": len(out["stacks"]),
    "by_rule": {str(k): v for k, v in agg.items()},
}
with open("out/e13_alpha.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)

print("stacks", out["summary"]["stacks"], "rounds", rounds_total)
for k, v in out["summary"]["by_rule"].items():
    print(f"{k:>6}  cells={v['cells']:>4}  viol={v['viol']:>3}  max_excess={v['max_excess']:+.4f}")
