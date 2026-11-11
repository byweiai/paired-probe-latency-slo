"""The exclusion ladder, on the same sample the manuscript's break-even line uses.

One number per stage, all on the same cells, ordered by what each stage assumes:

  1 nominal        point estimate of m and of F0, no allowance
  2 + band         F0 carries its radius; m stays at the point estimate
  3 + declared     F0 is evaluated at the declared upper limit of m
  4 + multiplicity stage 3 with a Bonferroni split over the stacks tested

Stage 4 is the number of exclusions the grid can certify.  If it is zero at
operational objectives, that is the paper's finding about *this grid's size*, not
about the endpoints: e23_certify.py runs the same criterion on a solo sample sized
for certification and reports what then survives.

The sample, the acceleration estimate and the break-even rule are held identical to
breakeven.py (out/e7_final + out/e14 solo, m95 from out/rebuild.json, tau = 3 s,
A = 0.95) so that stage 1 reproduces the printed line instead of quietly replacing
it with a second generation of the same quantity.
"""
import glob
import io
import json
import math
import os

import numpy as np

A, ALPHA = 0.95, 0.95
TARGET = A - (1 - ALPHA)
DELTA = 0.05
TAUS = [500, 1000, 2000, 3000, 5000, 10000]


def solo():
    d = {}
    for path in glob.glob("out/e7_final/*.jsonl") + glob.glob("out/e14/*.jsonl"):
        tag = os.path.basename(path).replace(".jsonl", "")
        for line in io.open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("arm") not in ("solo", "c1", None):
                continue
            if r.get("status") != 200 or not r.get("ttft_ms"):
                continue
            d.setdefault(tag, []).append(float(r["ttft_ms"]))
    return {k: np.array(v) for k, v in d.items()}


def main():
    x = solo()
    reb = json.load(io.open("out/rebuild.json", encoding="utf-8"))["probe"]["by_stack"]
    mu = {}
    for c in json.load(io.open("out/m_upper.json", encoding="utf-8"))["cells"]:
        mu[c["stack"]] = max(mu.get(c["stack"], 0.0), c["m_up"])

    cells = []
    for tag, v in reb.items():
        xs = x.get(tag)
        if xs is None or len(xs) < 30:
            continue
        n = len(xs)
        eps_req = math.sqrt(0.5 / n * math.log(2.0 / DELTA))
        m_point = v["m95"]
        m_up = mu.get(tag)
        for tau in TAUS:
            f_p = float((xs <= m_point * tau).mean())
            f_u = float((xs <= m_up * tau).mean()) if m_up else float("nan")
            cells.append(dict(stack=tag, tau=tau, n=n, m95=m_point, m_up=m_up,
                              F0_point=round(f_p, 4),
                              F0_declared=(round(f_u, 4) if m_up else None),
                              eps=round(eps_req, 4),
                              stage1=bool(f_p < TARGET),
                              stage2=bool(f_p + eps_req < TARGET),
                              stage3=bool(m_up and f_u + eps_req < TARGET)))
    K = len({c["stack"] for c in cells})
    eps_k = {c["stack"]: math.sqrt(0.5 / c["n"] * math.log(2.0 * K / DELTA))
             for c in cells}
    for c in cells:
        c["stage4"] = bool(c["m_up"] and c["F0_declared"] is not None
                           and c["F0_declared"] + eps_k[c["stack"]] < TARGET)

    counts = {}
    for tau in TAUS:
        s = [c for c in cells if c["tau"] == tau]
        counts[str(tau)] = {k: int(sum(c[k] for c in s)) for k in
                            ("stage1", "stage2", "stage3", "stage4")}
        counts[str(tau)]["stacks"] = len(s)

    json.dump({"meta": {"A": A, "alpha": ALPHA, "target": TARGET, "delta": DELTA,
                        "stacks": K, "taus": TAUS,
                        "source": "out/e7_final + out/e14 (solo), m95 from rebuild.json,"
                                  " m_up from m_upper.json"},
               "counts": counts, "cells": cells},
              io.open("out/ladder.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("%6s %6s | %8s %6s %9s %12s" % ("tau", "stacks", "nominal", "+band",
                                          "+declared", "+multiplicity"))
    for tau in TAUS:
        c = counts[str(tau)]
        print("%6d %6d | %8d %6d %9d %12d" % (tau, c["stacks"], c["stage1"],
                                              c["stage2"], c["stage3"], c["stage4"]))
    txt = io.open("paper/numbers.tex", encoding="utf-8").read()
    import re
    dead = re.search(r"\\def\\numStacksDead\{(\d+)", txt)
    if dead:
        print("manuscript numStacksDead = %s ; stage1 at 3 s = %d"
              % (dead.group(1), counts["3000"]["stage1"]))


if __name__ == "__main__":
    main()