"""Certify exclusions from the E23 solo sample.

The band is a Hoeffding radius over *prompts*, not over requests.  E23 issues one
request per prompt with a wording no other prompt shares, so the prompt is the
sampling unit and the radius needs no within-cluster model and no estimated design
effect: with n independent prompts each indicator lies in [0,1] and

    eps_F0 = sqrt( ln(2/delta_F0) / (2 n) ).

Error is allocated before the run, not after: delta_total = 0.05 is split in half
between the F0 band and the declared upper limit on m_alpha, and each half is
Bonferroni-split over the K stacks in the table.  The m_alpha limit is the
order-statistic bound already computed by m_upper.py (exact binomial choice of k,
not a normal approximation); the certificate evaluates F0 at the *upper* end of
that limit, which is the conservative direction.

Criterion (A = 0.95, alpha = 0.95):  F0_hat(m_up * tau) + eps_F0 < A - (1 - alpha).
Reported alongside: the nominal reading (no band), the request-level band, and a
cluster-level band over the 20 class x topic cells, which is the reading a referee
who refuses i.i.d. prompts would insist on.
"""
import glob
import io
import json
import math
import os
import sys

TAUS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000, 2000, 3000]
A = 0.95
ALPHA = 0.95
DELTA = 0.05
STACKS = ["ds-flash", "ds-v4pro"]


def load(tag):
    rows = []
    for f in sorted(glob.glob("out/e23/%s_*.jsonl" % tag)):
        for line in io.open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == 200 and r.get("ttft_ms"):
                rows.append(r)
    return rows


def ecdf(vals, t):
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v <= t) / len(vals)


def hoeff(n, delta):
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def main():
    mu = json.load(io.open("out/m_upper.json", encoding="utf-8"))
    # m_up is a property of the stack, not of the objective: R = T_solo / T_load
    # does not depend on tau, so the same declared limit serves every row.
    up = {}
    for c in mu["cells"]:
        k = c["stack"]
        up[k] = max(up.get(k, 0.0), c["m_up"])

    K = len(STACKS)
    d_f0 = DELTA / 2.0 / K
    out = {"meta": {"A": A, "alpha": ALPHA, "delta_total": DELTA, "stacks": K,
                    "delta_f0": d_f0, "source": "out/e23/*.jsonl",
                    "criterion": "F0_hat(m_up*tau) + eps_F0 < A-(1-alpha)"},
           "stacks": {}}

    for tag in STACKS:
        rows = load(tag)
        if not rows:
            continue
        vals = [r["ttft_ms"] for r in rows]
        n = len(vals)
        eps = hoeff(n, d_f0)
        cells = {}
        for r in rows:
            cells.setdefault((r.get("cls"), int(r["id"].split("-")[1]) % 5), []).append(r["ttft_ms"])
        C = len(cells)
        eps_cl = hoeff(C, d_f0)
        deltas = {}
        for tau in TAUS:
            m_up = up[tag]
            f = ecdf(vals, m_up * tau)
            deltas[str(tau)] = {
                "m_up": round(m_up, 4),
                "target_ms": round(m_up * tau, 1),
                "F0": round(f, 4),
                "margin_nominal": round(A - (1 - ALPHA) - f, 4),
                "margin_cert": round(A - (1 - ALPHA) - f - eps, 4),
                "cert_dead": bool(f + eps < A - (1 - ALPHA)),
            }
        half = n // 2
        out["stacks"][tag] = {
            "n_prompts": n, "cells": C,
            "eps_F0": round(eps, 4), "eps_cluster": round(eps_cl, 4),
            "F0_at_1000": round(ecdf(vals, 1000), 4),
            "F0_at_3000": round(ecdf(vals, 3000), 4),
            "split_half_F0_1000": [round(ecdf(vals[:half], 1000), 4),
                                   round(ecdf(vals[half:], 1000), 4)],
            "by_tau": deltas,
        }

    print("%-9s %5s %6s %6s | %-30s | %-30s" % ("stack", "n", "epsF0", "epsCl",
                                                "tightest certified objective",
                                                "tau=1000 ms"))
    for tag, s in out["stacks"].items():
        row = [tag, s["n_prompts"], s["eps_F0"], s["eps_cluster"]]
        line = "%-9s %5d %6.3f %6.3f" % tuple(row)
        dead = [int(t) for t in s["by_tau"] if s["by_tau"][t]["cert_dead"]]
        stable = [int(t) for t in s["by_tau"]
                  if s["by_tau"][t]["cert_dead"] and s["by_tau"][t]["margin_cert"] >= eps]
        s["certified_ceiling_ms"] = max(dead) if dead else 0
        s["stable_ceiling_ms"] = max(stable) if stable else 0
        if dead:
            d = s["by_tau"][str(max(dead))]
            line += " | <=%d ms (stable %d)" % (max(dead), s["stable_ceiling_ms"])
        else:
            line += " | none"
        d1 = s["by_tau"]["1000"]
        line += " | F0=%.3f cert=%+.3f %s" % (d1["F0"], d1["margin_cert"],
                                              "DEAD" if d1["cert_dead"] else "alive")
        print(line)


        print(line)

    io.open("out/e23_certify.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
