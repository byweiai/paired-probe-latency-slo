# -*- coding: utf-8 -*-
"""Metric-blinding split: raw vs gated, and gated vs gated."""
import io
import itertools
import json

d = json.load(io.open("out/e3_blind.json", encoding="utf-8"))
best_raw, best_gated = (0, None), (0, None)
for c in d["cells"]:
    mc = c["met_counts"]
    n = c["n_rounds"]
    for a, b in itertools.permutations(sorted(mc), 2):
        rate = abs(mc[a] - mc[b]) / n
        if "raw" in (a, b):
            if rate > best_raw[0]:
                best_raw = (rate, (c["tau"], c["A"], a, b, mc[a], mc[b], n))
        else:
            if rate > best_gated[0]:
                best_gated = (rate, (c["tau"], c["A"], a, b, mc[a], mc[b], n))
print("raw-vs-gated max %.4f" % best_raw[0], best_raw[1])
print("gated-vs-gated max %.4f" % best_gated[0], best_gated[1])
out = {"raw_max": round(best_raw[0], 4), "raw_at": best_raw[1],
       "gated_max": round(best_gated[0], 4), "gated_at": best_gated[1]}
io.open("out/e3_split.json", "w", encoding="utf-8").write(json.dumps(out, indent=1))
