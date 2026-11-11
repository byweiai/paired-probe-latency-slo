# -*- coding: utf-8 -*-
"""Per-stack break-even check: is F0(m95*tau) < A at tau = 3 s, A = 0.95?"""
import glob, io, json, os
import numpy as np

TAU, A = 3000.0, 0.95
solo = {}
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
        solo.setdefault(tag, []).append(float(r["ttft_ms"]))

reb = json.load(io.open("out/rebuild.json", encoding="utf-8"))["probe"]["by_stack"]
rows = []
for tag, v in reb.items():
    x = np.array(solo.get(tag, []))
    if len(x) < 30:
        rows.append((tag, v["m95"], len(x), float("nan"), float("nan"), None))
        continue
    q = float(np.quantile(x, A))
    f0 = float((x <= v["m95"] * TAU).mean())
    rows.append((tag, v["m95"], len(x), q, f0, f0 < A))

rows.sort(key=lambda r: r[1])
print("%-18s %6s %6s %9s %7s  %s" % ("stack", "m95", "n0", "Q0(95%)", "F0(m95*3s)", "dead"))
dead = 0
for tag, m95, n0, q, f0, verdict in rows:
    mark = "-" if verdict is None else ("DEAD" if verdict else "alive")
    if verdict:
        dead += 1
    print("%-18s %6.2f %6d %9.1f %7.3f  %s" % (tag, m95, n0, q, f0, mark))
print("dead stacks:", dead, "of", len(rows))

json.dump({"tau": TAU, "A": A, "dead": dead, "rows": [
    {"tag": t, "m95": m, "n0": n, "q0": (None if q != q else round(q, 1)),
     "f0": (None if f0 != f0 else round(f0, 4)), "dead": bool(v)}
    for t, m, n, q, f0, v in rows]},
    io.open("out/breakeven.json", "w", encoding="utf-8"), indent=1)
print("wrote out/breakeven.json")
