# -*- coding: utf-8 -*-
"""Recompute the SI coverage table (Table S3 of the supplement) from the archived probe records.

The caption states the rule: a combination is a (platform, model) pair, and the denial and timeout
rates are those observed at the highest concurrency level.  This script applies exactly that rule to
out/dual_channel_curves.json and compares the result with the rows printed in paper/supplementary.tex.
It exits 1 on any disagreement, so it can serve as a gate.

  py -3.13 check_si_cov.py           # verify only
  py -3.13 check_si_cov.py --table   # print the LaTeX body it would emit
"""
import collections
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, "out", "dual_channel_curves.json")
SI = os.path.join(ROOT, "paper", "supplementary.tex")

# the printed table names the platforms A..F; this is the only hand-kept mapping, and the script
# checks that the number of combinations per letter matches what the archive holds.
ORDER = ["<platform>", "<platform>", "<platform>", "<platform><provider><platform>",
         "<provider> V4 Flash <platform>", "<platform>"]


def rows_from_archive():
    recs = json.load(io.open(ARCHIVE, encoding="utf-8"))
    best = {}
    for r in recs:
        key = (r["vendor"], r["model"])
        cur = best.get(key)
        if cur is None or r["c"] > cur["c"]:
            best[key] = r
    by_vendor = collections.defaultdict(list)
    for (vendor, _), r in best.items():
        by_vendor[vendor].append(r)
    out = []
    for vendor in ORDER:
        tops = by_vendor.get(vendor, [])
        if not tops:
            continue
        th = [t["theta"] for t in tops]
        de = [t["delta_3000"] for t in tops]
        out.append((len(tops), min(th), max(th), min(de), max(de)))
    all_th = [r["theta"] for r in best.values()]
    all_de = [r["delta_3000"] for r in best.values()]
    return out, (len(best), min(all_th), max(all_th), min(all_de), max(all_de))


def printed_rows():
    src = io.open(SI, encoding="utf-8").read()
    m = re.search(r"\\caption\{Commercial probe coverage.*?\\end\{tabular\}", src, re.S)
    if not m:
        raise SystemExit("FAIL the coverage table is not where it used to be")
    block = m.group(0)
    rows = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.endswith("\\\\"):
            continue
        line = line[:-2].strip()
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        if not (cells[0].startswith("Platform") or cells[0] == "All"):
            continue
        rows.append((cells[0], cells))
    return rows


def fmt(lo, hi):
    return "%.2f--%.2f" % (lo, hi)


def main():
    computed, total = rows_from_archive()
    printed = printed_rows()
    letters = [chr(ord("A") + i) for i in range(len(computed))]
    problems = []

    if len(printed) != len(computed) + 1:
        problems.append("printed %d rows, archive gives %d platform rows + 1 total"
                        % (len(printed), len(computed)))

    for i, (n, tlo, thi, dlo, dhi) in enumerate(computed):
        want = ["Platform " + letters[i], str(n), fmt(tlo, thi), fmt(dlo, dhi)]
        got = printed[i][1] if i < len(printed) else None
        if got is None:
            problems.append("missing row for Platform " + letters[i])
        elif [g.replace(" ", "") for g in got] != [w.replace(" ", "") for w in want]:
            problems.append("Platform %s printed %s, archive gives %s" % (letters[i], got, want))

    n, tlo, thi, dlo, dhi = total
    want = ["All", str(n), fmt(tlo, thi), fmt(dlo, dhi)]
    got = printed[-1][1]
    if [g.replace(" ", "") for g in got] != [w.replace(" ", "") for w in want]:
        problems.append("All printed %s, archive gives %s" % (got, want))

    if "--table" in sys.argv:
        for i, (n, tlo, thi, dlo, dhi) in enumerate(computed):
            print("Platform %s & %d & %s & %s \\\\" % (letters[i], n, fmt(tlo, thi), fmt(dlo, dhi)))
        print("All & %d & %s & %s \\\\" % (total[0], fmt(total[1], total[2]), fmt(total[3], total[4])))

    if problems:
        for p in problems:
            print("FAIL " + p)
        return 1
    print("OK coverage table: %d platform rows, %d combinations, denial %.2f--%.2f, timeout %.2f--%.2f"
          % (len(computed), total[0], total[1], total[2], total[3], total[4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
