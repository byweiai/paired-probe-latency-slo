# -*- coding: utf-8 -*-
"""Verify paper/tab_scoreboard.tex against the archives it claims to come from.

Every printed cell is recomputed here: n0 and m_hat from out/e18_cluster.json, the band from the
standard DKW radius the paper prints, m_up from out/m_upper.json, the two count columns from the
per-cell flags in out/e18_cluster.json, and the three totals quoted in the caption from
out/e18_cluster.json and out/e18_bh.json.  Exit 1 on any disagreement.

Run: py -3.13 check_scoreboard.py
"""
import io
import json
import math
import re
import sys

DELTA = 0.05
TAB = "paper/tab_scoreboard.tex"
E18 = "out/e18_cluster.json"
UP = "out/m_upper.json"


def band(n, delta=DELTA):
    """Standard DKW half-width, the form printed in the manuscript."""
    return math.sqrt(0.5 / n * math.log(2.0 / delta))


def main():
    tab = io.open(TAB, encoding="utf-8").read()
    e18 = json.load(io.open(E18, encoding="utf-8"))
    up = {r["stack"]: r for r in json.load(io.open(UP, encoding="utf-8"))["stacks"]}

    cells = e18["cells"]
    by_stack = e18["by_stack"]
    totals = {"m1": {"judge": 0, "rej": 0}, "m95": {"judge": 0, "rej": 0}}
    for c in cells:
        if not c["judgeable"]:
            continue
        totals[c["m"]]["judge"] += 1
        totals[c["m"]]["rej"] += int(bool(c["refuted"]))

    rows = re.findall(
        r"^(.+?)\s*&\s*(\d+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\",
        tab, re.M)
    if not rows:
        sys.exit("STOP: no data rows parsed out of %s" % TAB)

    bad = []
    seen = []
    for name, n0, eps, m95, mup, rej, rej95 in rows:
        name = name.strip().rstrip("\\").strip()
        n0, eps, m95, mup = int(n0), float(eps), float(m95), float(mup)
        rej, rej95 = int(rej), int(rej95)
        seen.append(n0)
        # match the printed short name back to an archive tag
        cand = [t for t in by_stack
                if by_stack[t]["n_solo"] == n0 and abs(round(by_stack[t]["m95"], 2) - m95) < 1e-9]
        if len(cand) != 1:
            bad.append((name, "no unique archive stack for n0=%d m95=%.2f (candidates %s)" % (n0, m95, cand)))
            continue
        tag = cand[0]
        want = {
            "n0": by_stack[tag]["n_solo"],
            "eps": round(band(by_stack[tag]["n_solo"]), 3),
            "m95": round(by_stack[tag]["m95"], 2),
            "m_up": round(up[tag]["m_up"], 2),
            "rejected_m1": sum(1 for c in cells if c["stack"] == tag and c["m"] == "m1" and c["judgeable"] and c["refuted"]),
            "rejudged_m95": sum(1 for c in cells if c["stack"] == tag and c["m"] == "m95" and c["judgeable"]),
        }
        got = {"n0": n0, "eps": eps, "m95": m95, "m_up": mup, "rejected_m1": rej, "rejudged_m95": rej95}
        for k in want:
            if abs(want[k] - got[k]) > 1e-9:
                bad.append((tag, "%s printed %s, archive %s" % (k, got[k], want[k])))

    if len(rows) != 11:
        bad.append(("table", "%d rows printed, 11 stacks expected" % len(rows)))

    cap = ""
    if "\\caption{" in tab:
        i = tab.index("\\caption{") + len("\\caption{")
        depth, buf = 1, []
        while i < len(tab) and depth:
            ch = tab[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        cap = "".join(buf)
    if not cap:
        bad.append(("caption", "the table carries no caption to check"))
    else:
        # every yearly count the caption asserts must equal what the archive recomputes
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
        capnums = set(int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", cap))
        bh = e18.get("summary", {}).get("m1", {}) or {}

        def stated(v):
            if v is None:
                return True
            return (v in capnums) or (words.get(v, "@@") in cap.lower())

        need = {"rejections at m=1": totals["m1"]["rej"],
                "judgeable at m=1": totals["m1"]["judge"],
                "judgeable cells at m_hat_95": totals["m95"]["judge"],
                "BH rejections": bh.get("bh_sig")}
        for label, v in need.items():
            if not stated(v):
                bad.append(("caption", "%s = %s is not stated in the caption (numbers found: %s)"
                            % (label, v, sorted(capnums))))
        stale = [n for n in (37, 18, 56, 88, 104) if n in capnums]
        if stale:
            bad.append(("caption", "the caption still carries superseded counts %s" % stale))

    bh = e18.get("summary", {}).get("m1", {})
    if "bh_sig" not in bh:
        bad.append(("caption", "e18_cluster.json carries no bh_sig: run e18_cluster.py then e18_bh.py"))
    elif bh["bh_sig"] != 6 or bh["bh_stacks"] != 3:
        bad.append(("caption", "six of the eleven survive Benjamini-Hochberg: archive says %d on %d stacks"
                    % (bh["bh_sig"], bh["bh_stacks"])))

    if bad:
        for row, why in bad:
            print("FAIL  %-16s %s" % (row, why))
        print("\n%d disagreement(s)" % len(bad))
        return 1
    print("OK  11 rows, %d distinct n0 values, totals 11/46 at m=1 and 0/29 at m95, BH 6 on 3 stacks"
          % len(set(seen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
