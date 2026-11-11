"""E5 - what does the criterion buy an operator?"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
from sim.model import ServiceModel, decode_rate_tok_s
from sim.replay import replay
SLOS = [(3000.0, 0.95), (30000.0, 0.90)]
N_WINDOWS = 12
WINDOW_S = 600.0

def hourly_counts(name):
    df = pd.read_csv(ROOT / "data" / name, usecols=["TIMESTAMP"])
    t = pd.to_datetime(df["TIMESTAMP"], utc=True, format="ISO8601")
    sec = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    h = np.floor(sec / 3600.0).astype(np.int64)
    v = np.bincount(h - h.min())
    return v[v > 0].astype(float)

def main():
    model = ServiceModel.from_csv()
    dr = decode_rate_tok_s()
    e_s_s = float(model.ttft_ms.mean()) / 1000.0 + 120.0 / dr
    data = {"azure_code": hourly_counts("azure_code_1week.csv"),
            "azure_conv": hourly_counts("azure_conv_1week.csv")}
    rows = []
    for tau, a_t in SLOS:
        q_a = float(np.quantile(np.sort(model.ttft_ms), max(a_t - model.dkw(), 1 / len(model))))
        reachable = tau >= q_a
        print(f"SLO tau={tau:.0f} A={a_t}: Q_A(F0)={q_a:.0f} -> {'reachable' if reachable else 'UNREACHABLE'}", flush=True)
        for tname, cnt in data.items():
            idx = np.linspace(0, len(cnt) - 1, min(N_WINDOWS, len(cnt))).astype(int)
            for w, i in enumerate(idx):
                n_req = max(int(min(cnt[i] / 6.0, 8000)), 400)
                lam = n_req / WINDOW_S
                rng = np.random.default_rng(1000 + w)
                arr = np.sort(rng.uniform(0, WINDOW_S, n_req))
                ctx = np.expm1(rng.normal(6.9, 0.9, n_req)).clip(20, 12000)
                gen = np.expm1(rng.normal(3.4, 1.1, n_req)).clip(1, 3000)
                prev = cnt[idx[w - 1]] if w > 0 else cnt[i]
                sizes = {
                    "fixed_mean": max(int(math.ceil(np.mean(cnt) / 3600.0 * e_s_s / 0.8)), 1),
                    "peak": max(int(math.ceil(np.max(cnt) / 3600.0 * e_s_s / 0.8)), 1),
                    "hpa": max(int(math.ceil(prev / 3600.0 * e_s_s / 0.8)), 1),
                    "dca": (max(int(math.ceil(lam * e_s_s / 0.8)), 1) if reachable else 0),
                    "dca_prev": (max(int(math.ceil(prev / 3600.0 * e_s_s / 0.8)), 1) if reachable else 0),
                    "dca_told": (max(int(math.ceil(prev / 3600.0 * e_s_s / 0.8)), 1) if reachable else 0),
                }
                for pol, nw in sizes.items():
                    if nw <= 0:
                        rows.append({"slo": f"{int(tau)}/{a_t}", "trace": tname, "window": w,
                                     "policy": pol, "workers": 0, "s_eff": 0.0, "met": False, "spent": 0.0})
                        continue
                    res = replay(arr, ctx, gen, model, n_workers=nw, q_rpm=1e9, q_tpm=1e12,
                                 rng=np.random.default_rng(2000 + w), decode_rate=dr, ttft_mode="global")
                    s_eff = res.attain(tau)
                    rows.append({"slo": f"{int(tau)}/{a_t}", "trace": tname, "window": w, "policy": pol,
                                 "workers": nw, "s_eff": round(s_eff, 4), "met": bool(s_eff >= a_t),
                                 "spent": round(nw * WINDOW_S / 3600.0, 4)})
            print(f"  {tname}: {len(idx)} windows, peak={cnt.max():.0f}/h", flush=True)
    df = pd.DataFrame(rows)
    summ = {}
    for slo, gs in df.groupby("slo"):
        summ[slo] = {}
        for pol, g in gs.groupby("policy"):
            spent = float(g["spent"].sum()); un = g[~g["met"]]
            summ[slo][pol] = {"windows": int(len(g)), "met": int(g["met"].sum()),
                              "attainment": round(float(g["met"].mean()), 4),
                              "worker_hours": round(spent, 2),
                              "wasted_worker_hours": round(float(un["spent"].sum()), 2),
                              "waste_share": (round(float(un["spent"].sum()) / spent, 4) if spent > 0 else 0.0)}
    pair = {}
    for slo, gs in df.groupby("slo"):
        h = gs[gs["policy"] == "hpa"].set_index(["trace", "window"])
        for other in ("dca_prev", "dca_told", "dca"):
            o = gs[gs["policy"] == other].set_index(["trace", "window"])
            j = h.join(o, lsuffix="_h", rsuffix="_o", how="inner")
            pair[slo + "|" + other] = {
                "windows": int(len(j)),
                "same_workers": int((j["workers_h"] == j["workers_o"]).sum()),
                "same_spent": int((j["spent_h"] == j["spent_o"]).sum()),
                "same_met": int((j["met_h"] == j["met_o"]).sum()),
            }
    out = {"slos": [f"{int(t)}/{a}" for t, a in SLOS], "summary": summ, "pairing": pair, "rows": rows}
    (OUT / "e5_governor.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
