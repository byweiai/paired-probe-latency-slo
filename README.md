# Data and code for *Before You Scale*

This package backs the manuscript **Before You Scale: Deciding First-Token
Latency-SLO Attainability for LLM Serving with a Paired Probe** (Yue Lv,
China Mobile Communications Group Yunnan Co., Ltd.).

It contains the per-request probe records the paper's numbers are computed from,
the aggregated archives those computations produce, and the scripts that turn
one into the other. Every printed table, band, macro and ladder rung in the
manuscript can be regenerated from this package.

## What is here

```
out/            measurements and aggregates
  e16/          paired probe grid: eight stacks, five concurrency levels, two rounds
  e17/          cache-control arms and the no-load placebo
  e21/          the repeat with fresh prompt wording
  e23/          certification probes, one prompt per request
  e7_final/     scoreboard arms (solo and load) per stack
  e14/          dual-channel records across the platform-model combinations
  *.json        the aggregates the printed numbers are rebuilt from
  rounds.csv    per-round summary; vendor and model columns removed
archive_manifest.json   sha256 of every aggregate, as produced
analysis/       the scripts, unmodified from the working tree
requirements.txt
LICENSE-CODE (MIT)  LICENSE-DATA (CC BY 4.0)
```

Each record is one request: `id`, prompt class, HTTP status, time to first
token, end-to-end time, chunk count, arm (`solo` or `load`), concurrency,
round, and the stack label used in the paper. Request bodies are not included.

## Re-running the pipeline

```bash
python -m pip install -r requirements.txt
cp analysis/*.py .          # the scripts expect out/ as a sibling
python rebuild.py           # scoreboard inputs, per-stack probe records
python rebuild2.py          # second grid: cache-free, disjoint prompts
python e18_cluster.py && python e18_bh.py
python m_upper.py           # the acceleration bound m_up per stack
python e19_cachefree.py     # refutation test on the cache-free grid
python e14_ucb.py           # upper confidence bounds
python e13_alpha.py         # held-out envelope test
python e5_governor.py       # governance windows
python ladder.py            # the four-stage exclusion ladder
python e23_certify.py       # the certified instance
python breakeven.py         # break-even acceleration per stack
```

Run them in that order. `ladder.py` and `e23_certify.py` read
`m_upper.json`, so they come after it. The scripts write back into `out/`;
compare your results with `archive_manifest.json` to see whether you reproduced
the published aggregates.

## What is deliberately absent

- **Provider and model identities.** Stack labels are the paper's own; no model
  name, vendor name or institutional name appears in this package.
- **Request and response text.** Only timings, statuses and sizes are shipped.
- **The public traces.** The search-query and chat-session traces the analysis
  also reads are third-party data under their own licences; they are cited in
  the manuscript and are not redistributed here.
- **Working notes, superseded grids and drafts.** Only what the printed numbers
  depend on is included.

## Known limits on reproduction

- Six of the twelve stacks answered when the certification probes ran; four
  returned HTTP 402 (quota exhausted) and two returned HTTP 429 (rate limited),
  so the certified instance covers two stacks. This is stated in the
  manuscript, and it is why the certification targets are the tight ones.
- Measurements are wall-clock times from a single vantage point on a shared
  enterprise network, collected in November 2026. Absolute latencies are not
  portable; ratios between arms of the same stack are what the method uses.
- The probe uses four prompt classes with fixed output-length caps. A different
  prompt mix moves the solo distribution, and with it the verdicts.

## Licence

Code: MIT (`LICENSE-CODE`). Measurements and aggregates: CC BY 4.0
(`LICENSE-DATA`). If you use either, please cite the manuscript.
