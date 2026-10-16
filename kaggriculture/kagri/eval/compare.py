"""Compare seat-takeover results: where does a candidate lose money to the top player, and is B better than A?

    python -m kagri.eval.compare results/takeover_tape.jsonl results/takeover_v1.jsonl [results/takeover_v2.jsonl]

The first file is the reference arm (usually `--agent tape`, i.e. the top player's own recorded play on the same
worlds). For every other arm it prints:
  * the money gap to the reference by day (where the gap opens: Syed Asad Ali's "plot money by day");
  * revenue by product against the reference (one product often explains most of the gap: Arda Barís Özten);
  * a paired read against the previous arm: strict flips with a sign test, margin change, split into our money and
    the pinned opponent's.
"""
import json
import sys

import numpy as np

from kagri.eval.stats import paired

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if "err" not in r}


def main():
    paths = sys.argv[1:]
    arms = [load(p) for p in paths]
    ref = arms[0]
    for k, (p, arm) in enumerate(zip(paths, arms)):
        ids = sorted(set(arm) & set(ref))
        if not ids:
            continue
        print("\n== %s  (%d worlds)" % (p, len(ids)))
        gap = np.array([np.array(arm[i]["money_by_day"][:30]) - np.array(ref[i]["money_by_day"][:30])
                        for i in ids if len(arm[i].get("money_by_day", [])) >= 30 and len(ref[i].get("money_by_day", [])) >= 30])
        if len(gap):
            print("  money vs reference by day (mean):  " + "  ".join("d%d %+.0f" % (d, gap[:, d].mean()) for d in
                                                                     (2, 5, 8, 10, 12, 15, 18, 21, 24, 27, 29)))
        if k > 0:
            rv = {pr: np.mean([arm[i].get("revenue", {}).get(pr, 0.0) - ref[i].get("revenue", {}).get(pr, 0.0) for i in ids])
                  for pr in PRODUCTS}
            print("  revenue vs reference by product:  " + "  ".join("%s %+.0f" % (pr, v) for pr, v in
                                                                     sorted(rv.items(), key=lambda kv: kv[1])))
            prev = arms[k - 1]
            pr = paired({i: (prev[i]["me"], prev[i]["op"]) for i in ids if i in prev},
                        {i: (arm[i]["me"], arm[i]["op"]) for i in ids if i in prev})
            print("  vs %s: wins %d -> %d  flips +%d/-%d (p %.3f)  d_margin %+.0f +- %.0f  d_me %+.0f  d_opp %+.0f"
                  % (paths[k - 1], pr["wins_base"], pr["wins_cand"], pr["flips_up"], pr["flips_down"], pr["p"],
                     pr["d_margin"], pr["d_margin_se"], pr["d_me"], pr["d_op"]))


if __name__ == "__main__":
    main()
