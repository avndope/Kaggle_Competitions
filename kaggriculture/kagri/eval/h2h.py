"""Closed-loop paired head-to-head on the official engine: candidate vs a panel of opponents.

    python -m kagri.eval.h2h --cand agents/v1 --panel v16=/path/to/b_v16a hl0927=/path/to/hl0927.py \
        --seeds 24 --seed0 9000 --procs 12 --out results/h2h_v1.jsonl
    python -m kagri.eval.h2h --cand agents/v2 --base agents/v1 --panel ...   # adds a paired candidate-vs-base read

Every seed is played twice with the seats swapped. Learnings applied:
  * count seeds, not games: near-identical agents often produce mirrored seat-swapped games, so a seed is the
    independent unit (Arda); the report shows how many seeds were exact mirrors;
  * report every opponent separately and the worst one: a Bradley-Terry fit punishes one-sided losses to
    similar-rated teams (Yuan800), and a change can collapse against an unseen style (KevlarZanderChi);
  * read our money and the opponent's money separately (24th place, Chris Deotte: many gains are denial).
"""
import argparse
import os as _os

# One BLAS thread per worker: small matmuls run ~5x faster than with threaded BLAS (Arda Barís Özten).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json
import os
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from kagri.engine.runner import Game
from kagri.eval.agents import call, load_agent
from kagri.eval.stats import mean_se, paired, score, sign_test


def play(job):
    name_a, path_a, name_b, path_b, seed, seat_a = job
    fa, fb = load_agent(path_a, fresh=True), load_agent(path_b, fresh=True)
    g = Game(seed)
    fns = [None, None]
    fns[seat_a], fns[1 - seat_a] = fa, fb
    err = [0, 0]
    worst = [0.0, 0.0]
    while not g.done:
        acts = []
        for p in (0, 1):
            t0 = time.perf_counter()
            a, e = call(fns[p], g.observe(p), g.agent_config)
            worst[p] = max(worst[p], time.perf_counter() - t0)
            err[p] += e
            acts.append(a)
        g.step(acts)
    r = g.rewards()
    return dict(a=name_a, b=name_b, seed=seed, seat_a=seat_a, me=r[seat_a], op=r[1 - seat_a],
                err_a=err[seat_a], err_b=err[1 - seat_a], max_call_a=round(worst[seat_a], 3))


def report(rows, a):
    by = defaultdict(list)
    for r in rows:
        by[(r["a"], r["b"])].append(r)
    worst = None
    for (na, nb), rs in sorted(by.items()):
        seeds = defaultdict(list)
        for r in rs:
            seeds[r["seed"]].append(r)
        mirror = sum(len(v) == 2 and v[0]["me"] == v[1]["me"] and v[0]["op"] == v[1]["op"] for v in seeds.values())
        per_seed = [np.mean([score(r["me"], r["op"]) for r in v]) for v in seeds.values()]
        w = float(np.mean(per_seed))
        m, se = mean_se([r["me"] - r["op"] for r in rs])
        up = sum(s > 0.5 for s in per_seed)
        dn = sum(s < 0.5 for s in per_seed)
        print("  %-10s vs %-14s seeds %3d (mirrored %3d)  win %5.1f%%  seeds won/lost %d/%d (p %.3f)  margin %+7.0f +- %5.0f"
              "  our bank %6.0f  their bank %6.0f  errors %d/%d  slowest %.2fs"
              % (na, nb, len(seeds), mirror, 100 * w, up, dn, sign_test(up, dn), m, se, np.mean([r["me"] for r in rs]),
                 np.mean([r["op"] for r in rs]), sum(r["err_a"] for r in rs), sum(r["err_b"] for r in rs),
                 max(r["max_call_a"] for r in rs)))
        if na == "cand" and (worst is None or w < worst[1]):
            worst = (nb, w)
    if worst:
        print("  worst opponent for cand: %s (%.1f%%)" % (worst[0], 100 * worst[1]))
    if a.base:
        for nb in sorted({r["b"] for r in rows}):
            base = {(r["seed"], r["seat_a"]): (r["me"], r["op"]) for r in rows if r["a"] == "base" and r["b"] == nb}
            cand = {(r["seed"], r["seat_a"]): (r["me"], r["op"]) for r in rows if r["a"] == "cand" and r["b"] == nb}
            if base and cand:
                p = paired(base, cand)
                print("  cand vs base on %-14s n %d  flips +%d/-%d (p %.3f)  d_margin %+.0f +- %.0f  d_me %+.0f  d_opp %+.0f"
                      % (nb, p["n"], p["flips_up"], p["flips_down"], p["p"], p["d_margin"], p["d_margin_se"], p["d_me"], p["d_op"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default=None, help="optional second candidate for a paired comparison")
    ap.add_argument("--panel", nargs="+", required=True, help="NAME=PATH opponents (main.py, .py or a directory)")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [("cand", a.cand)] + ([("base", a.base)] if a.base else [])
    panel = [x.split("=", 1) for x in a.panel]
    jobs = [(na, pa, nb, pb, a.seed0 + k, s) for na, pa in arms for nb, pb in panel for k in range(a.seeds) for s in (0, 1)]
    print("%d games" % len(jobs), flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=2) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    report(rows, a)
    print("%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
