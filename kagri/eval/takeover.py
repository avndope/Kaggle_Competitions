"""Seat takeover on recorded top games: how much of a top player's money does our agent make in its own seat?

    python -m kagri.eval.takeover --agent agents/v1 --team "M & M & P & Q" --census data/bc/census.jsonl \
        --zips data/raw/*.zip --held-out --day 0 --n 60 --out results/takeover_v1.jsonl
    python -m kagri.eval.takeover --agent tape ...      # calibration: must reproduce every recording exactly

For each recorded game of --team: shops are pinned to the recorded ones; both seats follow the recording until the
start of day --day (0 = our agent plays the whole game), then our agent takes the team's seat while the opponent
keeps following its recorded actions. Before the fork the agent still observes every step, so its internal state
is the one it would have had.

Reported per game: our money, the pinned opponent's money, and the teacher's recorded money in that seat.
Summary: money share (ours / teacher's recorded), win rate against the pinned opponent next to the teacher's
recorded win rate, and the margin change.

Caveat (Arda Barış Özten, Zakaria Aala): the pinned opponent cannot react to us. Absolute numbers are biased once
our play moves the shared market; the paired comparison of two candidates is the meaningful read. Closed-loop
head-to-head (kagri/eval/h2h.py) is the gate.
"""
import argparse
import os as _os

# One BLAS thread per worker: small matmuls run ~5x faster than with threaded BLAS (Arda Barís Özten).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json
import os
import time
import zipfile
from multiprocessing import get_context

import numpy as np

from kagri.engine.runner import Game
from kagri.eval.agents import Tape, call, load_agent
from kagri.eval.stats import mean_se, score

_ZIPS = {}


def _replay(eid, zips):
    for zp in zips:
        z = _ZIPS.get(zp)
        if z is None:
            z = _ZIPS[zp] = zipfile.ZipFile(zp)
        try:
            return json.loads(z.read("%d.json" % eid))
        except KeyError:
            continue
    return None


def run_world(job):
    agent_path, eid, seat, day, zips = job
    rep = _replay(eid, zips)
    if rep is None:
        return dict(id=eid, err="replay not found")
    steps = rep["steps"]
    tapes = [[steps[t + 1][s].get("action") for t in range(719)] for s in (0, 1)]
    shops = steps[-1][0]["observation"]["town"]["unlocked_shops"]
    me, op = seat, 1 - seat
    fn = Tape(tapes[me]) if agent_path == "tape" else load_agent(agent_path, fresh=True)
    opp = Tape(tapes[op])
    g = Game(int(rep["info"]["seed"]), shops=shops)
    T0 = 24 * day
    errors, worst = 0, 0.0
    while not g.done:
        t = g.step_no
        o = g.observe(me)
        t1 = time.perf_counter()
        a, e = call(fn, o, g.agent_config)
        worst = max(worst, time.perf_counter() - t1)
        acts = [None, None]
        acts[me] = Tape(tapes[me])(o) if t < T0 else a
        errors += e if t >= T0 else 0
        acts[op] = opp(g.observe(op))
        g.step(acts)
    r = g.rewards()
    rec = [float(x) for x in rep["rewards"]]
    return dict(id=eid, seat=me, team=rep["info"]["TeamNames"][me], oteam=rep["info"]["TeamNames"][op],
                me=r[me], op=r[op], rec_me=rec[me], rec_op=rec[op], errors=errors, max_call=round(worst, 3))


def summarize(rows, title=""):
    rows = [r for r in rows if "err" not in r]
    if not rows:
        print(title, "no games")
        return {}
    share = [r["me"] / max(1.0, r["rec_me"]) for r in rows]
    win = [score(r["me"], r["op"]) for r in rows]
    rec_win = [score(r["rec_me"], r["rec_op"]) for r in rows]
    dmargin = [(r["me"] - r["op"]) - (r["rec_me"] - r["rec_op"]) for r in rows]
    s_m, s_se = mean_se(share)
    d_m, d_se = mean_se(dmargin)
    out = dict(n=len(rows), money_share=s_m, money_share_se=s_se, win=float(np.mean(win)), teacher_win=float(np.mean(rec_win)),
               d_margin_vs_teacher=d_m, d_margin_se=d_se, exact=sum(r["me"] == r["rec_me"] and r["op"] == r["rec_op"] for r in rows),
               agent_errors=sum(r["errors"] for r in rows), max_call=max(r["max_call"] for r in rows))
    print("%s n=%d  money share of teacher %.3f +- %.3f  win vs pinned opp %.1f%% (teacher recorded %.1f%%)  "
          "d_margin %+.0f +- %.0f  exact %d  agent errors %d  slowest call %.2fs"
          % (title, out["n"], s_m, s_se, 100 * out["win"], 100 * out["teacher_win"], d_m, d_se, out["exact"],
             out["agent_errors"], out["max_call"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="submission dir / main.py, or 'tape' for the calibration arm")
    ap.add_argument("--team", required=True, help="the top team whose seat we take")
    ap.add_argument("--census", required=True)
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--held-out", action="store_true", help="only validation episodes (id %% --val-mod == 0)")
    ap.add_argument("--val-mod", type=int, default=25)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    for line in open(a.census):
        r = json.loads(line)
        if a.team not in r["teams"] or r["teams"].count(a.team) != 1:
            continue
        if a.held_out and r["id"] % a.val_mod != 0:
            continue
        jobs.append((a.agent, r["id"], r["teams"].index(a.team), a.day, a.zips))
    jobs.sort(key=lambda j: (j[1] * 2654435761) % 2 ** 32)   # deterministic mix of opponents and days
    jobs = jobs[:a.n]
    print("%d worlds for %s (day %d fork)" % (len(jobs), a.team, a.day), flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=4) as pool:
        rows = pool.map(run_world, jobs, chunksize=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summarize(rows, os.path.basename(a.out))
    print("by opponent:")
    for ot in sorted({r.get("oteam") for r in rows if "err" not in r}, key=lambda t: -sum(r.get("oteam") == t for r in rows))[:8]:
        summarize([r for r in rows if r.get("oteam") == ot], "   vs %-24s" % ot[:24])
    print("%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
