"""kagri: re-simulate a recorded game on the official engine and record what the engine actually executed.

Learnings behind this file (see learnings.md):
  * 4th place: train on executed state changes, not requested orders. Top players send oversized orders that the
    engine clips silently (one filled only 2% of its requested sells); a clone trained on requests learns empty orders.
  * Yuan800: invalid requests are masked, not taught. A unit action the engine turns into a no-op gets no label.
  * Every game is checked step by step against its recording; a game that does not reproduce exactly is not used.

resim(replay) -> dict of arrays, T = 720 (index t = the action taken from observation t; t = 719 has none):
  x_unit [T,2,MAXU] i1   -1 no action entry, 0 the engine did nothing, 1 the action changed the game
  x_sell [T,2,9]    i2   executed SELL units per product
  x_seed [T,2,5]    i2   executed BUY_SEED units per crop
  x_buyp [T,2,2]    i2   executed BUY_PRODUCT units (WHEAT, FERTILIZER)
  x_anim [T,2,3]    i2   executed BUY_ANIMAL units (GOOSE, COW, SHEEP)
  x_hire [T,2]      i1   executed hires
  x_land [T,2]      i1   executed land purchases
  x_req_sell [T,2]  i4   requested SELL units (to measure how much a teacher over-orders)
  parity            i4   first step whose state differs from the recording (-1 = reproduced exactly)
"""
import numpy as np

from common import PRODUCTS, CROPS, ANIMALS, MAXU
from kagri.engine.runner import Game

T = 720
PIDX = {k: i for i, k in enumerate(PRODUCTS)}
CIDX = {k: i for i, k in enumerate(CROPS)}
AIDX = {k: i for i, k in enumerate(ANIMALS)}


def _same(g, rec):
    """Engine state after a step == the recorded observation of the next step (shared state + both privates)."""
    o0 = rec[0]["observation"]
    st = g.state[0].observation
    if st.farms != o0["farms"] or st.market != o0["market"] or st.town != o0["town"]:
        return False
    return all(g.state[s].observation.private == rec[s]["observation"]["private"] for s in (0, 1))


def resim(replay, check_every=1):
    steps = replay["steps"]
    assert len(steps) == T, len(steps)
    g = Game(int(replay["info"]["seed"]), log=True)
    out = dict(x_unit=np.full((T, 2, MAXU), -1, np.int8), x_sell=np.zeros((T, 2, 9), np.int16),
               x_seed=np.zeros((T, 2, 5), np.int16), x_buyp=np.zeros((T, 2, 2), np.int16),
               x_anim=np.zeros((T, 2, 3), np.int16), x_hire=np.zeros((T, 2), np.int8),
               x_land=np.zeros((T, 2), np.int8), x_req_sell=np.zeros((T, 2), np.int32))
    parity = -1 if _same(g, steps[0]) else 0
    for t in range(T - 1):
        acts = []
        for s in (0, 1):
            a = steps[t + 1][s].get("action")
            acts.append(a if isinstance(a, dict) else {})
            for o in (a or {}).get("market", []) if isinstance(a, dict) else []:
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    try:
                        out["x_req_sell"][t, s] += max(0, int(o[2]))
                    except (TypeError, ValueError):
                        pass
        g.step(acts)
        for s in (0, 1):
            ex = g.last_exec[s]
            for i, (act, changed) in enumerate(ex["units"][:MAXU]):
                out["x_unit"][t, s, i] = 1 if changed else 0
            for (op, item), n in ex["market"].items():
                if op == "SELL" and item in PIDX:
                    out["x_sell"][t, s, PIDX[item]] += n
                elif op == "BUY_SEED" and item in CIDX:
                    out["x_seed"][t, s, CIDX[item]] += n
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    out["x_buyp"][t, s, 0 if item == "WHEAT" else 1] += n
                elif op == "BUY_ANIMAL" and item in AIDX:
                    out["x_anim"][t, s, AIDX[item]] += n
            out["x_hire"][t, s] = ex["hires"]
            out["x_land"][t, s] = ex["land"]
        if parity < 0 and (t % check_every == 0 or t == T - 2) and not _same(g, steps[t + 1]):
            parity = t + 1
    rec = [float(x) for x in replay["rewards"]]
    if parity < 0 and g.rewards() != rec:
        parity = T - 1
    out["parity"] = np.array(parity, np.int32)
    return out
