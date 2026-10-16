"""Paired statistics used by every evaluation (learnings: paired seeds both seats, count seeds not games,
strict flips with a sign test, split every gain into our money vs the opponent's)."""
import math

import numpy as np


def sign_test(up, dn):
    """Two-sided exact sign test p-value for `up` improvements vs `dn` regressions."""
    n = up + dn
    if n == 0:
        return 1.0
    k = min(up, dn)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def score(me, op):
    return 1.0 if me > op else 0.5 if me == op else 0.0


def mean_se(x):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def paired(base, cand):
    """base, cand: {key: (me, op)} over the same worlds. Strict flips, sign test, d_margin, d_me, d_op."""
    keys = sorted(set(base) & set(cand))
    d0 = np.array([base[k][0] - base[k][1] for k in keys])
    d1 = np.array([cand[k][0] - cand[k][1] for k in keys])
    dme = np.array([cand[k][0] - base[k][0] for k in keys])
    dop = np.array([cand[k][1] - base[k][1] for k in keys])
    up = int(((d0 <= 0) & (d1 > 0)).sum())
    dn = int(((d0 > 0) & (d1 <= 0)).sum())
    m, se = mean_se(d1 - d0)
    return dict(n=len(keys), wins_base=int((d0 > 0).sum()), wins_cand=int((d1 > 0).sum()), flips_up=up, flips_down=dn,
                p=sign_test(up, dn), d_margin=m, d_margin_se=se, d_me=float(dme.mean()) if len(keys) else 0.0,
                d_op=float(dop.mean()) if len(keys) else 0.0)
