"""Leakage-safe out-of-fold target encoding for the exact-value (lookup-table) columns.

The S6E8 generator baked a per-value probability table into the lattice columns
(notifications_per_day, app_opens_per_day, age, ...).  A numeric split cannot reach it
(notifications_per_day has single-feature AUC 0.492 as a float, 0.749 per-value), and
LightGBM's native categorical grouping overfits it.  Target encoding maps each exact
value straight to its empirical probability, which is what the trees actually need.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

NA_CODE = -999_999


def _codes(s: pd.Series, step: float) -> np.ndarray:
    """Exact lattice value -> integer code. NaN -> NA_CODE (kept as its own level)."""
    v = np.round(s.to_numpy(dtype="float64") / step)
    return np.where(np.isnan(v), NA_CODE, v).astype("int64")


def _stats(codes, y, n_levels_map):
    """sum and count of y per code, as dense arrays indexed by dense level id."""
    lvl = n_levels_map[codes]
    n = n_levels_map.max() + 1
    cnt = np.bincount(lvl, minlength=n).astype("float64")
    ssum = np.bincount(lvl, weights=y, minlength=n).astype("float64")
    return ssum, cnt, lvl


def oof_target_encode(tr_codes, y_tr, other_codes_list, n_inner=10, smooth=20.0,
                      seed=0, prior=None, add_count=True):
    """Leave-inner-fold-out TE for the training rows; full-fold stats for everything else.

    Returns (te_tr, cnt_tr, [ (te_x, cnt_x) for each array in other_codes_list ]).
    """
    prior = float(y_tr.mean()) if prior is None else prior

    # Dense level ids shared across train and the arrays we must transform.
    # searchsorted, not an offset lookup table: interaction keys (a*100003 + b) span a
    # huge but sparse integer range, and indexing by (code - min) would try to allocate
    # an array the size of that range.
    allc = np.concatenate([tr_codes] + [c for c in other_codes_list]) if other_codes_list else tr_codes
    uniq = np.unique(allc)
    dense = lambda c: np.searchsorted(uniq, c)

    n = len(uniq)
    ltr = dense(tr_codes)
    tot_cnt = np.bincount(ltr, minlength=n).astype("float64")
    tot_sum = np.bincount(ltr, weights=y_tr.astype("float64"), minlength=n)

    # --- training rows: subtract their own inner fold (exact leave-fold-out) ---
    te_tr = np.empty(len(tr_codes))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    for _, fi in skf.split(np.zeros(len(y_tr)), y_tr):
        f_cnt = np.bincount(ltr[fi], minlength=n).astype("float64")
        f_sum = np.bincount(ltr[fi], weights=y_tr[fi].astype("float64"), minlength=n)
        num = tot_sum - f_sum + prior * smooth
        den = tot_cnt - f_cnt + smooth
        te_tr[fi] = (num / den)[ltr[fi]]
    cnt_tr = tot_cnt[ltr]

    # --- everything else: full-fold statistics ---
    full = (tot_sum + prior * smooth) / (tot_cnt + smooth)
    out = []
    for c in other_codes_list:
        lx = dense(c)
        out.append((full[lx], tot_cnt[lx]))

    return (te_tr, cnt_tr, out) if add_count else (te_tr, out)


def oof_target_encode_extra(tr_codes, y_tr, extra_codes, extra_soft, other_codes_list,
                            n_inner=10, smooth=20.0, seed=0, prior=None):
    """Target encoding whose statistics are reinforced by unlabelled rows carrying soft
    pseudo-labels.

    The point of this on S6E8: the per-value lookup tables are the whole model, and they
    are estimated from train alone. The 296k test rows add ~43% more support per lattice
    cell -- most valuable exactly where cells are thinnest and smoothing currently does
    the most damage.

    Soft labels, not hard: a pseudo-label of 0.62 contributes 0.62 to the cell's sum. That
    is the honest thing to add, and it avoids manufacturing false confidence at the
    decision boundary, which is precisely where these cells are hardest to estimate.

    The extra rows sit outside the leave-inner-fold-out subtraction because they are never
    themselves predicted -- their contribution is constant across the inner folds.
    """
    prior = float(y_tr.mean()) if prior is None else prior
    allc = np.concatenate([tr_codes, extra_codes] + [c for c in other_codes_list])
    uniq = np.unique(allc)
    dense = lambda c: np.searchsorted(uniq, c)
    n = len(uniq)

    ltr, lex = dense(tr_codes), dense(extra_codes)
    tot_cnt = np.bincount(ltr, minlength=n).astype("float64")
    tot_sum = np.bincount(ltr, weights=y_tr.astype("float64"), minlength=n)
    ex_cnt = np.bincount(lex, minlength=n).astype("float64")
    ex_sum = np.bincount(lex, weights=extra_soft.astype("float64"), minlength=n)

    te_tr = np.empty(len(tr_codes))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    for _, fi in skf.split(np.zeros(len(y_tr)), y_tr):
        f_cnt = np.bincount(ltr[fi], minlength=n).astype("float64")
        f_sum = np.bincount(ltr[fi], weights=y_tr[fi].astype("float64"), minlength=n)
        num = tot_sum - f_sum + ex_sum + prior * smooth
        den = tot_cnt - f_cnt + ex_cnt + smooth
        te_tr[fi] = (num / den)[ltr[fi]]

    full = (tot_sum + ex_sum + prior * smooth) / (tot_cnt + ex_cnt + smooth)
    out = [(full[dense(c)], (tot_cnt + ex_cnt)[dense(c)]) for c in other_codes_list]
    return te_tr, (tot_cnt + ex_cnt)[ltr], out


STEP = {"age": 1.0, "daily_screen_time_hours": .01, "social_media_hours": .01,
        "gaming_hours": .01, "work_study_hours": .01, "sleep_hours": .01,
        "notifications_per_day": 1.0, "app_opens_per_day": 1.0, "weekend_screen_time": .01}


def code_frame(df, cols, cat_cols=()):
    """Integer codes for every column we want to target-encode."""
    out = {}
    for c in cols:
        out[c] = _codes(df[c], STEP[c])
    for c in cat_cols:
        out[c] = pd.Categorical(df[c]).codes.astype("int64")  # NaN -> -1, its own level
    return out
