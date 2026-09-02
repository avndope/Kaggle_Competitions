"""Pseudo-labelling, two-stage and leakage-safe.

Why it is the right bet here: the per-value lookup tables ARE the model on this dataset,
and they are estimated from 691k train rows only. The 296k unlabelled test rows add ~43%
more support per lattice cell, which sharpens the estimate precisely where cells are thin.
That is a different kind of gain from another architecture, and it is sized for the
0.00087 gap to first place.

THE LEAKAGE TRAP, and how this avoids it. Pseudo-labels come from a model. If that model
was trained on all of train, then every test row's pseudo-label carries information about
the validation fold -- and feeding it into the encoder hands the validation fold's own
labels back to the student through the back door. CV goes up, the leaderboard does not.

So the teacher is retrained inside every outer fold, on that fold's training portion only.
A test pseudo-label used while scoring fold k was produced by a model that never saw
fold k. The cost is one extra model fit per fold; the alternative is a meaningless number.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode, oof_target_encode_extra, _codes
import features2 as F2
from impute import fit_impute

N_SPLITS, SEED, SMOOTH = 5, 42, 2.0
DEV = "cuda"
tr, te = load()
y = tr[TARGET].values.astype("int64")
dec_tr, dec_te = F2.load_decimals("train"), F2.load_decimals("test")
ftr, fte = F2.build(tr, dec_tr, F2.ALL_GROUPS), F2.build(te, dec_te, F2.ALL_GROUPS)
codes_tr, codes_te = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)

imp = fit_impute(tr, te, NUM)
for c, (a, b, r2) in imp.items():
    ftr[f"imp_{c}"] = a.astype("float32"); fte[f"imp_{c}"] = b.astype("float32")
    ftr[f"was_na_{c}"] = tr[c].isna().to_numpy().astype("float32")
    fte[f"was_na_{c}"] = te[c].isna().to_numpy().astype("float32")
    codes_tr[f"impte_{c}"] = _codes(pd.Series(a), 0.01)
    codes_te[f"impte_{c}"] = _codes(pd.Series(b), 0.01)
TE_COLS = NUM + list(CAT) + [f"impte_{c}" for c in imp]
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))

XP = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device=DEV,
          learning_rate=0.04, max_depth=8, min_child_weight=40, subsample=0.8,
          colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5, n_estimators=5000,
          early_stopping_rounds=150, n_jobs=8, random_state=SEED, verbosity=0)


def assemble(idx_tr, idx_va, ptr, pva, pte):
    mk = lambda idx, p, base, f: pd.concat(
        [base.iloc[idx][NUM].reset_index(drop=True), f.iloc[idx].reset_index(drop=True),
         pd.DataFrame(p)], axis=1)
    X = mk(idx_tr, ptr, tr, ftr); V = mk(idx_va, pva, tr, ftr)
    T = pd.concat([te[NUM].reset_index(drop=True), fte.reset_index(drop=True),
                   pd.DataFrame(pte)], axis=1)
    for d in (X, V, T):
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).astype("float32")
    return X, V, T


def plain_te(ti, vi):
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                         [codes_tr[c][vi], codes_te[c]],
                                         n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
        ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
    return ptr, pva, pte


def enriched_te(ti, vi, soft):
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode_extra(codes_tr[c][ti], y[ti], codes_te[c], soft,
                                               [codes_tr[c][vi], codes_te[c]],
                                               n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
        ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
    return ptr, pva, pte


def run(name, mode):
    """mode: 'base' | 'te' (enrich encoder) | 'rows' (add rows) | 'both'"""
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        # ---- teacher: trained on this fold's training rows only ----
        if mode != "base":
            ptr, pva, pte = plain_te(ti, vi)
            X, V, T = assemble(ti, vi, ptr, pva, pte)
            tch = xgb.XGBClassifier(**XP)
            tch.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
            soft = tch.predict_proba(T)[:, 1]
        else:
            soft = None

        # ---- student ----
        if mode in ("te", "both"):
            ptr, pva, pte = enriched_te(ti, vi, soft)
        else:
            ptr, pva, pte = plain_te(ti, vi)
        X, V, T = assemble(ti, vi, ptr, pva, pte)

        if mode in ("rows", "both"):
            # Soft pseudo-labels are fractional, and XGBClassifier only accepts integer
            # classes. XGBRegressor with a binary:logistic objective takes labels anywhere
            # in [0,1] and still returns calibrated probabilities -- which keeps the soft
            # labels soft instead of thresholding away the boundary cases that matter.
            Xs = pd.concat([X, T], ignore_index=True)
            ys = np.concatenate([y[ti].astype("float64"), soft])
            w = np.concatenate([np.ones(len(X)), np.full(len(T), 0.5)])
            m = xgb.XGBRegressor(**XP)
            m.fit(Xs, ys, sample_weight=w, eval_set=[(V, y[vi])], verbose=False)
            pv, pt = m.predict(V), m.predict(T)
        else:
            m = xgb.XGBClassifier(**XP)
            m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
            pv, pt = m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]
        oof[vi] = pv; tep += pt / N_SPLITS
        print(f"    fold {i+1}/{N_SPLITS} AUC={roc_auc_score(y[vi], oof[vi]):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.save(f"{OUT}/_partial_oof_PL{name}.npy", oof)          # this box sleeps
        np.save(f"{OUT}/_partial_test_PL{name}.npy", tep * N_SPLITS / (i + 1))
    auc = roc_auc_score(y, oof)
    print(f"  [{name}] >>> OOF = {auc:.6f}", flush=True)
    np.save(f"{OUT}/oof_PL{name}.npy", oof); np.save(f"{OUT}/test_PL{name}.npy", tep)
    return auc


ARMS = os.environ.get("ARMS", "base,te,rows,both").split(",")
res = {}
print("=== pseudo-labelling ablation (xgb GPU, 5-fold, teacher retrained per fold) ===", flush=True)
for nm in ARMS:
    print(f"\n-- {nm} --", flush=True)
    res[nm] = run(nm, nm)
for nm in ("base", "te", "rows", "both"):        # fold in arms completed earlier
    if nm not in res and os.path.exists(f"{OUT}/oof_PL{nm}.npy"):
        res[nm] = roc_auc_score(y, np.load(f"{OUT}/oof_PL{nm}.npy"))

print("\n===== SUMMARY =====")
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k:6s} {v:.6f}   ({v - res['base']:+.6f} vs base)")
json.dump(res, open(f"{OUT}/pseudo_report.json", "w"), indent=2, default=float)
