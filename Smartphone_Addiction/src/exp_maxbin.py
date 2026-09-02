"""max_bin sweep. Never tuned in this pipeline -- both boosters ran at their default
(~255) the entire time, while daily_screen_time has 1389 distinct values, weekend 1437,
social 721. Any two values landing in the same bin are indistinguishable to the tree.

The forum result is that max_bin and value-encoding are SUBSTITUTES: +0.0024 on raw
columns, but only +0.0005 once you already encode values, because both mechanisms do the
same job of telling individual values apart. This pipeline already target-encodes every
lattice column, so the small number is the one to expect here -- worth measuring rather
than assuming, since it is the difference between closing a third of the gap to first
place and closing none of it.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode, _codes
import features2 as F2
from impute import fit_impute

N_SPLITS, SEED, SMOOTH = 5, 42, 2.0
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

CACHE = {}
def fold_matrices(i, ti, vi):
    if i in CACHE: return CACHE[i]
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                         [codes_tr[c][vi], codes_te[c]], n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
        ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
    mk = lambda idx, p: pd.concat([tr.iloc[idx][NUM].reset_index(drop=True),
                                   ftr.iloc[idx].reset_index(drop=True), pd.DataFrame(p)], axis=1)
    X, V = mk(ti, ptr), mk(vi, pva)
    T = pd.concat([te[NUM].reset_index(drop=True), fte.reset_index(drop=True),
                   pd.DataFrame(pte)], axis=1)
    for d in (X, V, T):
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf,-np.inf], np.nan).astype("float32")
    CACHE[i] = (X, V, T)
    return CACHE[i]

def run(tag, algo, max_bin):
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        X, V, T = fold_matrices(i, ti, vi)
        if algo == "xgb":
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                  tree_method="hist", device="cuda", max_bin=max_bin,
                                  learning_rate=0.04, max_depth=8, min_child_weight=40,
                                  subsample=0.8, colsample_bytree=0.55, reg_lambda=5.0,
                                  reg_alpha=0.5, n_estimators=5000, early_stopping_rounds=150,
                                  n_jobs=8, random_state=SEED, verbosity=0)
            m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
            pv, pt = m.predict_proba(V)[:,1], m.predict_proba(T)[:,1]
        else:
            m = lgb.LGBMClassifier(objective="binary", metric="auc", max_bin=max_bin,
                                   learning_rate=0.035, num_leaves=96, min_child_samples=200,
                                   feature_fraction=0.55, bagging_fraction=0.8, bagging_freq=1,
                                   lambda_l1=0.5, lambda_l2=5.0, n_estimators=6000,
                                   verbose=-1, n_jobs=-1, seed=SEED)
            m.fit(X, y[ti], eval_set=[(V, y[vi])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            pv, pt = m.predict_proba(V)[:,1], m.predict_proba(T)[:,1]
        oof[vi] = pv; tep += pt / N_SPLITS
    auc = roc_auc_score(y, oof)
    print(f"  {tag:22s} OOF={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_MB{tag}.npy", oof); np.save(f"{OUT}/test_MB{tag}.npy", tep)
    return auc

res = {}
print("=== max_bin sweep, XGBoost (reference: default bins = 0.968430) ===", flush=True)
for mb in [256, 1024, 2047, 4096]:
    res[f"xgb_{mb}"] = run(f"xgb_{mb}", "xgb", mb)
print("\n=== max_bin sweep, LightGBM ===", flush=True)
for mb in [255, 1023, 2047]:
    res[f"lgb_{mb}"] = run(f"lgb_{mb}", "lgb", mb)

print("\n===== SUMMARY =====")
base = res.get("xgb_256")
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k:12s} {v:.6f}   ({v-base:+.6f} vs xgb default)")
json.dump(res, open(f"{OUT}/maxbin_report.json","w"), indent=2, default=float)
