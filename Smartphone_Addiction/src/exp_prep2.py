"""Round 2 preprocessing: imputation of the strong columns, and the missingness count.

Baseline here is the winner of round 1: all feature groups + decimals, TE smoothing 2.
"""
import os
import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
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
codes_tr, codes_te = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)
TE_COLS = NUM + list(CAT)
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))

print("=== fitting imputers (target never seen) ===", flush=True)
t0 = time.time()
imp = fit_impute(tr, te, NUM)
for k, (_, _, r2) in imp.items():
    print(f"  {k:26s} holdout R2 = {r2:.4f}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

IMP_TE = list(imp)   # target-encode the recovered values too


def run(name, groups, use_imp, use_nna):
    ftr = F2.build(tr, dec_tr, groups); fte = F2.build(te, dec_te, groups)
    if use_nna:
        ftr["n_na"] = tr[NUM + CAT].isna().sum(axis=1).to_numpy().astype("float32")
        fte["n_na"] = te[NUM + CAT].isna().sum(axis=1).to_numpy().astype("float32")
    ctr, cte = dict(codes_tr), dict(codes_te)
    if use_imp:
        for c, (a, b, _) in imp.items():
            ftr[f"imp_{c}"] = a.astype("float32"); fte[f"imp_{c}"] = b.astype("float32")
            ftr[f"was_na_{c}"] = tr[c].isna().to_numpy().astype("float32")
            fte[f"was_na_{c}"] = te[c].isna().to_numpy().astype("float32")
            ctr[f"impte_{c}"] = _codes(pd.Series(a), 0.01)
            cte[f"impte_{c}"] = _codes(pd.Series(b), 0.01)
    te_cols = TE_COLS + ([f"impte_{c}" for c in IMP_TE] if use_imp else [])

    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        ptr, pva, pte = {}, {}, {}
        for c in te_cols:
            t_, cn_, oth = oof_target_encode(ctr[c][ti], y[ti], [ctr[c][vi], cte[c]],
                                             n_inner=10, smooth=SMOOTH)
            (v_, cv_), (s_, cs_) = oth
            ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
            ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
        X = pd.concat([tr.iloc[ti][NUM].reset_index(drop=True),
                       ftr.iloc[ti].reset_index(drop=True), pd.DataFrame(ptr)], axis=1)
        V = pd.concat([tr.iloc[vi][NUM].reset_index(drop=True),
                       ftr.iloc[vi].reset_index(drop=True), pd.DataFrame(pva)], axis=1)
        T = pd.concat([te[NUM].reset_index(drop=True),
                       fte.reset_index(drop=True), pd.DataFrame(pte)], axis=1)
        for dd in (X, V, T):
            for c in dd.columns:
                dd[c] = pd.to_numeric(dd[c], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).astype("float32")
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                              tree_method="hist", device="cuda", learning_rate=0.04,
                              max_depth=8, min_child_weight=40, subsample=0.8,
                              colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                              n_estimators=5000, early_stopping_rounds=150,
                              n_jobs=8, random_state=SEED, verbosity=0)
        m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
        oof[vi] = m.predict_proba(V)[:, 1]; tep += m.predict_proba(T)[:, 1] / N_SPLITS
    auc = roc_auc_score(y, oof)
    print(f"  {name:24s} nfeat={X.shape[1]:3d}  OOF={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_Q_{name}.npy", oof); np.save(f"{OUT}/test_Q_{name}.npy", tep)
    return auc


res = {}
G = F2.ALL_GROUPS
print("\n=== round-2 ablation (baseline = round-1 winner, 0.968382) ===", flush=True)
res["base"] = run("base_smooth2", G, use_imp=False, use_nna=False)
res["nna"] = run("plus_n_na", G, use_imp=False, use_nna=True)
res["imp"] = run("plus_imputation", G, use_imp=True, use_nna=False)
res["imp_nna"] = run("imp_plus_nna", G, use_imp=True, use_nna=True)

print("\n===== SUMMARY =====")
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k:10s} {v:.6f}")
json.dump(res, open(f"{OUT}/prep2_ablation.json", "w"), indent=2, default=float)
