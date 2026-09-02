"""Periodicity + log1p features, per a top-competitor tip that they push past 0.97.

The mechanism, as far as I can tell: sin/cos at period p expose where a value sits WITHIN
its lattice step -- sin(2*pi*x/1.0) separates whole numbers from fractions, period 0.1
separates by first decimal, and so on. That is a continuous, richer encoding of the same
decimal-place artifact already measured here (whole-number rows obey the un-smeared source
rule far better).

So this may well be another SUBSTITUTE for value encoding, like max_bin was -- the tip came
from a pipeline without exact-value target encoding. Cheap enough to settle rather than
argue about.
"""
import sys, os, time, json, numpy as np, pandas as pd
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
base_tr, base_te = F2.build(tr, dec_tr, F2.ALL_GROUPS), F2.build(te, dec_te, F2.ALL_GROUPS)
codes_tr, codes_te = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)
imp = fit_impute(tr, te, NUM)
for c, (a, b, r2) in imp.items():
    base_tr[f"imp_{c}"] = a.astype("float32"); base_te[f"imp_{c}"] = b.astype("float32")
    base_tr[f"was_na_{c}"] = tr[c].isna().to_numpy().astype("float32")
    base_te[f"was_na_{c}"] = te[c].isna().to_numpy().astype("float32")
    codes_tr[f"impte_{c}"] = _codes(pd.Series(a), 0.01)
    codes_te[f"impte_{c}"] = _codes(pd.Series(b), 0.01)
TE_COLS = NUM + list(CAT) + [f"impte_{c}" for c in imp]
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))

HOURS = F2.FLOATC
PERIODS = [1.0, 0.5, 0.25, 0.1, 0.05]

def periodic(df):
    d = {}
    for c in HOURS:
        v = df[c].to_numpy(dtype="float64")
        for p in PERIODS:
            d[f"sin_{c}_{p}"] = np.sin(2*np.pi*v/p)
            d[f"cos_{c}_{p}"] = np.cos(2*np.pi*v/p)
    return pd.DataFrame(d, index=df.index).astype("float32")

def logs(df):
    return pd.DataFrame({f"log1p_{c}": np.log1p(df[c].to_numpy(dtype="float64"))
                         for c in NUM}, index=df.index).astype("float32")

CACHE = {}
def te_parts(i, ti, vi):
    if i in CACHE: return CACHE[i]
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                         [codes_tr[c][vi], codes_te[c]], n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
        ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
    CACHE[i] = (ptr, pva, pte)
    return CACHE[i]

def run(tag, extra_tr, extra_te):
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        ptr, pva, pte = te_parts(i, ti, vi)
        blocks_tr = [tr.iloc[ti][NUM].reset_index(drop=True), base_tr.iloc[ti].reset_index(drop=True)]
        blocks_va = [tr.iloc[vi][NUM].reset_index(drop=True), base_tr.iloc[vi].reset_index(drop=True)]
        blocks_te = [te[NUM].reset_index(drop=True), base_te.reset_index(drop=True)]
        if extra_tr is not None:
            blocks_tr.append(extra_tr.iloc[ti].reset_index(drop=True))
            blocks_va.append(extra_tr.iloc[vi].reset_index(drop=True))
            blocks_te.append(extra_te.reset_index(drop=True))
        X = pd.concat(blocks_tr + [pd.DataFrame(ptr)], axis=1)
        V = pd.concat(blocks_va + [pd.DataFrame(pva)], axis=1)
        T = pd.concat(blocks_te + [pd.DataFrame(pte)], axis=1)
        for d in (X, V, T):
            for c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf,-np.inf], np.nan).astype("float32")
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                              device="cuda", learning_rate=0.04, max_depth=8, min_child_weight=40,
                              subsample=0.8, colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                              n_estimators=5000, early_stopping_rounds=150, n_jobs=8,
                              random_state=SEED, verbosity=0)
        m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
        oof[vi] = m.predict_proba(V)[:,1]; tep += m.predict_proba(T)[:,1]/N_SPLITS
    auc = roc_auc_score(y, oof)
    print(f"  {tag:16s} nfeat={X.shape[1]:3d}  OOF={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_PD{tag}.npy", oof); np.save(f"{OUT}/test_PD{tag}.npy", tep)
    return auc

per_tr, per_te = periodic(tr), periodic(te)
log_tr, log_te = logs(tr), logs(te)
both_tr = pd.concat([per_tr, log_tr], axis=1); both_te = pd.concat([per_te, log_te], axis=1)

res = {}
print("=== periodicity / log1p ablation (reference base = 0.968430) ===", flush=True)
res["base"] = run("base", None, None)
res["periodic"] = run("periodic", per_tr, per_te)
res["log1p"] = run("log1p", log_tr, log_te)
res["both"] = run("both", both_tr, both_te)
print("\n===== SUMMARY =====")
for k, v in sorted(res.items(), key=lambda x:-x[1]):
    print(f"  {k:10s} {v:.6f}  ({v-res['base']:+.6f} vs base)")
json.dump(res, open(f"{OUT}/periodic_report.json","w"), indent=2, default=float)
