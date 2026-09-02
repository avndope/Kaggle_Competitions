"""RealMLP (pytabkit) on the T3+imputation feature matrix.

Added on a top-competitor tip that RealMLP/TabM clear 0.97 here. Whether or not it beats
the hand-rolled MLPs on its own, it is a genuinely different architecture, and on this
problem architecture diversity is the only thing that has repeatedly paid: the first NN was
worth +0.00054 to the blend while three more GBDT families were worth +0.00011.

RealMLP needs dense input, so NaN is filled with a sentinel well outside every column's
range rather than imputed -- the target encodings already carry the missing-value signal.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode, _codes
import features2 as F2
from impute import fit_impute
from pytabkit import RealMLP_TD_S_Classifier

N_SPLITS, SEED, SMOOTH = 5, 42, 2.0
EPOCHS = int(os.environ.get("EPOCHS", "64"))
tr, te = load(); y = tr[TARGET].values.astype("int64")
dec_tr, dec_te = F2.load_decimals("train"), F2.load_decimals("test")
ftr, fte = F2.build(tr, dec_tr, F2.ALL_GROUPS), F2.build(te, dec_te, F2.ALL_GROUPS)
ctr, cte = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)
imp = fit_impute(tr, te, NUM)
for c, (a, b, r2) in imp.items():
    ftr[f"imp_{c}"] = a.astype("float32"); fte[f"imp_{c}"] = b.astype("float32")
    ftr[f"was_na_{c}"] = tr[c].isna().to_numpy().astype("float32")
    fte[f"was_na_{c}"] = te[c].isna().to_numpy().astype("float32")
    ctr[f"impte_{c}"] = _codes(pd.Series(a), 0.01); cte[f"impte_{c}"] = _codes(pd.Series(b), 0.01)
TE_COLS = NUM + list(CAT) + [f"impte_{c}" for c in imp]
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))

oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
for i, (ti, vi) in enumerate(folds):
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(ctr[c][ti], y[ti], [ctr[c][vi], cte[c]],
                                         n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
    mk = lambda idx, p, base, f: pd.concat(
        [base.iloc[idx][NUM].reset_index(drop=True), f.iloc[idx].reset_index(drop=True),
         pd.DataFrame(p)], axis=1).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).astype("float32").fillna(-999.0)
    X = mk(ti, ptr, tr, ftr); V = mk(vi, pva, tr, ftr)
    T = pd.concat([te[NUM].reset_index(drop=True), fte.reset_index(drop=True),
                   pd.DataFrame(pte)], axis=1).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).astype("float32").fillna(-999.0)

    m = RealMLP_TD_S_Classifier(device="cuda", n_epochs=EPOCHS, verbosity=0, random_state=SEED + i)
    m.fit(X.to_numpy(), y[ti])
    oof[vi] = m.predict_proba(V.to_numpy())[:, 1]
    tep += m.predict_proba(T.to_numpy())[:, 1] / N_SPLITS
    print(f"  fold {i+1}/{N_SPLITS} AUC={roc_auc_score(y[vi], oof[vi]):.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/_partial_oof_realmlp.npy", oof)
    np.save(f"{OUT}/_partial_test_realmlp.npy", tep * N_SPLITS / (i + 1))

auc = roc_auc_score(y, oof)
print(f"[realmlp] >>> OOF AUC = {auc:.6f}", flush=True)
np.save(f"{OUT}/oof_realmlp.npy", oof); np.save(f"{OUT}/test_realmlp.npy", tep)
json.dump({"realmlp": auc}, open(f"{OUT}/realmlp_report.json", "w"), indent=2, default=float)
