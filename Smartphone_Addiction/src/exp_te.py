"""Target-encoding experiment: recover the generator's per-value lookup table."""
import os
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, add_features, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode

tr, te = load()
y = tr[TARGET].values.astype("int64")

TE_NUM = NUM[:]                       # all 9 lattice columns
PAIRS = [("notifications_per_day", "app_opens_per_day")]

codes_tr = code_frame(tr, TE_NUM, CAT)
codes_te = code_frame(te, TE_NUM, CAT)
for a, b in PAIRS:                    # 2-way interaction level
    k = f"{a}__X__{b}"
    codes_tr[k] = codes_tr[a] * 100003 + codes_tr[b]
    codes_te[k] = codes_te[a] * 100003 + codes_te[b]

ENG_TR = add_features(tr, level=1)
ENG_TE = add_features(te, level=1)
ENG_COLS = [c for c in ENG_TR.columns if c not in ENG_TR.columns.intersection(tr.columns)]

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=96,
              min_child_samples=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, n_estimators=6000, verbose=-1, n_jobs=-1, seed=0)


def build_fold(ti, vi, te_cols, use_eng, use_counts, smooth=20.0):
    """Returns (Xtr, Xva, Xte) dataframes with leakage-safe TE columns."""
    parts_tr, parts_va, parts_te = {}, {}, {}
    for c in te_cols:
        ctr = codes_tr[c][ti]
        te_tr, cnt_tr, others = oof_target_encode(
            ctr, y[ti], [codes_tr[c][vi], codes_te[c]], n_inner=10, smooth=smooth)
        (te_va, cnt_va), (te_tst, cnt_tst) = others
        parts_tr[f"te_{c}"], parts_va[f"te_{c}"], parts_te[f"te_{c}"] = te_tr, te_va, te_tst
        if use_counts:
            parts_tr[f"cn_{c}"], parts_va[f"cn_{c}"], parts_te[f"cn_{c}"] = cnt_tr, cnt_va, cnt_tst

    base = NUM + ([] if not use_eng else ENG_COLS)
    src_tr = ENG_TR if use_eng else tr
    src_te = ENG_TE if use_eng else te
    Xtr = pd.concat([src_tr.iloc[ti][base].reset_index(drop=True), pd.DataFrame(parts_tr)], axis=1)
    Xva = pd.concat([src_tr.iloc[vi][base].reset_index(drop=True), pd.DataFrame(parts_va)], axis=1)
    Xte = pd.concat([src_te[base].reset_index(drop=True), pd.DataFrame(parts_te)], axis=1)
    for d in (Xtr, Xva, Xte):
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return Xtr, Xva, Xte


def run(name, te_cols, use_eng=False, use_counts=True, smooth=20.0, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time(); iters = []
    for i, (ti, vi) in enumerate(skf.split(tr, y)):
        Xtr, Xva, Xte = build_fold(ti, vi, te_cols, use_eng, use_counts, smooth)
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xtr, y[ti], eval_set=[(Xva, y[vi])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[vi] = m.predict_proba(Xva)[:, 1]
        tep += m.predict_proba(Xte)[:, 1] / n_splits
        iters.append(m.best_iteration_)
        print(f"  [{name}] fold {i+1}/{n_splits} AUC={roc_auc_score(y[vi], oof[vi]):.6f} "
              f"it={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
    auc = roc_auc_score(y, oof)
    print(f"  [{name}] >>> OOF AUC = {auc:.6f}   iters={iters}", flush=True)
    np.save(f"{OUT}/oof_{name}.npy", oof); np.save(f"{OUT}/test_{name}.npy", tep)
    if name == "T2":
        print(pd.Series(m.feature_importances_, index=Xtr.columns).sort_values(ascending=False).to_string(), flush=True)
    return auc


res = {}
print("\n===== T1: raw numerics + TE(9 lattice cols) =====", flush=True)
res["T1"] = run("T1", TE_NUM, use_eng=False, use_counts=False)

print("\n===== T2: + TE(3 categoricals) + value counts =====", flush=True)
res["T2"] = run("T2", TE_NUM + CAT, use_eng=False, use_counts=True)

print("\n===== T3: + engineered features =====", flush=True)
res["T3"] = run("T3", TE_NUM + CAT, use_eng=True, use_counts=True)

print("\n===== T4: + 2-way TE (notifications x app_opens) =====", flush=True)
res["T4"] = run("T4", TE_NUM + CAT + [f"{a}__X__{b}" for a, b in PAIRS], use_eng=True, use_counts=True)

print("\n===== SUMMARY (raw-numeric baseline = 0.963748) =====")
for k, v in res.items():
    print(f"  {k}: OOF AUC = {v:.6f}   (+{v-0.963748:.6f})")
