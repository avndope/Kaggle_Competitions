"""Preprocessing ablation. Same model throughout (XGBoost on GPU, 5-fold) so every
difference is attributable to the features, not the algorithm.

Also sweeps target-encoding smoothing, which was fixed at 20 the whole time and never
tuned -- it matters most for the high-cardinality columns (daily has 1389 levels at ~430
rows each) and barely at all for notifications (231 levels at ~2700 rows each).
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode
import features2 as F2

N_SPLITS, SEED = 5, 42
DEV = "cuda"
tr, te = load()
y = tr[TARGET].values.astype("int64")
dec_tr, dec_te = F2.load_decimals("train"), F2.load_decimals("test")
codes_tr, codes_te = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)
TE_COLS = NUM + list(CAT)
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))


def run(name, groups, smooth=20.0, per_col_smooth=None):
    ftr = F2.build(tr, dec_tr, groups)
    fte = F2.build(te, dec_te, groups)
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        ptr, pva, pte = {}, {}, {}
        for c in TE_COLS:
            sm = (per_col_smooth or {}).get(c, smooth)
            t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                             [codes_tr[c][vi], codes_te[c]],
                                             n_inner=10, smooth=sm)
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
                              tree_method="hist", device=DEV, learning_rate=0.04,
                              max_depth=8, min_child_weight=40, subsample=0.8,
                              colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                              n_estimators=5000, early_stopping_rounds=150,
                              n_jobs=8, random_state=SEED, verbosity=0)
        m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
        oof[vi] = m.predict_proba(V)[:, 1]; tep += m.predict_proba(T)[:, 1] / N_SPLITS
    auc = roc_auc_score(y, oof)
    print(f"  {name:34s} nfeat={X.shape[1]:3d}  OOF={auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_P_{name}.npy", oof); np.save(f"{OUT}/test_P_{name}.npy", tep)
    return auc, X.columns.tolist(), m


res = {}
print("=== A. feature-group ablation (XGBoost GPU, 5-fold, smooth=20) ===", flush=True)
res["cur_all5"], _, _ = run("cur_all5groups", ("structural", "behavioral", "rule", "weekend"))
res["plus_dec"], _, mdl = run("plus_decimals", F2.ALL_GROUPS)
res["no_behav"], cols, mdl2 = run("struct_dec_rule_wke", ("structural", "decimals", "rule", "weekend"))
res["struct_dec"], _, _ = run("struct_dec_only", ("structural", "decimals"))
res["dec_only"], _, _ = run("decimals_only", ("decimals",))

best_groups = {"cur_all5": ("structural", "behavioral", "rule", "weekend"),
               "plus_dec": F2.ALL_GROUPS,
               "no_behav": ("structural", "decimals", "rule", "weekend"),
               "struct_dec": ("structural", "decimals"),
               "dec_only": ("decimals",)}[max(res, key=res.get)]
print(f"\n  best group set: {best_groups}", flush=True)

print("\n=== B. TE smoothing sweep on the winning group set ===", flush=True)
for sm in [2.0, 50.0, 200.0]:
    res[f"smooth{int(sm)}"], _, _ = run(f"smooth{int(sm)}", best_groups, smooth=sm)

# high-cardinality columns are thin per level, the integer lookups are not
HI = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
      "work_study_hours", "sleep_hours", "gaming_hours"]
res["mixed_smooth"], _, _ = run("mixed_smooth", best_groups, smooth=5.0,
                                per_col_smooth={c: 100.0 for c in HI})

print("\n===== SUMMARY (5-fold xgb; previous 5-fold xgb reference = 0.968603) =====")
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} {v:.6f}")
json.dump(res, open(f"{OUT}/prep_ablation.json", "w"), indent=2, default=float)

print("\n=== decimals feature importance in the full model ===")
imp = pd.Series(mdl.feature_importances_, index=cols if len(cols) == len(mdl.feature_importances_) else None)
try:
    print(imp.sort_values(ascending=False).head(20).to_string())
except Exception:
    pass
