"""Baseline: does feature engineering beat raw features? LightGBM, 5-fold CV."""
import os
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, prep, cv_run, TARGET, CAT, OUT

tr, te = load()
y = tr[TARGET].values

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, n_estimators=3000, verbose=-1, n_jobs=-1, seed=0,
)

def make_fp(feats):
    def fp(Xtr, ytr, Xva, yva, Xte):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xtr[feats], ytr, eval_set=[(Xva[feats], yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        vp = m.predict_proba(Xva[feats])[:, 1]
        tp = m.predict_proba(Xte[feats])[:, 1] if Xte is not None else None
        fp.best_iters.append(m.best_iteration_)
        fp.models.append(m)
        return vp, tp
    fp.best_iters, fp.models = [], []
    return fp

results = {}
for level in [0, 1, 2]:
    trf, tef, feats = prep(tr, te, level=level)
    print(f"\n===== FEATURE LEVEL {level}  ({len(feats)} features) =====", flush=True)
    fp = make_fp(feats)
    oof, tep, auc = cv_run(fp, trf, y, tef, name=f"lgb_L{level}")
    results[level] = auc
    print(f"  best_iters: {fp.best_iters}", flush=True)
    np.save(f"{OUT}/oof_lgb_L{level}.npy", oof)
    np.save(f"{OUT}/test_lgb_L{level}.npy", tep)
    if level == 1:
        imp = pd.Series(fp.models[0].feature_importances_, index=feats).sort_values(ascending=False)
        print("\n  --- gain importance (fold 1) ---")
        print(imp.to_string(), flush=True)

print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"  feature level {k}: OOF AUC = {v:.6f}")
