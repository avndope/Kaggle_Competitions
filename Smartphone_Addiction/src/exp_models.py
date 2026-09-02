"""Model zoo on the target-encoded feature set.

All models share one FoldBuilder pass per fold, so the (expensive) leakage-safe target
encoding is computed once and reused. Blending is done on ranks: the model families put
their probabilities on different scales and plain averaging loses to rank averaging here.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from common import load, TARGET, OUT
from pipeline import FoldBuilder

N_SPLITS = 5
SEED = 42
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
print(f"GPU: {USE_GPU}  (xgb device={XGB_DEVICE}, catboost {'GPU' if USE_GPU else 'CPU'}, lgb CPU)", flush=True)
tr, te = load()
y = tr[TARGET].values.astype("int64")
# T3 feature set: pair/2D encodings measured at zero gain (T5 fold1 0.967740 vs T3 0.967744),
# so they are off — they only dilute feature_fraction.
fb = FoldBuilder(tr, te, y, use_eng=True, use_counts=True, use_pairs=False)


# Budgets are deliberately moderate. These three exist to disagree with each other, and a
# blend gains far more from three differently-shaped models than from squeezing the last
# 1e-4 out of any one of them. Deep/slow settings (lr 0.02, 128 leaves, depth 9, 12k
# CatBoost iters) projected to ~3h for well under a 0.0005 single-model gain.
def m_lgb(X, yt, V, yv, T, seed):
    m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.035,
                           num_leaves=96, min_child_samples=200, feature_fraction=0.55,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
                           n_estimators=6000, verbose=-1, n_jobs=-1, seed=seed)
    m.fit(X, yt, eval_set=[(V, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1], m.best_iteration_


# XGBoost and CatBoost run on the RTX 3050; LightGBM's pip wheel is CPU-only and stays
# there, which is convenient -- the GPU models and the CPU model no longer contend.
def m_xgb(X, yt, V, yv, T, seed):
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                          device=XGB_DEVICE,
                          learning_rate=0.04, max_depth=8, min_child_weight=40,
                          subsample=0.8, colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                          n_estimators=5000, early_stopping_rounds=150, n_jobs=-1,
                          random_state=seed, verbosity=0)
    m.fit(X, yt, eval_set=[(V, yv)], verbose=False)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1], m.best_iteration


def m_cat(X, yt, V, yv, T, seed):
    kw = dict(task_type="GPU", devices="0") if USE_GPU else dict(thread_count=-1)
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.06,
                           depth=7, l2_leaf_reg=6.0, iterations=4000, random_seed=seed,
                           od_type="Iter", od_wait=150, verbose=0,
                           allow_writing_files=False, **kw)
    m.fit(X, yt, eval_set=(V, yv), use_best_model=True)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1], m.get_best_iteration()


MODELS = {"lgb": m_lgb, "xgb": m_xgb, "cat": m_cat}

oof = {k: np.zeros(len(tr)) for k in MODELS}
tep = {k: np.zeros(len(te)) for k in MODELS}
iters = {k: [] for k in MODELS}

def seed_avg(fn, X, yt, V, yv, T, base_seed):
    """Average N_SEEDS runs of one model. Cheapest reliable variance reduction there is,
    and the GPU speedup is what pays for it."""
    vs, ts, its = [], [], []
    for s in range(N_SEEDS):
        vp, tp, it = fn(X, yt, V, yv, T, base_seed + 1000 * s)
        vs.append(vp); ts.append(tp); its.append(it)
    return np.mean(vs, 0), np.mean(ts, 0), int(np.mean(its))


skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
t0 = time.time()
done_folds = 0
for i, (ti, vi) in enumerate(skf.split(tr, y)):
    X, V, T = fb.build(ti, vi, seed=SEED + i)
    print(f"\n--- fold {i+1}/{N_SPLITS}  ({X.shape[1]} features, {time.time()-t0:.0f}s) ---", flush=True)

    # LightGBM is CPU-only, XGB/CatBoost are on the GPU: run them at the same time rather
    # than queueing. Both release the GIL inside their native training loops.
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut = ex.submit(seed_avg, m_lgb, X, y[ti], V, y[vi], T, SEED + i)
        res = {}
        for k in ("xgb", "cat"):
            res[k] = seed_avg(MODELS[k], X, y[ti], V, y[vi], T, SEED + i)
        res["lgb"] = fut.result()

    for k in MODELS:
        vp, tp, it = res[k]
        oof[k][vi] = vp
        tep[k] += tp / N_SPLITS
        iters[k].append(it)
        print(f"    {k}: AUC={roc_auc_score(y[vi], vp):.6f}  it~{it}  ({time.time()-t0:.0f}s)", flush=True)
    # Checkpoint every fold. This box sleeps, so a 5-fold run spans many hours of wall
    # time; saving only at the end makes the whole thing all-or-nothing.
    done_folds = i + 1
    for k in MODELS:
        np.save(f"{OUT}/_partial_oof_{k}.npy", oof[k])
        np.save(f"{OUT}/_partial_test_{k}.npy", tep[k] * N_SPLITS / done_folds)
    np.save(f"{OUT}/_partial_meta.npy", np.array([done_folds]))
    print(f"    [checkpoint] {done_folds}/{N_SPLITS} folds saved", flush=True)

print("\n===== per-model OOF =====")
scores = {}
for k in MODELS:
    scores[k] = roc_auc_score(y, oof[k])
    print(f"  {k}: {scores[k]:.6f}   iters={iters[k]}")
    np.save(f"{OUT}/oof_{k}.npy", oof[k]); np.save(f"{OUT}/test_{k}.npy", tep[k])

# ---- rank-average blend, weights searched on OOF ----
R = {k: rankdata(oof[k]) / len(oof[k]) for k in MODELS}
best, bw = -1, None
for wl in np.arange(0, 1.01, 0.05):
    for wx in np.arange(0, 1.01 - wl + 1e-9, 0.05):
        wc = 1 - wl - wx
        b = wl * R["lgb"] + wx * R["xgb"] + wc * R["cat"]
        s = roc_auc_score(y, b)
        if s > best:
            best, bw = s, (round(wl, 2), round(wx, 2), round(wc, 2))
print(f"\n  best rank-blend OOF = {best:.6f}  weights lgb/xgb/cat = {bw}")

Rt = {k: rankdata(tep[k]) / len(tep[k]) for k in MODELS}
blend_te = bw[0] * Rt["lgb"] + bw[1] * Rt["xgb"] + bw[2] * Rt["cat"]
np.save(f"{OUT}/test_blend.npy", blend_te)
json.dump({"per_model": scores, "blend": best, "weights": bw, "iters": iters},
          open(f"{OUT}/models_report.json", "w"), indent=2, default=float)

sub = pd.DataFrame({"id": te["id"], TARGET: blend_te})
sub.to_csv(f"{OUT}/submission_blend.csv", index=False)
print(f"\n  wrote {OUT}/submission_blend.csv")
