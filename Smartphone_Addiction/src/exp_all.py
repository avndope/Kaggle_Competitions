"""Saturating run: CPU and GPU worked in parallel, 10-fold, all model families at once.

Two independent streams per fold, running concurrently:

  GPU stream  - XGBoost, CatBoost, and the embedding NNs, in sequence. Sequential because
                VRAM is the binding constraint (CatBoost alone peaks near 3.2 of 4 GB), not
                because the GPU is saturated.
  CPU stream  - LightGBM, sklearn HistGradientBoosting, and a logistic model on the encoded
                features, across the 16 cores.

Per-fold cost becomes max(GPU, CPU) instead of the sum, and the expensive leakage-safe
target encoding is built once per fold and shared by every model that wants a matrix.

Model choice follows the measured lesson: extra GBDT variants and extra seeds are worth
~nothing to the blend, decorrelated *shapes* are worth everything. HistGradientBoosting
(different binning/growth than the other three) and a plain logistic regression on the
target-encoded columns are here because they are wrong in different places, not because
they are individually strong.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from common import load, add_features, TARGET, CAT, NUM, OUT
from encoders import STEP
from pipeline import FoldBuilder
from nn_model import make_levels, encode, encode_cat, train_nn

N_SPLITS = int(os.environ.get("N_SPLITS", "10"))
SEED = 42
N_SEEDS = int(os.environ.get("N_SEEDS", "2"))
LGB_THREADS = int(os.environ.get("LGB_THREADS", "10"))   # leave cores to feed the GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
print(f"folds={N_SPLITS} seeds={N_SEEDS} device={DEVICE} lgb_threads={LGB_THREADS}", flush=True)

tr, te = load()
y = tr[TARGET].values.astype("int64")
fb = FoldBuilder(tr, te, y, use_eng=True, use_counts=True, use_pairs=False)

# ---- NN inputs: value->id maps and scaling use no target, so they are fold-independent ----
maps, sizes_d = make_levels(tr, te, NUM, STEP)
Xc_tr = np.hstack([encode(tr, NUM, STEP, maps),
                   encode_cat(tr, CAT, {c: {v: i + 1 for i, v in enumerate(
                       sorted(set(tr[c].dropna()) | set(te[c].dropna())))} for c in CAT})])
cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(set(tr[c].dropna()) | set(te[c].dropna())))}
            for c in CAT}
Xc_te = np.hstack([encode(te, NUM, STEP, maps), encode_cat(te, CAT, cat_maps)])
sizes = [sizes_d[c] for c in NUM] + [len(cat_maps[c]) + 1 for c in CAT]
ETR, ETE = add_features(tr, 1), add_features(te, 1)
CONT = NUM + [c for c in ETR.columns if c not in tr.columns]
A = ETR[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
B = ETE[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
mu, sd = A.mean(), A.std().replace(0, 1)
Xn_tr = np.hstack([((A - mu) / sd).fillna(0).to_numpy("float32"), A.isna().to_numpy("float32")])
Xn_te = np.hstack([((B - mu) / sd).fillna(0).to_numpy("float32"), B.isna().to_numpy("float32")])

NN_ARCH = {
    "nnC": dict(arch=dict(hidden=(256, 256, 256, 128), dropout=.10, emb_cap=16, act="relu"),
                lr=4e-3, bs=2048, epochs=16),
    "nnB": dict(arch=dict(hidden=(1024, 256), dropout=.25, emb_cap=40, act="gelu"),
                lr=2e-3, bs=8192, epochs=22),
}
MODELS = ["xgb", "cat", "lgb", "hgb", "lin"] + list(NN_ARCH)
oof = {k: np.zeros(len(tr)) for k in MODELS}
tep = {k: np.zeros(len(te)) for k in MODELS}


def avg(fn, n, *a):
    vs, ts = zip(*[fn(*a, SEED + 1000 * s) for s in range(n)])
    return np.mean(vs, 0), np.mean(ts, 0)


def f_xgb(X, yt, V, yv, T, s):
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                          device="cuda" if DEVICE == "cuda" else "cpu", learning_rate=0.04,
                          max_depth=8, min_child_weight=40, subsample=0.8, colsample_bytree=0.55,
                          reg_lambda=5.0, reg_alpha=0.5, n_estimators=5000,
                          early_stopping_rounds=150, n_jobs=4, random_state=s, verbosity=0)
    m.fit(X, yt, eval_set=[(V, yv)], verbose=False)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_cat(X, yt, V, yv, T, s):
    kw = dict(task_type="GPU", devices="0") if DEVICE == "cuda" else dict(thread_count=4)
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.06,
                           depth=7, l2_leaf_reg=6.0, iterations=4000, random_seed=s,
                           od_type="Iter", od_wait=150, verbose=0,
                           allow_writing_files=False, **kw)
    m.fit(X, yt, eval_set=(V, yv), use_best_model=True)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_lgb(X, yt, V, yv, T, s):
    m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.035, num_leaves=96,
                           min_child_samples=200, feature_fraction=0.55, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0, n_estimators=6000,
                           verbose=-1, n_jobs=LGB_THREADS, seed=s)
    m.fit(X, yt, eval_set=[(V, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_hgb(X, yt, V, yv, T, s):
    m = HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.06, max_leaf_nodes=48,
                                       min_samples_leaf=150, l2_regularization=5.0,
                                       early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=60, random_state=s)
    m.fit(X, yt)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_lin(X, yt, V, yv, T, s):
    # Linear on the encoded columns: maximally decorrelated from every tree in the blend.
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.5, max_iter=3000, n_jobs=2))
    p.fit(X, yt)
    return p.predict_proba(V)[:, 1], p.predict_proba(T)[:, 1]


def cpu_stream(X, yt, V, yv, T):
    out = {}
    out["lgb"] = avg(f_lgb, N_SEEDS, X, yt, V, yv, T)
    out["hgb"] = avg(f_hgb, 1, X, yt, V, yv, T)
    out["lin"] = avg(f_lin, 1, X, yt, V, yv, T)
    return out


def main():
  skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
  t0 = time.time()
  for i, (ti, vi) in enumerate(skf.split(tr, y)):
    X, V, T = fb.build(ti, vi, seed=SEED + i)
    print(f"\n--- fold {i+1}/{N_SPLITS} ({X.shape[1]} feats, {time.time()-t0:.0f}s) ---", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(cpu_stream, X, y[ti], V, y[vi], T)          # CPU: lgb + hgb + lin
        res["xgb"] = avg(f_xgb, N_SEEDS, X, y[ti], V, y[vi], T)     # GPU
        res["cat"] = avg(f_cat, N_SEEDS, X, y[ti], V, y[vi], T)     # GPU
        for nm, cfg in NN_ARCH.items():                              # GPU
            vs, ts = [], []
            for s in range(N_SEEDS):
                vp, tp, _ = train_nn(Xc_tr[ti], Xn_tr[ti], y[ti], Xc_tr[vi], Xn_tr[vi], y[vi],
                                     Xc_te, Xn_te, sizes, seed=SEED + i + 1000 * s,
                                     device=DEVICE, verbose=False, arch=cfg["arch"],
                                     lr=cfg["lr"], bs=cfg["bs"], epochs=cfg["epochs"])
                vs.append(vp); ts.append(tp)
            res[nm] = (np.mean(vs, 0), np.mean(ts, 0))
        res.update(fut.result())

    for k in MODELS:
        vp, tp = res[k]
        oof[k][vi] = vp
        tep[k] += tp / N_SPLITS
        print(f"    {k:4s} AUC={roc_auc_score(y[vi], vp):.6f}", flush=True)
    for k in MODELS:   # checkpoint every fold -- this box sleeps
        np.save(f"{OUT}/_partial_oof_{k}.npy", oof[k])
        np.save(f"{OUT}/_partial_test_{k}.npy", tep[k] * N_SPLITS / (i + 1))
    print(f"    [checkpoint] {i+1}/{N_SPLITS} folds ({time.time()-t0:.0f}s)", flush=True)

  print("\n===== 10-FOLD OOF =====")
  scores = {}
  for k in MODELS:
    scores[k] = roc_auc_score(y, oof[k])
    print(f"  {k:4s} {scores[k]:.6f}")
    np.save(f"{OUT}/oof_{k}10.npy", oof[k]); np.save(f"{OUT}/test_{k}10.npy", tep[k])
  json.dump(scores, open(f"{OUT}/models10_report.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
