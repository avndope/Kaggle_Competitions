"""Final run: best preprocessing, best models, 10-fold, CPU and GPU saturated.

Preprocessing settled by ablation (5-fold XGBoost, everything else held fixed):
  all feature groups          0.968307   (dropping the behavioral group costs -0.00020,
                                          so it stays despite community advice to the
                                          contrary -- that advice was measured without
                                          target encoding)
  + decimal-place features    0.968339   (+0.00003 only; exact-value TE already separates
                                          7.0 from 7.01, so it is mostly redundant here)
  + TE smoothing 20 -> 2      0.968382   (monotone in smoothing: 2 > 20 > 50 > 200)
  + model-based imputation    0.968430   (recovers daily at R2 0.818, then re-encodes the
                                          recovered value onto the 0.01 lattice so the
                                          per-value lookup exists for those rows again)
  n_na                        rejected   (-0.00001)

Models are the five that actually earned blend weight last time. HistGradientBoosting and
the logistic model are dropped: both scored zero weight, the first because it lands where
LightGBM already is, the second because it is too weak to pay for its difference.
"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode, _codes, STEP
import features2 as F2
from impute import fit_impute
from nn_model import make_levels, encode, encode_cat, train_nn

N_SPLITS = int(os.environ.get("N_SPLITS", "10"))
N_SEEDS = int(os.environ.get("N_SEEDS", "2"))
SEED, SMOOTH = 42, 2.0
LGB_THREADS = int(os.environ.get("LGB_THREADS", "10"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
print(f"folds={N_SPLITS} seeds={N_SEEDS} smooth={SMOOTH} device={DEVICE}", flush=True)

tr, te = load()
y = tr[TARGET].values.astype("int64")
dec_tr, dec_te = F2.load_decimals("train"), F2.load_decimals("test")
ftr, fte = F2.build(tr, dec_tr, F2.ALL_GROUPS), F2.build(te, dec_te, F2.ALL_GROUPS)

print("fitting imputers (target never seen)...", flush=True)
imp = fit_impute(tr, te, NUM)
codes_tr, codes_te = code_frame(tr, NUM, CAT), code_frame(te, NUM, CAT)
for c, (a, b, r2) in imp.items():
    print(f"  {c:26s} R2={r2:.4f}", flush=True)
    ftr[f"imp_{c}"] = a.astype("float32"); fte[f"imp_{c}"] = b.astype("float32")
    ftr[f"was_na_{c}"] = tr[c].isna().to_numpy().astype("float32")
    fte[f"was_na_{c}"] = te[c].isna().to_numpy().astype("float32")
    codes_tr[f"impte_{c}"] = _codes(pd.Series(a), 0.01)
    codes_te[f"impte_{c}"] = _codes(pd.Series(b), 0.01)
TE_COLS = NUM + list(CAT) + [f"impte_{c}" for c in imp]

# ---- NN inputs (no target involved, so fold-independent) ----
maps, sizes_d = make_levels(tr, te, NUM, STEP)
cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(set(tr[c].dropna()) | set(te[c].dropna())))}
            for c in CAT}
Xc_tr = np.hstack([encode(tr, NUM, STEP, maps), encode_cat(tr, CAT, cat_maps)])
Xc_te = np.hstack([encode(te, NUM, STEP, maps), encode_cat(te, CAT, cat_maps)])
sizes = [sizes_d[c] for c in NUM] + [len(cat_maps[c]) + 1 for c in CAT]
A = pd.concat([tr[NUM].reset_index(drop=True), ftr.reset_index(drop=True)], axis=1)
B = pd.concat([te[NUM].reset_index(drop=True), fte.reset_index(drop=True)], axis=1)
A = A.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
B = B.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
mu, sd = A.mean(), A.std().replace(0, 1)
Xn_tr = np.hstack([((A - mu) / sd).fillna(0).to_numpy("float32"), A.isna().to_numpy("float32")])
Xn_te = np.hstack([((B - mu) / sd).fillna(0).to_numpy("float32"), B.isna().to_numpy("float32")])
print(f"NN: {len(sizes)} embeddings, {Xn_tr.shape[1]} continuous", flush=True)

NN_ARCH = {"nnC": dict(arch=dict(hidden=(256, 256, 256, 128), dropout=.10, emb_cap=16, act="relu"),
                       lr=4e-3, bs=2048, epochs=16),
           "nnB": dict(arch=dict(hidden=(1024, 256), dropout=.25, emb_cap=40, act="gelu"),
                       lr=2e-3, bs=8192, epochs=22)}
MODELS = ["xgb", "cat", "lgb"] + list(NN_ARCH)
oof = {k: np.zeros(len(tr)) for k in MODELS}
tep = {k: np.zeros(len(te)) for k in MODELS}


def build_fold(ti, vi):
    ptr, pva, pte = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                         [codes_tr[c][vi], codes_te[c]], n_inner=10, smooth=SMOOTH)
        (v_, cv_), (s_, cs_) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = t_, v_, s_
        ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cn_, cv_, cs_
    X = pd.concat([tr.iloc[ti][NUM].reset_index(drop=True),
                   ftr.iloc[ti].reset_index(drop=True), pd.DataFrame(ptr)], axis=1)
    V = pd.concat([tr.iloc[vi][NUM].reset_index(drop=True),
                   ftr.iloc[vi].reset_index(drop=True), pd.DataFrame(pva)], axis=1)
    T = pd.concat([te[NUM].reset_index(drop=True),
                   fte.reset_index(drop=True), pd.DataFrame(pte)], axis=1)
    for d in (X, V, T):
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).astype("float32")
    return X, V, T


def f_xgb(X, yt, V, yv, T, s):
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                          device=DEVICE, learning_rate=0.04, max_depth=8, min_child_weight=40,
                          subsample=0.8, colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                          n_estimators=5000, early_stopping_rounds=150, n_jobs=4,
                          random_state=s, verbosity=0)
    m.fit(X, yt, eval_set=[(V, yv)], verbose=False)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_cat(X, yt, V, yv, T, s):
    kw = dict(task_type="GPU", devices="0") if DEVICE == "cuda" else dict(thread_count=4)
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.06,
                           depth=7, l2_leaf_reg=6.0, iterations=4000, random_seed=s,
                           od_type="Iter", od_wait=150, verbose=0, allow_writing_files=False, **kw)
    m.fit(X, yt, eval_set=(V, yv), use_best_model=True)
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def f_lgb(X, yt, V, yv, T, s):
    m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.035, num_leaves=96,
                           min_child_samples=200, feature_fraction=0.55, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0, n_estimators=6000,
                           verbose=-1, n_jobs=LGB_THREADS, seed=s)
    m.fit(X, yt, eval_set=[(V, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict_proba(V)[:, 1], m.predict_proba(T)[:, 1]


def avg(fn, n, *a):
    vs, ts = zip(*[fn(*a, SEED + 1000 * s) for s in range(n)])
    return np.mean(vs, 0), np.mean(ts, 0)


def main():
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for i, (ti, vi) in enumerate(skf.split(tr, y)):
        X, V, T = build_fold(ti, vi)
        print(f"\n--- fold {i+1}/{N_SPLITS} ({X.shape[1]} feats, {time.time()-t0:.0f}s) ---", flush=True)
        res = {}
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(avg, f_lgb, N_SEEDS, X, y[ti], V, y[vi], T)   # CPU
            res["xgb"] = avg(f_xgb, N_SEEDS, X, y[ti], V, y[vi], T)       # GPU
            res["cat"] = avg(f_cat, N_SEEDS, X, y[ti], V, y[vi], T)       # GPU
            for nm, cfg in NN_ARCH.items():                                # GPU
                vs, ts = [], []
                for s in range(N_SEEDS):
                    vp, tp, _ = train_nn(Xc_tr[ti], Xn_tr[ti], y[ti], Xc_tr[vi], Xn_tr[vi], y[vi],
                                         Xc_te, Xn_te, sizes, seed=SEED + i + 1000 * s,
                                         device=DEVICE, verbose=False, arch=cfg["arch"],
                                         lr=cfg["lr"], bs=cfg["bs"], epochs=cfg["epochs"])
                    vs.append(vp); ts.append(tp)
                res[nm] = (np.mean(vs, 0), np.mean(ts, 0))
            res["lgb"] = fut.result()
        for k in MODELS:
            vp, tp = res[k]
            oof[k][vi] = vp; tep[k] += tp / N_SPLITS
            print(f"    {k:4s} AUC={roc_auc_score(y[vi], vp):.6f}", flush=True)
        for k in MODELS:
            np.save(f"{OUT}/_partial_oof_F{k}.npy", oof[k])
            np.save(f"{OUT}/_partial_test_F{k}.npy", tep[k] * N_SPLITS / (i + 1))
        print(f"    [checkpoint] {i+1}/{N_SPLITS} ({time.time()-t0:.0f}s)", flush=True)

    print("\n===== FINAL 10-FOLD OOF =====")
    sc = {}
    for k in MODELS:
        sc[k] = roc_auc_score(y, oof[k])
        print(f"  {k:4s} {sc[k]:.6f}")
        np.save(f"{OUT}/oof_F{k}.npy", oof[k]); np.save(f"{OUT}/test_F{k}.npy", tep[k])
    json.dump(sc, open(f"{OUT}/final_report.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
