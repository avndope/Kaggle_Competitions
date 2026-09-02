"""Measure the CV<->LB offset locally, before spending any submission.

The offset splits into two independent parts:

  bagging (systematic) - OOF scores ONE fold-model on its held-out fold, but the
      submission averages all K fold-models. The averaged ensemble is genuinely better,
      so the LB sits above OOF by a fixed amount. Reproduce the exact structure locally:
      hold out a slice of train, run the K-fold pipeline on the rest, then score the
      averaged K models on the untouched slice. That is what the test set sees.

  sampling (noise) - the public LB is ~59,260 rows against 691,369 OOF rows. Resample
      public-sized subsets of the OOF predictions to get the distribution of scores the
      LB could show for a model whose true skill is fixed.

Only the first shifts the number; only the second explains why two submissions can swap
places for no reason.
"""
import os
import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from common import load, TARGET, CAT, NUM, OUT
from encoders import code_frame, oof_target_encode

N_PUBLIC = 59260          # 20% of the 296,302-row test set
tr, te = load()
y = tr[TARGET].values.astype("int64")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. sampling noise
print("=" * 72)
print("1. SAMPLING NOISE — what a public-sized slice does to a fixed model")
print("=" * 72)
blend_oof = None
try:
    rep = json.load(open(f"{OUT}/blend_report.json"))
    tot = 0
    for k, w in rep["weights"].items():
        o = np.load(f"{OUT}/oof_{k}.npy")
        tot = tot + w * (rankdata(o) / len(o))
    blend_oof = tot
    print(f"  reconstructed blend OOF AUC = {roc_auc_score(y, blend_oof):.6f}")
except Exception as e:
    print("  (blend unavailable, using xgb10)", e)
    blend_oof = np.load(f"{OUT}/oof_xgb10.npy")

full = roc_auc_score(y, blend_oof)
draws = []
for _ in range(400):
    idx = rng.choice(len(y), N_PUBLIC, replace=False)
    draws.append(roc_auc_score(y[idx], blend_oof[idx]))
draws = np.array(draws)
print(f"  full-OOF AUC (691,369 rows)      = {full:.6f}")
print(f"  public-sized slices ({N_PUBLIC:,} rows):")
print(f"    mean = {draws.mean():.6f}   sd = {draws.std():.6f}")
print(f"    95% of draws land in [{np.percentile(draws,2.5):.6f}, {np.percentile(draws,97.5):.6f}]")
print(f"  => a single LB reading carries +/- {1.96*draws.std():.5f} of pure noise")

# --------------------------------------------- 2. resolvable gap between two submissions
print("\n" + "=" * 72)
print("2. RESOLVABLE GAP — smallest CV difference the public LB could confirm")
print("=" * 72)
print("  sd(gap) = sd(move) * sqrt(2(1-rho)); correlated submissions share their noise")
pairs = [("xgb10", "Fxgb"), ("xgb10", "nnC10"), ("xgb10", "lgb_L0")]
for a, b in pairs:
    try:
        oa, ob = np.load(f"{OUT}/oof_{a}.npy"), np.load(f"{OUT}/oof_{b}.npy")
    except FileNotFoundError:
        continue
    ra, rb = rankdata(oa) / len(oa), rankdata(ob) / len(ob)
    ga, gb = [], []
    for _ in range(200):
        idx = rng.choice(len(y), N_PUBLIC, replace=False)
        ga.append(roc_auc_score(y[idx], ra[idx])); gb.append(roc_auc_score(y[idx], rb[idx]))
    ga, gb = np.array(ga), np.array(gb)
    rho = np.corrcoef(ga, gb)[0, 1]
    sd_gap = (ga - gb).std()
    print(f"  {a:8s} vs {b:8s}: rho(AUC est)={rho:.4f}  sd(gap)={sd_gap:.6f}  "
          f"resolvable@95% = {1.96*sd_gap:.5f}")

# ------------------------------------------------------- 3. the bagging offset itself
print("\n" + "=" * 72)
print("3. BAGGING OFFSET — the systematic part, measured by mimicking the test structure")
print("=" * 72)
codes_tr = code_frame(tr, NUM, CAT)
TE_COLS = NUM + list(CAT)
K = 10

dev_idx, hold_idx = train_test_split(np.arange(len(tr)), test_size=0.2,
                                     stratify=y, random_state=7)
print(f"  dev={len(dev_idx):,}  holdout={len(hold_idx):,} (never seen by any model)")


def matrices(ti, vi, hi):
    ptr, pva, pho = {}, {}, {}
    for c in TE_COLS:
        t_, cn_, oth = oof_target_encode(codes_tr[c][ti], y[ti],
                                         [codes_tr[c][vi], codes_tr[c][hi]],
                                         n_inner=10, smooth=2.0)
        (v_, _), (h_, _) = oth
        ptr[f"te_{c}"], pva[f"te_{c}"], pho[f"te_{c}"] = t_, v_, h_
    mk = lambda idx, p: pd.concat(
        [tr.iloc[idx][NUM].reset_index(drop=True), pd.DataFrame(p)], axis=1
    ).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return mk(ti, ptr), mk(vi, pva), mk(hi, pho)


oof_dev = np.zeros(len(dev_idx))
hold_preds = []
t0 = time.time()
for i, (a, b) in enumerate(StratifiedKFold(K, shuffle=True, random_state=42)
                           .split(dev_idx, y[dev_idx])):
    ti, vi = dev_idx[a], dev_idx[b]
    X, V, H = matrices(ti, vi, hold_idx)
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                          device="cuda", learning_rate=0.04, max_depth=8, min_child_weight=40,
                          subsample=0.8, colsample_bytree=0.55, reg_lambda=5.0, reg_alpha=0.5,
                          n_estimators=5000, early_stopping_rounds=150, n_jobs=8,
                          random_state=42, verbosity=0)
    m.fit(X, y[ti], eval_set=[(V, y[vi])], verbose=False)
    oof_dev[b] = m.predict_proba(V)[:, 1]
    hold_preds.append(m.predict_proba(H)[:, 1])
    print(f"    fold {i+1}/{K} ({time.time()-t0:.0f}s)", flush=True)

oof_auc = roc_auc_score(y[dev_idx], oof_dev)
single = np.mean([roc_auc_score(y[hold_idx], p) for p in hold_preds])
ens = roc_auc_score(y[hold_idx], np.mean(hold_preds, 0))
print(f"\n  (a) OOF AUC on dev, single model per fold      = {oof_auc:.6f}   <- what CV reports")
print(f"  (b) one fold-model, scored on holdout          = {single:.6f}")
print(f"  (c) average of all {K} fold-models on holdout   = {ens:.6f}   <- what the LB sees")
print(f"\n  BAGGING OFFSET (c - a) = {ens - oof_auc:+.6f}")
print(f"  ensembling gain (c - b) = {ens - single:+.6f}")
json.dump({"oof": oof_auc, "single_holdout": single, "ensemble_holdout": ens,
           "bagging_offset": ens - oof_auc, "noise_sd": float(draws.std())},
          open(f"{OUT}/offset_report.json", "w"), indent=2, default=float)

print("\n" + "=" * 72)
print(f"  PREDICTION: public LB ~= OOF {ens - oof_auc:+.5f}, +/- {1.96*draws.std():.5f} (95%)")
print("=" * 72)
