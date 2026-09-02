"""Run the entity-embedding MLP in CV and save OOF / test predictions for blending."""
import sys, os, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(int(os.environ.get("NN_THREADS", "8")))
DEVICE = os.environ.get("NN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device = {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, add_features, TARGET, CAT, NUM, OUT
from encoders import STEP
from nn_model import make_levels, encode, encode_cat, train_nn

N_SPLITS, SEED = 5, 42
tr, te = load()
y = tr[TARGET].values.astype("int64")

# ---- embedding inputs: every lattice column + the 3 real categoricals ----
maps, sizes = make_levels(tr, te, NUM, STEP)
Xc_tr, Xc_te = encode(tr, NUM, STEP, maps), encode(te, NUM, STEP, maps)
cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(set(tr[c].dropna()) | set(te[c].dropna())))}
            for c in CAT}
Xc_tr = np.hstack([Xc_tr, encode_cat(tr, CAT, cat_maps)])
Xc_te = np.hstack([Xc_te, encode_cat(te, CAT, cat_maps)])
sizes = [sizes[c] for c in NUM] + [len(cat_maps[c]) + 1 for c in CAT]
print("embedding table sizes:", dict(zip(NUM + CAT, sizes)), flush=True)

# ---- continuous inputs: raw numerics + engineered, standardised, NaN -> 0 + flag ----
ETR, ETE = add_features(tr, 1), add_features(te, 1)
CONT = NUM + [c for c in ETR.columns if c not in tr.columns]
A = ETR[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
B = ETE[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
mu, sd = A.mean(), A.std().replace(0, 1)
Xn_tr = np.hstack([((A - mu) / sd).fillna(0).to_numpy("float32"), A.isna().to_numpy("float32")])
Xn_te = np.hstack([((B - mu) / sd).fillna(0).to_numpy("float32"), B.isna().to_numpy("float32")])
print(f"continuous block: {Xn_tr.shape[1]} cols  |  embeddings: {len(sizes)}", flush=True)

oof = np.zeros(len(tr)); tep = np.zeros(len(te))
skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
t0 = time.time()
for i, (ti, vi) in enumerate(skf.split(tr, y)):
    print(f"\n--- NN fold {i+1}/{N_SPLITS} ({time.time()-t0:.0f}s) ---", flush=True)
    vp, tp, best = train_nn(Xc_tr[ti], Xn_tr[ti], y[ti], Xc_tr[vi], Xn_tr[vi], y[vi],
                            Xc_te, Xn_te, sizes, seed=SEED + i, device=DEVICE)
    oof[vi] = vp; tep += tp / N_SPLITS
    print(f"  fold best val AUC = {best:.6f}", flush=True)

auc = roc_auc_score(y, oof)
print(f"\n[NN] >>> OOF AUC = {auc:.6f}", flush=True)
np.save(f"{OUT}/oof_nn.npy", oof); np.save(f"{OUT}/test_nn.npy", tep)
