"""Several NN architectures, each saved separately so the blender can weight them apart.

The blend analysis is unambiguous about where the headroom is: adding three whole GBDT
families moved the blend +0.00011, while one NN was worth +0.00054. The GBDTs are all
wrong in the same places; the NN is not. So the payoff is in more *non-tree* variety,
not more boosting -- which is what this script buys, now that the GPU makes it cheap.

Each variant is deliberately shaped differently (width vs depth, embedding capacity,
activation, dropout) so they decorrelate rather than just re-running one model.
"""
import sys, os, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(int(os.environ.get("NN_THREADS", "8")))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import load, add_features, TARGET, CAT, NUM, OUT
from encoders import STEP
from nn_model import make_levels, encode, encode_cat, train_nn

N_SPLITS, SEED = 5, 42
N_SEEDS = int(os.environ.get("NN_SEEDS", "2"))
DEVICE = os.environ.get("NN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device={DEVICE} | seeds/variant={N_SEEDS}", flush=True)

VARIANTS = {
    "nnA": dict(arch=dict(hidden=(512, 256, 128), dropout=0.15, emb_cap=24, act="silu"),
                lr=3e-3, bs=4096, epochs=18),
    "nnB": dict(arch=dict(hidden=(1024, 256), dropout=0.25, emb_cap=40, act="gelu"),
                lr=2e-3, bs=8192, epochs=22),
    "nnC": dict(arch=dict(hidden=(256, 256, 256, 128), dropout=0.10, emb_cap=16, act="relu"),
                lr=4e-3, bs=2048, epochs=16),
}

tr, te = load()
y = tr[TARGET].values.astype("int64")

maps, sizes_d = make_levels(tr, te, NUM, STEP)
Xc_tr, Xc_te = encode(tr, NUM, STEP, maps), encode(te, NUM, STEP, maps)
cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(set(tr[c].dropna()) | set(te[c].dropna())))}
            for c in CAT}
Xc_tr = np.hstack([Xc_tr, encode_cat(tr, CAT, cat_maps)])
Xc_te = np.hstack([Xc_te, encode_cat(te, CAT, cat_maps)])
sizes = [sizes_d[c] for c in NUM] + [len(cat_maps[c]) + 1 for c in CAT]

ETR, ETE = add_features(tr, 1), add_features(te, 1)
CONT = NUM + [c for c in ETR.columns if c not in tr.columns]
A = ETR[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
B = ETE[CONT].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
mu, sd = A.mean(), A.std().replace(0, 1)
Xn_tr = np.hstack([((A - mu) / sd).fillna(0).to_numpy("float32"), A.isna().to_numpy("float32")])
Xn_te = np.hstack([((B - mu) / sd).fillna(0).to_numpy("float32"), B.isna().to_numpy("float32")])
print(f"embeddings {len(sizes)} | continuous {Xn_tr.shape[1]}", flush=True)

skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(tr, y))
results = {}

for name, cfg in VARIANTS.items():
    oof = np.zeros(len(tr)); tep = np.zeros(len(te)); t0 = time.time()
    for i, (ti, vi) in enumerate(folds):
        vs, ts = [], []
        for s in range(N_SEEDS):
            vp, tp, best = train_nn(
                Xc_tr[ti], Xn_tr[ti], y[ti], Xc_tr[vi], Xn_tr[vi], y[vi], Xc_te, Xn_te, sizes,
                seed=SEED + i + 1000 * s, device=DEVICE, verbose=False,
                arch=cfg["arch"], lr=cfg["lr"], bs=cfg["bs"], epochs=cfg["epochs"])
            vs.append(vp); ts.append(tp)
        oof[vi] = np.mean(vs, 0); tep += np.mean(ts, 0) / N_SPLITS
        print(f"  [{name}] fold {i+1}/{N_SPLITS} AUC={roc_auc_score(y[vi], oof[vi]):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.save(f"{OUT}/_partial_oof_{name}.npy", oof)
        np.save(f"{OUT}/_partial_test_{name}.npy", tep * N_SPLITS / (i + 1))
    auc = roc_auc_score(y, oof)
    results[name] = auc
    np.save(f"{OUT}/oof_{name}.npy", oof); np.save(f"{OUT}/test_{name}.npy", tep)
    print(f"  [{name}] >>> OOF AUC = {auc:.6f}\n", flush=True)

print("===== NN VARIANTS =====")
for k, v in results.items():
    print(f"  {k}: {v:.6f}")
