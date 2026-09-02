"""Rank-average blender with greedy forward selection on OOF AUC.

Rank space, not probability space: the model families here put their outputs on visibly
different scales, and AUC only cares about ordering, so averaging ranks removes a
nuisance degree of freedom that arithmetic averaging has to fight.

Greedy forward selection with replacement (Caruana) rather than a continuous weight
solve: it cannot pick negative weights, it stops adding a model the moment it stops
helping, and on a saturated leaderboard that conservatism is worth more than the last
1e-5 of OOF fit.
"""
import sys, glob, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from common import load, TARGET, OUT

tr, te = load()
y = tr[TARGET].values.astype("int64")

EXCLUDE = {"blend"}
oof_files = sorted(glob.glob(f"{OUT}/oof_*.npy"))
models = {}
for f in oof_files:
    name = os.path.basename(f)[4:-4]
    tf = f"{OUT}/test_{name}.npy"
    if name in EXCLUDE or not os.path.exists(tf):
        continue
    o, t = np.load(f), np.load(tf)
    if len(o) != len(tr) or len(t) != len(te) or not np.isfinite(o).all():
        print(f"  skip {name} (shape/finite mismatch)"); continue
    models[name] = (rankdata(o) / len(o), rankdata(t) / len(t), roc_auc_score(y, o))

print("=== candidates ===")
for k, (_, _, s) in sorted(models.items(), key=lambda x: -x[1][2]):
    print(f"  {k:26s} OOF AUC = {s:.6f}")

if not models:
    raise SystemExit("no candidate predictions found")

# ---- greedy forward selection with replacement (Caruana) ----
# Seeded with INIT_K copies of the best single model so the first increment is worth
# 1/(INIT_K+1) rather than 1/2. A partner that helps at weight 0.1 but hurts at 0.5 is
# the normal case among highly-correlated candidates, and a coarse start rejects it.
# Run a fixed budget and keep the best prefix rather than stopping on the first miss.
names = list(models)
INIT_K, N_ITER = 5, 80
best_k0 = max(names, key=lambda k: models[k][2])
cur = models[best_k0][0].copy()
counts = {k: 0 for k in names}; counts[best_k0] = INIT_K; n = INIT_K
best_overall, best_counts = models[best_k0][2], dict(counts)
print(f"  init: {INIT_K}x {best_k0}  OOF = {best_overall:.6f}")

for it in range(N_ITER):
    best_s, best_k = -1, None
    for k in names:
        s = roc_auc_score(y, (cur * n + models[k][0]) / (n + 1))
        if s > best_s:
            best_s, best_k = s, k
    cur = (cur * n + models[best_k][0]) / (n + 1)
    n += 1; counts[best_k] += 1
    if best_s > best_overall + 1e-9:
        best_overall, best_counts = best_s, dict(counts)
        print(f"  iter {it+1:2d}: +{best_k:24s} OOF = {best_s:.6f}  *")
    elif it < 12 or (it + 1) % 10 == 0:
        print(f"  iter {it+1:2d}: +{best_k:24s} OOF = {best_s:.6f}")

counts = best_counts
n = sum(counts.values())
weights = {k: v / n for k, v in counts.items() if v}
final_oof = sum(w * models[k][0] for k, w in weights.items())
final_te = sum(w * models[k][1] for k, w in weights.items())
score = roc_auc_score(y, final_oof)

print("\n=== final blend ===")
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k:26s} weight {w:.3f}")
print(f"  blend OOF AUC = {score:.6f}")
best_single = max(models.items(), key=lambda x: x[1][2])
print(f"  best single   = {best_single[1][2]:.6f} ({best_single[0]})")
print(f"  gain over best single = {score - best_single[1][2]:+.6f}")

sub = pd.DataFrame({"id": te["id"], TARGET: final_te / final_te.max()})
path = f"{OUT}/submission.csv"
sub.to_csv(path, index=False)
json.dump({"weights": weights, "oof_auc": score,
           "per_model": {k: v[2] for k, v in models.items()}},
          open(f"{OUT}/blend_report.json", "w"), indent=2, default=float)
print(f"\n  wrote {path}  ({len(sub)} rows)")
print(sub.head().to_string(index=False))
