"""How much of the blend's OOF score is just the weight-fitting fitting itself?

The greedy selector chooses weights on the same 691k OOF rows it then reports a score on,
so that score is in-sample. With 13 candidates the optimism should be small, but the gains
being claimed are ~0.0001, so 'small' needs a number rather than an assumption.

Split the OOF in half: fit weights on A, score on B (and vice versa). The gap between the
in-sample and out-of-sample blend score is the optimism.
"""
import sys, glob, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from common import load, TARGET, OUT

tr, _ = load()
y = tr[TARGET].values.astype("int64")

models = {}
for f in sorted(glob.glob(f"{OUT}/oof_*.npy")):
    name = os.path.basename(f)[4:-4]
    if not os.path.exists(f"{OUT}/test_{name}.npy"):
        continue
    o = np.load(f)
    if len(o) == len(y) and np.isfinite(o).all():
        models[name] = rankdata(o) / len(o)
print(f"candidates: {len(models)}")


def greedy(idx, names, init_k=5, n_iter=80):
    best0 = max(names, key=lambda k: roc_auc_score(y[idx], models[k][idx]))
    cur = models[best0][idx].copy()
    counts = {k: 0 for k in names}; counts[best0] = init_k; n = init_k
    best, bestc = roc_auc_score(y[idx], cur), dict(counts)
    for _ in range(n_iter):
        sc, bk = -1, None
        for k in names:
            s = roc_auc_score(y[idx], (cur * n + models[k][idx]) / (n + 1))
            if s > sc: sc, bk = s, k
        cur = (cur * n + models[bk][idx]) / (n + 1); n += 1; counts[bk] += 1
        if sc > best + 1e-9:
            best, bestc = sc, dict(counts)
    tot = sum(bestc.values())
    return {k: v / tot for k, v in bestc.items() if v}, best


names = list(models)
skf = StratifiedKFold(2, shuffle=True, random_state=0)
ins, outs, singles = [], [], []
for a, b in skf.split(np.zeros(len(y)), y):
    w, in_score = greedy(a, names)
    pred_b = sum(wt * models[k][b] for k, wt in w.items())
    out_score = roc_auc_score(y[b], pred_b)
    best_single_b = max(roc_auc_score(y[b], models[k][b]) for k in names)
    ins.append(in_score); outs.append(out_score); singles.append(best_single_b)
    print(f"  fit on half A: in-sample {in_score:.6f} | held-out half {out_score:.6f} "
          f"| best single on that half {best_single_b:.6f}")

print(f"\n  mean in-sample blend      = {np.mean(ins):.6f}")
print(f"  mean out-of-sample blend  = {np.mean(outs):.6f}")
print(f"  WEIGHT-FITTING OPTIMISM   = {np.mean(ins)-np.mean(outs):+.6f}")
print(f"  blend gain that survives  = {np.mean(outs)-np.mean(singles):+.6f}  "
      f"(vs best single on the same rows)")
json.dump({"in_sample": float(np.mean(ins)), "out_of_sample": float(np.mean(outs)),
           "optimism": float(np.mean(ins) - np.mean(outs)),
           "surviving_gain": float(np.mean(outs) - np.mean(singles))},
          open(f"{OUT}/blend_honesty.json", "w"), indent=2)
