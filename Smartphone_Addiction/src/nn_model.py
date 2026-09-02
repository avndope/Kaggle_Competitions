"""Entity-embedding MLP.

The generator's per-value lookup table is exactly what an embedding table represents:
one free parameter vector per exact lattice value, no splitting required. That makes this
model structurally different from the GBDTs -- it reads the table directly and learns the
smooth part in the dense layers -- so its errors decorrelate from theirs and it earns its
place in the blend rather than just adding another boosted tree.

Missing values are not imputed: each column gets a dedicated "missing" level, so the
network learns an embedding for absence the same way it learns one for any other value.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.metrics import roc_auc_score

NA_LEVEL = 0  # index 0 of every embedding table is reserved for NaN


def make_levels(tr, te, cols, steps):
    """Map each column's exact values to contiguous ids shared by train and test."""
    maps, sizes = {}, {}
    for c in cols:
        pool = pd.concat([tr[c], te[c]])
        v = np.round(pool.to_numpy(dtype="float64") / steps[c])
        u = np.unique(v[~np.isnan(v)])
        maps[c] = {k: i + 1 for i, k in enumerate(u)}
        sizes[c] = len(u) + 1
    return maps, sizes


def encode(df, cols, steps, maps):
    out = np.zeros((len(df), len(cols)), dtype="int64")
    for j, c in enumerate(cols):
        v = np.round(df[c].to_numpy(dtype="float64") / steps[c])
        out[:, j] = pd.Series(v).map(maps[c]).fillna(NA_LEVEL).astype("int64").to_numpy()
    return out


def encode_cat(df, cols, maps):
    out = np.zeros((len(df), len(cols)), dtype="int64")
    for j, c in enumerate(cols):
        out[:, j] = df[c].map(maps[c]).fillna(NA_LEVEL).astype("int64").to_numpy()
    return out


class EmbMLP(nn.Module):
    def __init__(self, sizes, n_cont, hidden=(512, 256, 128), dropout=0.15, emb_cap=24,
                 act="silu"):
        super().__init__()
        dims = [min(emb_cap, max(4, (s + 1) // 3)) for s in sizes]
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, d in zip(sizes, dims)])
        self.emb_drop = nn.Dropout(0.05)
        self.bn_cont = nn.BatchNorm1d(n_cont) if n_cont else None
        d_in = sum(dims) + n_cont
        Act = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[act]
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), Act(), nn.Dropout(dropout)]
            d_in = h
        layers += [nn.Linear(d_in, 1)]
        self.mlp = nn.Sequential(*layers)
        for e in self.embs:
            nn.init.normal_(e.weight, 0, 0.05)

    def forward(self, xc, xn):
        z = [e(xc[:, i]) for i, e in enumerate(self.embs)]
        z = self.emb_drop(torch.cat(z, 1))
        if self.bn_cont is not None:
            z = torch.cat([z, self.bn_cont(xn)], 1)
        return self.mlp(z).squeeze(1)


def train_nn(Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, Xc_te, Xn_te, sizes,
             epochs=18, bs=4096, lr=3e-3, wd=1e-5, seed=0, verbose=True, device=None,
             arch=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbMLP(sizes, Xn_tr.shape[1], **(arch or {})).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_steps = int(np.ceil(len(y_tr) / bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps, pct_start=0.25)
    lossf = nn.BCEWithLogitsLoss()

    # Whole dataset lives on the device: ~150MB of int64/float32 against 4GB of VRAM, so
    # there is no reason to pay per-batch host->device copies.
    to = lambda a: torch.from_numpy(a).to(dev)
    tc, tn, ty = to(Xc_tr), to(Xn_tr), to(y_tr.astype("float32"))
    vc, vn = to(Xc_va), to(Xn_va)
    sc, sn = to(Xc_te), to(Xn_te)

    @torch.no_grad()
    def predict(c, n_):
        model.eval()
        return torch.sigmoid(torch.cat([model(c[i:i+32768], n_[i:i+32768])
                                        for i in range(0, len(c), 32768)])).float().cpu().numpy()

    best, best_va, best_te, bad = -1, None, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ty), device=dev)
        t0 = time.time()
        for i in range(0, len(ty), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(tc[idx], tn[idx]), ty[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        pv = predict(vc, vn)
        a = roc_auc_score(y_va, pv)
        if verbose:
            print(f"      ep{ep+1:02d} val_auc={a:.6f} ({time.time()-t0:.0f}s)", flush=True)
        if a > best:
            best, bad = a, 0
            best_te, best_va = predict(sc, sn), pv
        else:
            bad += 1
            if bad >= 4:
                break
    return best_va, best_te, best
