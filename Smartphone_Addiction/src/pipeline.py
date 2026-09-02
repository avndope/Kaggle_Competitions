"""Reusable leakage-safe fold builder for the final models.

Feature blocks
  raw   : the 9 numeric columns as-is  (the 3 smooth screen-time columns carry
          genuinely monotone signal, so the float view still matters)
  te    : out-of-fold target encoding of each EXACT lattice value + the 3 categoricals
          -- this is where the generator's per-value lookup table lives
  cnt   : how many training rows back each encoded value (lets the model discount
          thinly-supported levels)
  eng   : structural features, chiefly other_h = daily - (social + gaming + work)
  pair  : 2-way target encoding for column pairs
"""
import numpy as np, pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_features, CAT, NUM
from encoders import code_frame, oof_target_encode

PAIRS = [("notifications_per_day", "app_opens_per_day"),
         ("daily_screen_time_hours", "social_media_hours"),
         ("notifications_per_day", "sleep_hours")]


class FoldBuilder:
    def __init__(self, tr, te, y, use_eng=True, use_counts=True, use_pairs=True,
                 smooth=20.0, n_inner=10):
        self.tr, self.te, self.y = tr, te, y
        self.use_eng, self.use_counts, self.use_pairs = use_eng, use_counts, use_pairs
        self.smooth, self.n_inner = smooth, n_inner

        self.ctr = code_frame(tr, NUM, CAT)
        self.cte = code_frame(te, NUM, CAT)
        self.te_cols = NUM + list(CAT)
        if use_pairs:
            for a, b in PAIRS:
                k = f"{a}__X__{b}"
                self.ctr[k] = self.ctr[a] * 100003 + self.ctr[b]
                self.cte[k] = self.cte[a] * 100003 + self.cte[b]
                self.te_cols.append(k)

        if use_eng:
            self.etr, self.ete = add_features(tr, 1), add_features(te, 1)
            self.eng_cols = [c for c in self.etr.columns if c not in tr.columns]
        else:
            self.etr, self.ete, self.eng_cols = tr, te, []

    def build(self, ti, vi, seed=0):
        ptr, pva, pte = {}, {}, {}
        for c in self.te_cols:
            te_tr, cnt_tr, others = oof_target_encode(
                self.ctr[c][ti], self.y[ti], [self.ctr[c][vi], self.cte[c]],
                n_inner=self.n_inner, smooth=self.smooth, seed=seed)
            (te_va, cnt_va), (te_tst, cnt_tst) = others
            ptr[f"te_{c}"], pva[f"te_{c}"], pte[f"te_{c}"] = te_tr, te_va, te_tst
            if self.use_counts:
                ptr[f"cn_{c}"], pva[f"cn_{c}"], pte[f"cn_{c}"] = cnt_tr, cnt_va, cnt_tst

        base = NUM + self.eng_cols
        X = pd.concat([self.etr.iloc[ti][base].reset_index(drop=True), pd.DataFrame(ptr)], axis=1)
        V = pd.concat([self.etr.iloc[vi][base].reset_index(drop=True), pd.DataFrame(pva)], axis=1)
        T = pd.concat([self.ete[base].reset_index(drop=True), pd.DataFrame(pte)], axis=1)
        for d in (X, V, T):
            for c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).astype("float32")
        return X, V, T
