"""Model-based imputation of the strong columns, then target-encode the imputed value.

Why this is worth trying at all: when daily_screen_time_hours is missing (13.9% of rows)
its target encoding falls back to the prior, so those rows lose the single most predictive
feature entirely. But daily is highly recoverable -- weekend_screen_time correlates 0.80
with it, and the generator's own constraint daily >= social+gaming+work bounds it from
below. Recover the value, snap it to the 0.01 lattice, and the per-value lookup becomes
available again for rows that previously had nothing.

Leakage: the imputers never see the target, so they can be fit once on train+test features
pooled. Only the *target encoding* of the recovered value has to stay inside the fold.
"""
import numpy as np, pandas as pd, lightgbm as lgb

IMPUTE_TARGETS = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"]


def fit_impute(tr, te, cols_all, targets=IMPUTE_TARGETS, seed=0):
    """Returns {col: (imputed_train, imputed_test, r2)} — value filled in everywhere,
    observed values preserved."""
    pool = pd.concat([tr[cols_all], te[cols_all]], ignore_index=True)
    n_tr = len(tr)
    out = {}
    for t in targets:
        feats = [c for c in cols_all if c != t]
        obs = pool[t].notna().to_numpy()
        X, yv = pool.loc[obs, feats], pool.loc[obs, t].to_numpy()
        cut = int(len(X) * 0.9)
        m = lgb.LGBMRegressor(objective="l2", learning_rate=0.06, num_leaves=96,
                              min_child_samples=100, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=1, n_estimators=600,
                              verbose=-1, n_jobs=-1, seed=seed)
        m.fit(X.iloc[:cut], yv[:cut], eval_set=[(X.iloc[cut:], yv[cut:])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = m.predict(pool[feats])
        ss_res = ((yv[cut:] - m.predict(X.iloc[cut:])) ** 2).sum()
        ss_tot = ((yv[cut:] - yv[cut:].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        filled = pool[t].to_numpy(dtype="float64").copy()
        filled[~obs] = pred[~obs]
        # respect the generator's hard constraint where the components are known
        if t == "daily_screen_time_hours":
            lo = (pool.social_media_hours.fillna(0) + pool.gaming_hours.fillna(0)
                  + pool.work_study_hours.fillna(0)).to_numpy()
            filled[~obs] = np.maximum(filled[~obs], lo[~obs])
        filled = np.round(filled, 2)          # snap back onto the 0.01 lattice
        out[t] = (filled[:n_tr], filled[n_tr:], r2)
    return out
