"""Shared data loading, feature engineering and CV harness for PS S6E8."""
import numpy as np, pandas as pd, time, os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("S6E8_DATA", os.path.join(_ROOT, "data"))
OUT  = os.environ.get("S6E8_OUT",  os.path.join(_ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

TARGET = "addicted_label"
CAT = ["gender", "stress_level", "academic_work_impact"]
NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
RAW = NUM + CAT


def load():
    tr = pd.read_csv(f"{DATA}/train.csv")
    te = pd.read_csv(f"{DATA}/test.csv")
    return tr, te


def add_features(df, level=1):
    """level 0 = raw only. Higher levels add progressively more engineered features."""
    d = df.copy()
    if level == 0:
        return d

    dsh = d.daily_screen_time_hours
    smh = d.social_media_hours
    gmh = d.gaming_hours
    wsh = d.work_study_hours
    wke = d.weekend_screen_time

    # --- structural: components sum never exceeds the daily total ---
    d["comp_sum"] = smh + gmh + wsh                 # accounted-for screen time
    d["other_h"]  = dsh - d.comp_sum                # unaccounted residual  (>= 0 always)
    d["nonsocial_h"] = dsh - gmh - wsh              # proxy for social + other (used when smh missing)

    # --- shares (composition of the day) ---
    d["social_share"] = smh / dsh
    d["gaming_share"] = gmh / dsh
    d["work_share"]   = wsh / dsh
    d["other_share"]  = d.other_h / dsh

    # --- weekend vs weekday: second noisy measurement of latent screen time ---
    d["wke_minus_dsh"] = wke - dsh
    d["wke_over_dsh"]  = wke / dsh
    d["screen_mean"]   = (dsh + wke) / 2
    d["screen_max"]    = np.maximum(dsh, wke)

    # --- the generator's underlying rule lives on (dsh, smh) ---
    d["rule_pos"] = ((dsh > 8) | (smh > 4)).astype("float32")
    d["rule_neg"] = ((dsh <= 6) & (smh <= 4)).astype("float32")
    d["d_minus_8"] = dsh - 8.0
    d["s_minus_4"] = smh - 4.0
    d["rule_margin"] = np.maximum(dsh - 8.0, (smh - 4.0) * 2.0)

    # --- intensity / behaviour ---
    d["notif_per_open"] = d.notifications_per_day / d.app_opens_per_day
    d["screen_per_open"] = dsh * 60 / d.app_opens_per_day
    d["awake_h"] = 24 - d.sleep_hours
    d["screen_share_awake"] = dsh / d.awake_h

    # --- missingness ---
    d["n_na"] = df[RAW].isna().sum(axis=1).astype("float32")
    if level >= 2:
        for c in RAW:
            d[f"na_{c}"] = df[c].isna().astype("float32")

    return d


def prep(tr, te, level=1):
    trf = add_features(tr, level)
    tef = add_features(te, level)
    feats = [c for c in trf.columns if c not in ("id", TARGET)]
    for c in CAT:
        cats = sorted(set(trf[c].dropna().unique()) | set(tef[c].dropna().unique()))
        dt = pd.CategoricalDtype(categories=cats)
        trf[c] = trf[c].astype(dt)
        tef[c] = tef[c].astype(dt)
    # keep only finite floats (ratios can produce inf when denominator is 0)
    for f in feats:
        if f not in CAT:
            trf[f] = pd.to_numeric(trf[f], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            tef[f] = pd.to_numeric(tef[f], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return trf, tef, feats


def cv_run(fit_predict, X, y, Xte=None, n_splits=5, seed=42, name="model", verbose=True):
    """fit_predict(Xtr,ytr,Xva,yva,Xte) -> (va_pred, te_pred or None)"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    tep = np.zeros(len(Xte)) if Xte is not None else None
    scores = []
    t0 = time.time()
    for i, (ti, vi) in enumerate(skf.split(X, y)):
        vp, tp = fit_predict(X.iloc[ti], y[ti], X.iloc[vi], y[vi], Xte)
        oof[vi] = vp
        if tp is not None:
            tep += tp / n_splits
        s = roc_auc_score(y[vi], vp)
        scores.append(s)
        if verbose:
            print(f"  [{name}] fold {i+1}/{n_splits}  AUC={s:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    full = roc_auc_score(y, oof)
    if verbose:
        print(f"  [{name}] OOF AUC = {full:.6f}   folds {np.mean(scores):.6f} +/- {np.std(scores):.6f}", flush=True)
    return oof, tep, full
