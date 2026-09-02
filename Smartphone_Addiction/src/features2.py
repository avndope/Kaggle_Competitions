"""Feature groups, toggleable, so each can be ablated on its own.

The organising finding: this dataset rewards features that describe *how the generator
made the row*, and punishes features that describe the fictional behaviour it depicts.
Trees already find ratios and differences via splits, so precomputing them only eats the
feature-sampling fraction.

Groups
  structural : other_h = daily - (social+gaming+work). Exact constraint of the generator
               (>= 0 on all 421,427 complete rows).
  decimals   : how many decimal places each float column carries. This is a per-row
               measure of how hard the generator smeared that row -- rows whose
               daily_screen_time is a whole number obey the original hard source rule far
               more closely (rule=0 region: 0.148 positive vs 0.328 for other rows).
  behavioral : shares, ratios, per-open rates. Hand-built psychology. Measured negative.
  rule       : distance to the source rule's thresholds.
  weekend    : weekend-vs-weekday screen-time comparisons.
"""
import os
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("S6E8_DATA", os.path.join(_ROOT, "data"))
FLOATC = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
          "work_study_hours", "sleep_hours", "weekend_screen_time"]
ALL_GROUPS = ("structural", "decimals", "behavioral", "rule", "weekend")


def _ndec(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return -1
    s = str(s)
    if s == "nan" or s == "":
        return -1
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def load_decimals(which):
    """Decimal-place count per float column, read from the raw CSV text.

    Must come from the original strings: pandas has already collapsed 7.0 and 7.00 by the
    time it hands back a float, and that distinction is the whole signal.
    """
    raw = pd.read_csv(f"{DATA}/{which}.csv", dtype=str, usecols=FLOATC)
    return pd.DataFrame({c: raw[c].map(_ndec).astype("int8") for c in FLOATC})


def build(df, dec, groups=ALL_GROUPS):
    d = pd.DataFrame(index=df.index)
    dsh, smh = df.daily_screen_time_hours, df.social_media_hours
    gmh, wsh = df.gaming_hours, df.work_study_hours
    wke = df.weekend_screen_time

    if "structural" in groups:
        d["comp_sum"] = smh + gmh + wsh
        d["other_h"] = dsh - d.comp_sum
        d["nonsocial_h"] = dsh - gmh - wsh

    if "decimals" in groups:
        for c in FLOATC:
            d[f"ndec_{c}"] = dec[c].to_numpy()
        w = (dec[FLOATC] == 0)
        d["n_whole"] = w.sum(axis=1).to_numpy().astype("float32")
        d["whole_dsh"] = w["daily_screen_time_hours"].to_numpy().astype("float32")
        d["whole_slp"] = w["sleep_hours"].to_numpy().astype("float32")
        d["whole_gam"] = w["gaming_hours"].to_numpy().astype("float32")

    if "behavioral" in groups:
        d["social_share"] = smh / dsh
        d["gaming_share"] = gmh / dsh
        d["work_share"] = wsh / dsh
        d["notif_per_open"] = df.notifications_per_day / df.app_opens_per_day
        d["screen_per_open"] = dsh * 60 / df.app_opens_per_day
        d["awake_h"] = 24 - df.sleep_hours
        d["screen_share_awake"] = dsh / (24 - df.sleep_hours)

    if "rule" in groups:
        d["d_minus_8"] = dsh - 8.0
        d["s_minus_4"] = smh - 4.0
        d["rule_margin"] = np.maximum(dsh - 8.0, (smh - 4.0) * 2.0)

    if "weekend" in groups:
        d["wke_minus_dsh"] = wke - dsh
        d["wke_over_dsh"] = wke / dsh
        d["screen_mean"] = (dsh + wke) / 2
        d["screen_max"] = np.maximum(dsh, wke)

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return d
