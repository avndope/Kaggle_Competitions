"""Download Kaggle's official daily top-episode datasets (kaggle/kaggriculture-episodes-YYYY-MM-DD).

    python scripts/download_episodes.py --days 2026-10-01:2026-10-13 [--out data/raw]
    python scripts/download_episodes.py --list

Each day is one zip (~850 MB) of full replay JSONs plus manifest.csv (episodes and agent ratings). Zips are kept
as they are and read member by member later (scripts/build_dataset.py), so nothing is unpacked to disk.
Authentication: the Kaggle access token in ~/.kaggle/access_token or $KAGGLE_TOKEN_FILE (Bearer token).
"""
import argparse
import csv
import datetime
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://www.kaggle.com/api/v1"
TOKEN_PATHS = [os.environ.get("KAGGLE_TOKEN_FILE", ""), os.path.expanduser("~/.kaggle/access_token")]


def token():
    for p in TOKEN_PATHS:
        if p and os.path.exists(p):
            return open(p).read().strip()
    sys.exit("no Kaggle access token found (set KAGGLE_TOKEN_FILE)")


def request(url, tok):
    return urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})


def list_days(tok):
    """manifest.csv of the index dataset: one row per published day."""
    url = API + "/datasets/download/kaggle/kaggriculture-episodes-index/manifest.csv"
    raw = urllib.request.urlopen(request(url, tok), timeout=120).read()
    if raw[:2] == b"PK":                       # served zipped
        import zipfile
        raw = zipfile.ZipFile(io.BytesIO(raw)).read("manifest.csv")
    return list(csv.DictReader(io.StringIO(raw.decode())))


def download(day, out, tok, retries=5):
    path = os.path.join(out, "%s.zip" % day)
    if os.path.exists(path) and os.path.getsize(path) > 1e6:
        return path, "exists"
    url = API + "/datasets/download/kaggle/kaggriculture-episodes-%s" % day
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request(url, tok), timeout=600) as r, open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + ".part", path)
            return path, "ok %.0f MB" % (os.path.getsize(path) / 1e6)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            time.sleep(10 * (attempt + 1))
            err = e
    return None, "failed: %r" % err


def day_range(spec):
    a, b = spec.split(":") if ":" in spec else (spec, spec)
    d, end = datetime.date.fromisoformat(a), datetime.date.fromisoformat(b)
    while d <= end:
        yield d.isoformat()
        d += datetime.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", help="YYYY-MM-DD or FROM:TO (inclusive)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    tok = token()
    if a.list or not a.days:
        rows = list_days(tok)
        print(json.dumps(rows[:3], indent=1), "...", len(rows), "days")
        return
    os.makedirs(a.out, exist_ok=True)
    for day in day_range(a.days):
        path, status = download(day, a.out, tok)
        print(day, status, flush=True)


if __name__ == "__main__":
    main()
