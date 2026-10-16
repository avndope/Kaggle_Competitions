"""Daily replay zips -> episode / label caches + census.jsonl + vocab.json + report.json.

    python scripts/build_dataset.py --zips data/raw/2026-10-*.zip --out data/bc [--procs 12] [--min-avg-score 0]

For every replay (read straight from the zip, nothing unpacked):
  1. keep it only with 720 steps, engine 1.32.7 and no ERROR / TIMEOUT / INVALID status;
  2. encode it (kagri/bc/extract.py, DECEM's lossless format);
  3. re-simulate it on the official engine (kagri/bc/resim.py): the game must reproduce the recording at every step,
     and the engine's execution record becomes the market labels;
  4. label it (kagri/bc/labels.py with the execution record).
Seats that replay a byte-identical action stream in two or more episodes are open-loop tapes (DECEM): they are
masked. A team gets its own conditioning embedding with at least --min-seats trainable seats.
Census rows match kagri/bc/dataset.py build_index (id, day, teams, rewards, mask, seat_days) plus extras.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kagri" / "bc"))
from extract import extract  # noqa: E402
from labels import make_labels  # noqa: E402
from resim import resim  # noqa: E402

BAD = ("ERROR", "TIMEOUT", "INVALID")
VERSIONS = ("1.32.7", "1.33.0")
MIN_BANK = 5000.0   # a side ending near the 3,000 start crashed or idled (e.g. version-check failures, 2-6 Oct)


def work(job):
    zpath, member, day, ep_dir, lab_dir, avg = job
    eid = int(member.split("/")[-1].split(".")[0])
    ep_path, lab_path = os.path.join(ep_dir, "%d.npz" % eid), os.path.join(lab_dir, "%d.npz" % eid)
    try:
        if os.path.exists(ep_path) and os.path.exists(lab_path):   # cached: metadata is stored in the label file
            lab = np.load(lab_path)
            if "meta" in lab.files:
                row = json.loads(str(lab["meta"]))
                row.update(day=day, avg_score=avg)
                if min(float(x) for x in row["rewards"]) < MIN_BANK:
                    return row, "passive_or_crashed_side"
                return row, "cached" if int(lab["parity"]) < 0 else "not_reproduced"
        with zipfile.ZipFile(zpath) as z:
            d = json.loads(z.read(member))
        st = d.get("steps", [])
        teams = d["info"].get("TeamNames") or []
        row = dict(id=eid, day=day, seed=d["info"].get("seed"), teams=teams, rewards=d.get("rewards"),
                   statuses=d.get("statuses"), avg_score=avg)
        if len(st) != 720 or len(teams) != 2:
            return row, "incomplete"
        # 1.33.0 (2-6 Oct) was a version-number-only bump of the same engine (Kaggle staff, discussion 745445);
        # such games are kept only if they reproduce exactly on the 1.32.7 engine (parity check below).
        if d.get("module_version") not in VERSIONS:
            return row, "other_version"
        if any(a.get("status") in BAD for s in st for a in s):
            return row, "error_status"
        if min(float(x) for x in d.get("rewards") or [0, 0]) < MIN_BANK:
            return row, "passive_or_crashed_side"
        if os.path.exists(ep_path) and os.path.exists(lab_path):
            lab = np.load(lab_path)
            row.update(parity=int(lab["parity"]), req_sell=lab["req_sell_tot"].tolist(), exec_sell=lab["exec_sell_tot"].tolist())
            return row, "cached" if int(lab["parity"]) < 0 else "not_reproduced"
        ep = extract(d)
        ex = resim(d)
        row["parity"] = int(ex["parity"])
        if int(ex["parity"]) >= 0:
            return row, "not_reproduced"
        lab = make_labels(ep, ex)
        lab["x_unit"] = ex["x_unit"]
        lab["parity"] = ex["parity"]
        lab["req_sell_tot"] = ex["x_req_sell"].sum(0).astype(np.int64)
        lab["exec_sell_tot"] = ex["x_sell"].sum((0, 2)).astype(np.int64)
        row.update(req_sell=lab["req_sell_tot"].tolist(), exec_sell=lab["exec_sell_tot"].tolist())
        lab["meta"] = np.array(json.dumps({k: row[k] for k in ("id", "seed", "teams", "rewards", "statuses", "parity",
                                                                "req_sell", "exec_sell")}))
        for path, arr in ((ep_path, ep), (lab_path, lab)):
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **arr)
            os.replace(tmp, path)
        return row, "ok"
    except Exception as e:  # noqa: BLE001
        return dict(id=eid, day=day), "error %r" % (e,)


def seat_hash(ep_path):
    a = np.load(ep_path)
    return [hashlib.sha1(a["uact"][:, s].tobytes() + a["mact"][:, s].tobytes() + a["nord"][:, s].tobytes()).hexdigest()
            for s in range(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--out", default="data/bc")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--min-avg-score", type=float, default=0.0, help="skip games whose average agent rating is lower")
    ap.add_argument("--exclude-teams", nargs="*", default=[], help="never train these teams' seats (e.g. your own)")
    ap.add_argument("--train-teams", nargs="*", default=[], help="if given, train only these teams' seats")
    ap.add_argument("--min-seats", type=int, default=50)
    ap.add_argument("--vocab", default=None, help="reuse a vocabulary (stage 2 / 3 of a run)")
    ap.add_argument("--keep-tapes", action="store_true")
    a = ap.parse_args()
    out, ep_dir, lab_dir = Path(a.out), Path(a.cache) / "ep", Path(a.cache) / "lab"
    for p in (out, ep_dir, lab_dir):
        p.mkdir(parents=True, exist_ok=True)
    jobs = []
    for zp in sorted(a.zips):
        day = Path(zp).stem
        with zipfile.ZipFile(zp) as z:
            man = {}
            if "manifest.csv" in z.namelist():
                man = {r["episode_id"]: float(r["avg_score"]) for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))}
            for m in z.namelist():
                if not m.endswith(".json"):
                    continue
                avg = man.get(m.split("/")[-1].split(".")[0])
                if avg is not None and avg < a.min_avg_score:
                    continue
                jobs.append((zp, m, day, str(ep_dir), str(lab_dir), avg))
    print("%d replays in %d zips" % (len(jobs), len(a.zips)), flush=True)
    t0 = time.time()
    rows, stat = [], collections.Counter()
    with Pool(a.procs, maxtasksperchild=50) as pool:
        for i, (row, status) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            stat[status.split(" ")[0]] += 1
            if status in ("ok", "cached"):
                rows.append(row)
            elif status.startswith("error"):
                print("  ", row.get("id"), status[:200], flush=True)
            if (i + 1) % 200 == 0:
                print("  %d/%d  %.0fs  %s" % (i + 1, len(jobs), time.time() - t0, dict(stat)), flush=True)
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: (r["day"], r["id"])):
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    rows = uniq
    excl, only = set(a.exclude_teams), set(a.train_teams)
    tape_seats = set()
    if not a.keep_tapes:
        with Pool(a.procs) as pool:
            hs = pool.map(seat_hash, [str(ep_dir / ("%d.npz" % r["id"])) for r in rows], chunksize=16)
        cnt = collections.Counter((r["teams"][s], h[s]) for r, h in zip(rows, hs) for s in range(2))
        tape_seats = {(r["id"], s) for r, h in zip(rows, hs) for s in range(2) if cnt[(r["teams"][s], h[s])] >= 2}
    for r in rows:
        r["mask"] = [t not in excl and (not only or t in only) and (r["id"], s) not in tape_seats
                     for s, t in enumerate(r["teams"])]
        r["seat_days"] = [r["day"], r["day"]]
    stat["open_loop_tape_seats"] = len(tape_seats)
    rows = [r for r in rows if any(r["mask"])]
    if a.vocab:
        vocab = json.load(open(a.vocab))
    else:
        seats = collections.Counter(t for r in rows for s, t in enumerate(r["teams"]) if r["mask"][s])
        vocab = ["<other>"] + sorted(t for t, n in seats.items() if n >= a.min_seats)
    json.dump(vocab, open(out / "vocab.json", "w"), ensure_ascii=False)
    (out / "census.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    trained = collections.Counter(t for r in rows for s, t in enumerate(r["teams"]) if r["mask"][s])
    wins = collections.Counter(r["teams"][s] for r in rows for s in range(2)
                               if r["mask"][s] and r["rewards"][s] > r["rewards"][1 - s])
    req = sum(sum(r.get("req_sell", [0, 0])) for r in rows)
    exe = sum(sum(r.get("exec_sell", [0, 0])) for r in rows)
    report = dict(replays=len(jobs), episodes=len(rows), status=dict(stat), seconds=round(time.time() - t0),
                  requested_vs_executed_sells=[req, exe], vocab=len(vocab),
                  days=dict(sorted(collections.Counter(r["day"] for r in rows).items())),
                  seats_by_team={t: [n, wins[t]] for t, n in trained.most_common()})
    json.dump(report, open(out / "report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "seats_by_team"}, indent=1, ensure_ascii=False))
    print("top teams (seats, wins):", list(report["seats_by_team"].items())[:15])


if __name__ == "__main__":
    main()
