"""Assemble a v13 bundle (flat directory, the layout tools/package_dir_agent.py expects).

    python3 build_v13.py OUT_DIR [--exploit] [--base PATH] [--cands N]

Contents: main.py (with _BUNDLE_SHA stamped), shadow.py, exploit.py (if --exploit),
engine_v1327.py (verbatim official engine, SHA-checked), base_v12u.py, cand_NN_<slug>.py,
v13_manifest.json. Candidates are listed in priority order below; --cands N keeps the first N.
"""
import argparse
import hashlib
import json
import os
import shutil

K = "/mnt/c/Users/avnbe/Documents/Claude/Avineesh/Upskilling/Kaggle/kaggriculture"
SRC = os.path.join(K, "scratch", "v13", "src")
ENGINE = os.path.join(K, ".venv", "Lib", "site-packages", "kaggle_environments", "envs",
                      "kaggriculture", "kaggriculture.py")
ENGINE_SHA = "bc8a54879ef02c7e"
V = os.path.join(K, "scratch", "v12")

# Priority = how likely the agent (or an exact copy) is to sit across from us at 2400+.
# v13.1 (2026-09-25): ladder agent logs showed hits on cha22 (x6), harvest-ledger and
# 7-turn-rescue, so those lead; the four versions published after the notebook lock follow.
CANDIDATES = [
    # v13.2 (2026-09-27): the 27 Sep master-engine-v4 / harvest-ledger version beats v12u 75% head-to-head.
    # v13.3 / v13.3b (2026-09-28) tried adding hl0927 forks and newer lineage versions; every addition
    # co-survives with hl-0927 for part of a game, the maximin exploit hedges, and the margin against the
    # far more common exact hl0927 drops (100% -> 95.8%). So the list below is the shipped v13.2 list.
    ("hl-0927", "agents_0927/haodou092--kaggriculture-harvest-ledger.py"),
    ("cha22", "agents_0924b/abhinav0370--cha22-agent.py"),
    ("harvest-ledger-0925", "agents_0925/haodou092--kaggriculture-harvest-ledger.py"),
    ("harvest-ledger", "agents_0924b/haodou092--kaggriculture-harvest-ledger.py"),
    ("7-turn-rescue", "agents_recent/dmitriigluzdov--kaggriculture-7-turn-rescue-historical-lb-2800.py"),
    ("master-engine-v4-0925", "agents_0925/guruprasaathas111--kaggriculture-top-2-master-engine-v4.py"),
    ("shepherds-0925", "agents_0925/haideptry--the-shepherds-ledger-herd-safe-sovereign.py"),
    ("evgendvorkin-0925", "agents_0925/evgendvorkin--kaggriculture.py"),
    ("one-more-wheat", "agents_recent/dmitriigluzdov--kaggriculture-one-more-wheat.py"),
    ("smarter-sales", "agents_recent/dmitriigluzdov--kaggriculture-more-wheat-smarter-sales.py"),
    ("v57", "agents_recent/ahmedberatozer--kaggriculture-v57-funding-order-invariant.py"),
    ("v56", "agents_recent/ahmedberatozer--kaggriculture-v56-smarter-seeds-and-fertilizer.py"),
    ("v54", "agents_recent/ahmedberatozer--kaggriculture-v54-productive-wheat-and-patient.py"),
    ("busyaprime-seedleak", "agents_recent/busyaprime--92-vs-the-best-public-farm-a-seed-leak-in-v54.py"),
    ("busyaprime-race", "agents_recent/busyaprime--v54-fork-wins-72-5-head-to-head-against-v54.py"),
    ("herdsafe-v3", "agents_0923/arsgorynich--herd-safe-v3-experimental-risk-aware-feed.py"),
    ("master-engine-v4", "agents_recent/guruprasaathas111--kaggriculture-top-2-master-engine-v4.py"),
    ("metav4-v13", "agents_recent/thomastschinkel--the-metav4-farm-submission-v13.py"),
    ("hybrid-2965", "agents_0924/haideptry--the-2965-master-hybrid-engine.py"),
    ("jaxa623-2802", "league_v3/jaxa623-2802-two-identical-agents-90.py"),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--exploit", action="store_true")
    ap.add_argument("--base", default=os.path.join(K, "submission", "main_v12u.py"))
    ap.add_argument("--cands", type=int, default=len(CANDIDATES))
    ap.add_argument("--only", default=None, help="comma list of candidate names to keep")
    args = ap.parse_args()

    assert sha(ENGINE).startswith(ENGINE_SHA), "engine is not kaggle-environments 1.32.7"
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)
    shutil.copy(ENGINE, os.path.join(args.out, "engine_v1327.py"))
    shutil.copy(os.path.join(os.path.dirname(ENGINE), "kaggriculture.json"), args.out)
    shutil.copy(os.path.join(SRC, "shadow.py"), args.out)
    if args.exploit:
        shutil.copy(os.path.join(SRC, "exploit.py"), args.out)
    shutil.copy(args.base, os.path.join(args.out, "base_v12u.py"))

    chosen = CANDIDATES[:args.cands]
    if args.only:
        keep = set(args.only.split(","))
        chosen = [c for c in CANDIDATES if c[0] in keep]
    seen, cands = set(), []
    for i, (name, rel) in enumerate(chosen):
        src = os.path.join(V, rel)
        h = sha(src)
        if h in seen:                       # identical agents would be the same hypothesis
            continue
        seen.add(h)
        fname = "cand_%02d_%s.py" % (i, name)
        shutil.copy(src, os.path.join(args.out, fname))
        cands.append({"name": name, "file": fname, "sha": h[:16]})
    manifest = {"base": "base_v12u.py", "exploit": bool(args.exploit), "candidates": cands,
                "load_first": 0.3, "load_budget": 0.25, "min_step": 96, "min_streak": 24}
    json.dump(manifest, open(os.path.join(args.out, "v13_manifest.json"), "w"), indent=1)

    main_src = open(os.path.join(SRC, "main.py"), encoding="utf-8").read()
    h = hashlib.sha256()
    for f in sorted(os.listdir(args.out)):
        h.update(f.encode())
        h.update(open(os.path.join(args.out, f), "rb").read())
    h.update(main_src.encode())
    bundle_sha = h.hexdigest()[:16]
    main_src = main_src.replace('_BUNDLE_SHA = "unset"', '_BUNDLE_SHA = "%s"' % bundle_sha)
    open(os.path.join(args.out, "main.py"), "w", encoding="utf-8").write(main_src)
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print("built %s: %d candidates, exploit=%s, bundle %s, %.1f MB"
          % (args.out, len(cands), args.exploit, bundle_sha, total / 1e6))


if __name__ == "__main__":
    main()
