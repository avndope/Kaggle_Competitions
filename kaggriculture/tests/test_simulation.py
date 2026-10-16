"""Simulation confidence checks. Run: .venv/bin/python -m pytest tests -q   (or python tests/test_simulation.py)

1. The vendored engine is byte-identical to kaggle-environments 1.32.7's.
2. Re-simulating recorded top games with both recorded action streams reproduces every step (shared state and
   both private states) and the final money exactly.
3. Pinning a recorded game's shops changes nothing for the recorded actions.
4. Re-simulating takes our engine's execution record: executed sells never exceed requested ones.
"""
import glob
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kagri", "bc"))

from kagri.engine.runner import ENGINE_PATH, Game  # noqa: E402

ENGINE_SHA256_PREFIX = "bc8a54879ef02c7e"   # kaggle_environments/envs/kaggriculture/kaggriculture.py, 1.32.7


def _replays(n=3):
    zips = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.zip")))
    if not zips:
        return []
    z = zipfile.ZipFile(zips[-1])
    names = [m for m in z.namelist() if m.endswith(".json")][:n]
    return [json.loads(z.read(m)) for m in names]


def test_engine_is_official():
    h = hashlib.sha256(open(ENGINE_PATH, "rb").read()).hexdigest()
    assert h.startswith(ENGINE_SHA256_PREFIX), h


def test_recorded_games_reproduce_exactly():
    from resim import resim
    reps = _replays()
    if not reps:
        print("skip: no replay zips in data/raw")
        return
    for rep in reps:
        ex = resim(rep)
        assert int(ex["parity"]) == -1, "episode %s diverged at step %d" % (rep["info"]["EpisodeId"], int(ex["parity"]))
        assert (ex["x_sell"].sum(-1) <= ex["x_req_sell"]).all()


def test_pinned_shops_change_nothing():
    for rep in _replays(2):
        steps = rep["steps"]
        g = Game(int(rep["info"]["seed"]), shops=steps[-1][0]["observation"]["town"]["unlocked_shops"])
        while not g.done:
            t = g.step_no
            g.step([steps[t + 1][s].get("action") or {} for s in (0, 1)])
        assert g.rewards() == [float(x) for x in rep["rewards"]]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
