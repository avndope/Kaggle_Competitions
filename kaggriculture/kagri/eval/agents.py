"""Agents for local evaluation: load a Kaggle main.py like the hosted runner does, or replay a recorded tape."""
import copy
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
_CACHE = {}


def load_agent(path, fresh=False):
    """Exec a submission file (or a directory's main.py) and return its last callable, as Kaggle's
    get_last_callable does. Each call with fresh=True gets its own namespace, so two copies of the same agent
    in one game do not share module state."""
    if os.path.isdir(path):
        path = os.path.join(path, "main.py")
    path = os.path.abspath(path)
    if not fresh and path in _CACHE:
        return _CACHE[path]
    src = open(path, encoding="utf-8").read()
    d = os.path.dirname(path)
    ns = {"__name__": "agent_%x" % (id(src) ^ len(_CACHE)), "__file__": path}
    before = set(sys.modules)
    sys.path.insert(0, d)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(compile(src, path, "exec"), ns)
    finally:
        os.chdir(cwd)
        sys.path.remove(d)
    for m in set(sys.modules) - before:   # bundle-local modules must not leak into the next agent
        f = getattr(sys.modules.get(m), "__file__", "") or ""
        if f and os.path.dirname(os.path.abspath(f)) == d:
            del sys.modules[m]
    fns = [v for v in ns.values() if callable(v)]
    assert fns, "no callable in %s" % path
    fn = fns[-1]
    if not fresh:
        _CACHE[path] = fn
    return fn


class Tape:
    """Replays one seat's recorded actions; action i is the one submitted on observation step i."""

    def __init__(self, actions):
        self.actions = actions

    def __call__(self, obs, config=None):
        i = int(obs["step"])
        a = self.actions[i] if i < len(self.actions) else None
        return copy.deepcopy(a) if isinstance(a, dict) else dict(PASS)


def call(fn, obs, cfg):
    """Call an agent with (obs, config) or (obs); return (action, error flag)."""
    try:
        try:
            a = fn(obs, cfg)
        except TypeError:
            a = fn(obs)
        return (a, 0) if isinstance(a, dict) else (dict(PASS), 1)
    except Exception:  # noqa: BLE001
        return dict(PASS), 1
