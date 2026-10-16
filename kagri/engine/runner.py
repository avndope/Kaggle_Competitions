"""Exact local runner for Kaggriculture on the verbatim official engine (kaggle-environments 1.32.7).

`kaggriculture_v1327.py` is the unmodified engine file from kaggle-environments 1.32.7 (Apache-2.0, see
LICENSE-kaggle-environments). It is exec'd into a private namespace; the only substitution is the
`resolve_episode_seed` import, replaced by an exact copy of that helper, so no kaggle_environments install is needed.

    g = Game(seed=123)                        # same world as the hosted episode with this seed
    while not g.done:
        g.step([agent0(g.observe(0)), agent1(g.observe(1))])
    g.rewards()                               # final money, the competition score

Extras over kaggle_environments:
  * shops=[...] pins the town's shop sequence (e.g. to a recorded game's). The engine still draws its RNG, so weed
    spawns stay exactly as in the real engine; only the drawn shop is replaced.
  * log=True records what the engine actually executed each step (Game.last_exec): per seat, every unit's action
    and whether it changed anything, executed market units per (op, item), hires and land purchases.
"""
import copy
import json
import os
import random
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(HERE, "kaggriculture_v1327.py")
SPEC = json.load(open(os.path.join(HERE, "kaggriculture.json")))
DEFAULT_CONFIG = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in SPEC["configuration"].items()}
EPISODE_STEPS = int(DEFAULT_CONFIG["episodeSteps"])

_IMPORT = "from kaggle_environments.utils import resolve_episode_seed"
# Exact copy of kaggle_environments.utils.resolve_episode_seed (1.32.7).
_SEED_SRC = '''
def resolve_episode_seed(env, *, config_key="seed", fallback=None):
    if not hasattr(env, "info") or env.info is None:
        env.info = {}
    seed = env.info.get("seed")
    config = env.configuration
    if seed is None:
        seed = getattr(config, config_key, None)
        if seed is None and isinstance(config, dict):
            seed = config.get(config_key)
    if seed is None:
        seed = fallback() if fallback is not None else random.randrange(2**31)
    try:
        setattr(config, config_key, None)
    except (AttributeError, TypeError):
        config[config_key] = None
    env.info["seed"] = seed
    return seed
'''


class Struct(dict):
    """dict with attribute access, like the Struct kaggle_environments hands to agents and the engine."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __deepcopy__(self, memo):
        return Struct({k: copy.deepcopy(v, memo) for k, v in self.items()})


def structify(x):
    if isinstance(x, dict):
        return Struct({k: structify(v) for k, v in x.items()})
    if isinstance(x, list):
        return [structify(v) for v in x]
    return x


def plain(x):
    return json.loads(json.dumps(x))


def load_engine():
    """A fresh namespace holding the official engine, with optional pinning / execution-log hooks installed."""
    src = open(ENGINE_PATH, encoding="utf-8").read()
    assert src.count(_IMPORT) == 1, "unexpected engine source"
    src = src.replace(_IMPORT, _SEED_SRC)
    ns = {"__name__": "kagri_engine_%x" % random.getrandbits(32), "__file__": ENGINE_PATH}
    exec(compile(src, ENGINE_PATH, "exec"), ns)
    _install_hooks(ns)
    return ns


def _install_hooks(E):
    """Wrap engine functions. Hooks read per-game settings from env.kagri (None = behave exactly as the engine)."""
    orig_eod, orig_unit, orig_commit = E["_end_of_day"], E["_apply_unit_action"], E["_commit_unit"]
    orig_hire, orig_land = E["_do_hire"], E["_do_buy_land"]
    ctx = E["__kagri_ctx__"] = {"g": None}

    def end_of_day(state, env, day):
        n0 = len(state[0].observation.town["unlocked_shops"])
        orig_eod(state, env, day)
        g = ctx["g"]
        if g is not None and g.pinned_shops is not None:
            town = state[0].observation.town["unlocked_shops"]
            for i in range(n0, len(town)):
                if i < len(g.pinned_shops):
                    town[i] = g.pinned_shops[i]

    def apply_unit(farm, private, idx, action, *a, **k):
        g = ctx["g"]
        if g is None or g.exec_log is None:
            return orig_unit(farm, private, idx, action, *a, **k)
        p = g._seat_of(farm)
        before = _unit_snapshot(farm, private, idx)
        orig_unit(farm, private, idx, action, *a, **k)
        g.exec_log[p]["units"].append((action, _unit_snapshot(farm, private, idx) != before))

    def commit(op, item, price, farm, private, market, *a, **k):
        ok = orig_commit(op, item, price, farm, private, market, *a, **k)
        g = ctx["g"]
        if ok and g is not None and g.exec_log is not None:
            m = g.exec_log[g._seat_of(farm)]["market"]
            key = (op, item)
            m[key] = m.get(key, 0) + 1
            g.exec_log[g._seat_of(farm)]["cash"].append((op, item, price))
        return ok

    def hire(farm, private, *a, **k):
        n0 = farm["hires_today"]
        orig_hire(farm, private, *a, **k)
        g = ctx["g"]
        if g is not None and g.exec_log is not None:
            g.exec_log[g._seat_of(farm)]["hires"] += farm["hires_today"] - n0

    def land(farm, *a, **k):
        n0 = len(farm["unlocked_quadrants"])
        orig_land(farm, *a, **k)
        g = ctx["g"]
        if g is not None and g.exec_log is not None:
            g.exec_log[g._seat_of(farm)]["land"] += len(farm["unlocked_quadrants"]) - n0

    E["_end_of_day"], E["_apply_unit_action"], E["_commit_unit"] = end_of_day, apply_unit, commit
    E["_do_hire"], E["_do_buy_land"] = hire, land


def _unit_snapshot(farm, private, idx):
    pos = farm["farmer"] if idx == 0 else (farm["hands"][idx - 1] if idx - 1 < len(farm["hands"]) else None)
    inv = private["inventories"][idx] if idx < len(private["inventories"]) else None
    tile = farm["tiles"][pos[1]][pos[0]] if pos is not None else None
    return json.dumps([pos, inv, private["shed"], private["seeds"], tile], sort_keys=True)


_ENGINE = None


def shared_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = load_engine()
    return _ENGINE


class Game:
    """One episode on the official engine. Agents see exactly the observation kaggle_environments builds."""

    def __init__(self, seed, config=None, shops=None, log=False, engine=None):
        self.E = engine or shared_engine()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        cfg["seed"] = int(seed)
        self.cfg = Struct(cfg)
        self.seed = int(seed)
        self.pinned_shops = list(shops) if shops is not None else None
        self.env = types.SimpleNamespace(configuration=self.cfg, info={}, done=False)
        self.state = [types.SimpleNamespace(observation=Struct(), action=None, status="ACTIVE", reward=0)
                      for _ in range(2)]
        self.exec_log = None
        self.last_exec = None
        self._log = bool(log)
        self._call(self.E["interpreter"])
        for s in self.state:
            s.observation.step = 0
            s.observation.remainingOverageTime = 60
        self.step_no = 0
        self.done = False
        self.agent_config = Struct({k: v for k, v in cfg.items() if k != "seed"})
        self.agent_config.seed = None

    def _call(self, fn):
        ctx = self.E["__kagri_ctx__"]
        ctx["g"] = self
        try:
            fn(self.state, self.env)
        finally:
            ctx["g"] = None

    def _seat_of(self, farm):
        farms = self.state[0].observation.farms
        return 0 if farm is farms[0] else 1

    def observe(self, player):
        """Agent observation for `player` (a private deep copy: agents cannot corrupt the game)."""
        o = self.state[player].observation
        return structify({"player": player, "farms": plain(o.farms), "private": plain(o.private),
                          "market": plain(o.market), "town": plain(o.town), "day": o.day, "hour": o.hour,
                          "step": self.step_no, "remainingOverageTime": o.get("remainingOverageTime", 60)})

    def step(self, actions):
        """Advance one turn with [action_0, action_1] (Kaggle action dicts)."""
        assert not self.done, "episode finished"
        for i in (0, 1):
            a = actions[i]
            self.state[i].action = plain(a) if isinstance(a, dict) else {}
        if self._log:
            self.exec_log = [{"units": [], "market": {}, "cash": [], "hires": 0, "land": 0} for _ in (0, 1)]
        self._call(self.E["interpreter"])
        self.last_exec, self.exec_log = self.exec_log, None
        self.step_no += 1
        for s in self.state:
            s.observation.step = self.step_no
        if self.step_no >= int(self.cfg["episodeSteps"]) - 1:
            self.done = True

    def money(self, player):
        return float(self.state[0].observation.farms[player]["money"])

    def rewards(self):
        return [self.money(0), self.money(1)]

    def shops(self):
        return list(self.state[0].observation.town["unlocked_shops"])

    def public_state(self):
        o = self.state[0].observation
        return {"farms": o.farms, "market": o.market, "town": o.town}

    def private_state(self, player):
        return self.state[player].observation.private


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def play(agents, seed, shops=None, config=None, on_step=None):
    """Play a full game with two callables agent(obs, config) -> action. Returns (rewards, errors per seat).
    An agent that raises plays PASS for that turn (and the error is counted)."""
    g = Game(seed, config=config, shops=shops)
    errors = [0, 0]
    while not g.done:
        acts = []
        for p in (0, 1):
            try:
                a = agents[p](g.observe(p), g.agent_config)
            except Exception:
                errors[p] += 1
                a = PASS
            acts.append(a if isinstance(a, dict) else PASS)
        if on_step is not None:
            on_step(g, acts)
        g.step(acts)
    return g.rewards(), errors
