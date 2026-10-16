# kaggriculture-bc

A behaviour-cloning agent for Kaggle's [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture), built from
the competition's top solutions (see `learnings.md` in the competition folder). It runs locally on a laptop GPU,
plays on the verbatim official engine, and is evaluated against the top teams' recorded games.

## What is in it

| Part | Where | What it does |
|---|---|---|
| Engine | `kagri/engine/` | The unmodified official engine (kaggle-environments 1.32.7, Apache-2.0) run in a private namespace. `Game` plays seeded games, can pin a recorded shop sequence, and logs what the engine actually executed. |
| Data | `scripts/download_episodes.py`, `scripts/build_dataset.py`, `kagri/bc/resim.py` | Downloads Kaggle's daily top-episode datasets, re-simulates every game step by step (a game that does not reproduce its recording exactly is dropped), and labels each turn with what the engine **executed** (4th place) with ignored unit actions masked (Yuan800). Open-loop tapes are dropped (DECEM). |
| Model | `kagri/bc/model.py` | DECEM's entity Transformer (5th place, MIT): global, item, unit and tile tokens with 2-D axial RoPE; units decoded in turn with a memory of earlier units' tasks; market quantities as classes. Conditioned on team, date and outcome. |
| Training | `kagri/bc/train.py`, `configs/` | Pretrain on all top seats, then specialise on one team (DECEM, Arda, 4th place). EMA weights only. |
| Agent | `scripts/build_agent.py`, `kagri/bc/policy.py` | A single-file NumPy `main.py` (Kaggle-ready): legality masks that replay the turn, expected-quantity market accumulators, sells against the shed after this turn's deliveries, night-overflow sell and last-day return-and-drop. |
| Evaluation | `kagri/eval/` | **Seat takeover** on held-out recorded top games (our money as a share of the top player's, in its own seat); **closed-loop head-to-head** against a panel, seat-swapped, counted per seed, sign-tested, with our money and theirs reported separately. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy pyarrow pyyaml
```

A Kaggle access token is read from `~/.kaggle/access_token` (or `$KAGGLE_TOKEN_FILE`).

## Run

```bash
# 1. data: Kaggle's daily top games (~850 MB per day); 1-13 Oct are the top teams' fixed final agents
.venv/bin/python scripts/download_episodes.py --days 2026-10-01:2026-10-13
.venv/bin/python scripts/build_dataset.py --zips data/raw/*.zip --out data/bc --procs 12

# 2. simulation check: re-simulating the recorded games must reproduce them exactly
.venv/bin/python -m kagri.eval.takeover --agent tape --team "M & M & P & Q" --census data/bc/census.jsonl \
    --zips data/raw/*.zip --n 40 --out results/takeover_tape.jsonl

# 3. train (RTX 3050 4 GB: ~3.3M parameters)
.venv/bin/python kagri/bc/train.py --config configs/pretrain.yaml --run pretrain
.venv/bin/python scripts/build_dataset.py --zips data/raw/*.zip --out data/bc_spec --train-teams "M & M & P & Q" \
    --vocab data/bc/vocab.json
.venv/bin/python kagri/bc/train.py --config configs/specialize.yaml --run specialize

# 4. agent
.venv/bin/python scripts/build_agent.py runs/specialize/best.pt agents/v1 --team "M & M & P & Q" --date 57

# 5. compare with the top replays, then closed loop
.venv/bin/python -m kagri.eval.takeover --agent agents/v1 --team "M & M & P & Q" --census data/bc/census.jsonl \
    --zips data/raw/*.zip --held-out --n 60 --out results/takeover_v1.jsonl
.venv/bin/python -m kagri.eval.h2h --cand agents/v1 --panel v16=/path/to/b_v16a hl0927=/path/to/hl0927.py \
    --seeds 24 --out results/h2h_v1.jsonl
```

## Results (first full run, 16 Oct 2026)

All numbers come from this repo's tools on a laptop (RTX 3050 4 GB, 16 cores).

**Simulation checks**
- `kagri/engine/kaggriculture_v1327.py` is byte-identical to kaggle-environments 1.32.7's engine.
- 6,444 of 6,449 top games (1-13 Oct) reproduce their recording at every step: shared state, both private states
  and final money. The other 5 had error statuses.
- Replaying the top players' own actions in the seat-takeover tool reproduces 204 / 204 held-out games to the dollar.

**Training:** 3.79M-parameter model, 40k steps × batch 192 (3.9 h). 94.0% of held-out unit actions and 57% of
whole turns match the recorded top players exactly.

**Seat takeover** (`kagri/eval/takeover.py`): the agent, conditioned on a team, plays that team's own held-out
seats with the opponent's recorded actions and the shops pinned. "Money share" is our final money divided by
what the team really made in the same seat.

| Seat taken | Games | Money share | Our win rate | Team's real win rate |
|---|---|---|---|---|
| M & M & P & Q (1st) | 48 | 1.026 ± 0.020 | 50.0% | 66.7% |
| DSM (3rd) | 53 | 1.001 ± 0.004 | 62.3% | 52.8% |
| Victor @ Tufa Labs (2nd) | 49 | 1.020 ± 0.009 | 55.1% | 40.8% |
| DECEM (5th) | 54 | 1.008 ± 0.008 | 72.2% | 48.1% |
| *our competition agent (v16), in the M & M & P & Q seats* | 48 | 0.910 ± 0.029 | 12.5% | 66.7% |

The pinned opponent cannot react, so read these as a screen; the closed-loop games below are the gate.

**Closed-loop head-to-head** (`kagri/eval/h2h.py`, every seed played with both seat orders):

| Agent | Opponent | Seeds won | Margin per game |
|---|---|---|---|
| conditioned on DSM | our competition agent (v16) | 16 / 16 | +$35,537 ± 1,079 |
| 10k-step checkpoint, conditioned on M & M & P & Q | v16 | 24 / 24 | +$27,708 ± 897 |
| 10k-step checkpoint, conditioned on M & M & P & Q | hl0927 (the most-copied public agent) | 24 / 24 | +$27,651 ± 927 |

**Which top team's style is strongest when both sides react?** A closed-loop round robin of the pretrained model
conditioned on different teams (row agent's win rate, 16 seeds × both seat orders):

| Style copied | vs 1st (M & M & P & Q) | vs 2nd (Victor) | vs 3rd (DSM) | vs 5th (DECEM) | Average |
|---|---|---|---|---|---|
| M & M & P & Q | — | 44% | 78% | 84% | **69%** |
| Victor @ Tufa Labs | 56% (n.s., +$734 ± 322) | — | 59% | 78% | **65%** |
| DSM | 22% (p 0.035) | 41% | — | 75% (p 0.04) | 46% |
| DECEM | 16% (p 0.007) | 22% (p 0.035) | 25% | — | 21% |

The order matches the real final leaderboard (1st, 2nd, 3rd, 5th): conditioning on a team reproduces that team's
relative strength.

Specialising on one team (3k extra steps on M & M & P & Q's seats) changed nothing measurable over conditioning the
pretrained model on that team (takeover flips +5 / −6).

Not included in the repo: the replay data (~10 GB of Kaggle zips, 1.4 GB of caches) and the trained weights; the
commands above rebuild both.

## Data notes

- Kaggle's daily datasets `kaggle/kaggriculture-episodes-YYYY-MM-DD` hold ~540 of the day's highest-rated games
  (median average agent rating ~2,850-3,030 in October), ~850 MB zipped. 2026-10-11 is missing on Kaggle (404).
- 2-6 Oct ran on a runtime labelled kaggle-environments 1.33.0, a version-number-only bump of the same engine
  (Kaggle staff, discussion 745445). Those games are kept only when they reproduce exactly on the 1.32.7 engine.
- Games where a side ends below $5,000 are dropped: in that window some agents crashed on a version check.
- Every kept game reproduces its recording at every step. The top teams request far more than executes: on
  1, 2 and 13 Oct they asked to sell 257M units and 7.7M sold, so labels are taken from the execution record.

## Credits and licences

- Engine: `kagri/engine/kaggriculture_v1327.py` and `kaggriculture.json` are unmodified files from
  [kaggle-environments](https://github.com/Kaggle/kaggle-environments) 1.32.7, Apache License 2.0
  (`kagri/engine/LICENSE-kaggle-environments`).
- Model, features, labels, policy, training loop and agent builder are adapted from DECEM's
  [kaggriculture-bc](https://github.com/Decem-Y/kaggriculture-bc) (5th place), MIT License
  (`third_party/LICENSE-DECEM-kaggriculture-bc`). Every adapted file says so in its header; our changes are marked
  `kagri:`.
- Method choices follow the public write-ups of the top teams; the sources are listed in `learnings.md`.
