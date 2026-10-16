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
