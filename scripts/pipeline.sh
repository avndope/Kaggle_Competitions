#!/usr/bin/env bash
# End-to-end run: data -> simulation checks -> pretrain -> specialise -> agents -> comparison with the top replays.
#
#   bash scripts/pipeline.sh                 # everything (several hours on an RTX 3050)
#   STEPS=8000 bash scripts/pipeline.sh      # a shorter pretrain
#   TEACHERS="M & M & P & Q|DSM" bash scripts/pipeline.sh
#
# Environment: DAYS (download range), PROCS (CPU workers), STEPS (pretrain steps), TEACHERS ('|'-separated),
# PANEL (h2h opponents as NAME=PATH, space-separated), DATE (date bucket the agents condition on; 57 = 2026-10-13).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAYS=${DAYS:-2026-10-01:2026-10-13}
PROCS=${PROCS:-12}
STEPS=${STEPS:-40000}
DATE=${DATE:-57}
TEACHERS=${TEACHERS:-"M & M & P & Q|DSM|Victor @ Tufa Labs"}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "== 1. data ($DAYS)"
$PY scripts/download_episodes.py --days "$DAYS"
$PY scripts/build_dataset.py --zips data/raw/*.zip --out data/bc --procs "$PROCS"

echo "== 2. simulation checks"
$PY tests/test_simulation.py
$PY -m kagri.eval.takeover --agent tape --team "M & M & P & Q" --census data/bc/census.jsonl --zips data/raw/*.zip \
    --n 48 --procs "$PROCS" --out results/takeover_tape.jsonl

echo "== 3. pretrain ($STEPS steps)"
$PY kagri/bc/train.py --config configs/pretrain.yaml --run pretrain --steps "$STEPS"

IFS='|' read -ra TS <<< "$TEACHERS"
for T in "${TS[@]}"; do
  slug=$(echo "$T" | tr -c 'A-Za-z0-9' '_' | tr -s '_' | sed 's/_$//')
  echo "== 4. teacher: $T ($slug)"
  # conditioned agent straight from the pretrained model (final EMA weights: choose by games, not validation loss)
  $PY scripts/build_agent.py runs/pretrain/last.pt "agents/pre_$slug" --team "$T" --date "$DATE"
  # specialised agent: fine-tune on this team's seats only
  $PY scripts/build_dataset.py --zips data/raw/*.zip --out "data/bc_spec_$slug" --train-teams "$T" \
      --vocab data/bc/vocab.json --procs "$PROCS"
  sed -e "s#data/bc_spec/census.jsonl#data/bc_spec_$slug/census.jsonl#" -e "s#\"M & M & P & Q\"#\"$T\"#" \
      -e "s#^teacher: .*#teacher: null#" configs/specialize.yaml > "configs/specialize_$slug.yaml"
  $PY kagri/bc/train.py --config "configs/specialize_$slug.yaml" --run "spec_$slug"
  $PY scripts/build_agent.py "runs/spec_$slug/best.pt" "agents/spec_$slug" --team "$T" --date "$DATE"
  $PY scripts/build_agent.py "runs/spec_$slug/last.pt" "agents/speclast_$slug" --team "$T" --date "$DATE"
  echo "== 5. compare with the top replays: $T's own held-out seats (the games decide which checkpoint to keep)"
  for A in "pre_$slug" "spec_$slug" "speclast_$slug"; do
    $PY -m kagri.eval.takeover --agent "agents/$A" --team "$T" --census data/bc/census.jsonl --zips data/raw/*.zip \
        --held-out --n 60 --procs "$PROCS" --out "results/takeover_$A.jsonl"
  done
done

if [ -n "${PANEL:-}" ]; then
  echo "== 6. closed-loop head-to-head"
  for A in agents/spec_*; do
    # shellcheck disable=SC2086
    $PY -m kagri.eval.h2h --cand "$A" --panel $PANEL --seeds 24 --procs "$PROCS" --out "results/h2h_$(basename "$A").jsonl"
  done
fi
