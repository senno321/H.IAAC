#!/usr/bin/env bash
# FedCS dynamic pruning sweep with two schedules and two seeds.
# Schedules: (10,50) and (30,70) | Seeds: 2,3 | Clients: 100 | Rounds: 100
#
# Usage:
#   ./run_exp/run_fedcs_dynamic_prune_sweep.sh [federation] [--skip-setup] [--dry-run] [--pf=VALUE] [--pl=VALUE]
#
# Examples:
#   ./run_exp/run_fedcs_dynamic_prune_sweep.sh
#   ./run_exp/run_fedcs_dynamic_prune_sweep.sh gpu-sim-lrc
#   ./run_exp/run_fedcs_dynamic_prune_sweep.sh --dry-run

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${HOME:-}" ] || [ ! -w "${HOME:-/nonexistent}" ]; then
  export HOME="$ROOT/.runhome"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CONFIG_HOME/matplotlib}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

FED="local-simulation-100"
SKIP_SETUP=false
DRY_RUN=false
PF="0.5"
PL="0.2"
for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=true ;;
    --dry-run)    DRY_RUN=true ;;
    --pf=*)       PF="${arg#*=}" ;;
    --pl=*)       PL="${arg#*=}" ;;
  esac
done
[[ -n "$1" && "$1" != --* ]] && FED="$1"

echo "=== FedCS dynamic pruning sweep ==="
echo "Federation: $FED"
echo "Schedules: 10,50 and 30,70"
echo "Seeds: 2, 3"
echo "FedCS pf=$PF | pl=$PL"
echo ""

if [ "$SKIP_SETUP" = false ] && [ "$DRY_RUN" = false ]; then
  echo ">> Setting num-clients=100 for model/profile generation..."
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak && HAS_BAK=true
    sed -i 's/^num-clients = .*/num-clients = 100/' pyproject.toml || true
  fi

  for SEED in 2 3; do
    echo ">> Creating model (seed=$SEED)..."
    PYTHONPATH=. python gen_profile/gen_sim_model.py \
      --config_file ./pyproject.toml \
      --seed "$SEED" \
      --name "Mobilenet_v2" \
      --sel "fedcs_dynamic" \
      --agg "fedavg" \
      --input-shape "(3,224,224)" \
      --num-classes "10"

    echo ">> Creating profiles (seed=$SEED, 100 clients)..."
    PYTHONPATH=. python gen_profile/gen_sim_profile.py --config_file ./pyproject.toml --seed $SEED
  done

  if [ "$HAS_BAK" = true ] && [ -f pyproject.toml.bak ]; then
    mv pyproject.toml.bak pyproject.toml
  fi
  echo ""
fi

ALPHA=0.1
MODEL="Mobilenet_v2"
INPUT_SHAPE="(3,224,224)"
NUM_CLASSES=10
N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10

for SEED in 2 3; do
  for SCHEDULE in "10,50" "30,70"; do
    echo "=== SEED=$SEED prune-rounds=$SCHEDULE ==="
    RUN_CONFIG="seed=$SEED num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" participants-name=\"constant\" aggregation-name=\"fedavg\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES pretrain-rounds=3 prune-rounds=\"$SCHEDULE\" pf=$PF pl=$PL batch-size=64 epochs=10"
    if [ "$DRY_RUN" = true ]; then
      echo "flwr run . $FED --run-config=\"$RUN_CONFIG\""
    else
      flwr run . "$FED" --run-config="$RUN_CONFIG"
    fi
    echo ""
  done
done

echo "=== Sweep completo! Total: 4 experimentos (2 seeds × 2 schedules) ==="
